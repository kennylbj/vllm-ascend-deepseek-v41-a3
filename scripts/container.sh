#!/usr/bin/env bash
# =============================================================================
# container.sh — manage the long-lived vLLM Ascend container (DSV41_RUNTIME=docker)
#
# Why: the rootfs runtime runs the image's binaries directly on the host and
# therefore needs host glibc >= the image's (Ubuntu 22.04 → 2.35). On older hosts
# (e.g. Kylin V10, glibc 2.28) the service must run inside a container created
# from the locally loaded image — no registry access needed.
#
# The container is an idle shell host (`sleep infinity`, --init). It shares the
# host's network, IPC and PID namespaces, so:
#   * ports are plain host ports (8021 router, 8020 supervisor, 8022/8023 ranks)
#   * host-side tools (watchdog.sh, kill_vllm.sh, doctor.sh) see the container's
#     processes in /proc under the same PIDs
# Only the parts that need the image's Python / sources (serve.sh start path,
# apply_patches.sh) run inside, through `docker exec`.
#
# Usage:
#   bash scripts/container.sh up          # create/start (idempotent) + apply patches
#   bash scripts/container.sh status
#   bash scripts/container.sh exec [--] <cmd...>   # run a command inside
#   bash scripts/container.sh shell
#   bash scripts/container.sh down         # stop the container (kills vLLM inside!)
#   bash scripts/container.sh recreate     # rm + up (after changing mounts/args)
#   bash scripts/container.sh print-cmd    # show the docker run command, run nothing
#   bash scripts/container.sh print-args   # same, one raw argument per line
#
# Settings (env or config/dsv41.local.env):
#   DSV41_IMAGE=quay.io/ascend/vllm-ascend:v0.30.0rc2-a3   DSV41_CONTAINER=dsv41
#   DSV41_STATE_DIR      host dir for /root/.cache (compile caches) and /root/ascend (CANN logs)
#   DSV41_DOCKER_PRIVILEGED=1   0 → explicit --device list instead of --privileged
#   DSV41_DOCKER_EXTRA_ARGS     extra `docker run` words (no spaces inside a value)
#   DSV41_DOCKER_EXTRA_MOUNTS   extra "-v" specs, space separated (host:container[:opts])
#   DSV41_CONTAINER_SOURCE_ENV=1  wrap `exec` with the image's set_env.sh chain
#   DSV41_SKIP_PATCHES=1        `up` does not run apply_patches.sh
#   DSV41_HOST_ROOT      prefix for host paths under /usr/local, /etc, /dev (tests only)
# =============================================================================
set -Eeuo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
. "$SELF_DIR/common.sh"

: "${DSV41_DOCKER_BIN:=docker}"
: "${DSV41_DOCKER_PRIVILEGED:=1}"
: "${DSV41_DOCKER_EXTRA_ARGS:=}"
: "${DSV41_DOCKER_EXTRA_MOUNTS:=}"
: "${DSV41_CONTAINER_SOURCE_ENV:=0}"
: "${DSV41_SKIP_PATCHES:=0}"
: "${DSV41_HOST_ROOT:=}"
# vllm-ascend commit the patch series was written against (informational check).
: "${DSV41_EXPECT_VLLM_ASCEND_COMMIT:=8fde6d4e3}"

HR="$DSV41_HOST_ROOT"
RUN_ARGS=()
MOUNTED_SAME_PATH=" "
MISSING_REQUIRED=0

die()  { echo "[container] ❌ $*" >&2; exit 1; }
warn() { echo "[container] ⚠ $*" >&2; }
info() { echo "[container] $*"; }

need_docker() {
  command -v "$DSV41_DOCKER_BIN" >/dev/null 2>&1 || die "找不到 $DSV41_DOCKER_BIN 命令"
  "$DSV41_DOCKER_BIN" info >/dev/null 2>&1 || die "docker daemon 不可用（systemctl status docker）"
}

container_state() {
  "$DSV41_DOCKER_BIN" inspect -f '{{.State.Status}}' "$DSV41_CONTAINER" 2>/dev/null || true
}

# ---- docker run argument assembly ------------------------------------------

# Mount a host directory at the same absolute path (creating it when needed),
# unless an enclosing directory is already mounted at the same path.
mount_same_path() {
  local dir="$1" opts="${2:-}" m
  for m in $MOUNTED_SAME_PATH; do
    case "$dir/" in "$m"/*) return 0 ;; esac
  done
  mkdir -p "$dir"
  RUN_ARGS+=(-v "$dir:$dir${opts:+:$opts}")
  MOUNTED_SAME_PATH="$MOUNTED_SAME_PATH$dir "
}

# Mount a host path (usually below a Ascend install dir) when it exists.
# $4 = "required" marks mounts without which the NPU cannot work.
mount_host() {
  local src="$1" dst="$2" opts="${3:-}" kind="${4:-optional}"
  if [ ! -e "$src" ]; then
    if [ "$kind" = "required" ]; then
      MISSING_REQUIRED=1
      warn "缺少必需的挂载源: $src"
    else
      warn "跳过不存在的挂载源: $src"
    fi
    return 0
  fi
  RUN_ARGS+=(-v "$src:$dst${opts:+:$opts}")
}

build_run_args() {
  local d m
  RUN_ARGS=(run -d --name "$DSV41_CONTAINER" --init --restart unless-stopped)

  if [ "$DSV41_DOCKER_PRIVILEGED" = "1" ]; then
    RUN_ARGS+=(--privileged)
  else
    for d in "$HR"/dev/davinci[0-9]* "$HR/dev/davinci_manager" "$HR/dev/devmm_svm" "$HR/dev/hisi_hdc"; do
      [ -e "$d" ] && RUN_ARGS+=(--device "$d")
    done
    RUN_ARGS+=(--security-opt seccomp=unconfined --security-opt label=disable
               --cap-add SYS_NICE --cap-add IPC_LOCK)
  fi

  RUN_ARGS+=(--net=host --ipc=host --pid=host
             --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576
             -w "$DSV41_PROJECT_ROOT"
             -e DSV41_IN_CONTAINER=1 -e DSV41_ROOTFS=/)

  # Same-path mounts: pid files, logs and the model resolve identically on the
  # host and in the container (the compile cache key depends on the model path).
  mount_same_path "$DSV41_PROJECT_ROOT"
  case "$DSV41_PID_DIR" in
    /|/tmp) die "docker 运行时 DSV41_PID_DIR 不能是 '$DSV41_PID_DIR'（会把宿主 $DSV41_PID_DIR 整个挂进容器）；请设置专用目录，如 /run/dsv41" ;;
  esac
  mount_same_path "$DSV41_PID_DIR"
  mount_same_path "$DSV41_RESULTS"
  mount_same_path "$DSV41_LOG_DIR"
  if [ -d "$DSV41_MODEL" ]; then
    mount_same_path "$DSV41_MODEL" ro
  else
    warn "模型目录不存在: $DSV41_MODEL"
  fi

  mkdir -p "$DSV41_STATE_DIR/cache" "$DSV41_STATE_DIR/ascend"
  RUN_ARGS+=(-v "$DSV41_STATE_DIR/cache:/root/.cache" -v "$DSV41_STATE_DIR/ascend:/root/ascend")

  # Host Ascend driver user-space, as in the official vllm-ascend docker recipe.
  mount_host "$HR/usr/local/Ascend/driver/lib64" /usr/local/Ascend/driver/lib64 "" required
  mount_host "$HR/usr/local/Ascend/driver/version.info" /usr/local/Ascend/driver/version.info
  mount_host "$HR/usr/local/Ascend/driver/tools/hccn_tool" /usr/local/Ascend/driver/tools/hccn_tool
  mount_host "$HR/usr/local/dcmi" /usr/local/dcmi
  mount_host "$HR/usr/local/bin/npu-smi" /usr/local/bin/npu-smi "" required
  mount_host "$HR/etc/ascend_install.info" /etc/ascend_install.info
  mount_host "$HR/etc/hccn.conf" /etc/hccn.conf
  mount_host "$HR/etc/localtime" /etc/localtime ro

  if [ -n "${DSV41_DOCKER_EXTRA_MOUNTS// /}" ]; then
    for m in $DSV41_DOCKER_EXTRA_MOUNTS; do RUN_ARGS+=(-v "$m"); done
  fi
  if [ -n "${DSV41_DOCKER_EXTRA_ARGS// /}" ]; then
    # shellcheck disable=SC2206
    local -a extra=( $DSV41_DOCKER_EXTRA_ARGS )
    RUN_ARGS+=("${extra[@]}")
  fi

  # The image entrypoint sources set_env.sh and then execs its argument; an idle
  # `sleep` is all we need — serve.sh builds its own environment.
  RUN_ARGS+=(--entrypoint sleep "$DSV41_IMAGE" infinity)
}

# ---- sub-commands -----------------------------------------------------------

cmd_print_cmd() {
  local a
  build_run_args
  printf '%s' "$DSV41_DOCKER_BIN"
  for a in "${RUN_ARGS[@]}"; do printf ' \\\n  %q' "$a"; done
  printf '\n'
}

cmd_print_args() {
  local a
  build_run_args
  for a in "${RUN_ARGS[@]}"; do printf '%s\n' "$a"; done
}

cmd_exec() {
  [ "${1:-}" = "--" ] && shift
  [ $# -gt 0 ] || die "用法: container.sh exec [--] <命令...>"
  need_docker
  [ "$(container_state)" = "running" ] || die "容器 $DSV41_CONTAINER 未运行；先执行: bash scripts/container.sh up"

  local -a args=(exec)
  local name
  # -it only for an interactive terminal, so Ctrl-C reaches `serve.sh run`
  # and background callers (watchdog, systemd) never block on stdin.
  if [ -t 0 ] && [ -t 1 ]; then args+=(-it); fi
  args+=(-w "$DSV41_PROJECT_ROOT")
  # Forward caller-side settings by name only (values stay out of `ps`). The
  # container has its own ROOTFS, PROC_TAG and Ascend environment.
  for name in $(compgen -e); do
    case "$name" in
      DSV41_ROOTFS|DSV41_IN_CONTAINER|DSV41_PROC_TAG|DSV41_HOST_ROOT) ;;
      DSV41_*|VLLM_*|HCCL_*|ROUTER_RESTART) args+=(-e "$name") ;;
    esac
  done

  if [ "$DSV41_CONTAINER_SOURCE_ENV" = "1" ]; then
    exec "$DSV41_DOCKER_BIN" "${args[@]}" "$DSV41_CONTAINER" bash -c '
      source /usr/local/Ascend/ascend-toolkit/set_env.sh &&
      source /usr/local/Ascend/cann-${CANN_VERSION}/share/info/ascendnpu-ir/bin/set_env.sh &&
      source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=${ATB_CXX_ABI:-1} &&
      exec "$@"' dsv41-exec "$@"
  fi
  exec "$DSV41_DOCKER_BIN" "${args[@]}" "$DSV41_CONTAINER" "$@"
}

cmd_shell() {
  cmd_exec -- bash
}

cmd_up() {
  need_docker
  "$DSV41_DOCKER_BIN" image inspect "$DSV41_IMAGE" >/dev/null 2>&1 \
    || die "本机没有镜像 $DSV41_IMAGE（离线环境请先 docker load -i <镜像.tar>）"
  [ -f "$DSV41_MODEL/config.json" ] || die "找不到模型: $DSV41_MODEL/config.json"

  local state
  state=$(container_state)
  case "$state" in
    running)
      info "容器 $DSV41_CONTAINER 已在运行"
      ;;
    "")
      build_run_args
      [ "$MISSING_REQUIRED" = "0" ] || die "缺少必需的 Ascend 驱动挂载源（见上方告警），无法创建容器"
      info "创建容器 $DSV41_CONTAINER (镜像 $DSV41_IMAGE)"
      "$DSV41_DOCKER_BIN" "${RUN_ARGS[@]}" >/dev/null
      ;;
    *)
      info "启动已存在的容器 $DSV41_CONTAINER (状态: $state)"
      "$DSV41_DOCKER_BIN" start "$DSV41_CONTAINER" >/dev/null
      ;;
  esac

  local i
  for i in $(seq 1 20); do
    "$DSV41_DOCKER_BIN" exec "$DSV41_CONTAINER" true >/dev/null 2>&1 && break
    sleep 1
  done
  "$DSV41_DOCKER_BIN" exec "$DSV41_CONTAINER" true >/dev/null 2>&1 || die "容器启动后不可 exec"

  local want have
  want=$("$DSV41_DOCKER_BIN" image inspect -f '{{.Id}}' "$DSV41_IMAGE" 2>/dev/null || true)
  have=$("$DSV41_DOCKER_BIN" inspect -f '{{.Image}}' "$DSV41_CONTAINER" 2>/dev/null || true)
  if [ -n "$want" ] && [ "$want" != "$have" ]; then
    warn "容器所用镜像与本机 $DSV41_IMAGE 不一致；如需切换请 recreate"
  fi

  local head
  head=$("$DSV41_DOCKER_BIN" exec "$DSV41_CONTAINER" git -C /vllm-workspace/vllm-ascend rev-parse HEAD 2>/dev/null || true)
  case "$head" in
    "$DSV41_EXPECT_VLLM_ASCEND_COMMIT"*) info "vllm-ascend 提交 ${head:0:9} 与补丁基线一致" ;;
    *) warn "容器内 vllm-ascend 提交为 '${head:-unknown}'，补丁基线为 $DSV41_EXPECT_VLLM_ASCEND_COMMIT；apply_patches.sh 可能冲突" ;;
  esac

  if [ "$DSV41_SKIP_PATCHES" != "1" ]; then
    info "应用补丁（幂等）"
    "$DSV41_DOCKER_BIN" exec -e DSV41_ROOTFS=/ -w "$DSV41_PROJECT_ROOT" "$DSV41_CONTAINER" \
      bash "$SELF_DIR/apply_patches.sh" apply
  fi
  info "✅ 容器就绪: $DSV41_CONTAINER"
}

cmd_down() {
  need_docker
  if [ -z "$(container_state)" ]; then
    info "容器 $DSV41_CONTAINER 不存在"
    return 0
  fi
  info "停止容器 $DSV41_CONTAINER（容器内的 vLLM / router 会随之终止；其它容器和进程不受影响）"
  "$DSV41_DOCKER_BIN" stop -t 30 "$DSV41_CONTAINER" >/dev/null
}

cmd_recreate() {
  need_docker
  if [ -n "$(container_state)" ]; then
    info "删除容器 $DSV41_CONTAINER"
    "$DSV41_DOCKER_BIN" rm -f "$DSV41_CONTAINER" >/dev/null
  fi
  cmd_up
}

cmd_status() {
  need_docker
  local state
  state=$(container_state)
  if [ -z "$state" ]; then
    echo "container: 不存在 ($DSV41_CONTAINER)"
    return 0
  fi
  echo "container: $state ($DSV41_CONTAINER)"
  echo "image    : $("$DSV41_DOCKER_BIN" inspect -f '{{.Config.Image}}' "$DSV41_CONTAINER" 2>/dev/null)"
  echo "started  : $("$DSV41_DOCKER_BIN" inspect -f '{{.State.StartedAt}}' "$DSV41_CONTAINER" 2>/dev/null)"
}

case "${1:-status}" in
  up)         cmd_up ;;
  down)       cmd_down ;;
  recreate)   cmd_recreate ;;
  status)     cmd_status ;;
  exec)       shift; cmd_exec "$@" ;;
  shell)      cmd_shell ;;
  print-cmd)  cmd_print_cmd ;;
  print-args) cmd_print_args ;;
  -h|--help)  sed -n '2,39p' "$0" ;;
  *) echo "用法: $0 {up|down|recreate|status|exec|shell|print-cmd|print-args}" >&2; exit 1 ;;
esac
