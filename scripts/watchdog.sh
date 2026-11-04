#!/usr/bin/env bash
# =============================================================================
# watchdog.sh — health monitor and auto-restart for scripts/serve.sh
#
# Behaviour:
#   * Poll GET /health every CHECK_INTERVAL seconds (200 = healthy).
#   * Process alive but /health failing → wait GRACE_PERIOD seconds, then
#     restart (covers model loading and transient stalls).
#   * Process gone → restart immediately.
#   * Server dies during startup (missing model/rootfs, port conflict) →
#     fail fast instead of waiting the full STARTUP_TIMEOUT.
#   * Restart failing → cool down FAILURE_COOLDOWN seconds and retry forever.
#
# Usage:
#   bash scripts/watchdog.sh start     # background watchdog
#   bash scripts/watchdog.sh stop      # stop watchdog only (server keeps running)
#   bash scripts/watchdog.sh status
#   bash scripts/watchdog.sh restart
#   bash scripts/watchdog.sh run       # foreground (debug)
# =============================================================================
set -uo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
. "$SELF_DIR/common.sh"

dsv41_unset_proxy

# ---- tunables --------------------------------------------------------------
: "${DSV41_CHECK_INTERVAL:=30}"
: "${DSV41_HEALTH_TIMEOUT:=10}"
: "${DSV41_GRACE_PERIOD:=180}"
: "${DSV41_STARTUP_TIMEOUT:=3600}"
: "${DSV41_STARTUP_POLL:=15}"
: "${DSV41_STARTUP_FAIL_FAST:=120}"
: "${DSV41_FAILURE_COOLDOWN:=120}"
: "${DSV41_MAX_LOG_SIZE:=10485760}"
: "${DSV41_MAX_LOG_KEEP:=5}"

SELF=$(readlink -f "$0")
SERVE_SCRIPT="${DSV41_SERVE_SCRIPT:-$SELF_DIR/serve.sh}"
LOG_DIR="$DSV41_LOG_DIR"
LOG_FILE="${DSV41_WATCHDOG_LOG:-$LOG_DIR/watchdog.log}"
HEALTH_URL="http://127.0.0.1:${DSV41_PORT}/health"

mkdir -p "$LOG_DIR" "$DSV41_RESULTS" "$DSV41_PID_DIR"

# ---- logging ---------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

rotate_log() {
  [ -f "$LOG_FILE" ] || return 0
  local size
  size=$(stat -c %s "$LOG_FILE" 2>/dev/null || echo 0)
  [ "$size" -lt "$DSV41_MAX_LOG_SIZE" ] && return 0
  local i
  for i in $(seq $((DSV41_MAX_LOG_KEEP - 1)) -1 1); do
    [ -f "${LOG_FILE}.$i" ] && mv -f "${LOG_FILE}.$i" "${LOG_FILE}.$((i + 1))"
  done
  mv -f "$LOG_FILE" "${LOG_FILE}.1"
}

is_healthy() {
  curl -sf --noproxy '*' --max-time "$DSV41_HEALTH_TIMEOUT" -o /dev/null "$HEALTH_URL" 2>/dev/null
}

# ---- watchdog process discovery -------------------------------------------
# Works even when the pid file was lost (manual start, tmp cleanup, reboot).
find_watchdog_pid() {
  local pid="" p cmd
  if [ -f "$DSV41_WATCHDOG_PID_FILE" ]; then
    pid=$(cat "$DSV41_WATCHDOG_PID_FILE" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
      case "$cmd" in
        *"watchdog.sh"*"run"*) echo "$pid"; return 0 ;;
      esac
    fi
  fi
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    [ "$pid" = "$$" ] && continue
    [ -r "$p/cmdline" ] || continue
    cmd=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
    case "$cmd" in
      *"watchdog.sh"*"run"*) echo "$pid"; return 0 ;;
    esac
  done
  return 1
}

# ---- start / stop ----------------------------------------------------------
start_service() {
  if is_healthy; then
    log "服务已健康 ($HEALTH_URL), 跳过启动"
    return 0
  fi
  dsv41_cleanup_residual
  # Container runtime: make sure the container is up (idempotent; also
  # re-applies patches) before serve.sh hands over to it.
  if [ "$DSV41_RUNTIME" = "docker" ]; then
    if ! bash "$SELF_DIR/container.sh" up >> "$LOG_FILE" 2>&1; then
      log "容器未就绪 (container.sh up 失败)，见 $LOG_FILE"
      return 1
    fi
  fi
  log "启动服务: bash $SERVE_SCRIPT (端口 $DSV41_PORT)"
  nohup bash "$SERVE_SCRIPT" >> "$LOG_FILE" 2>&1 &

  local waited=0 pid no_process_since=0
  while [ "$waited" -lt "$DSV41_STARTUP_TIMEOUT" ]; do
    if is_healthy; then
      pid=$(dsv41_find_serve_pid || true)
      log "服务已就绪 (pid=${pid:-unknown}, 等待 ${waited}s)"
      return 0
    fi
    sleep "$DSV41_STARTUP_POLL"
    waited=$((waited + DSV41_STARTUP_POLL))
    rotate_log

    # Detect a deterministic startup failure quickly: the serve process for
    # this port exited. Do not look at global VLLM:: processes here: another
    # instance on the node would mask the failure.
    pid=$(dsv41_find_serve_pid || true)
    if [ -z "$pid" ]; then
      if [ "$no_process_since" -eq 0 ]; then
        no_process_since=$waited
      elif [ $((waited - no_process_since)) -ge "$DSV41_STARTUP_FAIL_FAST" ]; then
        log "启动失败：等待 ${waited}s 后仍无 vLLM 进程，请查看日志 $LOG_FILE"
        return 1
      fi
    else
      no_process_since=0
    fi

    if [ $((waited % 300)) -eq 0 ]; then
      log "等待服务就绪... ${waited}s / ${DSV41_STARTUP_TIMEOUT}s"
    fi
  done
  log "启动超时 (${DSV41_STARTUP_TIMEOUT}s), /health 仍不可用"
  return 1
}

stop_router() {
  local pid
  pid=$(dsv41_find_router_pid || true)
  if [ -n "$pid" ]; then
    log "停止 router pid=$pid"
    kill "$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$DSV41_ROUTER_PID_FILE"
}

stop_service() {
  local pid
  pid=$(dsv41_find_serve_pid || true)
  if [ -n "$pid" ]; then
    log "停止服务 pid=$pid"
    kill "$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 2
    done
  fi
  stop_router
  dsv41_cleanup_residual
}

restart_service() {
  stop_service
  start_service
}

# ---- main loop -------------------------------------------------------------
run_loop() {
  log "watchdog 启动 (port=$DSV41_PORT, health=$HEALTH_URL, interval=${DSV41_CHECK_INTERVAL}s, grace=${DSV41_GRACE_PERIOD}s)"
  local unhealthy_since=0 ever_healthy=0
  while true; do
    rotate_log
    if is_healthy; then
      [ "$unhealthy_since" -ne 0 ] && log "服务恢复健康"
      unhealthy_since=0
      ever_healthy=1
      sleep "$DSV41_CHECK_INTERVAL"
      continue
    fi

    local pid
    pid=$(dsv41_find_serve_pid || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      local age limit
      age=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
      limit=$DSV41_GRACE_PERIOD
      # A watchdog attached to an already-loading server must not kill it
      # before the normal cold-start window has elapsed.
      if [ "$ever_healthy" -eq 0 ] && [ -n "$age" ] && [ "$age" -lt "$DSV41_STARTUP_TIMEOUT" ]; then
        limit=$DSV41_STARTUP_TIMEOUT
      fi
      if [ "$unhealthy_since" -eq 0 ]; then
        unhealthy_since=$(date +%s)
        log "服务进程在 (pid=$pid, age=${age:-?}s) 但 /health 异常, 进入 ${limit}s 宽限"
      fi
      local elapsed=$(( $(date +%s) - unhealthy_since ))
      if [ "$elapsed" -lt "$limit" ]; then
        sleep "$DSV41_CHECK_INTERVAL"
        continue
      fi
      log "宽限期已过 (${elapsed}s), 重启服务"
    else
      log "服务进程不存在, 重启服务"
    fi

    unhealthy_since=0
    if restart_service; then
      log "重启成功"
    else
      log "重启失败, 冷却 ${DSV41_FAILURE_COOLDOWN}s 后重试"
      sleep "$DSV41_FAILURE_COOLDOWN"
    fi
  done
}

# ---- commands --------------------------------------------------------------
cmd_start() {
  local existing
  existing=$(find_watchdog_pid || true)
  if [ -n "$existing" ]; then
    echo "$existing" > "$DSV41_WATCHDOG_PID_FILE"
    echo "watchdog 已在运行 (pid=$existing)"
    return 0
  fi
  rm -f "$DSV41_WATCHDOG_PID_FILE"
  nohup bash "$SELF" run >> "$LOG_FILE" 2>&1 &
  echo $! > "$DSV41_WATCHDOG_PID_FILE"
  echo "watchdog 已启动 (pid=$!, log=$LOG_FILE)"
}

cmd_stop() {
  local pid
  pid=$(find_watchdog_pid || true)
  if [ -n "$pid" ]; then
    kill "$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 10); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
    echo "watchdog 已停止 (pid=$pid，不影响 vLLM 服务)"
  else
    echo "watchdog 未运行"
  fi
  rm -f "$DSV41_WATCHDOG_PID_FILE"
}

cmd_status() {
  local wd_pid="" serve_pid="" router_pid="" health="不可达"
  wd_pid=$(find_watchdog_pid || true)
  serve_pid=$(dsv41_find_serve_pid || true)
  router_pid=$(dsv41_find_router_pid || true)
  is_healthy && health="HTTP 200"
  if [ -n "$wd_pid" ]; then echo "watchdog : 运行中 pid=$wd_pid"; else echo "watchdog : 未运行"; fi
  if [ -n "$serve_pid" ]; then echo "vllm     : 运行中 pid=$serve_pid"; else echo "vllm     : 未运行"; fi
  if [ -n "$router_pid" ]; then echo "router   : 运行中 pid=$router_pid"; else echo "router   : 未运行"; fi
  echo "health   : $health ($HEALTH_URL)"
  if [ -n "$serve_pid" ]; then
    echo "uptime   : $(ps -o etime= -p "$serve_pid" 2>/dev/null | tr -d ' ')"
  fi
}

cmd_restart() {
  cmd_stop
  sleep 1
  cmd_start
  echo "等待服务健康..."
  local i
  for i in $(seq 1 120); do
    is_healthy && { echo "服务已就绪"; return 0; }
    sleep 5
  done
  echo "⚠ 服务尚未就绪，watchdog 会自动继续等待/重启（查看 $LOG_FILE）"
}

case "${1:-status}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  status)  cmd_status ;;
  restart) cmd_restart ;;
  run)     run_loop ;;
  *) echo "用法: $0 {start|stop|status|restart|run}"; exit 1 ;;
esac
