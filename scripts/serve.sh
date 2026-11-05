#!/usr/bin/env bash
# =============================================================================
# serve.sh — start / stop / inspect the DeepSeek-V4.1-Flash-1017-w8a8 server
#
# Production configuration baked in:
#   * vLLM Ascend v0.30.0rc2 (DSV41_RUNTIME=native rootfs, or =docker container)
#   * DP2 / TP8 on one 8-card A3 node (16 dies), expert parallel
#   * Model Runner V2 + DSpark speculative decoding (K=5)
#   * FULL_DECODE_ONLY ACL graph with 6N-sized buckets
#   * 360448-token context (DSV41_MAX_MODEL_LEN), prefix caching,
#     engram CPU offload / DP shared memory, prompt_tokens_details in usage
#   * optional Bearer-token authentication (--api-key)
#
# Usage:
#   bash scripts/serve.sh                 # start in background
#   bash scripts/serve.sh run             # start in foreground (debug)
#   STOP=1 bash scripts/serve.sh          # stop this deployment (never other NPU jobs)
#   STATUS=1 bash scripts/serve.sh        # show status / readiness
#
# With DSV41_RUNTIME=docker the start path re-executes itself inside the
# container (scripts/container.sh); STOP/STATUS always run on the host.
# Prefer scripts/watchdog.sh for unattended operation.
# =============================================================================
set -Eeuo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
. "$SELF_DIR/common.sh"

# ---- tunables (override via environment or config/*.env) --------------------
: "${DSV41_MAX_MODEL_LEN:=360448}"                # 2816 x 128 (block-aligned)
: "${DSV41_MAX_NUM_BATCHED_TOKENS:=4096}"        # DP2+EP lockstep: smaller prefill chunk = shorter decode stall
: "${DSV41_MAX_NUM_SEQS:=32}"
: "${DSV41_GPU_MEM_UTIL:=0.92}"
: "${DSV41_BLOCK_SIZE:=128}"
: "${DSV41_SPEC_TOKENS:=5}"
: "${DSV41_USE_V2:=1}"
: "${DSV41_ENABLE_PREFIX_CACHING:=1}"
: "${DSV41_ENABLE_EXPERT_PARALLEL:=1}"
: "${DSV41_ENABLE_PROMPT_TOKENS_DETAILS:=1}"    # usage.prompt_tokens_details.cached_tokens
: "${DSV41_OFFLINE:=1}"                         # never reach out to model hubs / usage-stat servers
: "${DSV41_SAFETENSORS_LOAD_STRATEGY:=lazy}"
: "${DSV41_LOADER_THREADS:=128}"
: "${DSV41_ENGINE_READY_TIMEOUT_S:=36000}"
: "${DSV41_HCCL_BUFFSIZE:=1024}"
: "${DSV41_HCCL_OP_EXPANSION_MODE:=AIV}"
: "${DSV41_TOKENIZER_MODE:=deepseek_v41}"
: "${DSV41_REASONING_PARSER:=deepseek_v41}"
: "${DSV41_TOOL_CALL_PARSER:=deepseek_v41}"
: "${DSV41_CUDAGRAPH_SIZES:=1,2,4,6,8,12,16,18,24,30,32,40,48,54,56,64,72,78,80,88,96,102,104,112,120,126,128,136,144,150,152,160,168,174,176,184,192}"
: "${DSV41_EXTRA_SERVE_ARGS:=}"

: "${DSV41_LB_MODE:=internal}"                 # internal | external
: "${DSV41_DP_BASE_PORT:=$((DSV41_PORT + 1))}"  # rank0 port (external mode)
: "${DSV41_DP_SUPERVISOR_PORT:=$((DSV41_PORT - 1))}"
: "${DSV41_KV_EVENTS:=1}"
: "${DSV41_KV_EVENTS_BASE_PORT:=5557}"
: "${DSV41_ROUTER_SCRIPT:=$SELF_DIR/prefix_router.py}"
: "${DSV41_ROUTER_LOG:=$DSV41_RESULTS/router.log}"
: "${DSV41_ROUTER_MAX_SKEW:=8}"                # legacy policy only
# cost (default): score ranks in tokens - uncached prompt + queued prefill + KV saturation.
# legacy: longest prefix wins unless the active-request gap exceeds DSV41_ROUTER_MAX_SKEW.
: "${DSV41_ROUTER_POLICY:=cost}"
: "${DSV41_ROUTER_KV_HIGH:=0.90}"               # KV usage (0-1) at which a rank counts as saturated
: "${DSV41_ROUTER_SAT_PENALTY:=32768}"          # tokens added to a saturated / queueing rank's score
: "${DSV41_ROUTER_PREFILL_WEIGHT:=0.5}"         # weight of prompt tokens already queued for prefill
: "${DSV41_ROUTER_LOAD_INTERVAL:=2}"            # seconds between /metrics scrapes (0 disables)

RUN_ID="dsv41w8a8rc2-$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$DSV41_RESULTS/serve_${RUN_ID}.log"

stop_router() {
  local pid
  pid=$(dsv41_find_router_pid || true)
  if [ -n "$pid" ]; then
    echo "[serve] 停止 router pid=$pid"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$DSV41_ROUTER_PID_FILE"
}

start_router() {
  local i
  local -a rargs=(--listen "$DSV41_PORT" --host "$DSV41_HOST" --model "$DSV41_SERVED_NAME"
    --block-size "$DSV41_BLOCK_SIZE" --max-skew "$DSV41_ROUTER_MAX_SKEW"
    --policy "$DSV41_ROUTER_POLICY" --kv-high "$DSV41_ROUTER_KV_HIGH"
    --sat-penalty "$DSV41_ROUTER_SAT_PENALTY" --prefill-weight "$DSV41_ROUTER_PREFILL_WEIGHT"
    --load-interval "$DSV41_ROUTER_LOAD_INTERVAL")
  for i in $(seq 0 $((DSV41_DP - 1))); do
    rargs+=(--upstream "http://127.0.0.1:$((DSV41_DP_BASE_PORT + i))")
    if [ "$DSV41_KV_EVENTS" = "1" ]; then
      rargs+=(--kv-events "tcp://127.0.0.1:$((DSV41_KV_EVENTS_BASE_PORT + i))")
    fi
  done
  if [ -f "$DSV41_ROUTER_LOG" ] && [ "$(stat -c%s "$DSV41_ROUTER_LOG" 2>/dev/null || echo 0)" -gt 10485760 ]; then
    mv -f "$DSV41_ROUTER_LOG" "$DSV41_ROUTER_LOG.1"
  fi
  echo "[serve] 启动 prefix-aware router: 0.0.0.0:$DSV41_PORT -> $(for i in $(seq 0 $((DSV41_DP - 1))); do printf 'rank%d=127.0.0.1:%d ' "$i" "$((DSV41_DP_BASE_PORT + i))"; done)"
  nohup "$PY" "$DSV41_ROUTER_SCRIPT" "${rargs[@]}" >> "$DSV41_ROUTER_LOG" 2>&1 &
  echo $! > "$DSV41_ROUTER_PID_FILE"
  echo "[serve] router pid=$(cat "$DSV41_ROUTER_PID_FILE") 日志=$DSV41_ROUTER_LOG"
}

# ---- stop / status ---------------------------------------------------------
if [ "${STOP:-0}" = "1" ]; then
  pid=$(dsv41_find_serve_pid || true)
  if [ -n "$pid" ]; then
    echo "[serve] 停止 pid=$pid"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 2
    done
    kill -9 "$pid" 2>/dev/null || true
  else
    echo "[serve] 无运行实例"
  fi
  stop_router
  dsv41_cleanup_residual
  echo "[serve] 已停"
  exit 0
fi

if [ "${STATUS:-0}" = "1" ]; then
  pid=$(dsv41_find_serve_pid || true)
  rpid=$(dsv41_find_router_pid || true)
  if [ -n "$pid" ]; then
    echo "[serve] vLLM/DP supervisor 运行中 pid=$pid ($(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' '))"
  else
    echo "[serve] vLLM 未运行"
  fi
  if [ "$DSV41_LB_MODE" = "external" ]; then
    if [ -n "$rpid" ]; then
      echo "[serve] router 运行中 pid=$rpid"
    else
      echo "[serve] router 未运行"
    fi
  fi
  curl -s --noproxy '*' --max-time 5 "http://127.0.0.1:$DSV41_PORT/health" \
    -o /dev/null -w "health: HTTP %{http_code}\n" 2>/dev/null || echo "health: 不可达"
  exit 0
fi

# ---- container runtime hand-off ----------------------------------------------
# Everything below needs the image's Python / CANN, so with DSV41_RUNTIME=docker
# it runs inside the container (no-op on the native runtime or when already inside).
dsv41_reexec_in_container "$SELF_DIR/serve.sh" "$@"

# ---- preflight -------------------------------------------------------------
dsv41_unset_proxy
mkdir -p "$DSV41_RESULTS" "$DSV41_PID_DIR"

# DSV41_ROOTFS=/ (container runtime) must not produce "//usr/local/..." paths.
RFS=${DSV41_ROOTFS%/}

PYTHON_HOME_DIR=${DSV41_PYTHONHOME:-$(find "$RFS/usr/local" -maxdepth 1 -type d -name 'python3.12*' 2>/dev/null | sort -V | tail -1)}
CANN=${DSV41_CANN_HOME:-$(find "$RFS/usr/local/Ascend" -maxdepth 1 -type d -name 'cann-*' 2>/dev/null | sort -V | tail -1)}
PY="$PYTHON_HOME_DIR/bin/python3.12"

if [ -z "$PYTHON_HOME_DIR" ] || [ ! -x "$PY" ]; then
  echo "[serve] ❌ 找不到 rootfs Python: ${PYTHON_HOME_DIR:-<none>}/bin/python3.12" >&2
  echo "[serve]    先运行: bash scripts/setup_rootfs.sh；或设置 DSV41_PYTHONHOME" >&2
  exit 1
fi
if [ -z "$CANN" ] || [ ! -d "$CANN" ]; then
  echo "[serve] ❌ 找不到 CANN 目录: ${CANN:-<none>}" >&2
  echo "[serve]    设置 DSV41_CANN_HOME 指向 rootfs 内的 cann-* 目录" >&2
  exit 1
fi
if [ ! -f "$DSV41_MODEL/config.json" ]; then
  echo "[serve] ❌ 找不到模型: $DSV41_MODEL/config.json" >&2
  echo "[serve]    设置 DSV41_MODEL 指向 DeepSeek-V4.1-Flash-1017-w8a8 权重目录" >&2
  exit 1
fi

# Refuse to silently start a second instance on the same node.
# (DSV41_PROBE=1 only imports the stack and exits, so it may run next to a live server.)
if [ "${DSV41_ALLOW_MULTI:-0}" != "1" ] && [ "${ROUTER_RESTART:-0}" != "1" ] && [ "${DSV41_PROBE:-0}" != "1" ]; then
  existing=$(dsv41_find_serve_pid || true)
  if [ -n "$existing" ]; then
    echo "[serve] ❌ 端口 $DSV41_PORT 上已有 vLLM 实例 (pid=$existing)" >&2
    echo "[serve]    先 STOP=1 bash scripts/serve.sh；确实要多实例请设 DSV41_ALLOW_MULTI=1" >&2
    exit 1
  fi
  if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -qE ":${DSV41_PORT}([^0-9]|$)"; then
    echo "[serve] ❌ 端口 $DSV41_PORT 已被其它进程占用:" >&2
    ss -ltnp 2>/dev/null | grep -E ":${DSV41_PORT}([^0-9]|$)" >&2 || true
    exit 1
  fi
  if ! command -v ss >/dev/null 2>&1 && command -v netstat >/dev/null 2>&1 \
     && netstat -tln 2>/dev/null | grep -qE ":${DSV41_PORT}([^0-9]|$)"; then
    echo "[serve] ❌ 端口 $DSV41_PORT 已被其它进程占用:" >&2
    netstat -tlnp 2>/dev/null | grep -E ":${DSV41_PORT}([^0-9]|$)" >&2 || true
    exit 1
  fi
fi

if [ "${DSV41_ALLOW_MULTI:-0}" != "1" ] && [ "${ROUTER_RESTART:-0}" != "1" ] && [ "${DSV41_PROBE:-0}" != "1" ] && [ "$DSV41_LB_MODE" = "external" ]; then
  for p in "$DSV41_DP_SUPERVISOR_PORT" $(seq "$DSV41_DP_BASE_PORT" $((DSV41_DP_BASE_PORT + DSV41_DP - 1))); do
    busy=0
    if command -v ss >/dev/null 2>&1; then
      ss -ltn 2>/dev/null | grep -qE ":${p}([^0-9]|$)" && busy=1
    elif command -v netstat >/dev/null 2>&1; then
      netstat -tln 2>/dev/null | grep -qE ":${p}([^0-9]|$)" && busy=1
    fi
    if [ "$busy" = 1 ]; then
      echo "[serve] ❌ external 模式端口 $p 已被占用" >&2
      exit 1
    fi
  done
fi

# Warn (do not silently disable auth) when the server would be unauthenticated.
if [ -z "$DSV41_API_KEY" ]; then
  echo "[serve] ⚠ 未设置 DSV41_API_KEY：服务将不带鉴权，请仅在可信网络内使用" >&2
fi

# ---- process environment ---------------------------------------------------
unset NPU_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES 2>/dev/null || true

export VLLM_ENGINE_READY_TIMEOUT_S
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE
export HCCL_OP_EXPANSION_MODE="$DSV41_HCCL_OP_EXPANSION_MODE"
[ "$DSV41_USE_V2" = "1" ] && export VLLM_USE_V2_MODEL_RUNNER=1

# Air-gapped nodes: the model is a local directory; stop transformers / hub /
# vLLM from probing the internet (each probe otherwise waits for a timeout).
if [ "$DSV41_OFFLINE" = "1" ]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
fi

# Tag every process launched below (server, workers, router). Cleanup and
# kill_vllm.sh only touch tagged processes, never other vLLM/NPU jobs on the node.
export DSV41_PROC_TAG="$DSV41_INSTANCE_ID"

# jemalloc reduces allocator contention during long decode loops.
JEMALLOC=${DSV41_JEMALLOC:-$RFS/usr/lib/aarch64-linux-gnu/libjemalloc.so.2}
if [ -f "$JEMALLOC" ]; then
  export LD_PRELOAD="$JEMALLOC${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# Optimization gates. They are also defaulted to 1 inside the patched tree
# (spawned workers do not inherit os.environ), but exporting them keeps the
# configuration self-documenting and works when running without patches.
export VLLM_ASCEND_V41_TRITON_SLOT_MAPPING="${VLLM_ASCEND_V41_TRITON_SLOT_MAPPING:-1}"
export VLLM_ASCEND_ROPE_INDEX_SELECT="${VLLM_ASCEND_ROPE_INDEX_SELECT:-1}"
export VLLM_ASCEND_V41_MOE_MASK_RANGE="${VLLM_ASCEND_V41_MOE_MASK_RANGE:-1}"
export VLLM_ASCEND_V41_O_PROJ_2D="${VLLM_ASCEND_V41_O_PROJ_2D:-1}"
export VLLM_ASCEND_V41_QLI_NO_CANDIDATE="${VLLM_ASCEND_V41_QLI_NO_CANDIDATE:-1}"
export VLLM_ASCEND_V41_ENGRAM_WKV_TP="${VLLM_ASCEND_V41_ENGRAM_WKV_TP:-1}"

# ---- rootfs Python + CANN --------------------------------------------------
export PYTHONHOME="$PYTHON_HOME_DIR"
export PYTHONPATH="$RFS/vllm-workspace/vllm-ascend:$RFS/vllm-workspace/vllm"
# Use the concrete cann-9.2.0 directory, not the ascend-toolkit/latest symlink:
# the driver performs a host-path check that fails through the symlink.
export ASCEND_HOME_PATH="$CANN"
export ASCEND_TOOLKIT_HOME="$CANN"
export ASCEND_OPP_PATH="$CANN/opp"
export ASCEND_AICPU_PATH="$CANN"
export LD_LIBRARY_PATH="$CANN/lib64:$CANN/lib64/plugin/opskernel:$CANN/lib64/plugin/nnengine:$CANN/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:$RFS/usr/local/Ascend/driver/lib64:$RFS/usr/local/Ascend/driver/lib64/common:$RFS/usr/local/Ascend/driver/lib64/driver:$RFS/usr/local/lib:${LD_LIBRARY_PATH:-}"
export PATH="$CANN/bin:$PYTHON_HOME_DIR/bin:${PATH:-}"
export PYTHONPATH="$CANN/python/site-packages:$PYTHONPATH"

# Environment probe:  DSV41_PROBE=1 bash scripts/serve.sh
# Imports the stack with the exact serving environment (rootfs/container Python,
# CANN, host driver libs), prints versions and the visible NPU count, and exits.
if [ "${DSV41_PROBE:-0}" = "1" ]; then
  exec "$PY" -c '
import torch, torch_npu
print("torch", torch.__version__, "torch_npu", torch_npu.__version__)
print("npu_count", torch.npu.device_count())
import vllm, vllm_ascend
print("vllm", vllm.__version__, "vllm_ascend", getattr(vllm_ascend, "__version__", "?"))
'
fi

# ---- compose arguments -----------------------------------------------------
SPEC_JSON=$(printf '{"num_speculative_tokens":%s,"method":"dspark","enforce_eager":false}' "$DSV41_SPEC_TOKENS")
COMP_JSON=$(printf '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[%s]}' "$DSV41_CUDAGRAPH_SIZES")
LOADER_JSON=$(printf '{"enable_multithread_load":true,"num_threads":%s}' "$DSV41_LOADER_THREADS")
ADDITIONAL_JSON='{"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false},"multistream_overlap_shared_expert":true}'
ENGRAM_JSON='{"cpu_offload":true,"dp_shared_memory":true}'

ARGS=(
  serve "$DSV41_MODEL"
  --host "$DSV41_HOST"
  --max-model-len "$DSV41_MAX_MODEL_LEN"
  --max-num-batched-tokens "$DSV41_MAX_NUM_BATCHED_TOKENS"
  --served-model-name "$DSV41_SERVED_NAME"
  --gpu-memory-utilization "$DSV41_GPU_MEM_UTIL"
  --max-num-seqs "$DSV41_MAX_NUM_SEQS"
  --tensor-parallel-size "$DSV41_TP"
  --tokenizer-mode "$DSV41_TOKENIZER_MODE"
  --reasoning-parser "$DSV41_REASONING_PARSER"
  --tool-call-parser "$DSV41_TOOL_CALL_PARSER"
  --enable-auto-tool-choice
  --safetensors-load-strategy "$DSV41_SAFETENSORS_LOAD_STRATEGY"
  --model-loader-extra-config "$LOADER_JSON"
  --quantization ascend
  --block-size "$DSV41_BLOCK_SIZE"
  --compilation-config "$COMP_JSON"
  --speculative-config "$SPEC_JSON"
  --engram-config "$ENGRAM_JSON"
  --additional-config "$ADDITIONAL_JSON"
)

if [ "$DSV41_LB_MODE" = "external" ]; then
  ARGS+=(
    --data-parallel-size "$DSV41_DP"
    --data-parallel-size-local "$DSV41_DP"
    --data-parallel-multi-port-external-lb
    --port "$DSV41_DP_BASE_PORT"
    --data-parallel-supervisor-port "$DSV41_DP_SUPERVISOR_PORT"
  )
  if [ "$DSV41_KV_EVENTS" = "1" ]; then
    KV_JSON=$(printf '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:%s","topic":"kv-events"}' "$DSV41_KV_EVENTS_BASE_PORT")
    ARGS+=(--kv-events-config "$KV_JSON")
  fi
else
  ARGS+=(--data-parallel-size "$DSV41_DP" --port "$DSV41_PORT")
fi
[ -n "$DSV41_API_KEY" ] && ARGS+=(--api-key "$DSV41_API_KEY")
[ "$DSV41_ENABLE_EXPERT_PARALLEL" = "1" ] && ARGS+=(--enable-expert-parallel)
[ "$DSV41_ENABLE_PREFIX_CACHING" = "1" ] && ARGS+=(--enable-prefix-caching)
[ "$DSV41_ENABLE_PROMPT_TOKENS_DETAILS" = "1" ] && ARGS+=(--enable-prompt-tokens-details)
if [ -n "${DSV41_EXTRA_SERVE_ARGS// /}" ]; then
  # Deliberately simple word splitting: do not put spaces inside a value here.
  # shellcheck disable=SC2206
  EXTRA=( $DSV41_EXTRA_SERVE_ARGS )
  ARGS+=("${EXTRA[@]}")
fi

# ---- launch ----------------------------------------------------------------
echo "[serve] runtime=$DSV41_RUNTIME ROOTFS=$DSV41_ROOTFS instance=$DSV41_INSTANCE_ID offline=$DSV41_OFFLINE"
echo "[serve] model=$DSV41_MODEL tp=$DSV41_TP dp=$DSV41_DP port=$DSV41_PORT served=$DSV41_SERVED_NAME"
echo "[serve] api-key: $([ -n "$DSV41_API_KEY" ] && echo "已启用 (len=${#DSV41_API_KEY})" || echo "未启用 (仅限内网)")"
echo "[serve] V2=$DSV41_USE_V2 DSpark K=$DSV41_SPEC_TOKENS FULL_DECODE_ONLY max_len=$DSV41_MAX_MODEL_LEN prompt_tokens_details=$DSV41_ENABLE_PROMPT_TOKENS_DETAILS"
echo "[serve] gates: triton_slot=$VLLM_ASCEND_V41_TRITON_SLOT_MAPPING rope_idx=$VLLM_ASCEND_ROPE_INDEX_SELECT moe_mask=$VLLM_ASCEND_V41_MOE_MASK_RANGE qli=$VLLM_ASCEND_V41_QLI_NO_CANDIDATE engram_wkv_tp=$VLLM_ASCEND_V41_ENGRAM_WKV_TP o_proj_2d=$VLLM_ASCEND_V41_O_PROJ_2D"
echo "[serve] engram: cpu_offload=true dp_shared_memory=true"
echo "[serve] 日志：$LOG_FILE"

if [ "$DSV41_LB_MODE" = "external" ]; then
  "$PY" -c 'import aiohttp, zmq, msgspec' 2>/dev/null || {
    echo "[serve] ❌ external 模式需要 router 依赖: aiohttp pyzmq msgspec" >&2
    exit 1
  }
fi

if [ "${ROUTER_RESTART:-0}" = "1" ]; then
  if [ "$DSV41_LB_MODE" != "external" ]; then
    echo "[serve] ❌ ROUTER_RESTART=1 仅用于 DSV41_LB_MODE=external" >&2
    exit 1
  fi
  stop_router
  start_router
  exit 0
fi

if [ "${1:-}" = "run" ]; then
  # Keep the real Python PID and forward signals correctly.
  exec > >(tee -a "$LOG_FILE") 2>&1
  if [ "$DSV41_LB_MODE" = "external" ]; then
    start_router
    trap 'stop_router' EXIT INT TERM
    "$PY" -m vllm.entrypoints.cli.main "${ARGS[@]}"
    rc=$?
    stop_router
    exit $rc
  fi
  exec "$PY" -m vllm.entrypoints.cli.main "${ARGS[@]}"
fi

nohup "$PY" -m vllm.entrypoints.cli.main "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
cpid=$!
disown 2>/dev/null || true
echo "$cpid" > "$DSV41_SERVE_PID_FILE"
echo "[serve] 已提交 pid=$cpid，run_id=$RUN_ID"
if [ "$DSV41_LB_MODE" = "external" ]; then
  start_router
  echo "[serve] DP supervisor health: curl --noproxy '*' http://127.0.0.1:$DSV41_DP_SUPERVISOR_PORT/health"
  echo "[serve] rank endpoints: $(seq -f "http://127.0.0.1:%g" "$DSV41_DP_BASE_PORT" $((DSV41_DP_BASE_PORT + DSV41_DP - 1)) | tr '\n' ' ')"
fi
echo "[serve] 健康检查：curl --noproxy '*' http://127.0.0.1:$DSV41_PORT/health"
echo "[serve] 停服：STOP=1 bash $SELF_DIR/serve.sh"
