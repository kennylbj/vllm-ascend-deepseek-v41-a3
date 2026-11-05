#!/usr/bin/env bash
# Shared configuration for the Docker host launcher and the serving container.

dsv41_kv_supported() {
  [ "$DSV41_RUNTIME" = docker ] && [ "${DSV41_USE_V2:-1}" = 1 ] &&
    [ "${DSV41_ENABLE_PREFIX_CACHING:-1}" = 1 ]
}

if dsv41_kv_supported; then
  : "${DSV41_KV_OFFLOAD:=1}"
else
  : "${DSV41_KV_OFFLOAD:=0}"
fi
: "${DSV41_KV_DRAM_GIB_PER_WORKER:=8}"
: "${DSV41_KV_META_PORT:=5100}"
: "${DSV41_KV_STORE_PORT:=6100}"
: "${DSV41_KV_METRICS_PORT:=8100}"

dsv41_kv_error() {
  echo "[kv] ❌ $*" >&2
  return 1
}

dsv41_kv_validate() {
  local port event_base
  case "$DSV41_KV_OFFLOAD" in
    0) return 0 ;;
    1) ;;
    *) dsv41_kv_error 'DSV41_KV_OFFLOAD 必须为 0 或 1'; return 1 ;;
  esac
  if ! dsv41_kv_supported; then
    dsv41_kv_error 'KV offload 需要 Docker、Model Runner V2 和本地前缀缓存'
    return 1
  fi
  if [[ ! "$DSV41_KV_DRAM_GIB_PER_WORKER" =~ ^[1-9][0-9]{0,3}$ ]] \
     || [ "$DSV41_KV_DRAM_GIB_PER_WORKER" -gt 1024 ]; then
    dsv41_kv_error 'DSV41_KV_DRAM_GIB_PER_WORKER 必须为 1 到 1024 的整数'
    return 1
  fi
  for port in "$DSV41_KV_META_PORT" "$DSV41_KV_STORE_PORT" "$DSV41_KV_METRICS_PORT"; do
    if [[ ! "$port" =~ ^[1-9][0-9]{0,4}$ ]] || [ "$port" -lt 1025 ] || [ "$port" -gt 65535 ]; then
      dsv41_kv_error "Memcache 端口无效: ${port}（1025–65535）"
      return 1
    fi
    if [ "$port" = "$DSV41_PORT" ]; then
      dsv41_kv_error "Memcache 端口与推理服务端口冲突: $port"
      return 1
    fi
    if [ "${DSV41_LB_MODE:-internal}" = external ]; then
      event_base=${DSV41_KV_EVENTS_BASE_PORT:-5557}
      if [ "$port" = "$DSV41_DP_SUPERVISOR_PORT" ] ||
         (( port >= DSV41_DP_BASE_PORT && port < DSV41_DP_BASE_PORT + DSV41_DP )) ||
         { [ "${DSV41_KV_EVENTS:-1}" = 1 ] &&
           (( port >= event_base && port < event_base + DSV41_DP )); }; then
        dsv41_kv_error "Memcache 端口与 DP 服务或 KV events 端口冲突: $port"
        return 1
      fi
    fi
  done
  if [ "$DSV41_KV_META_PORT" = "$DSV41_KV_STORE_PORT" ] \
     || [ "$DSV41_KV_META_PORT" = "$DSV41_KV_METRICS_PORT" ] \
     || [ "$DSV41_KV_STORE_PORT" = "$DSV41_KV_METRICS_PORT" ]; then
    dsv41_kv_error 'Memcache 的三个端口必须互不相同'
    return 1
  fi
}

dsv41_kv_config_dir() {
  printf '%s/kv/%s/host_shm-%s-%s-%s-%s\n' "$DSV41_RESULTS" "$DSV41_CONTAINER" \
    "$DSV41_KV_META_PORT" "$DSV41_KV_STORE_PORT" "$DSV41_KV_METRICS_PORT" "$DSV41_KV_DRAM_GIB_PER_WORKER"
}

dsv41_kv_meta_config() {
  cat <<EOF
ock.mmc.log_level = info
ock.mmc.meta_service_url = tcp://127.0.0.1:$DSV41_KV_META_PORT
ock.mmc.meta_service.config_store_url = tcp://127.0.0.1:$DSV41_KV_STORE_PORT
ock.mmc.meta_service.metrics_url = http://127.0.0.1:$DSV41_KV_METRICS_PORT
EOF
}

dsv41_kv_local_config() {
  cat <<EOF
ock.mmc.log_level = info
ock.mmc.meta_service_url = tcp://127.0.0.1:$DSV41_KV_META_PORT
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:$DSV41_KV_STORE_PORT
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = host_shm
ock.mmc.local_service.dram.size = ${DSV41_KV_DRAM_GIB_PER_WORKER}GB
ock.mmc.local_service.max.dram.size = ${DSV41_KV_DRAM_GIB_PER_WORKER}GB
ock.mmc.local_service.hbm.size = 0GB
ock.mmc.local_service.storage.enabled = false
EOF
}

dsv41_kv_configs_match() {
  cmp -s "$1/mmc-meta.conf" <(dsv41_kv_meta_config) &&
    cmp -s "$1/mmc-local.conf" <(dsv41_kv_local_config)
}

dsv41_kv_meta_ready() {
  curl --noproxy '*' --fail --silent --max-time 2 -o /dev/null \
    "http://127.0.0.1:$DSV41_KV_METRICS_PORT/metrics"
}
