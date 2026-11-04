#!/usr/bin/env bash
# =============================================================================
# setup_rootfs.sh — download an Ascend vLLM image and turn it into a rootfs
#
# No docker / podman needed. Layers are pulled directly from the registry,
# whiteouts are honoured, and execute bits lost by the extraction are restored.
#
# Default image:
#   quay.nju.edu.cn/ascend/vllm-ascend:v0.30.0rc2-a3   (linux/arm64)
#
# Usage:
#   bash scripts/setup_rootfs.sh                 # extract to $DSV41_ROOTFS
#   bash scripts/setup_rootfs.sh --dry-run       # list layers only
#   bash scripts/setup_rootfs.sh --keep-layers   # keep downloaded archives
#   bash scripts/setup_rootfs.sh --force         # re-apply every layer
#
# Overrides:
#   DSV41_ROOTFS=/cache/dsv41-rc2-rootfs
#   REGISTRY=quay.nju.edu.cn REPO=ascend/vllm-ascend TAG=v0.30.0rc2-a3
# =============================================================================
set -Eeuo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
. "$SELF_DIR/common.sh"

if [ "$DSV41_RUNTIME" = "docker" ]; then
  echo "[rootfs] DSV41_RUNTIME=docker：服务运行在本机已加载的镜像 $DSV41_IMAGE 内，无需下载/解包 rootfs。"
  echo "[rootfs] 下一步: bash scripts/container.sh up"
  exit 0
fi

REGISTRY=${REGISTRY:-quay.nju.edu.cn}
REPO=${REPO:-ascend/vllm-ascend}
TAG=${TAG:-v0.30.0rc2-a3}
ROOTFS=${ROOTFS:-$DSV41_ROOTFS}
LAYERS_DIR=${LAYERS_DIR:-${ROOTFS}.layers}
KEEP_LAYERS=0
FORCE=0
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --rootfs)      ROOTFS="$2"; shift 2 ;;
    --layers-dir)  LAYERS_DIR="$2"; shift 2 ;;
    --registry)    REGISTRY="$2"; shift 2 ;;
    --repo)        REPO="$2"; shift 2 ;;
    --tag)         TAG="$2"; shift 2 ;;
    --keep-layers) KEEP_LAYERS=1; shift ;;
    --force)       FORCE=1; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)     sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

PYTHON=${PYTHON:-python3}
command -v "$PYTHON" >/dev/null 2>&1 || { echo "需要 python3" >&2; exit 1; }

ARGS=(
  --registry "$REGISTRY"
  --repo "$REPO"
  --tag "$TAG"
  --target "$ROOTFS"
  --layers-dir "$LAYERS_DIR"
)
[ "$KEEP_LAYERS" = 1 ] && ARGS+=(--keep-layers)
[ "$FORCE" = 1 ] && ARGS+=(--force)
[ "$DRY_RUN" = 1 ] && ARGS+=(--dry-run)

echo "[rootfs] image : $REGISTRY/$REPO:$TAG (linux/arm64)"
echo "[rootfs] target: $ROOTFS"
SKIP_EXTRACT=${SKIP_EXTRACT:-0}
provenance_matches() {
  python3 - "$ROOTFS/.dsv41-image" "$REGISTRY" "$REPO" "$TAG" <<'PY'
import json
import sys

path, registry, repo, tag = sys.argv[1:5]
try:
    data = json.load(open(path, encoding="utf-8"))
except Exception:
    raise SystemExit(1)
ok = (
    data.get("registry") == registry
    and data.get("repository") == repo
    and data.get("reference") == tag
    and data.get("platform") == "linux/arm64"
)
raise SystemExit(0 if ok else 1)
PY
}
if [ "$DRY_RUN" != 1 ] && [ "$FORCE" != 1 ] && [ "$SKIP_EXTRACT" = 1 ]; then
  echo "[rootfs] SKIP_EXTRACT=1，跳过下载（仅修复权限并校验）。"
elif [ "$DRY_RUN" != 1 ] && [ "$FORCE" != 1 ] && [ -f "$ROOTFS/.dsv41-image" ] \
     && provenance_matches; then
  echo "[rootfs] 检测到相同镜像已解包（.dsv41-image 匹配），跳过下载。"
  echo "[rootfs] 如需重新下载/解包，请加 --force。"
else
  if [ -f "$ROOTFS/.dsv41-image" ] && [ "$FORCE" != 1 ] && [ "$DRY_RUN" != 1 ]; then
    echo "[rootfs] 现有 .dsv41-image 与请求的镜像/TAG 不一致，重新解包。"
  fi
  "$PYTHON" "$SELF_DIR/oci_extract.py" "${ARGS[@]}"
fi

[ "$DRY_RUN" = 1 ] && exit 0

# ---- restore execute bits --------------------------------------------------
# Package extraction can lose the executable bit. vLLM needs to execute CANN
# binaries (bishengir-compile, triton-*-opt), and the NPU runtime mmaps shared
# objects that must be executable.
echo "[rootfs] restoring execute bits..."
ASCEND_ROOT="$ROOTFS/usr/local/Ascend"
if [ -d "$ASCEND_ROOT" ]; then
  find "$ASCEND_ROOT" -type f \( -name '*.so' -o -name '*.so.*' -o -name '*-opt' \) \
    -exec chmod +x {} + 2>/dev/null || true
  find "$ASCEND_ROOT" -type f -path '*/bin/*' -exec chmod +x {} + 2>/dev/null || true
  find "$ASCEND_ROOT" -type f -path '*/tools/*' -exec chmod +x {} + 2>/dev/null || true
fi
if [ -d "$ROOTFS/vllm-workspace" ]; then
  find "$ROOTFS/vllm-workspace" -type f \( -name '*.so' -o -name '*.so.*' \) \
    -exec chmod +x {} + 2>/dev/null || true
fi
PYHOME=$(find "$ROOTFS/usr/local" -maxdepth 1 -type d -name 'python3.12*' 2>/dev/null | sort -V | tail -1)
if [ -n "$PYHOME" ] && [ -d "$PYHOME/lib" ]; then
  find "$PYHOME/lib" -type f \( -name '*.so' -o -name '*.so.*' \) \
    -exec chmod +x {} + 2>/dev/null || true
fi
if [ -n "$PYHOME" ] && [ -d "$PYHOME/bin" ]; then
  chmod +x "$PYHOME/bin/"* 2>/dev/null || true
fi

# ---- verify ----------------------------------------------------------------
echo "[rootfs] verifying..."
missing=0
check_file() {
  if [ ! -e "$1" ]; then
    echo "  ❌ 缺少 $1" >&2
    missing=1
  fi
}
check_file "${PYHOME:-/nonexistent}/bin/python3.12"
CANNDIR=$(find "$ROOTFS/usr/local/Ascend" -maxdepth 1 -type d -name 'cann-*' 2>/dev/null | sort -V | tail -1)
check_file "$CANNDIR/set_env.sh"
check_file "$ROOTFS/vllm-workspace/vllm/vllm/__init__.py"
check_file "$ROOTFS/vllm-workspace/vllm-ascend/vllm_ascend/envs.py"
if ! ls "$ROOTFS"/vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C*.so >/dev/null 2>&1; then
  echo "  ⚠ 未找到预编译 vllm_ascend_C*.so（源码树可能不完整）" >&2
fi

if [ "$missing" != 0 ]; then
  echo "[rootfs] ❌ 解包不完整。可加 --force 重新应用所有层。" >&2
  exit 1
fi

echo "[rootfs] ✅ rootfs 就绪: $ROOTFS"
if [ -f "$ROOTFS/.dsv41-image" ]; then
  echo "[rootfs] 镜像溯源: $ROOTFS/.dsv41-image"
fi
echo "[rootfs] 下一步: bash scripts/apply_patches.sh apply"
