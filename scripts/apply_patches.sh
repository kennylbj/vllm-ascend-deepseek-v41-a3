#!/usr/bin/env bash
# =============================================================================
# apply_patches.sh — apply the DeepSeek-V4.1 optimization patch series
#
# The patches target the vllm-ascend source tree inside the extracted rc2
# rootfs. After applying them, six optimization gates are pinned to default 1
# because multiprocessing spawn workers do not inherit exported env vars.
#
# The script tracks each patch independently, so a partially applied series is
# detected instead of being silently treated as "done".
#
# Usage:
#   bash scripts/apply_patches.sh status
#   bash scripts/apply_patches.sh check
#   bash scripts/apply_patches.sh apply       # idempotent
#   bash scripts/apply_patches.sh verify
#   bash scripts/apply_patches.sh revert --force
#
# Targets can be overridden:
#   SRC=/path/to/vllm-ascend PATCH_DIR=/path/to/patches \
#     bash scripts/apply_patches.sh apply
# =============================================================================
set -Eeuo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
. "$SELF_DIR/common.sh"

# The patched sources live inside the image: with DSV41_RUNTIME=docker run
# everything in the container (no-op on the native runtime / inside the container).
dsv41_reexec_in_container "$SELF_DIR/apply_patches.sh" "$@"

SRC=${SRC:-${DSV41_ROOTFS%/}/vllm-workspace/vllm-ascend}
PATCH_DIR=${PATCH_DIR:-$SELF_DIR/patches/vllm-ascend-v41-rc2}
SERIES="$PATCH_DIR/series.conf"
MARKER_DIR="$SRC/.patches-applied.d"

GATES=(
  VLLM_ASCEND_ROPE_INDEX_SELECT
  VLLM_ASCEND_V41_TRITON_SLOT_MAPPING
  VLLM_ASCEND_V41_MOE_MASK_RANGE
  VLLM_ASCEND_V41_O_PROJ_2D
  VLLM_ASCEND_V41_QLI_NO_CANDIDATE
  VLLM_ASCEND_V41_ENGRAM_WKV_TP
)

patches() { grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$SERIES"; }
is_git_tree() { git -C "$SRC" rev-parse --is-inside-work-tree >/dev/null 2>&1; }
marker_for() { echo "$MARKER_DIR/$1.done"; }

content_state() {
  python3 "$SELF_DIR/patch_utils.py" content-state --src "$SRC" --patch "$PATCH_DIR/$1" 2>/dev/null || echo unknown
}

# Print one of: applied | applicable | conflict | unknown
patch_state() {
  local f="$1" content
  if [ -f "$(marker_for "$f")" ]; then
    echo applied
    return
  fi
  if is_git_tree; then
    if git -C "$SRC" apply --check "$PATCH_DIR/$f" >/dev/null 2>&1; then
      echo applicable
      return
    fi
    if git -C "$SRC" apply --reverse --check "$PATCH_DIR/$f" >/dev/null 2>&1; then
      echo applied
      return
    fi
    # Gate-default pinning changes lines introduced by the patch, so the
    # reverse check can fail even when the patch is fully applied.
    content=$(content_state "$f")
    case "$content" in
      applied) echo applied ;;
      partial) echo conflict ;;
      *)       echo conflict ;;
    esac
    return
  fi
  content=$(content_state "$f")
  if [ "$content" = "applied" ]; then
    echo applied
  elif [ "$content" = "partial" ]; then
    echo conflict
  elif command -v patch >/dev/null 2>&1 \
       && patch -p1 -R --dry-run --batch -d "$SRC" < "$PATCH_DIR/$f" >/dev/null 2>&1; then
    echo applied
  else
    echo unknown
  fi
}

apply_one() {
  local f="$1"
  if is_git_tree; then
    git -C "$SRC" apply "$PATCH_DIR/$f"
  elif command -v patch >/dev/null 2>&1; then
    patch -p1 -d "$SRC" --forward --batch < "$PATCH_DIR/$f"
  else
    echo "[apply] ❌ 无法应用 $f（既不是 git 树，也没有 patch 命令）" >&2
    return 1
  fi
  mkdir -p "$MARKER_DIR"
  date '+%Y-%m-%d %H:%M:%S' > "$(marker_for "$f")"
}

# Pin gate defaults to 1 (idempotent).
fix_gate_defaults() {
  python3 - "$SRC" <<'PY'
import re
import os
import sys

src = sys.argv[1]
env_gates = [
    "VLLM_ASCEND_ROPE_INDEX_SELECT",
    "VLLM_ASCEND_V41_TRITON_SLOT_MAPPING",
    "VLLM_ASCEND_V41_MOE_MASK_RANGE",
    "VLLM_ASCEND_V41_O_PROJ_2D",
    "VLLM_ASCEND_V41_QLI_NO_CANDIDATE",
    "VLLM_ASCEND_V41_ENGRAM_WKV_TP",
]
mod_gates = {
    "vllm_ascend/models/deepseek_v41/engram/layer.py": "VLLM_ASCEND_V41_ENGRAM_WKV_TP",
    # model.py has its own independent read of the same gate (weight sharding).
    "vllm_ascend/models/deepseek_v41/model.py": "VLLM_ASCEND_V41_ENGRAM_WKV_TP",
    "vllm_ascend/models/deepseek_v41/indexer.py": "VLLM_ASCEND_V41_QLI_NO_CANDIDATE",
    "vllm_ascend/ops/fused_moe/token_dispatcher.py": "VLLM_ASCEND_V41_MOE_MASK_RANGE",
}
changed = []

p = f"{src}/vllm_ascend/envs.py"
s = open(p, encoding="utf-8").read()
origin = s
for gate in env_gates:
    pat = re.compile(r'_strict_binary_env\(\s*"%s"\s*\)' % re.escape(gate))
    s2 = pat.sub(f'_strict_binary_env("{gate}", "1")', s)
    if s2 != s:
        changed.append(f"envs.py:{gate}")
        s = s2
if s != origin:
    open(p, "w", encoding="utf-8").write(s)

for rel, gate in mod_gates.items():
    p = f"{src}/{rel}"
    if not os.path.exists(p):
        print(f"[apply] ⚠ 跳过不存在的文件: {rel}")
        continue
    s = open(p, encoding="utf-8").read()
    pat = re.compile(r'os\.environ\.get\(\s*"%s"(?:\s*,\s*"[01]")?\s*\)' % re.escape(gate))
    s2 = pat.sub(f'os.environ.get("{gate}", "1")', s)
    if s2 != s:
        open(p, "w", encoding="utf-8").write(s2)
        changed.append(rel)

if changed:
    print("[apply] gate 默认值已改为 1: " + ", ".join(changed))
else:
    print("[apply] gate 默认值已是 1，无需改动")
PY
}

apply_patches() {
  [ -f "$SRC/vllm_ascend/envs.py" ] || {
    echo "[apply] ❌ 目标不存在: $SRC/vllm_ascend/envs.py" >&2
    echo "[apply]    先运行 bash scripts/setup_rootfs.sh" >&2
    exit 1
  }
  local f state
  for f in $(patches); do
    state=$(patch_state "$f")
    case "$state" in
      applied)
        echo "[apply] 已应用，跳过 $f"
        mkdir -p "$MARKER_DIR"
        [ -f "$(marker_for "$f")" ] || date '+%Y-%m-%d %H:%M:%S' > "$(marker_for "$f")"
        ;;
      applicable|unknown)
        echo "[apply] applying $f"
        apply_one "$f"
        ;;
      conflict)
        echo "[apply] ❌ $f 与当前源码冲突（既不能正向应用，也不能反向应用）。" >&2
        echo "[apply]    可能已手工修改相关文件，或 rootfs 版本不匹配。" >&2
        echo "[apply]    排查: git -C \"$SRC\" apply --check \"$PATCH_DIR/$f\"" >&2
        exit 1
        ;;
    esac
  done
  fix_gate_defaults
  echo "[apply] 完成。重启服务生效: bash scripts/watchdog.sh restart"
}

check_patches() {
  local rc=0 f state
  for f in $(patches); do
    state=$(patch_state "$f")
    case "$state" in
      applicable) echo "  [可应用]       $f" ;;
      applied)    echo "  [已应用]       $f" ;;
      unknown)    echo "  [无法判定]     $f（非 git 树且反向 dry-run 失败）"; rc=1 ;;
      conflict)   echo "  [冲突/需处理]  $f"; rc=1 ;;
    esac
  done
  return $rc
}

revert_all() {
  local force="${2:-}"
  local files
  files=$(grep -h "^+++ b/" "$PATCH_DIR"/*.patch | sed 's#^+++ b/##' | sort -u)
  echo "[revert] 将恢复以下文件到 git HEAD（丢弃补丁与 gate 改动）:"
  echo "$files" | sed 's/^/    /'
  echo
  if [ "$force" != "--force" ]; then
    echo "❌ 该操作会丢弃这些文件的本地改动。确认请加 --force"
    return 1
  fi
  if ! is_git_tree; then
    echo "[revert] ❌ $SRC 不是 git 工作树，无法自动回滚；请重新解包 rootfs" >&2
    return 1
  fi
  # shellcheck disable=SC2086
  git -C "$SRC" checkout -- $files
  rm -rf "$MARKER_DIR"
  echo "[revert] 完成"
}

status() {
  echo "SRC       : $SRC"
  echo "PATCH_DIR : $PATCH_DIR"
  if [ -f "$SRC/vllm_ascend/envs.py" ] && grep -q "VLLM_ASCEND_V41_TRITON_SLOT_MAPPING" "$SRC/vllm_ascend/envs.py"; then
    echo "patches   : ✅ 已应用"
  else
    echo "patches   : ❌ 未应用"
  fi
  echo "--- 每个补丁 ---"
  check_patches || true
  echo "--- gate 默认值 ---"
  python3 - "$SRC" "${GATES[@]}" <<'PY'
import re
import sys

src = sys.argv[1]
gates = sys.argv[2:]
path = f"{src}/vllm_ascend/envs.py"
try:
    s = open(path, encoding="utf-8").read()
except OSError:
    print("  (envs.py 不存在)")
    raise SystemExit(0)
for gate in gates:
    m = re.search(r'_strict_binary_env\(\s*"%s"(?:\s*,\s*"(\d)")?\s*\)' % re.escape(gate), s)
    if not m:
        print(f"  {gate:<42} (未定义)")
    else:
        print(f"  {gate:<42} default={m.group(1) or '0'}")
PY
}

verify_gates() {
  local rc=0
  python3 - "$SRC" "${GATES[@]}" <<'PY' || rc=$?
import os
import re
import sys

src, gates = sys.argv[1], sys.argv[2:]
env_path = f"{src}/vllm_ascend/envs.py"
s = open(env_path, encoding="utf-8").read()
problems = []
for gate in gates:
    m = re.search(r'_strict_binary_env\(\s*"%s"(?:\s*,\s*"(\d)")?\s*\)' % re.escape(gate), s)
    if not m:
        problems.append(f"{gate}: envs.py 未定义")
    elif (m.group(1) or "0") != "1":
        problems.append(f"{gate}: envs.py default={m.group(1) or '0'}")

module_gates = {
    "vllm_ascend/models/deepseek_v41/engram/layer.py": "VLLM_ASCEND_V41_ENGRAM_WKV_TP",
    "vllm_ascend/models/deepseek_v41/model.py": "VLLM_ASCEND_V41_ENGRAM_WKV_TP",
    "vllm_ascend/models/deepseek_v41/indexer.py": "VLLM_ASCEND_V41_QLI_NO_CANDIDATE",
    "vllm_ascend/ops/fused_moe/token_dispatcher.py": "VLLM_ASCEND_V41_MOE_MASK_RANGE",
}
for rel, gate in module_gates.items():
    path = f"{src}/{rel}"
    if not os.path.exists(path):
        problems.append(f"{rel}: 文件不存在")
        continue
    text = open(path, encoding="utf-8").read()
    m = re.search(r'os\.environ\.get\(\s*"%s"(?:\s*,\s*"([01])")?\s*\)' % re.escape(gate), text)
    if not m:
        problems.append(f"{rel}: 未读取 {gate}")
    elif (m.group(1) or "0") != "1":
        problems.append(f"{rel}: {gate} default={m.group(1) or '0'}")

if problems:
    print("[verify] ❌ " + "; ".join(problems))
    raise SystemExit(1)
print("[verify] ✅ 6 个 gate 已定义且默认值为 1（含模块级读取）")
PY
  return $rc
}

case "${1:-status}" in
  apply)  apply_patches ;;
  check)  check_patches ;;
  verify) verify_gates ;;
  revert) revert_all "$@" ;;
  status) status; verify_gates || true ;;
  *) echo "用法: $0 {status|check|apply|verify|revert --force}"; exit 1 ;;
esac
