#!/usr/bin/env bash
# Apply or verify both DeepSeek-V4.1 patch series in the serving source trees.
# SRC and PATCH_DIR select a single component for local testing.
set -Eeuo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
. "$SELF_DIR/common.sh"

# container.sh forwards DSV41_* settings to the serving container.
export DSV41_PATCH_SRC="${SRC:-${DSV41_PATCH_SRC:-}}"
export DSV41_PATCH_DIR="${PATCH_DIR:-${DSV41_PATCH_DIR:-}}"
dsv41_reexec_in_container "$SELF_DIR/apply_patches.sh" "$@"

args=(--rootfs "$DSV41_ROOTFS")
[ -z "$DSV41_PATCH_SRC" ] || args+=(--src "$DSV41_PATCH_SRC")
[ -z "$DSV41_PATCH_DIR" ] || args+=(--patch-dir "$DSV41_PATCH_DIR")
exec python3 "$SELF_DIR/patch_install.py" "${args[@]}" "$@"
