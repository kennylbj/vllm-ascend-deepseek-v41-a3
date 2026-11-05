# DeepSeek-V4.1 optimization patches for vLLM Ascend v0.30.0rc2

These patches are applied to the `vllm-ascend` source tree inside the
`v0.30.0rc2-a3` rootfs and are enabled by `scripts/serve.sh`.

Baseline: **vllm-ascend v0.30.0rc2** (`8fde6d4e3`) shipped in
`quay.nju.edu.cn/ascend/vllm-ascend:v0.30.0rc2-a3`, paired with **vllm 0.30.0**.

The series is intentionally small: only changes that are actually executed by
the V4.1 path on A3 are kept.

| File | Purpose |
|------|---------|
| `0001-envs-gates.patch` | Define six `VLLM_ASCEND_V41_*` / `ROPE_INDEX_SELECT` switches in `envs.py` |
| `0002-dsa-v41.patch` | DSA-V4.1 metadata builder optimizations + Triton slot-mapping gate |
| `0004-model-opts.patch` | Engram WKV TP sharding, indexer QLI mode 3, MoE mask-range dispatch, RoPE `index_select` |
| `0005-packed-cache-slot-mapping-scalar.patch` | Build the packed-cache slot mapping with a single scalar Triton launch instead of an eager op chain |
| `series.conf` | Application order |

Patches touch Python only. The prebuilt `.so` files in the image remain valid.

## Why the gate defaults must be `1`

`VLLM_WORKER_MULTIPROC_METHOD=spawn` means engine workers do **not** inherit
`os.environ`. Exported `VLLM_ASCEND_V41_*` variables are therefore silently
lost in the worker processes (vLLM logs an "Unknown vLLM environment variable"
warning while the gate evaluates to `0`).

`scripts/apply_patches.sh apply` therefore performs a second, idempotent step:

* `vllm_ascend/envs.py` — `_strict_binary_env("GATE")` → `_strict_binary_env("GATE", "1")` (6 gates)
* `engram/layer.py`, `model.py`, `indexer.py`, `token_dispatcher.py` — module-level `os.environ.get("GATE", "0")` → `..., "1")` (4 gates)
  (`layer.py` and `model.py` read the Engram WKV gate independently; both must be pinned)

`scripts/apply_patches.sh verify` checks all six definitions and their defaults.

## Removed patches

| File | Reason |
|------|--------|
| `0003-dsa-v1-o-proj-2d.patch` | V4.1 uses `AscendDSAV41Impl`; `dsa_v1.AscendDSAImpl._wo_a_bmm` is not executed |

## KV pool correctness patch

`0101-kvpool-mrv2-resume.patch` clears the preemption marker when MRV2 resumes
with a full block table. Later scheduler steps can then append block deltas
without replacing the table and writing incorrect KV under a pool key.
The Docker offload launcher checks the imported scheduler for this fix.

## Usage

```bash
bash scripts/apply_patches.sh status
bash scripts/apply_patches.sh check
bash scripts/apply_patches.sh apply      # idempotent
bash scripts/apply_patches.sh verify
bash scripts/apply_patches.sh revert --force
```

Manual equivalent:

```bash
cd "$DSV41_ROOTFS/vllm-workspace/vllm-ascend"
for f in $(grep -vE '^\s*#|^\s*$' <repo>/scripts/patches/vllm-ascend-v41-rc2/series.conf); do
  git apply --verbose <repo>/scripts/patches/vllm-ascend-v41-rc2/"$f"
done
# then pin the gate defaults as described above
```

## Provenance

The patches were originally derived from the CJYLZS DeepSeek-V4.1 reference
repository (`vllm-ascend-dsv41-a2`) and subsequently modified and validated in
this deployment. `0002-dsa-v41.patch` contains local changes. All patched files
are from the Apache-2.0 licensed vllm-ascend project; see `NOTICE`.
