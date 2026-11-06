# 补丁与优化 gate

rc2-a3 镜像内有两套补丁基线。vllm-ascend 使用 `v0.30.0rc2`（`8fde6d4e3`），
包含四个优化补丁和已有的 `0101`。vLLM 使用 `ced6857af`，包含一个 tokenizer 修复。

```
scripts/patches/vllm-ascend-v41-rc2/
├── 0001-envs-gates.patch
├── 0002-dsa-v41.patch
├── 0004-model-opts.patch
├── 0005-packed-cache-slot-mapping-scalar.patch
├── 0101-kvpool-mrv2-resume.patch
├── series.conf
└── README.md

scripts/patches/vllm-ced6857af/
├── 0001-deepseek-v41-encoding-safety.patch
└── series.conf
```

---

## 1. 补丁一览

### 0001 — 定义优化开关（`vllm_ascend/envs.py`）

在 vLLM 的环境变量注册表里新增：

| Gate | 作用 |
|---|---|
| `VLLM_ASCEND_ROPE_INDEX_SELECT` | cos/sin 缓存改用 `index_select` |
| `VLLM_ASCEND_V41_TRITON_SLOT_MAPPING` | 使用单次 Triton launch 构建 packed-cache 槽位映射 |
| `VLLM_ASCEND_V41_MOE_MASK_RANGE` | 连续 expert 区间比较，替代逐 token mask |
| `VLLM_ASCEND_V41_O_PROJ_2D` | 预留开关；当前补丁序列没有消费点，保持默认 1 作为后续 A/B 位 |
| `VLLM_ASCEND_V41_QLI_NO_CANDIDATE` | 每层固定走 mode 3，关闭两级候选机制 |
| `VLLM_ASCEND_V41_ENGRAM_WKV_TP` | Engram WKV 按输出维在 TP 组内分片 |

补丁本身保持 `Default: 0`，以保留上游 A/B 能力；`apply_patches.sh` 会在应用后把
默认值固化为 `1`（原因见 §3）。不定义 `0001` 时，后面的补丁无法读到这些开关。

### 0002 — DSA-V4.1 metadata / attention（`vllm_ascend/attention/dsa_v41.py`）

- 引入 Triton 槽位映射分支（受 `VLLM_ASCEND_V41_TRITON_SLOT_MAPPING` 控制）；
- 优化 `AscendDSAV41MetadataBuilder` 中每个 decode step 都要执行的元数据构造：
  减少 Python 侧 eager op 链与重复张量操作；
- 保留原来的 eager 路径作为回退。

槽位、有效位掩码和索引由当前 `slot_mapping` 逐次推导。只有最终坐标和 flat mapping
在同一 runner step 的缓存组内共享，键包含 `num_input_tokens`，避免不同形状复用旧映射。

### 0004 — 模型与算子微优化

| 文件 | 改动 |
|---|---|
| `models/deepseek_v41/engram/layer.py` | Engram WKV 按输出维 TP 分片（每 die 省约 550 MB，可换 KV 容量） |
| `models/deepseek_v41/indexer.py` | QLI 关闭候选机制，固定 mode 3 |
| `models/deepseek_v41/model.py` | V4.1 主干适配上述开关 |
| `ops/fused_moe/token_dispatcher.py` | 连续 expert 区间 mask-range 分发 |
| `ops/rope_dsv4.py` | RoPE cos/sin 走 `index_select` |

这些改动都发生在 import 期或图捕获内，**只能重启切换，热改环境变量无效**。

### 0005 — packed cache slot mapping 标量化

`vllm_ascend/ops/triton/packed_cache_slot_mapping.py`：

- 上游在 A3 上走 eager 链（约 100 个小算子，每步按 KV cache group 重复约 21 次，
  且每个 group 只有 6 个 token），设备大部分时间空转；
- 补丁把该逻辑改成单次 Triton kernel launch，直接写 flat slot / page / row；
- 顺带修正 int32/int64 比较与 padding 尾部有效性判断。

这是 6N 图桶之外最重要的 decode 侧收益之一。

### vLLM — DeepSeek V4.1 tokenizer

转义消息与工具参数中作为普通文本传入的控制 token。工具参数在 JSON 解码后转义，
消息和上下文由编码器复制后处理。空正文的 developer 消息仍可渲染所附 tools。

---

## 2. 已移除的补丁

| 补丁 | 移除原因 |
|---|---|
| `0003-dsa-v1-o-proj-2d.patch` | V4.1 使用 `AscendDSAV41Impl`，不执行 `dsa_v1.AscendDSAImpl._wo_a_bmm` |

### Memcache KV 池补丁

`0101-kvpool-mrv2-resume.patch` 已加入补丁序列。MRV2 恢复请求时会提交完整的 block 表。
补丁在此时清除抢占标记，避免后续增量覆盖完整的表，将池 key 关联到错误的 KV。
Docker offload 启动时会检查实际导入的调度器是否包含此修复。

---

## 3. 为什么要把 gate 默认值改成 1

`scripts/serve.sh` 会导出：

```bash
export VLLM_ASCEND_V41_TRITON_SLOT_MAPPING=1
export VLLM_ASCEND_ROPE_INDEX_SELECT=1
export VLLM_ASCEND_V41_MOE_MASK_RANGE=1
export VLLM_ASCEND_V41_O_PROJ_2D=1
export VLLM_ASCEND_V41_QLI_NO_CANDIDATE=1
export VLLM_ASCEND_V41_ENGRAM_WKV_TP=1
```

但 vLLM 使用 `VLLM_WORKER_MULTIPROC_METHOD=spawn` 启动 EngineCore/Worker。
spawn 子进程**不继承父进程的 `os.environ`**：

- 日志里可能出现 `Unknown vLLM environment variable` 警告；
- worker 内 `envs.py` 读到的是源码默认值 `0`；
- 服务正常起来、不报错，但所有 gate 优化**静默失效**。

因此 `apply_patches.sh apply` 在 `git apply` 之后执行幂等的源码替换：

| 位置 | 改动 |
|---|---|
| `vllm_ascend/envs.py` | `_strict_binary_env("GATE")` → `_strict_binary_env("GATE", "1")`（6 处） |
| `engram/layer.py` | `os.environ.get("VLLM_ASCEND_V41_ENGRAM_WKV_TP", "0")` → `..., "1")` |
| `model.py` | 同上（权重按 TP rank 切分读取，必须与 layer.py 同步开启） |
| `indexer.py` | `... QLI_NO_CANDIDATE ..., "0")` → `..., "1")` |
| `token_dispatcher.py` | `... MOE_MASK_RANGE ..., "0")` → `..., "1")` |


验证：

```bash
make verify
```

---

## 4. 应用 / 回滚机制

```bash
bash scripts/apply_patches.sh status    # 报告各组件状态
bash scripts/apply_patches.sh check     # 只读预演两套补丁
bash scripts/apply_patches.sh apply     # 应用 + 固化默认值（幂等）
make verify                            # 校验记录、序列和已部署源码
bash scripts/apply_patches.sh revert --force
```

细节：

- 默认管理 `$DSV41_ROOTFS/vllm-workspace/` 下的 `vllm-ascend` 与 `vllm`。
  用 `SRC` 和 `PATCH_DIR` 调试单组件。
- 每棵源码树的 `.patches-applied.d/manifest.json` 记录基线、有序补丁 SHA-256 和安装结果。
  旧 `.done` 只有在 Git 基线与实际源码可验证时才能迁移。已装旧版 `0002` 的实例从干净镜像或 rootfs 重装。
- 序列变更、源码漂移或 Git 基线变化会使 `make verify` 失败。Git 基线一致时，可用旧记录的文件清单回滚后重装。
  无 Git 的干净解包树使用 `patch -p1` 安装，回滚通过重新解包完成。

### A3 专项验收

摘流并停止 watchdog 后，重建 Docker 容器或准备干净的 native rootfs。安装后执行
`make verify` 和 `make probe`，启动服务后执行 `make smoke` 和以下专项测试。

```bash
DSV41_CONTAINER_SOURCE_ENV=1 bash scripts/container.sh exec -- \
  env PYTHONPATH=/vllm-workspace/vllm-ascend:/vllm-workspace/vllm \
  python3 -m unittest discover -s tests/ut -v
```

native 部署用服务所用的 rootfs Python 和 CANN 环境执行同一个 unittest 命令。测试读取
`DSV41_BASE_URL`、`DSV41_SERVED_NAME` 和 `DSV41_API_KEY`，覆盖连续 builder 调用、
tokenizer 及在线 chat 和 tokenize 接口。验收失败时从干净镜像或 rootfs 恢复上一版发布。

---

## 5. 升级评估

`series.conf` 记录了从 `v0.30.0rc2` 到上游 main（当时 452 个提交）的文件级 churn：

| 文件 | 提交数 | 行变更 | 冲突风险 |
|---|---:|---:|---|
| `dsa_v41.py` | 5 | +153/−4 | 中 |
| `model.py` | 7 | +46/−1 | 中 |
| `rope_dsv4.py` | 1 | +24/−1 | 低 |
| `engram/layer.py` | 0 | — | 无 |
| `indexer.py` | 0 | — | 无 |
| `token_dispatcher.py` | 0 | — | 无 |

升级到新镜像时：

1. 用新 rootfs 运行 `apply_patches.sh check`，先看哪些补丁不再适用；
2. 优先保留 `0001`（开关定义）与 `0005`（槽位标量化）；
3. 若上游已合并某项优化，删除对应补丁并在文档中注明；
4. 每次改动补丁后必须跑 `bench_8k.py` 对比**每步时延**，不要只看吞吐。
