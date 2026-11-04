# 性能与调优

所有数据来自单机 8×Ascend 910C（A3，16 die），生产配置：
DP2/TP8、Model Runner V2、DSpark K=5、FULL_DECODE_ONLY、1M 上下文上限、
`--max-num-batched-tokens 16384`、6 个优化 gate、
`HCCL_OP_EXPANSION_MODE=AIV`、jemalloc、external LB + prefix router。

> **默认上下文已改为 360448**（`DSV41_MAX_MODEL_LEN`）。本文所有吞吐与显存/KV
> 数据都是在 `--max-model-len 1048576` 下实测的；360448 下 KV 预算更宽松（可容纳更多
> 并发长请求），但**没有重新采集**，换算前请以启动日志里的
> `Free memory on device` / KV cache 行为准。

> **数据批次**：下面的 2K/8K 表在 2026-11-02 用生产 **16384** 批次、
> external LB + prefix router 重新采集，测试期间机器无其它在线流量。
> 2K 分两组：单前缀（热前缀，`bench_2k.py`）和双前缀 A/B（`bench_dual.py`，
> 两个长前缀分别落在两个 DP rank）。router 对相同前缀做亲和，**单前缀低并发
> 可能只压到一个 rank，不能当作 DP2 上限**。8K 的 `bench_8k.py` 聚合值是
> `单流 × 并发` 估算；8K→1024 双前缀表是 wall-clock 实测。

---

## 1. 2K prompt / 256 output

接口：`/v1/completions`，temperature=0，流式，开启 usage；
2026-11-02，16384 批次 + external LB/router，每个并发重复 3 次取中位数。

### 1.1 单前缀（热前缀）

同一个 ≈2K prompt 发给所有请求；router 会优先命中已有 KV 前缀的 rank。

| 并发 | 总吞吐 (tok/s) | 单流 (tok/s) | TTFT (ms) |
|---:|---:|---:|---:|
| 1 | 87.9 | 98.9 | 308 |
| 2 | 163.7 | 92.4 | 309 |
| 4 | 317.5 | 90.4 | 358 |
| 8 | 473.6 | 66.3 | 335 |
| 16 | 1038.8 | 72.5 | 347 |
| 32 | 1831.6 | 72.3 | 912 |

```bash
python3 scripts/bench_2k.py --concurrency 1,2,4,8,16,32 --max-tokens 256 --repeats 3
```

### 1.2 双前缀 A/B（DP2 均衡）

两个互不共享首块的长前缀交替发送，两个 rank 各持有一个前缀，更接近多租户 /
多 system prompt 的并发场景。

| 并发 | 总吞吐 (tok/s) | 单流 (tok/s) | TTFT (ms) |
|---:|---:|---:|---:|
| 2 | 192.1 | 112.1 | 353 |
| 4 | 360.4 | 105.6 | 383 |
| 8 | 672.9 | 102.5 | 496 |
| 16 | 1084.7 | 83.4 | 394 |
| 32 | 2253.4 | 81.3 | 401 |

```bash
python3 scripts/bench_dual.py --concurrency 2,4,8,16,32 --max-tokens 256 --repeats 3
```

观察：

- 单前缀 c2 只有 163.7 tok/s，而双前缀 c2 为 192.1 tok/s；c32 分别为
  1831.6 和 **2253.4 tok/s**。热前缀的亲和路由会牺牲一部分 DP2 利用率，
  容量规划应优先参考双前缀/混合负载。
- 单前缀 c32 的 TTFT 升到 912 ms，主要是热前缀集中 + 32 并发排队；双前缀
  c32 为 401 ms。
- 单流吞吐在 c1–c4 最高（约 90–112 tok/s），c32 降到约 72–81 tok/s，是
  吞吐/时延的正常取舍；`--max-num-seqs 32` 与图桶上限对齐。

---

## 2. 8K prompt / 推测解码

8K 场景使用 `scripts/bench_8k.py`，脚本会：

1. 先发一轮 warm-up，让 `/metrics` 产生推测解码统计；
2. 从 `spec_decode_num_accepted_tokens_total` 与
   `spec_decode_num_draft_tokens_total` 计算接受率；
3. 换算 `tokens_per_step = 1 + acceptance_rate × K`；
4. 用 decode 时间 / 前向次数得到每步时延；
5. 输出每步 ms、单流 tok/s、聚合 tok/s、TTFT。

### 2.1 单前缀（`bench_8k.py` 估算）

```bash
python3 scripts/bench_8k.py --concurrency 1,4,8,16
```

| 并发 | 512 总吞吐 | 512 单流 | 512 TTFT | 1024 总吞吐 | 1024 单流 | 1024 TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 107.2 | 107.2 | 283 ms | 109.2 | 109.2 | 254 ms |
| 4 | 398.0 | 99.5 | 504 ms | 414.5 | 103.6 | 530 ms |
| 8 | 673.2 | 84.2 | 508 ms | 826.4 | 103.3 | 502 ms |
| 16 | 1246.5 | 77.9 | 740 ms | 1440.4 | 90.0 | 505 ms |

- 接受率：512 输出 **43.1%**，1024 输出 **43.7%**；tokens/step 分别约
  **3.15** 和 **3.18**。
- 脚本的“聚合”是 `单流 × 并发` 的估算值，不是 wall-clock 总 token 吞吐。

### 2.2 双前缀 8K→1024（wall-clock 实测）

```bash
python3 scripts/bench_dual.py --prompt-repeat 11 --max-tokens 1024 \
  --concurrency 2,4,8,16 --repeats 3
```

| 并发 | 总吞吐 (tok/s) | 单流 (tok/s) | TTFT (ms) |
|---:|---:|---:|---:|
| 2 | 213.7 | 112.0 | 434 |
| 4 | 410.5 | 106.8 | 345 |
| 8 | 777.1 | 104.8 | 523 |
| 16 | 1337.2 | 90.9 | 664 |

> 8K 数据的绝对值强依赖接受率抽签与 batch 组成，**不同进程之间的聚合吞吐不可直接
> 对比**。比较优化收益时请固定 prompt、temperature=0、并发，并优先看**每步时延**。
> 2.1 与 2.2 的聚合口径不同，不要直接横向比较；双前缀 2.2 更适合评估 DP2 实际
> 并发能力。

---

## 3. 关键优化的实测收益

| 优化 | 收益（参考） | 说明 |
|---|---|---|
| Model Runner V2（`VLLM_USE_V2_MODEL_RUNNER=1`） | 单流 +36%，并发 32 +68% | 相对 V1；需要补丁与 V2 兼容 |
| `HCCL_OP_EXPANSION_MODE=AIV` | 每步 −10.8% | 通信展开从 HOST 下沉到 Vector Core |
| jemalloc `LD_PRELOAD` | 每步 −3.2%～5.1% | 分配器争用下降 |
| PGO 版 libpython（可选） | 每步 −3.2%～5.1% | 必须在目标机编译，本仓库不含产物 |
| 6 个 `VLLM_ASCEND_V41_*` gate | 合计数个百分点 | 槽位映射标量化、QLI mode 3、Engram WKV TP 分片等 |
| DSpark K=5 | 显著提升单流 | 训练即 5-token 块；K 增大反而更慢 |
| `FULL_DECODE_ONLY` + 6N 桶 | 避免 draft 回退 eager | 缺 6N 桶收益大幅缩水 |
| `multistream_overlap_shared_expert` | 每步 −1.3%～4.6% | 共享专家多流；与 fused MC2 互斥 |

> 百分比随负载、batch 形状与接受率波动，只用于判断方向，不应作为容量规划的唯一依据。

---

## 4. 参数取舍

### 上下文长度

`--max-model-len` 直接决定 KV cache 上限。**默认 360448**；下面的容量数字是在 1M
+ 16384 批次下的实测值：每 die KV ≈6.7 GiB（1,268,538 tokens/DP rank）；8192 批次时
约 9.7 GiB（2,091,049 tokens/DP rank）。缩短上下文可以换取更多并发余量，但会进入编译
缓存 key（改了就冷启动）：

```bash
DSV41_MAX_MODEL_LEN=1048576 bash scripts/serve.sh   # 回到 1M
DSV41_MAX_MODEL_LEN=262144 bash scripts/serve.sh
```

### 并发与批大小

| 参数 | 调大 | 调小 |
|---|---|---|
| `--max-num-seqs` | 聚合吞吐↑，单流时延↑，图桶需求↑ | 单流时延↓，吞吐↓ |
| `--max-num-batched-tokens` | prefill 吞吐↑，TTFT 抖动↑，激活峰值↑、KV↓ | TTFT 更稳，长 prompt 吞吐↓，KV↑ |
| `--gpu-memory-utilization` | KV 容量↑，OOM 风险↑ | 更安全，容量↓ |

`max-num-seqs` 与 `max-num-batched-tokens` 都参与调度器 hash，修改等于冷启动。

#### 8192 → 16384（实测容量）

| `max-num-batched-tokens` | 峰值激活 | KV cache | KV tokens / DP rank | 1M 上下文并发 |
|---:|---:|---:|---:|---:|
| 8192 | ≈4.1 GiB | ≈9.7 GiB | 2,091,049 | 1.99× |
| **16384（生产）** | **≈8.2 GiB** | **≈6.7 GiB** | **1,268,538** | **1.21×** |
| 32768（未采用） | ≈16.4 GiB（线性外推） | 预算为负 | < 1M | 无法满足 1M 上下文 |

32768 要让总占用回到 55.15 GiB 预算内，必须把 `max-model-len` 降到约 512K；
如果坚持 1M 上下文，**不要开 32768**。16384 是当前 1M 上下文下能保住的
最大批次。

### 图桶

默认 37 桶：

```
1,2,4,6,8,12,16,18,24,30,32,40,48,54,56,64,72,78,80,88,96,102,104,112,
120,126,128,136,144,150,152,160,168,174,176,184,192
```

- 6 的倍数保证 `num_reqs × (1+K)` 命中；
- 桶越密，图捕获时间与显存开销越高；
- 桶越少，越多 batch 形状退回 eager，decode 变慢。

### 推测解码

```bash
DSV41_SPEC_TOKENS=5     # 默认；DSpark 训练块大小
```

更大的 K 会增加 draft 开销；若接受率不足，净收益可能为负。
通过 `/metrics` 的接受率判断是否需要调整。

---

## 5. 测量注意事项

1. **预热**：首个请求包含算子编译/图捕获，必须丢弃。
2. **温度 0**：降低采样随机性，但 MoE 路由与并发调度仍会带来波动。
3. **不要跨进程比吞吐**：接受率抽签差异会盖过优化收益；优先比较每步时延。
4. **固定 prompt**：prefix cache 命中会显著改变 TTFT；压测前清理或统一预热。
5. **观察显存**：`npu-smi info` 与日志中的 `Free memory on device` 行可用于判断
   是否触及 HBM 上限。
6. **NFS 抖动**：权重在 NFS 上时，首次加载与页回收会引入额外方差。
7. **router 前缀亲和**：相同 prompt 会被粘到已有 KV 前缀的 rank；测 DP2 并发上限
   要用 `bench_dual.py` 的双前缀，或人为制造两个/多个长前缀，否则低并发可能只
   压到一个 rank。
8. **聚合口径**：`bench_8k.py` 的 aggregate 是 `单流 × 并发` 估算；
   `bench_dual.py` 按实际总 token / wall time 统计，跨脚本不要直接比吞吐。

---

## 6. 容量规划参考

| 指标 | 数值 |
|---|---|
| 单 die HBM | 64 GB（可用 ≈61.3 GiB） |
| 权重 / die | ≈38.3 GiB |
| KV cache / die @1M + 16384 批次 | ≈6.7 GiB（1,268,538 tokens/DP rank） |
| KV cache / die @1M + 8192 批次 | ≈9.7 GiB（2,091,049 tokens/DP rank） |
| 峰值激活 @16384 | ≈8.2 GiB |
| 峰值激活 @8192 | ≈4.1 GiB |
| 非 torch + 图 | ≈3.8–4.0 GiB |
| 典型整机聚合吞吐 | ≈2.25k tok/s @ 并发 32 / 2K 双前缀（实测）；≈1.34k @ 并发 16 / 8K→1024 双前缀（实测） |

2K 单热前缀 @c32 约 1.83k tok/s；双前缀约 2.25k，说明热前缀亲和会牺牲部分
DP2 利用率。需要更高并发时，优先评估缩短 `max-model-len` 或增加节点，而不是继续抬高
`gpu-memory-utilization`——后者会把 OOM 风险转成运行时抖动。
