# Prefix / KV-aware router（external LB）

DP2 下两个 EngineCore 的 **KV cache 互相独立**。默认的 `--data-parallel-size 2`
由一个 ApiServer 按 inflight / waiting / KV 使用率做负载均衡，同一段前缀可能落到
不同 rank，prefix cache 只能各自命中，跨 rank 无法复用。

本仓库提供 `scripts/prefix_router.py`：把 DP server 拆成两个独立的 API 端口，
由路由器根据每个 rank 的 **KV cache block 事件**把相同前缀的请求粘到同一 rank。
这是可选的生产路由模式，也是生产配置使用的模式。

> 不启用时保持 `DSV41_LB_MODE=internal`，行为与早期单端口 DP 完全一致。

## 1. 拓扑与端口

```
                         ┌─────────────────────────────────────┐
 client :8021 ──────────▶│ prefix_router.py                    │
 (OpenAI/SSE)            │  /v1/*  /health  /metrics            │
                         └───────┬─────────────────┬───────────┘
                                 │ longest prefix  │
                    ┌────────────▼───┐      ┌──────▼─────────┐
                    │ APIServer DP0  │      │ APIServer DP1  │
                    │ :8022          │      │ :8023          │
                    └───────┬────────┘      └──────┬─────────┘
                            │ KV events            │ KV events
                       tcp://*:5557          tcp://*:5558
                            │                      │
                    ┌───────▼────────┐      ┌──────▼─────────┐
                    │ EngineCore DP0 │      │ EngineCore DP1 │
                    │ TP8 / EP       │      │ TP8 / EP       │
                    └────────────────┘      └────────────────┘

              DPSupervisor :8020 只提供聚合 /health
```

| 端口 | 进程 | 用途 |
|---:|---|---|
| `8021` | `prefix_router.py` | 客户端入口；SDK / 客户端都指向这里 |
| `8020` | `DP supervisor` | 子 rank 存活聚合，供运维探活 |
| `8022` | `APIServer DP0` | rank0 OpenAI 兼容 API |
| `8023` | `APIServer DP1` | rank1 OpenAI 兼容 API |
| `5557` / `5558` | EngineCore ZMQ PUB | 每个 rank 的 KV cache 事件 |

端口均可覆盖：

```bash
DSV41_PORT=8021                  # router
DSV41_DP_SUPERVISOR_PORT=8020
DSV41_DP_BASE_PORT=8022          # rank0；rankN = base + N
DSV41_KV_EVENTS_BASE_PORT=5557
```

> 单机 2 个 DP rank 时 `--data-parallel-size-local=2`；多机部署需要按节点拆分
> local size 与 start rank，本仓库当前只验证单机 A3。

## 2. 启用

```bash
# config/dsv41.local.env
DSV41_LB_MODE=external
DSV41_DP=2
DSV41_TP=8
DSV41_KV_EVENTS=1
DSV41_DP_SUPERVISOR_PORT=8020
DSV41_DP_BASE_PORT=8022
DSV41_KV_EVENTS_BASE_PORT=5557
DSV41_ROUTER_POLICY=cost          # cost（默认）| legacy，见 3.2
DSV41_ROUTER_KV_HIGH=0.90         # KV 使用率 ≥ 此值视为饱和
DSV41_ROUTER_SAT_PENALTY=32768    # 饱和/排队 rank 的罚分（token）
DSV41_ROUTER_PREFILL_WEIGHT=0.5   # 已排队 prefill token 的权重
DSV41_ROUTER_LOAD_INTERVAL=2      # 抓取 /metrics 的间隔（秒），0 关闭
DSV41_ROUTER_MAX_SKEW=8           # 仅 legacy 使用

bash scripts/serve.sh
STATUS=1 bash scripts/serve.sh
```

`serve.sh` 会：

1. 用 `--data-parallel-multi-port-external-lb` 启动 DP supervisor；
2. 子进程自动得到 `--data-parallel-rank 0/1`、端口 `8022/8023`；
3. 给 vLLM 传 `--kv-events-config`；publisher 会按 rank 自动把端口偏移到
   `5557/5558`；
4. 启动 router，监听 `DSV41_PORT`，把请求转发到两个 rank；
5. 健康检查仍以 `http://127.0.0.1:8021/health` 为准。router 在**两个 rank 都
   健康**时才返回 200；任一 rank 未就绪时返回 503。

只重启路由器（不动 vLLM，约 1 秒中断）：

```bash
ROUTER_RESTART=1 bash scripts/serve.sh
```

## 3. 路由算法

### 3.1 KV 事件 → 前缀指纹链

vLLM 的 `BlockStored` 事件包含：

- `block_hashes`：本次新写入的 block hash；
- `parent_block_hash`：前缀链上的父 block；
- `token_ids`：这些 block 的 token；
- `block_size`：默认 128。

router 不依赖 vLLM 内部的 hash 算法，而是自己维护一条 **token 指纹链**：

```
digest_0 = BLAKE2b(tokens[0:128])
digest_i = BLAKE2b(digest_{i-1} || tokens[i*128:(i+1)*128])
```

每个 rank 保存按插入顺序排列的 digest 集合。收到请求时：

1. 调一次 `/tokenize` 得到 prompt token（在健康 rank 间轮询，避免所有 tokenize
   压在 rank0 的 API server 上；按 messages/prompt 缓存，避免重复 tokenize）；
2. 计算 prompt 的前缀 digest 链；
3. 对每个 rank 求“从第一个 block 开始连续命中的最长长度”；
4. 按 3.2 的策略在各 rank 之间打分选择。

同一 block 会被 12 个 KV cache group 各发一次事件；router 用引用计数去重，
`BlockRemoved` 逐次递减，避免某个 group 移除就让整段前缀“消失”。

### 3.2 打分策略（`DSV41_ROUTER_POLICY=cost`，默认）

对每个健康 rank 计算一个分数，**单位全部是 token**，取最低者（并列取 active 更少者）：

```
score(rank) = 该 rank 上未命中的 prompt token
            + PREFILL_WEIGHT × 该 rank 上已派发、尚未产出首个响应字节的 prefill token
            + SAT_PENALTY    （若该 rank 饱和：KV 使用率 ≥ KV_HIGH，或 num_requests_waiting > 0）
```

- 前缀命中越长，未命中 token 越少，所以亲和性自然保留；
- 已排队的 prefill 越多分数越高，一批冷请求会自动摊到两个 rank（router 自己记账，
  不依赖抓取延迟）；
- 饱和判断来自每 `LOAD_INTERVAL` 秒抓取的 rank `/metrics`
  （`vllm:kv_cache_usage_perc`、`vllm:num_requests_waiting`）；数据超过 10 s 视为过期并忽略；
- 命中很长的请求（命中 token > `SAT_PENALTY`）即使目标 rank 已饱和也留在原 rank，
  命中很短的请求则会被分流，避免继续堆在已经排队的 rank 上。

三个参数都是经验值（默认 `KV_HIGH=0.90`、`SAT_PENALTY=32768`、`PREFILL_WEIGHT=0.5`），
需要用自己的负载调；改完 `ROUTER_RESTART=1 bash scripts/serve.sh` 即可生效。

**为什么不能只按请求数均衡。** 2026-11-04 的生产日志（见 `docs/performance.md` 的
“DP lockstep 与 prefill 停顿”）里，平均 prompt 约 66–84K token，每个 rank 的 KV 只有
848,184 token，实际并发被 KV 而不是 `--max-num-seqs` 限制。09:40–10:40 rank0 的 KV
长期 86–87%（p90 约 97%）并有约 3 个请求排队，rank1 只有 46–58%；两边 active 请求数
只差 1–2 个，所以按请求数的旧策略看不到这个失衡，rank0 因 KV 抖动命中率从 84% 掉到 58%。

`/router/stats` 的 `spills` 是“没有选中命中最长的 rank”的次数，`sat_avoid` 是“因为饱和罚分
改变了选择”的次数，`fallback` 是完全没有命中的请求数。

**legacy 策略**（`DSV41_ROUTER_POLICY=legacy`，回退用）：命中最长者优先；并列取 active 少者；
没有命中时先查 `sticky` 表（`prompt body hash → rank`，最多 50k 条，LRU），再取 active 最少者；
目标 rank 的 active 比另一个多出 `DSV41_ROUTER_MAX_SKEW`（默认 8）时强制切换。

> 这是一个**前缀亲和（prefix-affinity）**路由器，不是全局最优调度器。它能做的是：减少
> 重复 prefill、避免某个 rank KV 打满。它**不能**消除 DP+EP 的 lockstep 停顿（一个 rank 的
> 长 prefill 会拖慢两个 rank 的 decode），那部分见 `docs/performance.md`。

### 3.3 KV 事件的边界情况

- **短于 128 token 的 prompt 不会产生 `BlockStored`**，因此没有亲和信息；
  这类请求只按 prefill 排队量 / 饱和 / active 打分。
- 事件流从 router 启动时开始消费。router 重启后，旧 KV cache 不会重放，
  需要新请求重新填充 digest；如需干净状态可重启 vLLM 或调用 vLLM 的
  reset prefix cache 接口。
- `AllBlocksCleared` 会清空该 rank 的 digest 表（计入 `digest_resets`）。
- 事件积压时 publisher 有 HWM。router 对**当前存活**的 digest 设 200k 上限，超出时
  淘汰最旧的 10%（计入 `digest_evicted`，并打 WARNING）。正常情况下存活 digest 只有
  KV 池大小量级（本部署约 8K），不会触发。
  - 旧版本按“累计插入次数”计数，约每 20 万个新 block 就会把整个 rank 的 digest 全部
    清空（vLLM 不重放事件，之后该 rank 的已缓存前缀对 router 不可见）。现已修复；
    按 2026-11-04 日志里 vLLM 的未命中 token 计数推算，那次运行里每个 rank 大约各发生过
    一次（旧版不记录，无法直接确认）。新版用 `digest_evicted` / `digest_resets` 可见。

## 4. 观测

```bash
# 健康（两个 rank 都健康才 200）
curl --noproxy '*' http://127.0.0.1:8021/health

# 路由统计；配置了 API key 后必须带 Bearer
curl --noproxy '*' http://127.0.0.1:8021/router/stats \
  -H "Authorization: Bearer $DSV41_API_KEY"
```

返回示例（数值仅为示意）：

```json
{
  "policy": "cost",
  "ranks": [
    {"url": "http://127.0.0.1:8022", "active": 9, "healthy": true, "cached_prefix_digests": 8139,
     "inflight_tokens": 612000, "pending_prefill": 41000, "kv_usage": 0.77, "waiting": 0},
    {"url": "http://127.0.0.1:8023", "active": 9, "healthy": true, "cached_prefix_digests": 5224,
     "inflight_tokens": 540000, "pending_prefill": 0, "kv_usage": 0.52, "waiting": 0}
  ],
  "digest_evicted": 0,
  "digest_resets": 0,
  "requests": 2132,
  "affinity_hits": 2020,
  "fallback": 105,
  "spills": 4,
  "sat_avoid": 11,
  "events": 5715276
}
```

- `cached_prefix_digests`：该 rank 当前可匹配的前缀块数；
- `inflight_tokens`：正在该 rank 上的请求的 prompt token 之和（KV 占用的近似）；
- `pending_prefill`：已派发但尚未出首个响应字节的未命中 prompt token（prefill 排队量）；
- `kv_usage` / `waiting`：最近一次抓到的 `/metrics`（`null` = 还没抓到或指标名不符）；
- `affinity_hits`：选中了命中最长的 rank 的请求数；
- `spills`：有命中、但因负载打分选了别的 rank 的请求数；
- `sat_avoid`：因饱和罚分而改变了选择的请求数；
- `fallback`：完全没有命中的请求数（legacy 下还包括 skew 切换）；
- `digest_evicted` / `digest_resets`：digest 表被安全上限淘汰的条数 / vLLM 清空前缀缓存的次数；
- `events`：消费的 KV 事件条数。

`GET /metrics` 会把两个 rank 的 Prometheus 文本合并（去重 HELP/TYPE），
因此 `prefix_cache_hits_total`、`num_requests_running` 等仍可直接抓取。
单 rank 指标也可直接访问 `:8022/metrics`、`:8023/metrics`。

## 5. 故障与回滚

| 症状 | 原因 | 处理 |
|---|---|---|
| `/health` 503，但两个 rank `/health` 200 | router 尚未收到健康轮询结果 | 等 5 s；检查 `logs/router.log` |
| `/health` 503，某个 rank 不健康 | 子 rank 启动失败 / 崩溃 | 看 `logs/serve_*.log`；watchdog 会整栈重启 |
| `/router/stats` 401 | 已启用 stats token | 带 `Authorization: Bearer $DSV41_API_KEY` |
| 路由全部 fallback、`cached_prefix_digests=0` | 没有 KV 事件 / prompt < 128 token / router 刚重启 | 发一个 >128 token 的请求；检查 5557/5558 |
| 端口 5557/5558 看不到 | 本机 `ss` 在某些 netns 下漏报 ZMQ | `grep -E ':15B5\\|:15B6' /proc/net/tcp`（5557/5558 十六进制） |
| 一个 rank KV 长期 >90% 且有排队，另一个很空 | 请求数均衡看不到 token / KV 负载 | 用 `cost`（默认）；看 `/router/stats` 的 `kv_usage` / `waiting` / `spills` / `sat_avoid`；调 `DSV41_ROUTER_SAT_PENALTY`、`DSV41_ROUTER_KV_HIGH` |
| `kv_usage` / `waiting` 一直是 `null` | `/metrics` 没抓到或指标名不符 | `curl :8022/metrics` 看是否含 `vllm:kv_cache_usage_perc` 与 `vllm:num_requests_waiting`；`router.log` 有 warning |
| `router.log` 出现 `stale connection ... retrying same rank` | 复用了被 uvicorn（5 s）关掉的 keep-alive 连接 | 已用 4 s keep-alive + 同 rank 重试，偶发无需处理 |
| 新策略表现不如预期 | 打分参数不适合当前负载 | `DSV41_ROUTER_POLICY=legacy ROUTER_RESTART=1 bash scripts/serve.sh`（只重启 router，约 1 s） |
| 想完全回退到 vLLM 内置 DP 均衡 | 不需要前缀亲和 | `DSV41_LB_MODE=internal` + `STOP=1 bash scripts/serve.sh` 后重启 |

router 只做转发，不修改返回体；上游 401/404/429 等状态码和 SSE 流会原样透传。
API key 仍由两个 rank 的 `--api-key` 校验，router 会把客户端的 `Authorization`
头转发给 `/tokenize` 与推理接口。

## 6. 安全说明

- `/router/stats` 默认要求 `DSV41_API_KEY`（也可用 `DSV41_ROUTER_STATS_TOKEN`
  单独指定）；未设置 key 时才匿名开放。
- `/metrics` 与传统 vLLM 一样默认无鉴权；如需限制，请在本地网络层控制
  `/metrics`、`/router/stats` 的访问。
- router 进程命令行**不再包含 API key**；key 通过环境变量继承。
