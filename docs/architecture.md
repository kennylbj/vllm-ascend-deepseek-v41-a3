# 架构与运行时

## 1. 总体拓扑

生产默认使用 external LB + prefix router（`DSV41_LB_MODE=external`）：

```
                         ┌─────────────────────────────────────┐
 OpenAI 兼容客户端 ──────▶│ prefix_router.py (0.0.0.0:8021)     │
 (HTTP / SSE)             │  longest-prefix + least-active      │
                         └───────┬─────────────────┬───────────┘
                                 │                 │
                    ┌────────────▼───┐      ┌──────▼─────────┐
                    │ APIServer DP0  │      │ APIServer DP1  │
                    │ 0.0.0.0:8022   │      │ 0.0.0.0:8023   │
                    └───────┬────────┘      └──────┬─────────┘
                            │ KV events            │ KV events
                       tcp://*:5557          tcp://*:5558
                            │                      │
                    ┌───────▼────────┐      ┌──────▼─────────┐
                    │ EngineCore DP0 │      │ EngineCore DP1 │
                    │ TP8 / EP       │      │ TP8 / EP       │
                    └───────┬────────┘      └──────┬─────────┘
                            │                      │
        ┌───────────────────┴───┐      ┌───────────┴───────────┐
        ▼        ▼        ▼      ▼      ▼        ▼        ▼      ▼
     Worker   Worker  Worker  Worker Worker  Worker  Worker  Worker
     DP0TP0   DP0TP1  ...     DP0TP7 DP1TP0  DP1TP1  ...     DP1TP7
        │                                                       │
        └──────────────── 16 × Ascend 910C die ─────────────────┘
```

`DPSupervisor` 监听 `:8020`，仅聚合两个子 rank 的 `/health`。
`DSV41_LB_MODE=internal` 时回退到单端口 8021 + vLLM 内置 DP 负载均衡，
拓扑为“客户端 → 一个 ApiServer → 两个 EngineCore”；两种模式共用同一套
rootfs、补丁和启动参数，只改变 DP 的端口布局。

- **DP2 × TP8** 正好用满 16 个 die（8 张 910C，每张 2 die）。
- `--enable-expert-parallel`：384 个 routed experts 按全局 EP 组切分。
- external 模式：router 按 KV 前缀亲和选择 rank0/rank1，再转发到独立 ApiServer；
- internal 模式：ApiServer 内部的 DP 协调器按 inflight/waiting/KV 使用率分发。
- 所有 NPU worker 的进程名以 `VLLM::` 开头，这是清理脚本的可靠匹配依据
  （`pgrep -f` 会误匹配 shell 自身命令行，不能用于杀进程）。

## 2. 为什么是“裸 rootfs”

官方 Ascend 推理镜像把以下内容打包在一起：

| 内容 | 位置（rootfs 内） |
|---|---|
| CANN 9.2.0 | `/usr/local/Ascend/cann-9.2.0` |
| NPU driver 用户态库 | `/usr/local/Ascend/driver/lib64` |
| Python 3.12.13 | `/usr/local/python3.12.13` |
| vLLM 0.30.0 源码 + 预编译 rust/C 扩展 | `/vllm-workspace/vllm` |
| vLLM Ascend 0.30.0rc2 源码 + `.so` | `/vllm-workspace/vllm-ascend` |

本方案默认不用容器运行时（`DSV41_RUNTIME=native`）。离线 / 旧 glibc 宿主改用
`DSV41_RUNTIME=docker`：镜像里的二进制在宿主 glibc 2.28 上无法运行，所以服务跑在
由本机镜像创建的长驻容器里（`--net=host --ipc=host --pid=host`），控制面脚本
（watchdog / kill_vllm / doctor）仍在宿主执行，因为共享 PID 命名空间后它们能直接看到容器内进程。
详见 [`offline-deployment.md`](offline-deployment.md)。

native 运行时：

1. 直接按 OCI 层解包到宿主目录；
2. 通过 `PYTHONHOME` / `PYTHONPATH` / `LD_LIBRARY_PATH` / `ASCEND_HOME_PATH`
   指向 rootfs 内部；
3. 宿主 NPU 驱动设备节点直连，不映射、不隔离设备。

好处：

- 与宿主系统的 Python / vLLM / CANN 完全隔离，不影响同机其它模型；
- 不需要 docker/podman，A3 notebook 这类受限环境也能跑；
- 镜像里的预编译算子直接复用，无需重新编译。

### 环境变量要点

| 变量 | 值 | 说明 |
|---|---|---|
| `PYTHONHOME` | `$ROOTFS/usr/local/python3.12*`（自动探测，可 `DSV41_PYTHONHOME` 覆盖） | 避免用到宿主 Python 标准库 |
| `PYTHONPATH` | CANN python/site-packages + vllm-ascend + vllm | 顺序不能反 |
| `ASCEND_HOME_PATH` | `$ROOTFS/usr/local/Ascend/cann-*`（自动探测，可 `DSV41_CANN_HOME` 覆盖） | **必须用具体版本目录**，不能指向 `ascend-toolkit/latest` 符号链接，否则驱动的主机路径检查会失败 |
| `LD_LIBRARY_PATH` | CANN lib64 / opskernel / nnengine / driver lib64 | 缺一不可 |
| `LD_PRELOAD` | `libjemalloc.so.2` | 可选，显著降低长 decode 的分配器抖动 |

## 3. 请求链路

1. 客户端 POST `/v1/chat/completions` 到 router `:8021`；
2. router 调 `/tokenize` 取 prompt token，计算 128-token 前缀指纹链；按两个 rank
   的 KV 事件选择最长命中者，平局选 active 更少者；无命中则走 sticky/最少 active；
3. 请求转发到 rank 的独立 ApiServer `:8022` 或 `:8023`；
4. ApiServer 做 reasoning parser / tool parser；
5. 调度器按 prefix cache 命中与 KV 块分配决定 batch；
6. EngineCore 执行 `execute_model`，通过 HCCL 在 TP/EP 组内通信；
7. DSpark draft 模型先产 K=5 个候选 token，target 模型一次前向验证；
8. 采样后流式返回 SSE chunk，经 router 原样透传；`reasoning_content` 与
   `content` 分离。

decode 阶段命中 `FULL_DECODE_ONLY` ACL graph；prefill 走 eager/编译混合路径。
图桶按 `num_reqs × (1 + K)` 设计，K=5 时以 6 为步长。

## 4. 端口、PID 与日志

| 项 | 默认值 | 覆盖变量 |
|---|---|---|
| 客户端 / router 端口 | 8021 | `DSV41_PORT` |
| DP supervisor 端口 | 8020 | `DSV41_DP_SUPERVISOR_PORT` |
| rank0 / rank1 API 端口 | 8022 / 8023 | `DSV41_DP_BASE_PORT` |
| KV 事件 ZMQ 端口 | 5557 / 5558 | `DSV41_KV_EVENTS_BASE_PORT` |
| ApiServer 鉴权 | `--api-key` | `DSV41_API_KEY` |
| 服务 PID（supervisor） | `/tmp/dsv41_serve.pid` | `DSV41_SERVE_PID_FILE` |
| router PID | `/tmp/dsv41_router.pid` | `DSV41_ROUTER_PID_FILE` |
| watchdog PID | `/tmp/dsv41_watchdog.pid` | `DSV41_WATCHDOG_PID_FILE` |
| 服务日志 | `logs/serve_<timestamp>.log` | `DSV41_RESULTS` |
| router 日志 | `logs/router.log` | `DSV41_ROUTER_LOG` |
| watchdog 日志 | `logs/watchdog.log` | `DSV41_LOG_DIR` |

PID 文件缺失时，脚本会扫描 `/proc/*/cmdline`：external 模式匹配
`--data-parallel-multi-port-external-lb` 的 supervisor，internal 模式匹配
`vllm.entrypoints.cli.main ... --port <PORT>`；router 按 `prefix_router.py` 匹配。

## 5. 显存布局（实测，单 die）

> 下表为 `--max-model-len 1048576` 下的实测值；默认上下文现为 360448，KV cache 预算
> 相应更宽松，未重新采集。

| 项 | 8192 批次 | 16384 批次（当前） |
|---|---:|---:|
| 权重 | ≈38.3 GiB | ≈38.7 GiB |
| 峰值激活 | ≈4.1 GiB | ≈8.2 GiB |
| 非 torch 内存（CANN/HCCL 等） | ≈3.1 GiB | ≈1.5 GiB |
| ACL graph | ≈0.7–0.9 GiB | ≈0.7 GiB |
| KV cache（1M 上限） | ≈9.7 GiB | ≈6.7 GiB |
| KV token 容量 / DP rank | ≈2.09M | **1,268,538** |
| 1M 上下文并发 | ≈1.99× | ≈1.21× |
| `gpu-memory-utilization` | 0.90（55.15 GiB 预算） | 0.90（55.15 GiB 预算） |

16384 批次用 KV 余量换 prefill 吞吐；32768 需要约 16.4 GiB 激活，在 1M 上下文
下放不进 0.90 预算（即使把利用率抬到 0.98，KV 也不足 1M），因此生产停在 16384。

如修改 `--max-model-len`、`--max-num-seqs` 或 `--max-num-batched-tokens`，KV cache
与图内存都会变化；日志里的 `Free memory on device` 行会给出可选
`--kv-cache-memory` 建议值。

## 6. 启动阶段

| 阶段 | 典型耗时 | 日志标志 |
|---|---|---|
| Python / CANN 初始化 | 数秒 | `Available plugins for group vllm.platform_plugins` |
| 权重加载（142 shard） | ≈4–5 min | `Loading safetensors checkpoint shards` |
| 引擎初始化（KV cache / warmup） | ≈70 s | `init engine (profile, create kv cache, warmup model) took ...` |
| ACL graph 捕获 | 数十秒 | `ACL graph replay is active for the V2 target model` |
| HTTP 就绪 | — | `Application startup complete.` + `/health` 200 |

冷启动总时长约 6 分钟；有编译缓存时略快，但权重加载仍是主要开销。
