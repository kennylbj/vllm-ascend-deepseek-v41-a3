# 故障排查

先跑这四条，能定位大部分问题：

```bash
bash scripts/doctor.sh              # 只读预检
STATUS=1 bash scripts/serve.sh
bash scripts/watchdog.sh status
tail -n 200 logs/serve_*.log
```

---

## 1. 启动阶段

| 症状 | 原因 | 修复 |
|---|---|---|
| `ModuleNotFoundError: No module named 'acl'` | `PYTHONPATH` 缺 CANN 的 python/site-packages | 使用 `scripts/serve.sh`（已包含），不要手写 env |
| `ImportError: cannot import name 'is_fake_tensor'` | rootfs 解包未处理 OCI whiteout，旧 torch 文件残留 | `bash scripts/setup_rootfs.sh --force` 重新解包 |
| `Permission denied: bishengir-compile` / `triton-mlir-opt` | 解包丢失执行位 | `bash scripts/setup_rootfs.sh` 会修复；或手工 `chmod +x` |
| `Free memory on device ... less than desired GPU memory utilization` | 启动时有其它 worker 占用 HBM | `bash scripts/kill_vllm.sh` 后重启 |
| worker 卡在 `D` 状态（`__lock_page`） | 权重在 NFS 上并发 mmap 触发 page-lock 死锁 | 把权重复制到本地 NVMe；或减少并发加载线程 |
| ApiServer 超时退出但 worker 仍在 | 大模型加载/编译超过默认超时 | 已内置 `VLLM_ENGINE_READY_TIMEOUT_S=36000`；勿手工降低 |
| `Address already in use` | 旧 ApiServer/EngineCore 残留 | `bash scripts/kill_vllm.sh` |
| `No module named vllm_ascend` | `PYTHONPATH` 顺序/路径错误 | 用 `serve.sh`；确认 rootfs 解包完整 |
| 启动极慢或卡在权重加载 | NFS 抖动、页缓存不足、CPU 绑核冲突 | 本地盘 + 足够内存；确认 `free -g` |
| `ASCEND_HOME_PATH` 相关错误 | 指向了 `ascend-toolkit/latest` 符号链接 | 指向具体目录 `cann-9.2.0`（`serve.sh` 已这样做） |

---

## 2. 运行阶段

| 症状 | 原因 | 修复 |
|---|---|---|
| `/health` 长期非 200，但进程存活 | 引擎仍在初始化/捕获图 | 等待；首次约 6 分钟。超过 180 s 由 watchdog 宽限 |
| 请求 401 | 服务启用了 `--api-key`，客户端未带 Bearer | 加 `Authorization: Bearer $DSV41_API_KEY` |
| 请求 404 `model not found` | `model` 字段与 `--served-model-name` 不一致 | 必须完全一致：`dsv41` |
| 输出乱码 | `--block-size` 不是 128 | 恢复 `DSV41_BLOCK_SIZE=128` |
| 推理明显变慢，日志无报错 | gate 静默失效 / 图桶缺失 / draft 回退 eager | `apply_patches.sh verify`；检查图桶含 6N |
| 日志出现 `Unknown vLLM environment variable` | spawn worker 不继承 env；gate 未固化默认值 | 重新 `apply_patches.sh apply` 并重启 |
| `apply_patches.sh status` 某些补丁显示冲突 | 补丁序列部分应用 / 手工改过目标文件 / 镜像版本不符 | `check` 定位；必要时 `revert --force` 后重新 apply |
| `verify` 报 `model.py` gate 默认值 0 | Engram WKV TP 在 `layer.py` 与 `model.py` 各读一次，旧版固化逻辑漏了后者 | 重新运行 `apply_patches.sh apply`（已修复映射） |
| 长请求偶发 OOM | KV cache 或激活峰值超出预算 | 降低 `DSV41_MAX_MODEL_LEN` / `MAX_NUM_SEQS` / `GPU_MEM_UTIL` |
| watchdog 反复重启 | 服务确定性启动失败（权重损坏 / 补丁冲突 / 端口占用） | 先 `watchdog.sh stop`，手工 `serve.sh run` 看完整错误 |
| `/metrics` 里推测解码计数为 0 | 未启用 DSpark / 指标尚未产生 | 确认 `--speculative-config` 与 warm-up 请求 |
| TTFT 忽高忽低 | prefix cache 命中差异、并发排队 | 统一 prompt/并发后再比较 |
| `/health` 503，但 8022/8023 各自 200 | router 健康轮询尚未完成 | 等 5 s；看 `logs/router.log` |
| `/router/stats` 401 | stats 默认要求 `DSV41_API_KEY` | 带 `Authorization: Bearer ...` |
| `cached_prefix_digests=0` | 请求 <128 token / router 刚重启 / 无 KV 事件 | 发 >128 token 请求；确认 5557/5558 |
| 路由长期只打一个 rank | 热前缀；或 `legacy` 策略下 skew 太小 | 默认 `cost` 策略会按 token / KV 负载分流；`legacy` 下调大 `DSV41_ROUTER_MAX_SKEW` |
| 两 rank 流量不均但健康 | 前缀亲和的正常代价 | 看 `/router/stats` 的 `inflight_tokens` / `kv_usage`；需要纯吞吐可回退 `DSV41_LB_MODE=internal` |
| 总输出 tok/s 大幅波动、`Running` 不低但 `generation throughput` 只有个位数 | DP+EP lockstep：一个 rank 的长 prefill 分块拖慢两个 rank 的 decode；或一个 rank KV 被打满 | `python3 scripts/analyze_serve_log.py logs/serve_*.log`，对照 `docs/performance.md` 第 7 节 |
| router 起不来 | 缺 aiohttp/pyzmq/msgspec | 用 rootfs Python 启动（serve.sh 已保证）或 `pip install -r requirements.txt` |

router 诊断：

```bash
tail -n 100 logs/router.log
curl --noproxy '*' http://127.0.0.1:8021/router/stats \
  -H "Authorization: Bearer $DSV41_API_KEY"
grep -E ':15B5|:15B6' /proc/net/tcp        # 5557 / 5558 的十六进制
ss -ltnp | grep -E ':(8020|8022|8023)'    # ZMQ 端口用 /proc/net/tcp 更可靠
ROUTER_RESTART=1 bash scripts/serve.sh    # 只重启 router，不动 vLLM
```

---

## 3. 清理与进程

| 症状 | 原因 | 修复 |
|---|---|---|
| `pgrep -f "Worker_DP"` 杀错进程 | 命令行自身被匹配 | 用 `ps -eo comm` 匹配 `VLLM::` 前缀；脚本已如此实现 |
| 停服务后端口仍被占用 | ApiServer/EngineCore 子进程残留 | `bash scripts/kill_vllm.sh`（只清理本部署的进程，含孤儿 worker） |
| NPU 仍显示 `VLLMWorker_DP` | 可能是本部署的残留，也可能是同机其它模型 | 先 `kill_vllm.sh`；仍在的多半不属于本部署（脚本刻意不动它们）。确认整机只跑本服务时才用 `kill_vllm.sh --all` |
| 端口释放但 HBM 未释放 | 进程处于不可中断状态 | 等待或重启节点；确认没有 NFS 卡死 |
| watchdog 把服务反复拉起 | 手工 `STOP=1` 时 watchdog 仍在运行 | 先 `watchdog.sh stop` 再停服务 |

诊断命令：

```bash
ps -eo pid,ppid,comm,etime | grep -E 'VLLM::|vllm.entrypoints'
npu-smi info | grep -E 'VLLMWorker|/ 65536'
ss -ltnp | grep -E ':(8020|8021|8022|8023)'
grep -E ':15B5|:15B6' /proc/net/tcp   # 5557/5558
```

---

## 4. rootfs / 镜像

| 症状 | 原因 | 修复 |
|---|---|---|
| 解包中断 | 网络或磁盘失败 | 重新执行 `setup_rootfs.sh`，已完成层会跳过 |
| blob digest mismatch | 下载不完整 / 镜像被替换 | 删除 `*.layers/L*.tar` 后重跑；确认 `.dsv41-image` 的 digest |
| 磁盘不足 | 层缓存 + 解包同时占空间 | 确保 ≥25 GiB；默认每层解包后删除压缩包 |
| rootfs 树缺少 `.so` | 镜像层缺失或解包异常 | `setup_rootfs.sh --force`；检查 `<rootfs>/.dsv41-image` |
| `git apply` 冲突 | 补丁与镜像版本不匹配 | `apply_patches.sh check` 定位；重新生成补丁 |
| 回滚失败 | rootfs 不是 git 工作树 | 重新解包 rootfs |

---

## 4b. docker 运行时（离线 / 旧 glibc 宿主）

| 症状 | 原因 | 修复 |
|---|---|---|
| `version 'GLIBC_2.3x' not found`（直接用 rootfs 跑） | 宿主 glibc 2.28 < 镜像的 2.35 | 改用 `DSV41_RUNTIME=docker`（`docs/offline-deployment.md`） |
| `container.sh up`：`本机没有镜像` | 镜像未加载或 tag 不同 | `docker images`；`docker load -i`；或设置 `DSV41_IMAGE` |
| `缺少必需的挂载源 …driver/lib64 / npu-smi` | 宿主驱动路径不在默认位置 | 确认 `/usr/local/Ascend/driver/lib64`、`/usr/local/bin/npu-smi` 存在 |
| `DSV41_PID_DIR 不能是 /tmp` | docker 运行时要把 pid 目录挂进容器 | `DSV41_PID_DIR=/run/dsv41` |
| `容器 dsv41 未运行` | 容器被停止/删除 | `bash scripts/container.sh up` |
| 改了挂载/参数没生效 | 容器是创建时固化的 | `bash scripts/container.sh recreate`（补丁会自动重新应用） |
| `DSV41_PROBE=1 bash scripts/serve.sh` 报 `ImportError` / `libXXX.so` | 驱动库未挂载或环境缺失 | 看报错库名；`container.sh shell` 里 `ldd`；必要时 `DSV41_CONTAINER_SOURCE_ENV=1` |
| `Free memory on device … less than desired GPU memory utilization` | 同机其它进程占用了这些 die 的 HBM | 停掉占用者，或调低 `DSV41_GPU_MEM_UTIL` |
| `serve.sh run` 里 Ctrl-C 没停掉服务 | docker exec 不转发信号（非 TTY 时） | 在交互终端运行（自动 `-it`），或另开终端 `STOP=1 bash scripts/serve.sh` |
| 重启宿主后服务没起来 | 未启用 systemd 单元 | `systemctl enable --now dsv41-container` |

---

## 5. 启动日志里的两条“已知无害”警告

DeepSeek-V4.1 启动时经常出现下面两条警告。经核对源码，它们**不影响本模型**：

### `NPU Triton causal_conv1d_update is unavailable`

```text
NPU Triton causal_conv1d_update is unavailable (cannot import name
'causal_conv1d_update_npu' from vllm_ascend.ops.triton.mamba.causal_conv1d);
falling back to the PyTorch implementation, which syncs per request and
therefore stalls ACL graph capture at decode-FULL.
```

原因：

- 上游 rc2 中 `vllm_ascend/ops/triton/mamba/causal_conv1d.py` 只保留
  `extract_last_width`；`causal_conv1d_update_npu` 已在
  `f0a938956 [Refactor][Ops] Remove causal_conv1d_update_npu` 中删除；
- 之后的 GLM-5.3 提交又在 `patch_triton.py` 里无条件尝试导入该符号，属于上游
  遗留的无效绑定；
- 该 `causal_conv1d_update` 只被 **GDN / KDA** 模型（GLM-5.3、Qwen3-Next、
  Kimi-KDA 等）调用。 DeepSeek-V4.1 的模型代码不引用它。

结论：**对 DeepSeek-V4.1 无影响，无需处理。**  
如果以后在同一 rootfs 上跑 GLM-5.3/KDA，才需要改用
`fla_npu.ops.ascendc.causal_conv1d_update` 或恢复删除前的 Triton 实现。

### `Failed to import the DeepSelect extension (vllm._deepselect_C)`

原因：

- `vllm._deepselect_C` 是基于 **CUDA 12.9+ / SM100/SM103** 编译的可选扩展；
- Ascend 镜像不会构建它，CMake 中该 target 在非 CUDA 架构下本来就是空目标；
- NPU 上 `SparseIndexerTopk` 会自动选择昇腾可用的 `per_row`
  （`ops.top_k_per_row_decode`）；`deep_select` 只在 CUDA 上作为显式可选项。

结论：**预期行为，DeepSeek-V4.1 使用 `per_row` 路径，无需处理。**

## 6. 获取帮助所需信息

提交问题时请附带：

```bash
uname -a
npu-smi info
cat "$DSV41_ROOTFS/.dsv41-image"
bash scripts/apply_patches.sh status
bash scripts/watchdog.sh status
tail -n 300 "$(ls -t logs/serve_*.log | head -1)"
```

请先脱敏：日志里可能含有请求内容、IP 与 API key（key 不会出现在 serve 日志中，
但配置文件和 shell history 可能包含）。
