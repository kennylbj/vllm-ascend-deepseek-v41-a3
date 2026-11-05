# 离线部署指南：Kylin V10 + docker 运行时

本文面向**没有外网、宿主较老**的 A3 服务器（目标机：Kylin V10 aarch64，glibc 2.28，
Python 3.7.9，Docker 29.3.1，镜像 `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3` 已 `docker load`），
给出从零到可服务、再到日常运维的完整步骤。每一步都有**判定点（Gate）**，失败时先看
[第 12 节](#12-故障排查)。

> 同机若已有别的模型在占用这些 NPU，**启动本服务前请先自行停掉**（见第 6 节）。
> 本项目的所有清理动作只处理本部署自己的进程，不会去动其它任务。

---

## 1. 结论与原理

### 1.1 要不要在服务器上装 Python 3.12.13？

**不要。** Python 3.12.13 以及 torch / torch_npu / vLLM / vllm-ascend / CANN 都在镜像里
（`/usr/local/python3.12.13`），服务直接用容器里的这套。宿主的 Python 3.7.9 只是偶尔
跑一下标准库脚本（如 `tests/smoke_test.py`），不需要装任何包。

### 1.2 为什么不能像默认流程那样“解包 rootfs 直接跑”

默认（`DSV41_RUNTIME=native`）会把镜像里的 `python3.12` 等二进制直接放到宿主上执行，
用的是**宿主的** glibc。实际检查结果：

| 项 | 宿主 | 镜像 |
|---|---|---|
| OS | Kylin V10 | Ubuntu 22.04.5 |
| glibc | **2.28** | **2.35** |
| Python | 3.7.9 | 3.12.13 |

镜像二进制需要 `GLIBC_2.3x` 符号，在 2.28 上会直接报 `version 'GLIBC_2.3x' not found`。
所以改用 **docker 运行时**：服务跑在由本机镜像创建的容器里，用容器自己的 glibc；
宿主 NPU 驱动用户态库以 bind mount 方式挂进去（Ascend 官方 docker 用法，驱动库向后兼容）。

### 1.3 结构

```
宿主（Kylin）                               容器 dsv41（Ubuntu 22.04 镜像）
────────────────────────────                ───────────────────────────────────
watchdog.sh / kill_vllm.sh / doctor.sh      sleep infinity（--init）
        │  读 /proc、curl /health                 ▲
        │  （共享 PID 命名空间）                  │ docker exec
serve.sh ──(DSV41_RUNTIME=docker)──────────────▶ serve.sh（容器内重新执行）
apply_patches.sh ──────────────────────────────▶   ├─ vLLM supervisor + 2×TP8 workers
container.sh up/exec/shell/down                    └─ prefix_router.py
```

关键设计：

- 容器使用 `--net=host --ipc=host --pid=host --privileged`。端口就是宿主端口
  （router 8021、supervisor 8020、rank 8022/8023、KV 事件 5557/5558）。
- 因为共享 PID 命名空间，宿主上的 watchdog / kill_vllm / doctor 直接能看到容器里的
  进程（同样的 PID），所以这些控制脚本**不需要**进容器。只有需要镜像 Python / 源码的
  部分（`serve.sh` 启动路径、`apply_patches.sh`）会自动 `docker exec` 进容器。
- 仓库、模型、日志、pid 目录都按**同一绝对路径**挂进容器，宿主和容器里路径一致
  （vLLM 编译缓存与模型路径相关，不要事后改路径）。
- 补丁直接打在容器内的 `/vllm-workspace/vllm-ascend`（容器可写层）；`container.sh up`
  每次都会幂等地重新应用，所以重建容器也不会丢。

---

## 2. 目标机事实（已核对）

| 项 | 值 |
|---|---|
| 内核 / OS | 4.19.90，Kylin V10 aarch64，SELinux Disabled，firewalld inactive |
| NPU | 16 个 die（`/dev/davinci0..15` + `davinci_manager`，`devmm_svm`、`hisi_hdc` 存在） |
| 驱动 | 25.5.1（`/usr/local/Ascend/driver/version.info`；README 基线 25.2.1） |
| 内存 | 1020 GiB |
| Docker | 29.3.1，overlay2，cgroup v1，Docker Root `/var/lib/docker`（`/` 剩余约 94 GB） |
| 镜像 | `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`，23.99 GB；容器内 vllm-ascend 提交 `8fde6d4e3`（与补丁基线一致） |
| 权重 | `/data/DeepSeek-V4.1-Flash-1017-w8a8`（本地 XFS，`/data` 剩余约 912 GB），142 个 shard |
| 宿主工具 | git / patch / curl / ss 均存在 |
| 镜像内工具 | python3(=3.12.13) / bash / ps / netstat / curl / patch / git / awk / find / sort 存在；**没有 `ss`**（脚本会退回 `netstat`） |

---

## 3. 安装仓库

仓库放在 `/opt/vllm-ascend-deepseek-v41-a3`（systemd 单元默认假设此路径）。离线机用
U 盘 / scp 拷贝即可，不需要 git、不需要 pip：

```bash
# 在有网络的机器上打包
tar czf vllm-ascend-deepseek-v41-a3.tgz vllm-ascend-deepseek-v41-a3

# 在目标机上
mkdir -p /opt && tar xzf vllm-ascend-deepseek-v41-a3.tgz -C /opt
cd /opt/vllm-ascend-deepseek-v41-a3
```

## 4. 配置

```bash
cp config/dsv41.docker.env.example config/dsv41.local.env
openssl rand -hex 32            # 生成 API key，填进 DSV41_API_KEY
vi config/dsv41.local.env
```

目标机上需要确认/修改的项：

| 变量 | 预设值 | 说明 |
|---|---|---|
| `DSV41_RUNTIME` | `docker` | 切换到容器运行时 |
| `DSV41_IMAGE` | `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3` | 必须与 `docker images` 中一致 |
| `DSV41_MODEL` | `/data/DeepSeek-V4.1-Flash-1017-w8a8` | 以相同路径挂入容器（只读） |
| `DSV41_API_KEY` | 需要你填 | 非空即启用 Bearer 鉴权 |
| `DSV41_STATE_DIR` | `/data/dsv41` | 容器 `/root/.cache`（编译缓存）与 `/root/ascend`（CANN 日志）落在这里 |
| `DSV41_RESULTS` / `DSV41_LOG_DIR` | `/data/dsv41/logs` | 服务 / router / watchdog 日志 |
| `DSV41_PID_DIR` | `/run/dsv41` | **必须是专用目录，不能是 `/tmp`** |
| `DSV41_MAX_MODEL_LEN` | `360448` | 默认上下文长度，见第 10 节 |
| `DSV41_GPU_MEM_UTIL` | `0.92` | 每个 die 需要的空闲 HBM 占比 |
| `DSV41_KV_OFFLOAD` | `1` | Docker 预设启用 Memcache DRAM KV offload |
| `DSV41_KV_DRAM_GIB_PER_WORKER` | `8` | DP2/TP8 共 128 GiB，见第 10 节的内存预算 |
| `DSV41_DOCKER_PRIVILEGED` | `1` | `--privileged`；设 0 退回显式 `--device` + 放宽 seccomp（备用，未充分实测） |
| `DSV41_CLEAN_ALL_VLLM` | `0` | 保持 0：只清理本部署进程 |

配置优先级：**导出的环境变量 > `config/dsv41.local.env` > `config/dsv41.env` > 脚本默认值**。

---

## 5. 创建容器

```bash
bash scripts/container.sh print-cmd     # 只打印 docker run 命令，先过目
bash scripts/container.sh up            # 创建/启动容器、应用补丁、启动 Memcache MetaService
bash scripts/container.sh status
```

`up` 做的事：检查镜像和模型 → `docker run -d --name dsv41 --init --restart unless-stopped …`
→ 等容器可 exec → 核对镜像一致性与 vllm-ascend 提交（应为 `8fde6d4e3`）→
`apply_patches.sh apply` → 启用 KV offload 时启动并检查 MetaService。典型输出末尾为
`✅ 容器就绪: dsv41`。

挂载清单（宿主路径存在才挂，缺失的可选项会打印 `⚠ 跳过不存在的挂载源`）：

| 宿主 | 容器 | 说明 |
|---|---|---|
| 仓库目录 | 同路径 | 脚本、配置、日志 |
| `DSV41_MODEL` | 同路径（ro） | 权重 |
| `DSV41_PID_DIR` / `DSV41_RESULTS` / `DSV41_LOG_DIR` | 同路径 | pid 与日志 |
| `$DSV41_STATE_DIR/cache`、`/ascend` | `/root/.cache`、`/root/ascend` | 编译缓存、CANN 日志 |
| `/usr/local/Ascend/driver/lib64` | 同路径 | **必需**，驱动用户态库 |
| `/usr/local/bin/npu-smi` | 同路径 | **必需** |
| `…/driver/version.info`、`…/driver/tools/hccn_tool`、`/usr/local/dcmi`、`/etc/ascend_install.info`、`/etc/hccn.conf`、`/etc/localtime` | 同路径 | 官方 docker 用法所需 |

### Gate 1：环境探针

```bash
make probe        # 等价于 DSV41_PROBE=1 bash scripts/serve.sh
# 或：bash scripts/doctor.sh --probe
```

用**和正式服务完全相同的环境**在容器里 import 整套栈。期望输出：

```
torch 2.10.0+cpu torch_npu 2.10.0.post7
npu_count 16
vllm 0.30.0 vllm_ascend <版本或 ?>
```

`npu_count` 必须是 16。若 `ImportError` / 找不到 `libXXX.so`，说明驱动库挂载或环境有问题，
见第 12 节。探针只 import 库并查询设备数，不加载模型（一般也不占用 HBM），所以可以在旧模型
还在运行时先做；如果担心，等旧模型停掉后再做。

### 预检

```bash
bash scripts/doctor.sh
```

docker 运行时会检查：docker daemon、镜像、容器、容器内 Python/CANN/源码/`vllm_ascend_C*.so`、
驱动库与 `npu-smi` 是否可见、模型是否可见、宿主与容器 glibc（仅供记录）、补丁与 6 个 gate、
权重 142 个 shard、API key、端口、磁盘。不应有 ❌。

---

## 6. 腾出 NPU，再启动

当前这些 die 的 HBM 被正在运行的模型占着（`npu-smi info` 里每个 die 约 63 GB / 65 GB）。
**由你手动停掉旧模型**之后再启动本服务；本项目不会替你停它。确认已释放：

```bash
npu-smi info | grep -E 'HBM|/ 65536'     # 每个 die 的 HBM-Usage 应降到几 GB
ss -ltn | grep -E ':(8020|8021|8022|8023|5557|5558)\b' || echo "ports free"
```

同机还保留其它小模型时：

- 本服务需要 16 个 die 全部参与（DP2 × TP8）；每个 die 需要至少
  `DSV41_GPU_MEM_UTIL × 总 HBM` 的**空闲**显存，否则报
  `Free memory on device … less than desired GPU memory utilization`。
  小模型占得少时，可适当调低 `DSV41_GPU_MEM_UTIL`（如 0.85）为它们留出空间；这会改变 KV cache
  容量，需要重新观察启动日志。
- 端口 8020–8023、5557、5558 不能被占用；与旧模型冲突时改 `DSV41_PORT` 系列变量。
- 本项目的清理动作**只**处理自己的进程（见第 9 节），不会终止小模型。

---

## 7. 首次启动

先前台启动一次，直接看到全部日志（Ctrl-C 可中止）：

```bash
bash scripts/serve.sh run
```

时间线（与默认文档一致；上下文长度改变后首次需重新编译，可能略久）：

| 阶段 | 典型耗时 | 日志标志 |
|---|---|---|
| Python / CANN 初始化 | 数秒 | `Available plugins for group vllm.platform_plugins` |
| 权重加载（142 shard） | ≈4–5 min | `Loading safetensors checkpoint shards` |
| 引擎初始化 / KV cache / warmup | ≈70 s | `init engine (profile, create kv cache, warmup model) took …` |
| ACL graph 捕获 | 数十秒 | `ACL graph replay is active for the V2 target model` |
| 就绪 | — | `Application startup complete.` |

日志同时写入 `/data/dsv41/logs/serve_*.log`。启动参数回显里应能看到
`max_len=360448 prompt_tokens_details=1`、`offline=1`、`runtime=docker`。

两条已知无害警告（`causal_conv1d_update` / `vllm._deepselect_C`）见
[`troubleshooting.md`](troubleshooting.md) 第 5 节。

### Gate 2：就绪

```bash
curl --noproxy '*' -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8021/health     # 200
curl --noproxy '*' -s http://127.0.0.1:8021/v1/models -H "Authorization: Bearer $DSV41_API_KEY"
```

router 只有两个 rank 都健康才返回 200。

### Gate 3：功能验证

```bash
export DSV41_API_KEY=$(sed -n 's/^DSV41_API_KEY=//p' config/dsv41.local.env)
python3 tests/smoke_test.py             # 宿主 Python 3.7 即可（仅标准库）
# 若宿主 python3 有问题：bash scripts/container.sh exec python3 tests/smoke_test.py
```

期望 `[PASS]`。再确认 `prompt_tokens_details` 与前缀缓存（同一长前缀请求两次，第二次
`cached_tokens` 应明显大于 0）：

```bash
python3 - <<'PY'
import json, os, urllib.request
base = "http://127.0.0.1:8021"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
body = {"model": "dsv41",
        "messages": [{"role": "system", "content": "请阅读以下内容。" * 400},
                     {"role": "user", "content": "用一句话回答：1+1 等于几？"}],
        "max_tokens": 64, "temperature": 0}
for i in (1, 2):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json",
                                  "Authorization": "Bearer " + os.environ["DSV41_API_KEY"]})
    print(i, json.load(opener.open(req, timeout=600))["usage"])
PY
```

router 统计：`curl --noproxy '*' http://127.0.0.1:8021/router/stats -H "Authorization: Bearer $DSV41_API_KEY"`。

---

## 8. 托管与开机自启

### 8.1 watchdog（手动）

```bash
bash scripts/serve.sh run               # 若前台实例还在，先 Ctrl-C 结束它
bash scripts/watchdog.sh start          # 自动拉起服务并持续监控
bash scripts/watchdog.sh status
```

watchdog 在宿主运行：轮询 `/health`；服务异常时先清理**本部署**的残留进程，再确认容器在运行
（`container.sh up`），然后重启服务。冷启动宽限期、失败冷却等参数见 `config/dsv41.docker.env.example`。

### 8.2 systemd（推荐，开机自启）

```bash
cp systemd/dsv41-container.service /etc/systemd/system/
# 可选：/etc/dsv41.env 放覆盖用的环境变量
systemctl daemon-reload
systemctl enable --now dsv41-container
systemctl status dsv41-container
```

单元流程：`container.sh up`（保证容器存在并已打补丁）→ `watchdog.sh start`。
`systemctl stop` 只停 watchdog，**不会**停容器和模型；容器本身带 `--restart unless-stopped`，
宿主重启后 Docker 会拉起容器，systemd 单元再拉起 watchdog 和服务。

**建议演练一次**：`reboot` 后约 6–10 分钟内 `/health` 应恢复 200。

---

## 9. 日常运维

| 目的 | 命令 |
|---|---|
| 总体状态 | `bash scripts/doctor.sh` / `STATUS=1 bash scripts/serve.sh` / `bash scripts/watchdog.sh status` / `bash scripts/container.sh status` |
| 看日志 | `tail -f /data/dsv41/logs/serve_*.log`，`router.log`，`watchdog.log` |
| 重启服务 | `bash scripts/watchdog.sh restart` |
| 只重启 router | `ROUTER_RESTART=1 bash scripts/serve.sh` |
| 停服务（watchdog 在时先停它） | `bash scripts/watchdog.sh stop && STOP=1 bash scripts/serve.sh` |
| 停 watchdog + 服务 | `bash scripts/kill_vllm.sh`（先 `--dry-run` 看会处理哪些进程） |
| 进入容器 | `bash scripts/container.sh shell` |
| 在容器里跑压测 | `bash scripts/container.sh exec python3 scripts/bench_2k.py` |
| 改服务参数（上下文、批大小等） | 改 `config/dsv41.local.env` → `bash scripts/watchdog.sh restart` |
| 改挂载 / 路径 / 镜像 | 改配置 → `bash scripts/container.sh recreate`（补丁自动重打） |
| 停容器 | `bash scripts/container.sh down`（容器内服务会终止，不影响宿主上其它容器/进程） |

### 清理范围（只动自己）

`serve.sh` 在启动 vLLM 和 router 前导出 `DSV41_PROC_TAG=dsv41-<端口>`，服务的所有进程
（含 spawn 出来的 worker、变成孤儿的 worker）都继承它。以下动作**只**处理：serve 主进程树、
router，以及带这个标记的进程：

- watchdog 重启前的 `dsv41_cleanup_residual`
- `STOP=1 bash scripts/serve.sh`
- `bash scripts/kill_vllm.sh`（默认；`PORT=` 可进一步限定到某个端口）

同机其它 vLLM / NPU 任务（包括名字同样是 `VLLM::` 的 worker）不会被终止。需要旧的“整机清扫”
行为时才显式使用：`bash scripts/kill_vllm.sh --all` 或 `DSV41_CLEAN_ALL_VLLM=1`。
`doctor.sh` 会区分“本部署进程”和“本机其它 `VLLM::` 进程”。

---

## 10. 关键参数：本次的默认值

| 参数 | 默认 | 说明 |
|---|---|---|
| `--max-model-len` | **360448** | `DSV41_MAX_MODEL_LEN`（= 2816 × 128，与 `--block-size 128` 对齐）。比 1M 更宽松的 KV 预算；改动会进入编译缓存 key，**首次启动需要冷启动**。需要 1M 时设 `1048576`。 |
| `--enable-prompt-tokens-details` | **on** | `DSV41_ENABLE_PROMPT_TOKENS_DETAILS=1`；响应 `usage.prompt_tokens_details.cached_tokens`。流式请求需 `stream_options.include_usage`。 |
| 离线环境变量 | on | `DSV41_OFFLINE=1`：`HF_HUB_OFFLINE`、`TRANSFORMERS_OFFLINE`、`HF_DATASETS_OFFLINE`、`VLLM_NO_USAGE_STATS`、`DO_NOT_TRACK`，避免启动时探测外网超时 |
| 其余 | 同 [`deployment.md`](deployment.md) 第 6 节 | DP2/TP8、V2 runner、DSpark K=5、FULL_DECODE_ONLY 37 桶、prefix caching、Engram CPU offload、`--block-size 128` 等 |

文档里的吞吐 / 显存实测数据都是在 1M 上下文下采集的；360448 下未重新采集，实际 KV 容量
以启动日志为准。自行调整长度时建议取 128 的整数倍（与 block size 对齐）。

---

### Memcache DRAM KV offload

Docker 预设默认启用 KV offload，要求 Model Runner V2 和本地前缀缓存。复用已有前缀时从
DRAM 加载 KV，执行中的请求仍需要 NPU KV 空间。

**内存预算。** 目标机共有 1020 GiB。历史启动日志中，两张 Engram 表各约 12.87 GiB/TP
分片，两个本机 DP 共享 8 个分片，估算约占 206 GiB。KV 池按 16 个 worker 各 8 GiB
配置，名义容量为 128 GiB。扣除两者后约有 686 GiB 可供模型加载、权重页缓存、系统和
运行时使用，实际余量以目标机测量为准。

1. 首次上线前可用 `DSV41_KV_OFFLOAD=0` 跑基线。在目标机确认推理容器与 `DSV41_IMAGE`
   使用相同镜像 ID，确认客户端使用 `model=dsv41`，并从启动日志记录各 DP engine 的
   `NPU KV cache size`。模型与 Engram 加载后，建议在代表性负载下确认宿主 `MemAvailable`
   仍至少有 256 GiB（128 GiB 池容量加 128 GiB 余量）。同时检查 `df -h /dev/shm`，以及
   `docker inspect -f '{{.HostConfig.Memory}}' dsv41` 显示的容器内存限额。
2. 在维护窗口停止 watchdog，再执行 `bash scripts/container.sh down`。在实际使用的环境文件
   （例如 systemd 的 `/etc/dsv41.env`）检查 `DSV41_KV_OFFLOAD=1` 和
   `DSV41_KV_DRAM_GIB_PER_WORKER=8`，旧配置里的显式值会覆盖脚本默认值。然后按原入口启动。
   手动启动依次运行 `bash scripts/container.sh up` 和 `bash scripts/serve.sh`，两次使用同一配置。
   systemd 启动会执行这两个步骤。配置目录位于 `${DSV41_RESULTS}/kv/${DSV41_CONTAINER}/`。
3. 从服务日志核对 16 个 worker 注册容量，并用
   `curl --noproxy '*' -fsS http://127.0.0.1:8100/metrics` 核对
   `memcache_total_capacity_bytes` 约为 128 GiB。默认 5100 / 6100 / 8100 分别为 MetaService、
   config store 和 metrics 端口。发生冲突时通过 Docker 环境模板中的 `DSV41_KV_*_PORT` 改端口。
   若启动失败，查看 `docker logs dsv41-mmc-meta` 和启动器日志。

**实机验收。** 在隔离测试环境临时设置 `VLLM_SERVER_DEV_MODE=1` 并将服务绑定到 loopback。
直连一个 DP 端口（默认 external LB 下为 8022 或 8023），发送固定的约 90k token 前缀，
等请求和池写入完成后执行
`curl --noproxy '*' -fsS -X POST 'http://127.0.0.1:8022/reset_prefix_cache?reset_external=false'`。
确认返回 `{"success":true}` 后向同一 DP 端口重发请求。检查
`vllm:external_prefix_cache_hits_total` 的增量以及 Memcache 成功读取的计数，另用不同前缀
做负对照。在受控 KV 压力下确认实际发生抢占和恢复，并将恢复后及后续池命中的贪心首 token
与同条件 offload 关闭的结果比较。验收结束后关闭开发模式。

**上线判据。** 固定新默认值和预热方法，在并发 16 / 24 / 32 各交替跑三轮开关对照。
逐 DP engine 记录本地与外部命中、重新计算的 prefill tokens、抢占、池驱逐、p95 首 token
延迟和解码吞吐。并发 24 和 32 的三轮中位数均应显示 prefill 重算量和 p95 首 token 延迟下降，
解码吞吐降幅不超过 5%，并通过上述正确性验证。

**关闭或恢复。** 停止 watchdog 后执行 `bash scripts/container.sh down`。如需回滚，设置
`DSV41_KV_OFFLOAD=0` 后按原入口启动，验证 `/health`、`/v1/models` 和推理。
开启时独立监控 Memcache 指标端口。MetaService 故障后，完整重启推理和池以重新注册，
按冷池处理。调整容量或端口也使用完整的 `down` / `up` 流程。

## 11. 升级、回滚与卸载

| 场景 | 操作 |
|---|---|
| 换镜像 tag | 先 `docker load` 新镜像，改 `DSV41_IMAGE`，`container.sh recreate`；补丁基线是 vllm-ascend `8fde6d4e3`，`up` 会在提交不一致时告警，`apply_patches.sh check` 暴露冲突 |
| 回滚补丁 | `bash scripts/apply_patches.sh revert --force`（在容器内的 git 树上执行，需 `up` 之后） |
| 换权重 | 保持路径不变；换路径等于一次完整冷启动，并需 `container.sh recreate`（挂载变了） |
| 卸载 | `bash scripts/kill_vllm.sh`；`systemctl disable --now dsv41-container`；`bash scripts/container.sh down && docker rm dsv41`；按需删除 `/data/dsv41`、`/run/dsv41` |

---

## 12. 故障排查

先跑：

```bash
bash scripts/doctor.sh
tail -n 200 /data/dsv41/logs/serve_*.log
bash scripts/container.sh status
```

| 症状 | 原因 | 修复 |
|---|---|---|
| `container.sh up`：`本机没有镜像` | tag 不同或未 load | `docker images`；`docker load -i <tar>`；设置 `DSV41_IMAGE` |
| `缺少必需的挂载源 …` | 驱动 / npu-smi 不在默认位置 | 确认 `/usr/local/Ascend/driver/lib64`、`/usr/local/bin/npu-smi` |
| `DSV41_PID_DIR 不能是 /tmp` | docker 运行时 pid 目录要整体挂载 | `DSV41_PID_DIR=/run/dsv41` |
| 探针 `ImportError` / `libXXX.so: cannot open` | 驱动库未挂载或环境缺失 | `container.sh shell` 后 `ldd` 对应库；确认挂载；必要时 `DSV41_CONTAINER_SOURCE_ENV=1`（按镜像入口脚本 source `set_env.sh` 链）后重试 |
| 探针 `npu_count` < 16 | 设备未进容器 | `ls /dev/davinci*`（应 16 个 + manager）；确认 `DSV41_DOCKER_PRIVILEGED=1` 且已 `recreate` |
| `Free memory on device … less than desired …` | 这些 die 的 HBM 被其它进程占用 | 停掉占用者，或调低 `DSV41_GPU_MEM_UTIL` |
| `Address already in use` | 8020–8023 / 5557 / 5558 被占用 | 释放端口或改端口配置 |
| 启动阶段长时间无输出、疑似访问外网 | 离线变量没生效 | 日志里确认 `offline=1`；`DSV41_OFFLINE=1` |
| 启动报 `max_model_len` / block 相关错误 | 长度超出 KV 预算或未与 block 对齐 | 确认长度是 128 的整数倍（默认 360448 = 2816 × 128）；或调小 / 回到 `1048576` 后看日志里的 KV 容量 |
| `serve.sh` 报 `容器 dsv41 未运行` | 容器停了 | `bash scripts/container.sh up` |
| 改了挂载没生效 | 容器参数创建时固化 | `bash scripts/container.sh recreate` |
| watchdog 反复重启 | 服务确定性启动失败 | `watchdog.sh stop` 后 `serve.sh run` 看完整错误 |
| 请求 401 / 404 | 缺 Bearer / `model` 名不一致 | 带 `Authorization`；`model` 必须是 `dsv41` |
| Ctrl-C 没停掉 `serve.sh run` | 非交互 TTY 下 docker exec 不转发信号 | 交互终端里运行（自动 `-it`），或另开终端 `STOP=1 bash scripts/serve.sh` |

更多通用问题见 [`troubleshooting.md`](troubleshooting.md)（含 docker 运行时专节 4b）。

---

## 13. 上线检查清单

- [ ] `docker images` 能看到 `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`
- [ ] `config/dsv41.local.env`：`DSV41_RUNTIME=docker`、模型路径、**强随机** `DSV41_API_KEY`、`DSV41_PID_DIR=/run/dsv41`
- [ ] `container.sh up` 输出 `vllm-ascend 提交 8fde6d4e3 与补丁基线一致` 与 `✅ 容器就绪`
- [ ] **Gate 1**：`make probe` → `npu_count 16`
- [ ] `apply_patches.sh verify` 显示 6 个 gate 默认值均为 1；`doctor.sh` 无 ❌
- [ ] 旧模型已由你手动停止，HBM 已释放，端口空闲
- [ ] **Gate 2**：`/health` = 200，`/v1/models` 列出 `dsv41`
- [ ] **Gate 3**：`tests/smoke_test.py` PASS；`usage.prompt_tokens_details.cached_tokens` 在重复前缀上 > 0
- [ ] watchdog / systemd 已启用，重启宿主演练通过
- [ ] 服务只监听内网；防火墙（当前 firewalld 未运行）按需放行 8021

---

## 附：相对默认流程的改动

| 项 | 默认（native） | docker 运行时 |
|---|---|---|
| 获取镜像内容 | `setup_rootfs.sh` 从 registry 拉 OCI 层 | 本机 `docker load` 的镜像，无需下载 |
| 运行位置 | 宿主直接执行 rootfs 的 Python | 长驻容器内 |
| `DSV41_ROOTFS` | 解包目录 | 容器内固定 `/`（脚本自动处理，不再出现 `//usr/local`） |
| 补丁 | 打在 rootfs | 打在容器可写层，`container.sh up` 幂等重打 |
| 控制脚本 | 宿主 | 仍在宿主（共享 PID 命名空间） |
| 清理范围 | 旧默认：整机 `VLLM::` | **仅本部署**（`DSV41_PROC_TAG`），两种运行时一致 |

## 附：验证状态

- 脚本、配置与本文的命令依据代码审阅和目标机上的只读检查结果编写；容器命令的组装与
  “清理只动自己的进程”有单元测试（`tests/test_container.py`，其中进程相关用例仅在 Linux 运行）。
- **尚未在真实 A3 + Kylin 上端到端验证**。首次部署请严格按 Gate 1 → 2 → 3 推进，并把
  各 Gate 的输出保留下来；若任一步失败，把对应输出与 `serve_*.log` 一并附上即可快速定位。
