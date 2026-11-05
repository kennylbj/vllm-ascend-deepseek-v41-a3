# 部署指南

本文从一台空白的 8×Ascend 910C（A3）节点开始，完成 rootfs 解包、权重准备、
补丁应用、服务启动、验证与托管。每一步都给出**可观察的判定点**，便于定位失败环节。

> 本文是 **rootfs 直跑**（`DSV41_RUNTIME=native`）流程，要求宿主 glibc ≥ 2.35 且能访问
> registry。**离线服务器或宿主 glibc 较旧（如 Kylin V10 的 2.28）请看
> [`offline-deployment.md`](offline-deployment.md)**（docker 运行时，使用本机已加载的镜像）。

---

## 0. 前置条件

| 项 | 要求 | 检查命令 |
|---|---|---|
| 架构 | aarch64 | `uname -m` |
| NPU | 8×910C（16 die），驱动可用 | `npu-smi info` |
| 权限 | root（或可访问 NPU 设备与 rootfs 目录） | `id -u` |
| 磁盘 | 权重 ≈ 728 GB；rootfs ≈ 21 GB（外加 25 GB 临时空间） | `df -h /cache /models` |
| 端口 | 8021/8020/8022/8023（external 模式）+ 5557/5558 未被占用 | `ss -ltnp \| grep -E ':(8020\|8021\|8022\|8023\|5557\|5558)'` |
| 内存 | 建议 ≥ 1 TB（Engram CPU offload + 权重页缓存） | `free -g` |

启动前确认没有其它 vLLM 占用 NPU：

```bash
npu-smi info | grep -E "VLLMWorker|/ 65536"
bash scripts/kill_vllm.sh --dry-run     # 只列出会处理哪些进程
```

> 清理逻辑默认只处理本部署的进程（`DSV41_CLEAN_ALL_VLLM=0`，serve.sh 启动的进程带
> `DSV41_PROC_TAG`），不会误杀同机其它模型服务；只有显式设置 `DSV41_CLEAN_ALL_VLLM=1`
> 或 `kill_vllm.sh --all` 才会终止本机所有 `VLLM::` worker。

---

## 1. 获取代码与配置

```bash
git clone <repo-url> vllm-ascend-deepseek-v41-a3
cd vllm-ascend-deepseek-v41-a3

cp config/dsv41.env.example config/dsv41.local.env
$EDITOR config/dsv41.local.env
```

`config/dsv41.local.env` 至少修改：

```bash
DSV41_ROOTFS=/cache/dsv41-rc2-rootfs
DSV41_MODEL=/models/DeepSeek-V4.1-Flash-1017-w8a8
DSV41_API_KEY=<长随机字符串>      # 生成: openssl rand -hex 32
```

配置优先级：**导出的环境变量 > `config/dsv41.local.env` > `config/dsv41.env` > 脚本默认值**。
所有脚本都会自动 source 这两个文件。

---

## 2. 准备权重

权重发布页：ModelScope `Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8`
（W8A8_DYNAMIC，142 个 `quant_model_weights-*.safetensors`，约 728 GB）。

```bash
pip install modelscope
modelscope download --model Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8 \
  --local_dir models/DeepSeek-V4.1-Flash-1017-w8a8
```

若 `DSV41_MODEL` 指向其它位置，把它复制/软链过去即可。校验：

```bash
ls models/DeepSeek-V4.1-Flash-1017-w8a8/config.json
ls models/DeepSeek-V4.1-Flash-1017-w8a8/quant_model_weights-*.safetensors | wc -l   # 142
du -sh models/DeepSeek-V4.1-Flash-1017-w8a8                                          # ≈728G
```

建议：

- **优先本地 NVMe**。权重放在 NFS 上时，启动阶段 16 个 worker 并发 mmap 容易触发
  page-lock 死锁（进程进入 `D` 状态且不可中断）。NFS 仅适合只读挂载与多机复用。
- 权重目录一旦确定不要改。编译缓存与模型路径相关，改路径会导致全量冷启动。

---

## 3. 解包 rootfs（无 docker）

```bash
bash scripts/setup_rootfs.sh --dry-run     # 先看层列表与大小
bash scripts/setup_rootfs.sh               # 下载 + 解包 + 修复执行位 + 校验
```

默认镜像：

```
quay.nju.edu.cn/ascend/vllm-ascend:v0.30.0rc2-a3   (linux/arm64)
```

自定义：

```bash
REGISTRY=quay.nju.edu.cn REPO=ascend/vllm-ascend TAG=v0.30.0rc2-a3 \
  bash scripts/setup_rootfs.sh
# 或
bash scripts/setup_rootfs.sh --rootfs /cache/dsv41-rc2-rootfs --keep-layers
```

脚本行为：

1. 请求 manifest list，选择 `linux/arm64` 子 manifest；
2. 逐层下载 blob，**下载时校验 sha256**；
3. 按顺序解包，处理 `.wh.<name>` 与 `.wh..wh..opq`（OCL whiteout）；
4. 每层的完成标记写入 `<rootfs>.layers/.done/<digest>`，中断后可续跑；
5. 解包完成后恢复 `.so`、`*-opt`、CANN `bin/` 的执行位；
6. 写入 `<rootfs>/.dsv41-image` 记录 registry/tag/resolved digest。

预期结果：

```bash
du -sh /cache/dsv41-rc2-rootfs                  # ≈21G
cat /cache/dsv41-rc2-rootfs/.dsv41-image
ls /cache/dsv41-rc2-rootfs/usr/local/Ascend/    # cann-9.2.0 / driver / ...
ls /cache/dsv41-rc2-rootfs/vllm-workspace/      # vllm / vllm-ascend
```

> **为什么不 `tar -x` 完事**：OCI 层用 whiteout 标记删除下层文件。忽略 whiteout
> 会留下旧版 torch/vllm 文件，典型症状是
> `ImportError: cannot import name 'is_fake_tensor'`。`setup_rootfs.sh` 已处理。

> **权限**：解包工具可能丢失执行位，表现为
> `Permission denied: bishengir-compile` 或 `triton-mlir-opt`。脚本会自动修复；
> 手工排查可用 `ls -l` 查看 CANN `bin/` 与 `tools/` 下目标文件。

---

## 4. 应用优化补丁

```bash
bash scripts/apply_patches.sh status
bash scripts/apply_patches.sh check
bash scripts/apply_patches.sh apply
bash scripts/apply_patches.sh verify
```

`apply` 是幂等的：已应用则跳过 `git apply`，但仍会执行 gate 默认值固化。
`verify` 必须输出：

```
[verify] ✅ 6 个 gate 均已定义且默认值为 1
```

补丁内容与原理见 [`patches.md`](patches.md)。

回滚：

```bash
bash scripts/apply_patches.sh revert --force
```

> 回滚只恢复补丁涉及的文件到 rootfs 内 git HEAD。若 rootfs 不是 git 工作树，
> 请重新解包。

---

## 5. 启动前预检

```bash
bash scripts/doctor.sh
```

只读检查：架构/NPU、rootfs 与执行位、权重分片、补丁与 gate 默认值、端口、
API key、磁盘与代理环境。失败项会给出对应修复入口；建议每次重启前执行。

## 6. 关键启动参数

| 参数 | 值 | 作用 |
|---|---|---|
| `--data-parallel-size` | 2 | 16 die = DP2 × TP8 |
| `--tensor-parallel-size` | 8 | 每 DP rank 8 die |
| `--enable-expert-parallel` | on | 384 routed experts 分布在 16 die |
| `--max-model-len` | 360448（`DSV41_MAX_MODEL_LEN`，= 2816 × 128） | 默认上下文；1048576（1M）在 16384 批次下 KV cache 约 6.7 GiB/die（实测） |
| `--enable-prompt-tokens-details` | on | `usage.prompt_tokens_details.cached_tokens`（`DSV41_ENABLE_PROMPT_TOKENS_DETAILS=0` 关闭） |
| `--max-num-seqs` | 32 | 与图捕获桶上限对齐 |
| `--max-num-batched-tokens` | 4096（`DSV41_MAX_NUM_BATCHED_TOKENS`） | prefill 分块大小；越大 prefill 越快但 DP lockstep 下 decode 停顿越长（16384 时单步约 2.5 s）。基准数据在 16384 下采集；32768 放不进 1M 上下文预算 |
| `--data-parallel-multi-port-external-lb` | on | external 模式：supervisor 8020 + rank 8022/8023 |
| `--kv-events-config` | ZMQ, `tcp://*:5557` | publisher 按 DP rank 自动偏移到 5557/5558 |
| `--block-size` | 128 | V4.1 混合注意力要求；16 会导致输出乱码 |
| `--quantization` | ascend | W8A8_DYNAMIC |
| `--tokenizer-mode` | deepseek_v41 | V4.1 tokenizer |
| `--reasoning-parser` / `--tool-call-parser` | deepseek_v41 | 输出 `reasoning_content` |
| `--speculative-config` | `dspark`, K=5, `enforce_eager=false` | draft 模型入图 |
| `--compilation-config` | `FULL_DECODE_ONLY` + 37 桶 | decode 全图；含 6N 桶 |
| `--engram-config` | `cpu_offload`, `dp_shared_memory` | 24 heads 不能整除 16 die |
| `--additional-config` | CPU binding / 共享专家多流 / 关 npugraph_ex | 见下 |
| `VLLM_USE_V2_MODEL_RUNNER` | 1 | Model Runner V2 |
| `HCCL_OP_EXPANSION_MODE` | AIV | 集合通信展开从 HOST 下沉到 Vector Core |
| `HCCL_BUFFSIZE` | 1024 | HCCL 缓冲 |
| `LD_PRELOAD` | libjemalloc.so.2 | 降低长 decode 的分配器争用 |

完整默认值与覆盖方式见 `scripts/serve.sh` 顶部与 `config/dsv41.env.example`。

### 为什么是 37 个图桶

DSpark 每个 decode step 处理 `num_reqs × (1 + K)` 个 token。K=5 时批大小是 6 的
倍数，因此桶列表除了上游常用的 2 的幂，还必须补齐 `6N`：
`6, 12, 18, 24, 30, 48, 54, 78, 102, 126, 150, 174, 192` 等。
缺少这些桶时 draft 步骤会退回 eager 执行，推测解码收益大幅下降。

### 为什么 gate 默认值必须是 1

`VLLM_WORKER_MULTIPROC_METHOD=spawn` 的子进程不继承 `os.environ`。仅在 shell 里
`export VLLM_ASCEND_V41_*` 不够——worker 内读到的是默认值 0。`apply_patches.sh`
把 6 个 gate 的源码默认值固化为 1，启动脚本仍会 export 作为自文档与兜底。

---

## 7. 启动与验证

### 手工启动

```bash
bash scripts/serve.sh
STATUS=1 bash scripts/serve.sh
```

首次启动约 **6 分钟**（权重加载 ≈5 分钟 + 引擎初始化/KV cache/图捕获 ≈1 分钟）。
日志：

```bash
tail -f logs/serve_*.log
```

external 模式下会同时启动 `prefix_router.py`（8021）、DP supervisor（8020）和两个
rank API（8022/8023）。就绪判定仍以 router 的 `/health` 为准；router 只有在两个
rank 都健康时才返回 200（不需要 API key）：

```bash
curl --noproxy '*' -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8021/health
curl --noproxy '*' http://127.0.0.1:8021/router/stats \
  -H "Authorization: Bearer $DSV41_API_KEY"
```

只重启路由器（不动 vLLM）：

```bash
ROUTER_RESTART=1 bash scripts/serve.sh
```

路由细节、事件格式与故障处理见 [`router.md`](router.md)。

### 冒烟测试

```bash
python3 tests/smoke_test.py
```

期望输出包含：

```
[1/3] /health              HTTP 200
[2/3] /v1/models           HTTP 200 models=['dsv41']
[3/3] /v1/chat/completions HTTP 200
[PASS] (reasoning parser active)
```

### 吞吐压测（可选）

```bash
pip install aiohttp
python3 scripts/bench_2k.py
python3 scripts/bench_8k.py
# DP2 双前缀并发（router 前缀亲和下更接近整机上限）
python3 scripts/bench_dual.py --concurrency 2,4,8,16,32
```

参考结果见 [`performance.md`](performance.md)。

---

## 8. 生产托管（watchdog）

```bash
bash scripts/watchdog.sh start
bash scripts/watchdog.sh status
```

`watchdog.sh start` 会自动发现 pid 文件丢失但仍在运行的实例，避免重复启动。
`serve.sh` 默认也会拒绝在已占用端口上再起一个实例（`DSV41_ALLOW_MULTI=1` 可关闭）。


行为：

| 情况 | 动作 |
|---|---|
| `/health` 200 | 每 30 s 复检 |
| 进程在但 `/health` 异常 | 宽限 180 s（加载/抖动），仍异常则重启 |
| 进程消失 | 立即清理残留并重启 |
| serve 进程在启动阶段退出 | 默认 120 s 内快速判定失败，冷却后重试 |
| 重启后 3600 s 仍未就绪 | 冷却 120 s 后重试，不放弃 |

日志：`logs/watchdog.log`（10 MB 轮转，保留 5 份）。
PID：`<DSV41_PID_DIR>/dsv41_watchdog.pid`、`<DSV41_PID_DIR>/dsv41_serve.pid`
（supervisor）、`<DSV41_PID_DIR>/dsv41_router.pid`。

停止：

```bash
bash scripts/watchdog.sh stop      # 只停 watchdog
STOP=1 bash scripts/serve.sh       # 停服务（含残留清理）
bash scripts/kill_vllm.sh          # watchdog + 服务 + NPU worker 全清
```

### systemd（可选）

仓库提供 `systemd/dsv41-watchdog.service`。安装：

```bash
sudo cp systemd/dsv41-watchdog.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dsv41-watchdog
sudo systemctl status dsv41-watchdog
```

单元假定仓库位于 `/opt/vllm-ascend-deepseek-v41-a3`，配置位于
`/etc/dsv41.env`。按实际路径修改后再安装。

---

## 9. 升级与重建

| 场景 | 操作 |
|---|---|
| 重建 rootfs | `bash scripts/setup_rootfs.sh --force`，然后重新 `apply_patches.sh apply` |
| 更换镜像 tag | `TAG=<new> BASE_ROOTFS=<new-dir> bash scripts/setup_rootfs.sh`，补丁可能需要重新生成 |
| 更新权重 | 替换权重目录，保持路径不变；否则接受一次完整冷启动 |
| 改补丁 | 编辑 `scripts/patches/...`，删除 `<rootfs>/vllm-workspace/vllm-ascend/.patches-applied` 后重新 apply |
| 升级 vllm-ascend | 重新核对补丁上下文；`apply_patches.sh check` 会提前暴露冲突 |

---

## 10. 上线检查清单

- [ ] `npu-smi info` 全部健康，启动前无残留 `VLLMWorker_DP`
- [ ] 权重 142 个 shard 完整，`config.json` 可读
- [ ] `setup_rootfs.sh` 校验通过，`.dsv41-image` 记录正确 digest
- [ ] `apply_patches.sh verify` 显示 6 个 gate 默认值均为 1（含 `model.py`）
- [ ] `scripts/doctor.sh` 无失败项
- [ ] `DSV41_API_KEY` 已设置为强随机值
- [ ] `/health` = 200，`tests/smoke_test.py` PASS
- [ ] 压测吞吐与 [`performance.md`](performance.md) 同量级
- [ ] watchdog 已启动，日志无反复重启
- [ ] 服务只监听本机/内网地址，未直接监听不可信网络
