SHELL := /bin/bash
ROOT  := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

.PHONY: help doctor setup patches verify serve status stop kill watchdog-start watchdog-stop watchdog-status router-restart smoke bench bench8 analyze test lint clean container-up container-down container-status container-shell container-recreate container-cmd probe

help:
	@echo "vllm-ascend-deepseek-v41-a3"
	@echo
	@echo "  --- DSV41_RUNTIME=docker（离线 / 旧 glibc 宿主，见 docs/offline-deployment.md）---"
	@echo "  make container-up    创建/启动容器并应用补丁（幂等）"
	@echo "  make container-status 容器状态"
	@echo "  make container-shell 进入容器"
	@echo "  make container-down  停止容器（容器内 vLLM 随之终止）"
	@echo "  make container-recreate 删除并重建容器（改了挂载/参数后）"
	@echo "  make container-cmd   只打印 docker run 命令"
	@echo "  make probe           用服务环境 import torch_npu / vllm_ascend 并统计 NPU 数"
	@echo
	@echo "  make setup           解包 rootfs（下载 OCI 镜像层；docker 运行时无需）"
	@echo "  make patches         应用优化补丁（幂等）"
	@echo "  make verify          校验补丁与 6 个 gate"
	@echo "  make serve           启动服务（后台）"
	@echo "  make status          服务状态"
	@echo "  make stop            停止服务（含残留清理）"
	@echo "  make kill            停止 watchdog + 服务 + NPU worker"
	@echo "  make watchdog-start  启动 watchdog（生产推荐）"
	@echo "  make watchdog-status watchdog 状态"
	@echo "  make watchdog-stop   停止 watchdog（不影响服务）"
	@echo "  make router-restart  只重启 prefix router（不动 vLLM）"
	@echo "  make doctor          只读预检（rootfs/权重/补丁/端口）"
	@echo "  make smoke           冒烟测试"
	@echo "  make bench           2K 吞吐压测"
	@echo "  make bench8          8K 推测解码压测"
	@echo "  make analyze LOG=... serve 日志分析（停顿占比 / lockstep / KV 饱和）"
	@echo "  make test            单元测试（patch 状态 / OCI 解包 / router / 日志分析）"
	@echo "  make lint            bash -n / py_compile"

doctor:
	bash $(ROOT)/scripts/doctor.sh

container-up:
	bash $(ROOT)/scripts/container.sh up

container-down:
	bash $(ROOT)/scripts/container.sh down

container-status:
	bash $(ROOT)/scripts/container.sh status

container-shell:
	bash $(ROOT)/scripts/container.sh shell

container-recreate:
	bash $(ROOT)/scripts/container.sh recreate

container-cmd:
	bash $(ROOT)/scripts/container.sh print-cmd

probe:
	DSV41_PROBE=1 bash $(ROOT)/scripts/serve.sh

setup:
	bash $(ROOT)/scripts/setup_rootfs.sh

patches:
	bash $(ROOT)/scripts/apply_patches.sh apply

verify:
	bash $(ROOT)/scripts/apply_patches.sh status

serve:
	bash $(ROOT)/scripts/serve.sh

status:
	STATUS=1 bash $(ROOT)/scripts/serve.sh

stop:
	STOP=1 bash $(ROOT)/scripts/serve.sh

kill:
	bash $(ROOT)/scripts/kill_vllm.sh

watchdog-start:
	bash $(ROOT)/scripts/watchdog.sh start

watchdog-status:
	bash $(ROOT)/scripts/watchdog.sh status

watchdog-stop:
	bash $(ROOT)/scripts/watchdog.sh stop

router-restart:
	ROUTER_RESTART=1 bash $(ROOT)/scripts/serve.sh

smoke:
	python3 $(ROOT)/tests/smoke_test.py

bench:
	python3 $(ROOT)/scripts/bench_2k.py

bench8:
	python3 $(ROOT)/scripts/bench_8k.py

analyze:
	@test -n "$(LOG)" || { echo "usage: make analyze LOG=logs/serve_xxx.log [ARGS='--since 09:30']"; exit 2; }
	python3 $(ROOT)/scripts/analyze_serve_log.py $(LOG) $(ARGS)

test:
	python3 -m unittest discover -s $(ROOT)/tests -v

lint:
	@for f in $(ROOT)/scripts/*.sh $(ROOT)/examples/*.sh; do bash -n "$$f" || exit 1; done
	@python3 -m py_compile $(ROOT)/scripts/*.py $(ROOT)/tests/*.py $(ROOT)/examples/*.py
	@# The air-gapped host only has Python 3.7: helper scripts must still parse there.
	@python3 -c "import ast,glob,sys; [ast.parse(open(f,encoding='utf-8').read(), f, feature_version=(3,7)) for f in glob.glob('$(ROOT)/scripts/*.py')+glob.glob('$(ROOT)/tests/*.py')+glob.glob('$(ROOT)/examples/*.py')]; print('py3.7 syntax ok')"
	@command -v shellcheck >/dev/null 2>&1 && shellcheck -S warning $(ROOT)/scripts/*.sh $(ROOT)/examples/*.sh || true
	@echo "lint ok"

clean:
	rm -rf $(ROOT)/logs/*.log
	find $(ROOT) -type d -name __pycache__ -prune -exec rm -rf {} +
