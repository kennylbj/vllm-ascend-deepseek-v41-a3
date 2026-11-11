"""Exercise Docker KV offload startup without an Ascend host."""

import json
import os
import pathlib
import shlex
import sys
import tempfile
import time
import unittest

from test_container import CONTAINER_SH, HAVE_BASH, ROOT, clean_env, run


@unittest.skipUnless(HAVE_BASH, "bash required")
class KVOffloadTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = pathlib.Path(tmp.name).resolve()
        self.bin = self.base / "bin"
        self.bin.mkdir()
        model = self.base / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        self.conf = self.base / "logs/kv/dsv41-test/host_shm-5100-6100-8100-8"
        self.calls = self.base / "docker.log"
        self.sidecar = self.base / "sidecar.json"
        self.env = clean_env(
            DSV41_RUNTIME="docker", DSV41_CONTAINER="dsv41-test",
            DSV41_DOCKER_BIN=str(self.bin / "docker"), DSV41_MODEL=str(model),
            DSV41_RESULTS=str(self.base / "logs"), DSV41_PID_DIR=str(self.base / "run"),
            DSV41_SKIP_PATCHES="1", DSV41_API_KEY="", MOCK_DIR=str(self.base),
            PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}",
        )
        self.stub("docker", f"#!{sys.executable}\n" + """
import json
import os
import sys
from pathlib import Path

base = Path(os.environ["MOCK_DIR"])
state = base / "sidecar.json"
args = sys.argv[1:]
with (base / "docker.log").open("a") as log:
    print(" ".join(args), file=log)
if args[:2] == ["container", "inspect"]:
    if not state.exists():
        sys.exit(1)
    meta = json.loads(state.read_text())
    fields = {
        "{{.Id}}": "meta-id",
        '{{index .Config.Labels "io.dsv41.kv.owner"}}': meta["owner"],
        '{{index .Config.Labels "io.dsv41.kv.config"}}': meta["config"],
        "{{.Image}}": meta["image"],
        "{{.State.Status}}": "running",
    }
    print(fields[args[3]])
elif args[0] == "run":
    labels = dict(args[i+1].split("=", 1) for i, a in enumerate(args) if a == "--label")
    state.write_text(json.dumps({
        "owner": labels["io.dsv41.kv.owner"],
        "config": labels["io.dsv41.kv.config"],
        "image": args[-3],
    }))
    print("meta-id")
elif args[0] == "inspect":
    print("running" if args[2] == "{{.State.Status}}" else "sha256:test-image")
elif args[0] == "image":
    print("sha256:test-image")
elif args[0] == "rm":
    state.unlink()
""")
        self.stub("curl", '#!/bin/bash\n[[ "${MOCK_NOT_READY:-0}" == 0 ]]\n')
        self.stub("sleep", "#!/bin/bash\nexit 0\n")
        self.stub("ss", '#!/bin/bash\nif [[ "${MOCK_BUSY:-0}" == 1 ]]; then echo "LISTEN 0 32 127.0.0.1:5100 0.0.0.0:*"; fi\n')

    def stub(self, name, contents):
        path = self.bin / name
        path.write_text(contents)
        path.chmod(0o755)

    def container(self, command="up", **overrides):
        return run(["bash", CONTAINER_SH, command], dict(self.env, **overrides))

    def assert_ok(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_disabled_does_not_create_pool(self):
        for overrides in ({"DSV41_KV_OFFLOAD": "0"}, {"DSV41_RUNTIME": "native"},
                          {"DSV41_USE_V2": "0"}, {"DSV41_ENABLE_PREFIX_CACHING": "0"}):
            with self.subTest(**overrides):
                self.assert_ok(self.container(**overrides))
                self.assertNotIn("mmc-meta", self.calls.read_text())
                self.assertFalse(self.conf.exists())

    def test_up_reuses_matching_pool_and_down_cleans_it_after_vllm(self):
        self.assert_ok(self.container())
        self.assert_ok(self.container())
        config = (self.conf / "mmc-local.conf").read_text()
        for value in ("127.0.0.1:5100", "127.0.0.1:6100", "protocol = host_shm",
                      "local_service.dram.size = 8GB", "local_service.max.dram.size = 8GB",
                      "hbm.size = 0GB", "storage.enabled = false"):
            self.assertIn(value, config)
        self.assertEqual((self.conf.parent / "ready").read_text().strip(), str(self.conf))
        self.assertEqual(self.calls.read_text().count("run -d --pull=never"), 1)
        self.assert_ok(self.container("down", DSV41_KV_OFFLOAD="0"))
        self.assertLess(self.calls.read_text().index("stop -t 30 dsv41-test"),
                        self.calls.read_text().index("rm -f meta-id"))
        self.assertFalse(self.sidecar.exists())
        self.assertFalse((self.conf.parent / "ready").exists())

    def test_occupied_port_fails_before_start(self):
        result = self.container(MOCK_BUSY="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("5100 已被占用", result.stderr)
        self.assertFalse(self.sidecar.exists())

    def test_configuration_change_requires_down(self):
        self.assert_ok(self.container())
        result = self.container(DSV41_KV_DRAM_GIB_PER_WORKER="4")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("配置不匹配", result.stderr)
        self.assertEqual(self.calls.read_text().count("run -d --pull=never"), 1)

    def test_foreign_sidecar_is_neither_reused_nor_removed(self):
        self.sidecar.write_text(json.dumps({"owner": "foreign", "config": "", "image": "sha256:test-image"}))
        result = self.container()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("归属、镜像或配置不匹配", result.stderr)
        self.assert_ok(self.container("down"))
        self.assertTrue(self.sidecar.exists())

    def test_meta_timeout_does_not_publish_readiness(self):
        result = self.container(MOCK_NOT_READY="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("未就绪", result.stderr)
        self.assertFalse((self.conf.parent / "ready").exists())

    def test_serving_requires_patch_and_preserves_cann_path(self):
        self.assert_ok(self.container())
        cann = self.base / "cann"
        cann.mkdir()
        self.stub("python3.12", f"""#!/bin/bash
if [[ "$1" == -m ]]; then
  printf '%s\n' "$@" > "$MOCK_DIR/serve.args"
  printf '%s\n' "$PYTHONHASHSEED" "$MMC_LOCAL_CONFIG_PATH" "$LD_LIBRARY_PATH" "$HCCL_BUFFSIZE" > "$MOCK_DIR/serve.env"
  touch "$MOCK_DIR/serve.done"
  exit 0
fi
unset PYTHONHOME
export PYTHONPATH="$MOCK_DIR/modules"
exec {shlex.quote(sys.executable)} "$@"
""")
        modules = self.base / "modules"
        memcache = modules / "memcache_hybrid"
        memcache.mkdir(parents=True)
        (memcache / "__init__.py").write_text("class DistributedObjectStore: pass\n")
        for package in ("memcache_hybrid", "memfabric_hybrid"):
            dist = modules / f"{package}-1.3.0.dist-info"
            dist.mkdir()
            (dist / "METADATA").write_text(f"Name: {package}\nVersion: 1.3.0\n")
        scheduler = modules / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py"
        scheduler.parent.mkdir(parents=True)
        scheduler.write_text("class KVPoolScheduler:\n    def _process_new_request(self, request): pass\n")

        env = dict(self.env, DSV41_IN_CONTAINER="1", DSV41_ALLOW_MULTI="1",
                   DSV41_ROOTFS="/", DSV41_PYTHONHOME=str(self.base), DSV41_CANN_HOME=str(cann),
                   DSV41_HCCL_BUFFSIZE="768", HCCL_BUFFSIZE="200")
        command = ["bash", str(ROOT / "scripts/serve.sh")]
        result = run(command, env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("缺少 0101-kvpool-mrv2-resume", result.stderr)
        self.assertFalse((self.base / "serve.args").exists())

        scheduler.write_text("class KVPoolScheduler:\n    def _process_new_request(self, request):\n"
                             "        self._preempted_req_ids.discard(request.req_id)\n")
        self.assert_ok(run(command, env))
        done = self.base / "serve.done"
        deadline = time.monotonic() + 5
        while not done.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(done.exists())
        args = (self.base / "serve.args").read_text().splitlines()
        config = json.loads(args[args.index("--kv-transfer-config") + 1])
        self.assertEqual(config, {
            "kv_connector": "AscendStoreConnector", "kv_role": "kv_both",
            "kv_load_failure_policy": "fail", "kv_connector_extra_config": {
                "backend": "memcache", "use_layerwise": False, "lookup_rpc_port": "8021",
            },
        })
        self.assertIn("--enable-prefix-caching", args)
        seed, config_path, ld_path, hccl_buffsize = (self.base / "serve.env").read_text().splitlines()
        self.assertEqual(seed, "0")
        self.assertEqual(config_path, str(self.conf / "mmc-local.conf"))
        self.assertIn(str(cann / "lib64"), ld_path.split(":"))
        self.assertEqual(hccl_buffsize, "768")
