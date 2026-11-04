#!/usr/bin/env python3
"""Tests for the container runtime helpers (no docker, no NPU required).

* scripts/container.sh print-args  — composition of the `docker run` command
* scripts/container.sh exec        — env forwarding, using a fake `docker`
* scripts/common.sh + kill_vllm.sh — cleanup is scoped to this deployment

Run:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import pathlib
import shutil
import signal
import subprocess
import tempfile
import unittest
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTAINER_SH = str(ROOT / "scripts" / "container.sh")
COMMON_SH = str(ROOT / "scripts" / "common.sh")
KILL_SH = str(ROOT / "scripts" / "kill_vllm.sh")

HAVE_BASH = shutil.which("bash") is not None
HAVE_PROC = os.path.isdir("/proc/self") and os.path.exists("/proc/self/environ")


def clean_env(**extra: str) -> dict:
    """Environment without any DSV41_*/PORT leakage from the developer's shell."""
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("DSV41_", "VLLM_", "HCCL_")) and k not in ("PORT", "STOP", "STATUS", "ROUTER_RESTART")
    }
    env.update(extra)
    return env


def run(cmd: list, env: dict, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, timeout=60, **kwargs,
    )


@unittest.skipUnless(HAVE_BASH, "bash required")
class ContainerArgsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(self.tmp.name).resolve()
        self.base = base
        self.host_root = base / "host"
        driver = self.host_root / "usr/local/Ascend/driver"
        (driver / "lib64").mkdir(parents=True)
        (driver / "tools").mkdir()
        (driver / "tools" / "hccn_tool").write_text("x")
        (self.host_root / "usr/local/dcmi").mkdir(parents=True)
        (self.host_root / "usr/local/bin").mkdir(parents=True)
        (self.host_root / "usr/local/bin/npu-smi").write_text("x")
        (self.host_root / "etc").mkdir()
        (self.host_root / "etc/ascend_install.info").write_text("x")
        (self.host_root / "dev").mkdir()
        (self.host_root / "dev/davinci0").write_text("")
        (self.host_root / "dev/davinci_manager").write_text("")
        # deliberately absent: version.info, hccn.conf, localtime
        self.model = base / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        self.env = clean_env(
            DSV41_RUNTIME="docker",
            DSV41_HOST_ROOT=str(self.host_root),
            DSV41_MODEL=str(self.model),
            DSV41_PID_DIR=str(base / "run"),
            DSV41_RESULTS=str(base / "logs"),
            DSV41_LOG_DIR=str(base / "logs"),
            DSV41_STATE_DIR=str(base / "state"),
            DSV41_IMAGE="example.invalid/img:tag",
            DSV41_CONTAINER="dsv41-test",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def args(self, **extra: str):
        env = dict(self.env)
        env.update(extra)
        proc = run(["bash", CONTAINER_SH, "print-args"], env)
        return proc, proc.stdout.splitlines()

    def test_core_flags(self) -> None:
        proc, args = self.args()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(args[0], "run")
        for flag in ("--privileged", "--pid=host", "--net=host", "--ipc=host", "--init"):
            self.assertIn(flag, args)
        self.assertIn("DSV41_ROOTFS=/", args)
        self.assertIn("DSV41_IN_CONTAINER=1", args)
        self.assertEqual(args[-4:], ["--entrypoint", "sleep", "example.invalid/img:tag", "infinity"])

    def test_same_path_mounts(self) -> None:
        _, args = self.args()
        self.assertIn("%s:%s:ro" % (self.model, self.model), args)
        self.assertIn("%s/run:%s/run" % (self.base, self.base), args)
        self.assertIn("%s/logs:%s/logs" % (self.base, self.base), args)  # once: RESULTS == LOG_DIR
        self.assertEqual(sum(1 for a in args if a == "%s/logs:%s/logs" % (self.base, self.base)), 1)
        self.assertTrue([a for a in args if a.split(":")[0] == a.split(":")[-1]
                         and os.path.realpath(a.split(":")[0]) == str(ROOT)])
        self.assertIn("%s/state/cache:/root/.cache" % self.base, args)
        self.assertIn("%s/state/ascend:/root/ascend" % self.base, args)

    def test_ascend_mounts_are_guarded(self) -> None:
        proc, args = self.args()
        hr = self.host_root
        self.assertIn("%s/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64" % hr, args)
        self.assertIn("%s/usr/local/bin/npu-smi:/usr/local/bin/npu-smi" % hr, args)
        self.assertIn("%s/usr/local/dcmi:/usr/local/dcmi" % hr, args)
        self.assertFalse([a for a in args if "hccn.conf" in a])
        self.assertFalse([a for a in args if "version.info" in a])
        self.assertIn("hccn.conf", proc.stderr)  # skipped with a warning

    def test_pid_dir_must_not_be_tmp(self) -> None:
        proc, _ = self.args(DSV41_PID_DIR="/tmp")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("DSV41_PID_DIR", proc.stderr)

    def test_unprivileged_fallback_lists_devices(self) -> None:
        _, args = self.args(DSV41_DOCKER_PRIVILEGED="0")
        self.assertNotIn("--privileged", args)
        self.assertIn("%s/dev/davinci0" % self.host_root, args)
        self.assertIn("%s/dev/davinci_manager" % self.host_root, args)
        self.assertIn("seccomp=unconfined", args)

    def test_extra_args_and_mounts(self) -> None:
        _, args = self.args(DSV41_DOCKER_EXTRA_ARGS="--cpus=64 --hostname=dsv41",
                            DSV41_DOCKER_EXTRA_MOUNTS="/a:/b:ro /c:/d")
        for expected in ("--cpus=64", "--hostname=dsv41", "/a:/b:ro", "/c:/d"):
            self.assertIn(expected, args)


@unittest.skipUnless(HAVE_BASH, "bash required")
class ContainerExecTest(unittest.TestCase):
    def test_exec_forwards_names_not_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = pathlib.Path(tmp, "docker")
            fake.write_text(
                "#!/usr/bin/env bash\n"
                "case \"$1\" in\n"
                "  info) exit 0 ;;\n"
                "  inspect) echo running ;;\n"
                "  exec) shift; printf '%s\\n' \"$@\" ;;\n"
                "esac\n"
            )
            fake.chmod(0o755)
            env = clean_env(
                DSV41_RUNTIME="docker",
                DSV41_DOCKER_BIN=str(fake),
                DSV41_CONTAINER="dsv41-test",
                DSV41_API_KEY="super-secret-key",
                DSV41_ROOTFS="/host/only/rootfs",
                DSV41_PROC_TAG="must-not-leak",
                VLLM_LOGGING_LEVEL="DEBUG",
                ROUTER_RESTART="1",
                ASCEND_HOME_PATH="/host/ascend",
            )
            proc = run(["bash", CONTAINER_SH, "exec", "--", "bash", "/x/serve.sh", "run"], env)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            out = proc.stdout.splitlines()
            self.assertEqual(out[-3:], ["bash", "/x/serve.sh", "run"])
            self.assertIn("DSV41_API_KEY", out)
            self.assertIn("VLLM_LOGGING_LEVEL", out)
            self.assertIn("ROUTER_RESTART", out)
            self.assertNotIn("super-secret-key", proc.stdout)
            for banned in ("DSV41_ROOTFS", "DSV41_PROC_TAG", "DSV41_IN_CONTAINER", "ASCEND_HOME_PATH"):
                self.assertNotIn(banned, out)
            self.assertEqual(os.path.realpath(out[out.index("-w") + 1]), str(ROOT))


@unittest.skipUnless(HAVE_BASH and HAVE_PROC, "needs bash and /proc (Linux)")
class ScopedCleanupTest(unittest.TestCase):
    """Other tenants' vLLM / NPU processes must never be selected for cleanup."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.instance = "dsv41-test-" + uuid.uuid4().hex[:8]
        self.env = clean_env(
            DSV41_INSTANCE_ID=self.instance,
            DSV41_PID_DIR=self.tmp.name,
            DSV41_PORT="58021",  # nothing listens here; keeps pattern matches away from real servers
        )
        self.procs = []

    def tearDown(self) -> None:
        for p in self.procs:
            try:
                p.send_signal(signal.SIGKILL)
                p.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        self.tmp.cleanup()

    def spawn(self, exe: str, env: dict) -> subprocess.Popen:
        p = subprocess.Popen([exe, "300"], env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(p)
        return p

    def test_instance_pids_only_returns_tagged(self) -> None:
        sleep = shutil.which("sleep")
        tagged = self.spawn(sleep, dict(self.env, DSV41_PROC_TAG=self.instance))
        other_tag = self.spawn(sleep, dict(self.env, DSV41_PROC_TAG="someone-else"))
        untagged = self.spawn(sleep, self.env)
        proc = run(["bash", "-c", 'source "$1"; dsv41_instance_pids', "_", COMMON_SH], self.env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        pids = set(proc.stdout.split())
        self.assertIn(str(tagged.pid), pids)
        self.assertNotIn(str(other_tag.pid), pids)
        self.assertNotIn(str(untagged.pid), pids)

    def test_kill_vllm_default_spares_foreign_vllm_workers(self) -> None:
        real_sleep = os.path.realpath(shutil.which("sleep"))
        if os.path.basename(real_sleep) == "busybox":
            self.skipTest("busybox sleep cannot be renamed")
        fake_worker = pathlib.Path(self.tmp.name, "VLLM::Fake_DP0")  # comm is capped at 15 chars
        shutil.copy(real_sleep, str(fake_worker))
        foreign = self.spawn(str(fake_worker), self.env)  # looks like a vLLM worker, not ours
        ours = self.spawn(str(fake_worker), dict(self.env, DSV41_PROC_TAG=self.instance))

        def dry_run(*flags: str) -> set:
            proc = run(["bash", KILL_SH, "--only-vllm", "--dry-run", *flags], self.env)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            pids = set()
            for line in proc.stdout.splitlines():
                if "[dry-run] kill -TERM" in line:
                    pids.update(line.split("kill -TERM", 1)[1].split())
            return pids

        scoped = dry_run()
        self.assertIn(str(ours.pid), scoped)
        self.assertNotIn(str(foreign.pid), scoped)

        everything = dry_run("--all")
        self.assertIn(str(ours.pid), everything)
        self.assertIn(str(foreign.pid), everything)


if __name__ == "__main__":
    unittest.main()
