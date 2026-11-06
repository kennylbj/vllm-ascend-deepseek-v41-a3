"""Exercise patch installation on disposable source trees."""

import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from test_container import ROOT, clean_env, run


@unittest.skipUnless(shutil.which("git") and shutil.which("bash"), "needs Git and Bash")
class PatchInstallTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()
        self.project = self.base / "project"
        scripts = self.project / "scripts"
        scripts.mkdir(parents=True)
        for name in ("apply_patches.sh", "patch_install.py", "common.sh"):
            shutil.copy2(ROOT / "scripts" / name, scripts)
        shutil.copy2(ROOT / "Makefile", self.project)
        self.ascend = self.base / "vllm-workspace" / "vllm-ascend"
        self.vllm = self.base / "vllm-workspace" / "vllm"
        self.ascend_patches = scripts / "patches" / "vllm-ascend-v41-rc2"
        self.vllm_patches = scripts / "patches" / "vllm-ced6857af"
        envs = "gates = {}\n"
        patch = ROOT / "scripts/patches/vllm-ascend-v41-rc2/0001-envs-gates.patch"
        additions = "\n".join(line[1:] for line in patch.read_text().splitlines()
                              if line.startswith("+") and not line.startswith("+++"))
        gates = re.findall(r'_strict_binary_env\(\s*"([^"]+)"', additions)
        self.assertEqual(len(gates), 6)
        self.env_file = self.ascend / "vllm_ascend/envs.py"
        self.env_file.parent.mkdir(parents=True)
        self.env_file.write_text(envs)
        pinned = ''.join(f'    "{gate}": lambda: _strict_binary_env("{gate}"),\n'
                         for gate in gates)
        self.envs_with_gates = "gates = {\n" + pinned + "}\n"
        for name, gate in {
            "models/deepseek_v41/engram/layer.py": "ENGRAM_WKV_TP",
            "models/deepseek_v41/model.py": "ENGRAM_WKV_TP",
            "models/deepseek_v41/indexer.py": "QLI_NO_CANDIDATE",
            "ops/fused_moe/token_dispatcher.py": "MOE_MASK_RANGE",
        }.items():
            path = self.ascend / "vllm_ascend" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('gate = os.environ.get("VLLM_ASCEND_V41_{}", "0")\n'.format(gate))
        self.slot = self.ascend / "slot.py"
        self.slot.write_text("slot = 0\n")
        self.vllm.mkdir()
        self.encoder = self.vllm / "encode.py"
        self.encoder.write_text("text = 'original'\n")
        self.make_patch(self.ascend_patches, "0001-gates.patch", "vllm_ascend/envs.py",
                        envs, self.envs_with_gates)
        self.make_patch(self.ascend_patches, "0002-slot.patch", "slot.py",
                        "slot = 0\n", "slot = 1\n")
        self.make_patch(self.vllm_patches, "0001-encoder.patch", "encode.py",
                        "text = 'original'\n", "text = 'escaped'\n")
        for tree in (self.ascend, self.vllm):
            self.git(tree, "init", "-q")
            self.git(tree, "add", ".")
            self.git(tree, "commit", "-qm", "baseline")
        self.env = clean_env(DSV41_RUNTIME="native", DSV41_ROOTFS=str(self.base),
                             SRC="", PATCH_DIR="")

    def git(self, tree, *args):
        return subprocess.check_output(
            ["git", "-C", str(tree), "-c", "user.name=Fixture",
             "-c", "user.email=fixture@example.invalid", *args], text=True,
        ).strip()

    def make_patch(self, directory, name, path, old, new):
        directory.mkdir(parents=True, exist_ok=True)
        lines = difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                     fromfile="a/" + path, tofile="b/" + path)
        (directory / name).write_text("".join(lines))
        with (directory / "series.conf").open("a") as series:
            series.write(name + "\n")

    def patches(self, *args, **env):
        return run(["bash", str(self.project / "scripts/apply_patches.sh"), *args],
                   dict(self.env, **env))

    def record(self, tree):
        return tree / ".patches-applied.d/manifest.json"

    def assert_ok(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assert_failed(self, result, diagnostic):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(diagnostic, result.stderr)

    def test_clean_install_and_repeat(self):
        self.assert_ok(self.patches("check"))
        self.assertFalse(self.record(self.ascend).exists())
        self.assert_failed(self.patches("verify"), "缺少补丁校验记录")
        self.assert_ok(self.patches("apply"))
        self.assert_ok(run(["make", "verify"], self.env, cwd=self.project))
        marker = self.record(self.ascend)
        before = marker.stat().st_mtime_ns
        self.assert_ok(self.patches("apply"))
        self.assertEqual(marker.stat().st_mtime_ns, before)
        record = json.loads(marker.read_text())
        self.assertEqual(record["baseline"]["head"], self.git(self.ascend, "rev-parse", "HEAD"))
        self.assertEqual(record["files"]["vllm_ascend/envs.py"][0],
                         hashlib.sha256(self.env_file.read_bytes()).hexdigest())
        self.assertEqual(self.env_file.read_text().count(', "1")'), 6)
        self.assertEqual(self.encoder.read_text(), "text = 'escaped'\n")

    def test_resumes_matching_legacy_prefix(self):
        self.git(self.ascend, "apply", str(self.ascend_patches / "0001-gates.patch"))
        old = self.record(self.ascend).with_name("0001-gates.patch.done")
        old.parent.mkdir()
        old.write_text("old timestamp\n")
        result = self.patches("apply")
        self.assert_ok(result)
        self.assertIn("1/2", result.stdout)
        self.assert_ok(self.patches("verify"))

    def test_resumes_interrupted_gate_pinning(self):
        for patch in ("0001-gates.patch", "0002-slot.patch"):
            self.git(self.ascend, "apply", str(self.ascend_patches / patch))
        self.env_file.write_text(self.env_file.read_text().replace('"),', '", "1"),'))
        self.assert_ok(self.patches("apply"))
        self.assert_ok(self.patches("verify"))
        module = self.ascend / "vllm_ascend/models/deepseek_v41/model.py"
        self.assertIn(', "1")', module.read_text())

    def test_stale_legacy_marker_does_not_hide_old_patch(self):
        self.slot.write_text("slot = 99\n")
        old = self.record(self.ascend).with_name("0002-slot.patch.done")
        old.parent.mkdir()
        old.write_text("old timestamp\n")
        self.assert_failed(self.patches("apply"), "补丁前缀不匹配")
        self.assertEqual(self.slot.read_text(), "slot = 99\n")
        self.assertFalse(self.record(self.vllm).exists())

    def test_second_component_conflict_leaves_first_pristine(self):
        self.encoder.write_text("unrecognized local edit\n")
        self.assert_failed(self.patches("apply"), "vllm")
        self.assertEqual(self.slot.read_text(), "slot = 0\n")
        self.assertFalse(self.record(self.ascend).exists())

    def test_restored_source_with_manifest_fails_strict_verify(self):
        self.assert_ok(self.patches("apply"))
        self.git(self.ascend, "checkout", "HEAD", "--", "slot.py")
        self.assert_failed(run(["make", "verify"], self.env, cwd=self.project), "源码与补丁记录不一致")
        self.assert_failed(self.patches("apply"), "源码与补丁记录不一致")
        self.assertEqual(self.slot.read_text(), "slot = 0\n")

    def test_series_reorder_and_removal_require_explicit_revert(self):
        self.assert_ok(self.patches("apply"))
        series = self.ascend_patches / "series.conf"
        series.write_text("0002-slot.patch\n0001-gates.patch\n")
        self.assert_failed(self.patches("verify"), "补丁序列已变更")
        series.write_text("0001-gates.patch\n")
        (self.ascend_patches / "0002-slot.patch").unlink()
        outside = self.ascend / "unrelated.txt"
        outside.write_text("keep\n")
        self.assert_failed(self.patches("apply"), "补丁序列已变更")
        self.assert_failed(self.patches("revert"), "--force")
        self.assert_ok(self.patches("revert", "--force"))
        self.assertEqual(self.slot.read_text(), "slot = 0\n")
        self.assertEqual(self.encoder.read_text(), "text = 'original'\n")
        self.assertEqual(outside.read_text(), "keep\n")
        self.assertFalse(self.record(self.ascend).exists())

    def test_new_head_rejects_verify_and_revert_before_changes(self):
        self.assert_ok(self.patches("apply"))
        self.git(self.vllm, "commit", "--allow-empty", "-qm", "different baseline")
        self.assert_failed(self.patches("verify"), "源码基线已变更")
        self.assert_failed(self.patches("revert", "--force"), "Git 基线")
        self.assertEqual(self.slot.read_text(), "slot = 1\n")

    def test_custom_revert_preserves_gate_file_it_did_not_change(self):
        module = self.ascend / "vllm_ascend/models/deepseek_v41/model.py"
        module.write_text("# no gate to pin\n")
        self.git(self.ascend, "add", str(module))
        self.git(self.ascend, "commit", "-qm", "custom module")
        overrides = {"SRC": str(self.ascend), "PATCH_DIR": str(self.ascend_patches)}
        self.assert_ok(self.patches("apply", **overrides))
        module.write_text("# unrelated edit after apply\n")
        self.assert_ok(self.patches("verify", **overrides))
        self.assert_ok(self.patches("revert", "--force", **overrides))
        self.assertEqual(module.read_text(), "# unrelated edit after apply\n")

    @unittest.skipUnless(shutil.which("patch"), "needs patch for extracted trees")
    def test_extracted_tree_single_component_override(self):
        shutil.rmtree(self.vllm / ".git")
        overrides = {"SRC": str(self.vllm), "PATCH_DIR": str(self.vllm_patches)}
        self.assert_ok(self.patches("apply", **overrides))
        self.assert_ok(self.patches("verify", **overrides))
        self.assertEqual(self.slot.read_text(), "slot = 0\n")
        self.assert_failed(self.patches("revert", "--force", **overrides), "Git 基线")
        self.encoder.write_text("text = 'original'\n")
        self.assert_failed(self.patches("verify", **overrides), "源码与补丁记录不一致")


if __name__ == "__main__":
    unittest.main()
