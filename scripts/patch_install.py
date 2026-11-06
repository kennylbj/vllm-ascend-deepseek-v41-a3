#!/usr/bin/env python3
"""Install the DeepSeek V4.1 patch series and verify the deployed files."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

PATCHES = Path(__file__).resolve().parent / "patches"
ASCEND = "vllm-ascend"
ASCEND_SERIES = PATCHES / "vllm-ascend-v41-rc2"
VLLM_SERIES = PATCHES / "vllm-ced6857af"
ENV_GATES = tuple("VLLM_ASCEND_" + name for name in (
    "ROPE_INDEX_SELECT", "V41_TRITON_SLOT_MAPPING", "V41_MOE_MASK_RANGE",
    "V41_O_PROJ_2D", "V41_QLI_NO_CANDIDATE", "V41_ENGRAM_WKV_TP",
))
MODULE_GATES = {
    "vllm_ascend/models/deepseek_v41/engram/layer.py": "V41_ENGRAM_WKV_TP",
    "vllm_ascend/models/deepseek_v41/model.py": "V41_ENGRAM_WKV_TP",
    "vllm_ascend/models/deepseek_v41/indexer.py": "V41_QLI_NO_CANDIDATE",
    "vllm_ascend/ops/fused_moe/token_dispatcher.py": "V41_MOE_MASK_RANGE",
}
ENV_FILE = "vllm_ascend/envs.py"
GATE_FILES = (ENV_FILE, *MODULE_GATES)
MARKER_DIR = ".patches-applied.d"


class PatchError(Exception):
    pass


def command(args, cwd, *, data=None, required=True):
    result = subprocess.run(args, cwd=cwd, input=data, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if required and result.returncode:
        raise PatchError(result.stderr.decode(errors="replace").strip()
                         or result.stdout.decode(errors="replace").strip()
                         or "{} failed".format(args[0]))
    return result


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_path(root, name):
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or not path.parts
            or any(part in (".git", MARKER_DIR, ".patches-applied") for part in path.parts)):
        raise PatchError("无效补丁路径: {}".format(name))
    target = root / path
    if target.resolve() != target:
        raise PatchError("补丁路径包含符号链接: {}".format(target))
    return target


def file_state(root, paths):
    state = {}
    for name in paths:
        path = source_path(root, name)
        if path.exists():
            mode = path.stat().st_mode
            if not stat.S_ISREG(mode):
                raise PatchError("补丁目标不是普通文件: {}".format(path))
            state[name] = [sha(path.read_bytes()), bool(mode & stat.S_IXUSR)]
        else:
            state[name] = None
    return state


def atomic_write(path, data, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".dsv41-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            os.fchmod(fh.fileno(), mode)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def patch_paths(data):
    # Git unified diffs use the old and new headers for deletions/additions.
    # Pure rename/copy headers also identify both paths.
    headers = (b"--- a/", b"+++ b/", b"rename from ", b"rename to ",
               b"copy from ", b"copy to ")
    paths = set()
    for line in data.splitlines():
        for prefix in headers:
            if line.startswith(prefix):
                paths.add(os.fsdecode(line[len(prefix):].split(b"\t", 1)[0]))
                break
    return paths


def head_of(root):
    if shutil.which("git") is None:
        return None
    top = command(["git", "rev-parse", "--show-toplevel"], root, required=False)
    if top.returncode or Path(os.fsdecode(top.stdout.rstrip())).resolve() != root:
        return None
    return command(["git", "rev-parse", "HEAD"], root).stdout.decode().strip()


def checkout_files(root, head, paths, dest):
    if not paths:
        return
    tree = command(["git", "--literal-pathspecs", "ls-tree", "-r", "-z",
                    head, "--", *sorted(paths)], root).stdout
    for item in tree.split(b"\0"):
        if not item:
            continue
        header, filename = item.split(b"\t", 1)
        mode, kind, blob = header.decode().split()
        name = os.fsdecode(filename)
        if name not in paths or kind != "blob" or mode not in ("100644", "100755"):
            raise PatchError("无法从 Git 基线读取普通文件: {}".format(name))
        data = command(["git", "cat-file", "blob", blob], root).stdout
        atomic_write(source_path(dest, name), data, int(mode, 8) & 0o777)


def gate_updates(root, *, required):
    rules = {
        ENV_FILE: [(r'(_strict_binary_env\(\s*"{}")(?:\s*,\s*"[01]")?\s*\)', gate)
                   for gate in ENV_GATES],
    }
    for name, gate in MODULE_GATES.items():
        rules[name] = [(r'(os\.environ\.get\(\s*"{}")(?:\s*,\s*"[01]")?\s*\)',
                        "VLLM_ASCEND_" + gate)]
    updates = {}
    for name, patterns in rules.items():
        path = source_path(root, name)
        data = path.read_bytes() if path.is_file() else b""
        updated = data.decode("utf-8")
        for pattern, gate in patterns:
            updated, count = re.subn(pattern.format(gate), r'\1, "1")', updated)
            if required and not count:
                raise PatchError("{} 缺少 gate {}".format(name, gate))
        if updated.encode() != data:
            updates[name] = updated.encode()
    return updates


def pin_gates(root, *, required):
    for name, data in gate_updates(root, required=required).items():
        path = source_path(root, name)
        atomic_write(path, data, stat.S_IMODE(path.stat().st_mode))


class Component:
    def __init__(self, name, root, directory, *, required_gates=False):
        self.name = name
        self.root = root.resolve()
        self.directory = directory
        self.pin = name == ASCEND
        self.required_gates = required_gates
        self.marker = self.root / MARKER_DIR
        self.manifest = self.marker / "manifest.json"

    def load(self):
        if self.marker.is_symlink() or self.manifest.is_symlink():
            raise PatchError("补丁记录路径包含符号链接: {}".format(self.marker))
        self.head = head_of(self.root)
        self.record = None
        if self.manifest.exists():
            self.record = json.loads(self.manifest.read_text(encoding="utf-8"))
            if (not isinstance(self.record, dict) or self.record.get("version") != 1
                    or self.record.get("component") != self.name
                    or not isinstance(self.record.get("baseline"), dict)
                    or not isinstance(self.record.get("files"), dict)
                    or not isinstance(self.record.get("series"), list)):
                raise PatchError("无效补丁记录: {}".format(self.manifest))

    def read_series(self):
        self.patches = []
        self.paths = set()
        self.series = []
        names = set()
        for line in (self.directory / "series.conf").read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            if Path(name).name != name or not name.endswith(".patch") or name in names:
                raise PatchError("无效或重复的补丁名称: {}".format(name))
            names.add(name)
            data = (self.directory / name).read_bytes()
            paths = patch_paths(data)
            if self.head is not None:
                # Git also reports paths in mode-only and binary patches.
                stats = command(["git", "apply", "--numstat", "-z"], self.root, data=data).stdout
                paths.update(os.fsdecode(entry.split(b"\t", 2)[2])
                             for entry in stats.split(b"\0") if entry)
            if not paths:
                raise PatchError("补丁未声明受影响文件: " + name)
            self.paths.update(paths)
            self.series.append([name, sha(data)])
            self.patches.append((name, data))
        if not self.series:
            raise PatchError("补丁序列为空: {}".format(self.directory))
        self.patch_targets = self.paths.copy()
        if self.pin:
            self.paths.update(GATE_FILES)

    def check_record(self):
        if self.record["baseline"].get("head") != self.head:
            raise PatchError("源码基线已变更，请重建容器或重新解包 rootfs")
        targets = set(self.record["files"])
        if self.record["series"] != self.series or not self.patch_targets <= targets <= self.paths:
            raise PatchError("补丁序列已变更，请 revert --force 后重新应用，或重建源码树")
        if self.record["files"] != file_state(self.root, targets):
            raise PatchError("已安装源码与补丁记录不一致，请重建或显式回滚")
        if self.pin and gate_updates(self.root, required=self.required_gates):
            raise PatchError("Ascend gate 默认值应为 1")

    def apply_patch(self, root, name, data):
        if self.head is not None:
            args = ["git", "apply", "--whitespace=nowarn"]
        else:
            args = ["patch", "-p1", "--forward", "--batch", "--fuzz=0", "--no-backup-if-mismatch"]
        try:
            command(args, root, data=data)
        except PatchError as exc:
            raise PatchError("{} 无法应用，请核对基线或重建源码树\n{}".format(name, exc)) from exc

    def preflight(self, scratch):
        self.read_series()
        if self.record is not None:
            self.check_record()
            self.before = file_state(self.root, self.paths)
            self.expected = self.record
            self.start = len(self.patches)
            return "已验证"
        self.before = file_state(self.root, self.paths)
        if (self.root / ".patches-applied").exists():
            raise PatchError("旧的单文件补丁记录无法确认安装范围，请重建源码树")
        if self.head is not None:
            retired = {p.stem for p in self.marker.glob("*.done")} - {name for name, _ in self.patches}
            if retired:
                raise PatchError("旧记录含已移除的补丁，请重建源码树: " + ", ".join(sorted(retired)))
            checkout_files(self.root, self.head, self.paths, scratch)
            baseline = {"head": self.head}
        else:
            if self.marker.exists():
                raise PatchError("无 Git 基线可验证旧补丁记录，请重新解包 rootfs")
            for name, value in self.before.items():
                if value is not None:
                    target = source_path(scratch, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path(self.root, name), target)
            baseline = {"head": None, "files": self.before}
        start = 0 if file_state(scratch, self.paths) == self.before else None
        for index, (name, data) in enumerate(self.patches, 1):
            self.apply_patch(scratch, name, data)
            for path in scratch.rglob("*"):
                if (path.is_file() or path.is_symlink()) and path.relative_to(scratch).as_posix() not in self.paths:
                    raise PatchError("{} 写入了未记录的路径: {}".format(name, path.relative_to(scratch)))
            if file_state(scratch, self.paths) == self.before:
                start = index
        targets = self.patch_targets.copy()
        if self.pin:
            updates = gate_updates(scratch, required=self.required_gates)
            targets.update(updates)
            # Recognize interruption during gate pinning as well as legacy trees
            # whose final patch was installed with all defaults already pinned.
            if self.head is not None and start is None:
                unpinned = file_state(scratch, self.paths)
                if all(self.before[name] == value or (
                    name in updates and self.before[name] == [sha(updates[name]), value[1]]
                ) for name, value in unpinned.items()):
                    start = len(self.patches)
            pin_gates(scratch, required=self.required_gates)
        final = file_state(scratch, self.paths)
        if self.before == final:
            start = len(self.patches)
        if start is None:
            raise PatchError("源码与基线或已应用的补丁前缀不匹配，请重建容器或重新解包 rootfs")
        self.start = start
        self.expected = {
            "version": 1, "component": self.name, "baseline": baseline,
            "series": self.series, "files": {name: final[name] for name in targets},
        }
        if start == len(self.patches) and self.before == final:
            return "源码已应用，运行 apply 写入校验记录"
        return "可应用（已应用 {}/{}）".format(start, len(self.patches))

    def install(self):
        if head_of(self.root) != self.head or file_state(self.root, self.paths) != self.before:
            raise PatchError("预检后源码已变化，请重试")
        for name, data in self.patches[self.start:]:
            self.apply_patch(self.root, name, data)
        if self.pin:
            pin_gates(self.root, required=self.required_gates)
        self.record = self.expected
        self.check_record()

    def save(self):
        if not self.manifest.exists():
            atomic_write(self.manifest, (json.dumps(self.record, indent=2, sort_keys=True)
                                         + "\n").encode())

    def prepare_revert(self, scratch):
        if self.record is None:
            self.read_series()
            if self.head is not None and not self.marker.exists() and not (self.root / ".patches-applied").exists():
                checkout_files(self.root, self.head, self.paths, scratch)
                if file_state(self.root, self.paths) == file_state(scratch, self.paths):
                    return
            raise PatchError("缺少可回滚的补丁记录，请重建容器或重新解包 rootfs")
        if self.head is None or self.record["baseline"].get("head") != self.head:
            raise PatchError("当前 Git 基线无法用于回滚，请重建容器或重新解包 rootfs")
        paths = set(self.record["files"])
        checkout_files(self.root, self.head, paths, scratch)
        file_state(self.root, paths)

    def revert(self, scratch):
        if self.record is None:
            return
        if head_of(self.root) != self.head:
            raise PatchError("预检后 Git 基线已变化，请重试")
        for name, state in file_state(scratch, self.record["files"]).items():
            path = source_path(self.root, name)
            if state is None:
                if path.exists():
                    path.unlink()
            else:
                base = source_path(scratch, name)
                atomic_write(path, base.read_bytes(), stat.S_IMODE(base.stat().st_mode))
        # Keep the record until all affected files have been restored.
        for marker in self.marker.glob("*.done"):
            marker.unlink()
        self.manifest.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootfs", type=Path, required=True)
    parser.add_argument("--src", type=Path)
    parser.add_argument("--patch-dir", type=Path)
    parser.add_argument("action", nargs="?", default="status",
                        choices=("status", "check", "apply", "verify", "revert"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workspace = args.rootfs / "vllm-workspace"
    if args.src or args.patch_dir:
        source = args.src or workspace / ASCEND
        if (source / ENV_FILE).is_file():
            name = ASCEND
        elif (source / "vllm/__init__.py").is_file():
            name = "vllm"
        else:
            name = "custom"
        default_series = VLLM_SERIES if name == "vllm" else ASCEND_SERIES
        components = [Component(name, source, args.patch_dir or default_series)]
    else:
        components = [
            Component(ASCEND, workspace / ASCEND, ASCEND_SERIES, required_gates=True),
            Component("vllm", workspace / "vllm", VLLM_SERIES),
        ]
    label = args.action
    try:
        if args.action == "revert" and not args.force:
            raise PatchError("回滚将丢弃补丁涉及文件的工作树改动，请显式传入 --force")
        with ExitStack() as stack:
            # Serialize installer operations on the same source trees without
            # creating any lock files, including for read-only checks.
            for root in sorted({component.root for component in components}):
                label = root
                fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                stack.callback(os.close, fd)
                fcntl.flock(fd, fcntl.LOCK_EX if args.action in ("apply", "revert") else fcntl.LOCK_SH)
            work = []
            for component in components:
                label = component.name
                component.load()
                if args.action == "verify":
                    component.read_series()
                    if component.record is None:
                        raise PatchError("缺少补丁校验记录，请先运行 apply")
                    component.check_record()
                else:
                    scratch = Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
                    work.append((component, scratch))
                    if args.action == "revert":
                        component.prepare_revert(scratch)
                    else:
                        print("[{}] {}: {}".format(args.action, label, component.preflight(scratch)))
            if args.action == "apply":
                for component, _ in work:
                    label = component.name
                    component.install()
                for component, _ in work:
                    label = component.name
                    component.save()
            if args.action == "revert":
                for component, scratch in work:
                    label = component.name
                    component.revert(scratch)
            if args.action in ("apply", "verify"):
                print("[{}] ✅ 补丁记录与源码校验通过".format(args.action))
    except (PatchError, OSError, ValueError) as exc:
        print("[{}] ❌ {}: {}".format(args.action, label, exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
