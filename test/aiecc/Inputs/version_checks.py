# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util
import pathlib
import re
import subprocess
import sys
import tempfile

script = pathlib.Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    source = root / "source"
    source.mkdir()
    header = root / "AIECCVersion.h"

    spec = importlib.util.spec_from_file_location("version_helper", sys.argv[2])
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper._HERE = source
    python_version = root / "_build_version.py"

    def stamp():
        subprocess.run(
            [
                "cmake",
                f"-DSOURCE_DIR={source}",
                f"-DOUTPUT_FILE={header}",
                "-P",
                script,
            ],
            check=True,
        )
        content = header.read_text()
        helper.write_build_metadata(python_version)
        values = {}
        exec(python_version.read_text(), values)
        revision = re.search(r'AIECC_GIT_SHA "(.*)"', content)[1]
        dirty = int(re.search(r"AIECC_GIT_DIRTY (-?\d+)", content)[1])
        assert values["revision"] == (None if revision == "unknown" else revision)
        assert values["dirty"] == (None if dirty == -1 else bool(dirty))
        assert values["version"]
        return content

    assert 'AIECC_GIT_SHA "unknown"' in stamp()
    assert "AIECC_GIT_DIRTY -1" in stamp()
    subprocess.run(["git", "init", "-q", source], check=True)

    def git(*args):
        return subprocess.check_output(["git", "-C", source, *args], text=True).strip()

    tracked = source / "tracked.txt"
    tracked.write_text("original\n")
    git("add", "tracked.txt")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "fixture",
    )
    revision = git("rev-parse", "HEAD")
    clean = stamp()
    assert f'AIECC_GIT_SHA "{revision}"' in clean
    assert "AIECC_GIT_DIRTY 0" in clean
    modified = header.stat().st_mtime_ns
    python_modified = python_version.stat().st_mtime_ns
    assert stamp() == clean
    assert header.stat().st_mtime_ns == modified
    assert python_version.stat().st_mtime_ns == python_modified
    (source / "untracked.txt").write_text("not a source change\n")
    assert stamp() == clean
    tracked.write_text("modified\n")
    assert "AIECC_GIT_DIRTY 1" in stamp()
    git("add", "tracked.txt")
    assert "AIECC_GIT_DIRTY 1" in stamp()
    git("checkout", "HEAD", "--", "tracked.txt")
    assert stamp() == clean

    dependency = root / "dependency"
    dependency.mkdir()
    subprocess.run(["git", "init", "-q", dependency], check=True)
    (dependency / "file.txt").write_text("dependency\n")

    def dependency_git(path, *args):
        subprocess.run(["git", "-C", path, *args], check=True, capture_output=True)

    dependency_git(dependency, "add", "file.txt")
    identity = [
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
    ]
    dependency_git(dependency, *identity, "commit", "-qm", "dependency")
    git(
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(dependency),
        "dependency",
    )
    git(*identity, "commit", "-qm", "add dependency")
    assert "AIECC_GIT_DIRTY 0" in stamp()
    checkout = source / "dependency"
    (checkout / "file.txt").write_text("build patch\n")
    assert "AIECC_GIT_DIRTY 0" in stamp()
    dependency_git(checkout, "add", "file.txt")
    dependency_git(checkout, *identity, "commit", "-qm", "different revision")
    assert "AIECC_GIT_DIRTY 1" in stamp()
