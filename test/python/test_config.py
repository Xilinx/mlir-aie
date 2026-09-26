# test_config.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
# REQUIRES: peano

"""aie.utils.config tool resolution, exercised against real binaries.

Coverage includes:

  * Discovery order runs the real resolver in a real subprocess, over real
    executables placed on a real PATH. Reading the environment and execing a
    candidate is the behaviour under test, so an in-process stand-in would
    only test the stand-in.

  * The llvm-* resolvers are then pointed at a real AIE object. Their whole
    reason to exist is that the AIEngine e_machine is unknown to GNU binutils,
    and the only way to show the resolved binary clears that bar is to run it.

Host compiler discovery separately checks Peano exclusion and explicit CXX
overrides. These tests need no NPU; the real-object tests compile with Peano.
"""

import os
import subprocess
import sys

import aie.utils.config as config
import pytest
from aie.utils.compile.utils import compile_cxx_core_function

_KERNEL_SOURCE = """
extern "C" void helper_fn(int *p) { *p += 1; }
extern "C" void add_one(int *p) { helper_fn(p); }
"""

# The synthetic-PATH tests stand up fake tools as shell scripts, which needs a
# POSIX shebang. The resolver's Windows spelling is covered by the real-object
# tests, which run whatever the platform actually bundles.
posix_only = pytest.mark.skipif(
    os.name == "nt", reason="fake tools are shebang shell scripts"
)

_FAKE = "config._find_llvm_tool('llvm-faketool', 'AIE_FAKETOOL_PATH')"


@pytest.fixture(scope="module")
def aie_object(tmp_path_factory):
    """Compile a real AIE object file exporting two known external symbols."""
    tmp_dir = tmp_path_factory.mktemp("aie_object")
    source = tmp_dir / "add_one.cc"
    source.write_text(_KERNEL_SOURCE)
    obj = tmp_dir / "add_one.o"
    compile_cxx_core_function(str(source), "aie2p", str(obj))
    assert obj.exists()
    return obj


def test_host_cxx_skips_peano_on_path(tmp_path, monkeypatch):
    peano = tmp_path / "peano"
    host = tmp_path / "host"
    for directory in (peano / "bin", host):
        directory.mkdir(parents=True)
        compiler = directory / config._executable_name("clang++")
        compiler.touch(mode=0o755)
    monkeypatch.setattr(config.config, "peano_install_dir", str(peano))
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setenv("PATH", os.pathsep.join([str(peano / "bin"), str(host)]))
    assert os.path.samefile(
        config.host_cxx_path(), host / config._executable_name("clang++")
    )

    monkeypatch.setenv("PATH", str(peano / "bin"))
    with pytest.raises(RuntimeError, match="Could not find a host C\\+\\+ compiler"):
        config.host_cxx_path()


def test_host_cxx_honors_explicit_override(tmp_path, monkeypatch):
    compiler = tmp_path / config._executable_name("custom-cxx")
    compiler.touch(mode=0o755)
    monkeypatch.setenv("CXX", str(compiler))
    assert config.host_cxx_path() == str(compiler)

    monkeypatch.setenv("CXX", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="CXX is set"):
        config.host_cxx_path()


@pytest.mark.parametrize(
    "suffix,target,flag",
    [
        (".dll", "x86_64-pc-windows-msvc", "-Wl,/Brepro"),
        (".dll", "x86_64-w64-windows-gnu", "-Wl,--no-insert-timestamp"),
        (".dll", "x86_64-w64-mingw32", "-Wl,--no-insert-timestamp"),
        (".so", None, None),
    ],
)
def test_host_shared_library_link_flags(monkeypatch, tmp_path, suffix, target, flag):
    from aie.utils.compile import utils as compile_utils

    calls = []

    def query_target(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=target + "\n")

    monkeypatch.setattr(compile_utils, "SHARED_LIB_SUFFIX", suffix)
    monkeypatch.setattr(config, "host_cxx_path", lambda: "host-cxx")
    monkeypatch.setattr(compile_utils.subprocess, "run", query_target)
    command = compile_utils.host_shared_lib_cmd(
        tmp_path / "source.cpp", tmp_path / f"dispatch{suffix}", opt="-O2"
    )
    if flag:
        assert flag in command
        assert calls == [["host-cxx", "-dumpmachine"]]
    else:
        assert not any(arg.startswith("-Wl,") for arg in command)
        assert not calls


def _write_tool(directory, name, exit_code=0):
    """Create a real executable that reports a version and exits `exit_code`."""
    directory.mkdir(parents=True, exist_ok=True)
    tool = directory / name
    tool.write_text(f"#!/bin/sh\necho '{name} 1.0'\nexit {exit_code}\n")
    tool.chmod(0o755)
    return tool


def _resolve(expr, path_dirs=None, llvm_bin=None, **env_overrides):
    """Evaluate `expr` against aie.utils.config in a fresh interpreter.

    Returns (stdout, stderr, returncode). The subprocess is the point: the
    resolver reads the real process environment and the real PATH, exactly as
    it does during a build. `llvm_bin` stands in for the LLVM bin directory
    recorded in the generated configure.py.
    """
    env = {**os.environ, **env_overrides}
    if path_dirs is not None:
        env["PATH"] = os.pathsep.join(str(d) for d in path_dirs)
    setup = (
        ""
        if llvm_bin is None
        else f"config.config.llvm_tools_binary_dir = {str(llvm_bin)!r}\n"
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import aie.utils.config as config\n{setup}print({expr})",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.stdout.strip(), proc.stderr.strip(), proc.returncode


@posix_only
def test_env_override_is_used(tmp_path):
    tool = _write_tool(tmp_path / "override", "llvm-faketool")
    stdout, _, code = _resolve(_FAKE, AIE_FAKETOOL_PATH=str(tool))
    assert code == 0
    assert stdout == str(tool)


def test_env_override_missing_file_raises(tmp_path):
    _, stderr, code = _resolve(
        _FAKE, AIE_FAKETOOL_PATH=str(tmp_path / "does-not-exist")
    )
    assert code != 0
    assert "AIE_FAKETOOL_PATH" in stderr


@posix_only
def test_env_override_beats_path(tmp_path):
    on_path = _write_tool(tmp_path / "bin", "llvm-faketool")
    preferred = _write_tool(tmp_path / "override", "llvm-faketool")
    stdout, _, code = _resolve(
        _FAKE, path_dirs=[on_path.parent], AIE_FAKETOOL_PATH=str(preferred)
    )
    assert code == 0
    assert stdout == str(preferred)


@posix_only
def test_broken_candidate_is_skipped_for_a_working_one(tmp_path):
    """A tool that exists but exits nonzero must not win over a working one.

    This is what motivates execing candidates at all: a broken binary emits
    nothing rather than an error, which a caller reads as "this object defines
    no symbols".
    """
    broken = _write_tool(tmp_path / "broken", "llvm-faketool", exit_code=1)
    working = _write_tool(tmp_path / "working", "llvm-faketool")
    stdout, _, code = _resolve(_FAKE, path_dirs=[broken.parent, working.parent])
    assert code == 0
    assert stdout == str(working)


@posix_only
def test_hanging_candidate_is_skipped_for_a_working_one(tmp_path):
    hanging = _write_tool(tmp_path / "hanging", "llvm-faketool")
    hanging.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    working = _write_tool(tmp_path / "working", "llvm-faketool")
    stdout, _, code = _resolve(_FAKE, path_dirs=[hanging.parent, working.parent])
    assert code == 0
    assert stdout == str(working)


@posix_only
@pytest.mark.parametrize("name", ["llvm-faketool", "llvm-faketool-18"])
@pytest.mark.parametrize("kind", ["directory", "fifo", "non_executable"])
def test_path_ignores_non_executable_files(tmp_path, name, kind):
    candidate = tmp_path / name
    if kind == "directory":
        candidate.mkdir()
    elif kind == "fifo":
        os.mkfifo(candidate, 0o755)
    else:
        candidate.write_text("#!/bin/sh\nexit 0\n")
        candidate.chmod(0o644)
    _, stderr, code = _resolve(_FAKE, path_dirs=[tmp_path])
    assert code != 0
    assert "Could not find llvm-faketool" in stderr


@posix_only
def test_versioned_spelling_used_when_bare_name_absent(tmp_path):
    tool = _write_tool(tmp_path / "bin", "llvm-faketool-18")
    stdout, _, code = _resolve(_FAKE, path_dirs=[tool.parent])
    assert code == 0
    assert stdout == str(tool)


@posix_only
def test_bare_name_preferred_over_versioned(tmp_path):
    bare = _write_tool(tmp_path / "bin", "llvm-faketool")
    versioned = _write_tool(tmp_path / "versioned", "llvm-faketool-18")
    stdout, _, code = _resolve(_FAKE, path_dirs=[versioned.parent, bare.parent])
    assert code == 0
    assert stdout == str(bare)


@posix_only
def test_highest_version_wins(tmp_path):
    older = _write_tool(tmp_path / "older", "llvm-faketool-9")
    newest = _write_tool(tmp_path / "bin", "llvm-faketool-18")
    stdout, _, code = _resolve(_FAKE, path_dirs=[older.parent, newest.parent])
    assert code == 0
    assert stdout == str(newest)


@posix_only
def test_broken_versioned_candidate_is_skipped(tmp_path):
    broken = _write_tool(tmp_path / "bin", "llvm-faketool-18", exit_code=1)
    working = _write_tool(tmp_path / "bin", "llvm-faketool-9")
    stdout, _, code = _resolve(_FAKE, path_dirs=[broken.parent])
    assert code == 0
    assert stdout == str(working)


@posix_only
def test_same_version_preserves_path_order(tmp_path):
    preferred = _write_tool(tmp_path / "first", "llvm-faketool-18")
    other = _write_tool(tmp_path / "second", "llvm-faketool-18")
    stdout, _, code = _resolve(_FAKE, path_dirs=[preferred.parent, other.parent])
    assert code == 0
    assert stdout == str(preferred)


@posix_only
def test_all_candidates_broken_returns_one_anyway(tmp_path):
    """With nothing runnable, hand back a candidate rather than "not found".

    The caller then surfaces the tool's own failure, which says why it broke;
    a "could not find llvm-faketool" would point at the wrong problem.
    """
    broken = _write_tool(tmp_path / "bin", "llvm-faketool", exit_code=1)
    stdout, _, code = _resolve(_FAKE, path_dirs=[broken.parent])
    assert code == 0
    assert stdout == str(broken)


def test_nothing_found_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    _, stderr, code = _resolve(_FAKE, path_dirs=[empty])
    assert code != 0
    assert "Could not find llvm-faketool" in stderr
    assert f"PATH directories: {empty}" in stderr


@posix_only
def test_configured_llvm_bin_dir_is_searched(tmp_path):
    """A build tree bundles no llvm-objcopy; the LLVM it was built against has one."""
    empty = tmp_path / "empty"
    empty.mkdir()
    tool = _write_tool(tmp_path / "llvm" / "bin", "llvm-faketool")
    stdout, _, code = _resolve(_FAKE, path_dirs=[empty], llvm_bin=tool.parent)
    assert code == 0
    assert stdout == str(tool)


@posix_only
def test_configured_llvm_bin_dir_beats_path(tmp_path):
    on_path = _write_tool(tmp_path / "bin", "llvm-faketool")
    configured = _write_tool(tmp_path / "llvm" / "bin", "llvm-faketool")
    stdout, _, code = _resolve(
        _FAKE, path_dirs=[on_path.parent], llvm_bin=configured.parent
    )
    assert code == 0
    assert stdout == str(configured)


def test_missing_configured_llvm_bin_dir_is_not_searched(tmp_path):
    """A wheel records its build machine's LLVM directory, absent on this one."""
    empty = tmp_path / "empty"
    empty.mkdir()
    gone = tmp_path / "build-machine" / "bin"
    _, stderr, code = _resolve(_FAKE, path_dirs=[empty], llvm_bin=gone)
    assert code != 0
    assert "Could not find llvm-faketool" in stderr
    assert str(gone) not in stderr


def test_nm_path_lists_symbols_of_a_real_aie_object(aie_object):
    from aie.utils.config import nm_path

    out = subprocess.run(
        [nm_path(), "--defined-only", "--extern-only", str(aie_object)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "add_one" in out
    assert "helper_fn" in out


def test_objcopy_path_renames_symbols_in_a_real_aie_object(tmp_path, aie_object):
    from aie.utils.config import nm_path, objcopy_path

    obj = tmp_path / "renamed.o"
    obj.write_bytes(aie_object.read_bytes())
    symbol_map = tmp_path / "symbols.map"
    symbol_map.write_text("add_one p_add_one\nhelper_fn p_helper_fn\n")

    subprocess.run(
        [objcopy_path(), f"--redefine-syms={symbol_map}", str(obj)],
        check=True,
        capture_output=True,
    )
    out = subprocess.run(
        [nm_path(), "--defined-only", "--extern-only", str(obj)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "p_add_one" in out
    assert "p_helper_fn" in out
    # The originals are gone, not merely joined by the prefixed spellings.
    assert "T add_one" not in out
    assert "T helper_fn" not in out


def test_ar_path_archives_a_real_aie_object(tmp_path, aie_object):
    from aie.utils.config import ar_path

    archive = tmp_path / "libadd_one.a"
    subprocess.run(
        [ar_path(), "rcs", str(archive), str(aie_object)],
        check=True,
        capture_output=True,
    )
    out = subprocess.run(
        [ar_path(), "t", str(archive)], check=True, capture_output=True, text=True
    ).stdout
    assert "add_one.o" in out


@posix_only
def test_aiecc_path_env_override(tmp_path):
    fake_aiecc = _write_tool(tmp_path / "bin", "aiecc")
    stdout, _, code = _resolve("config.aiecc_path()", AIECC_PATH=str(fake_aiecc))
    assert code == 0
    assert stdout == str(fake_aiecc)


def test_aiecc_path_env_override_missing_file_raises(tmp_path):
    _, stderr, code = _resolve(
        "config.aiecc_path()", AIECC_PATH=str(tmp_path / "does-not-exist")
    )
    assert code != 0
    assert "AIECC_PATH" in stderr
