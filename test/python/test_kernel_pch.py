# test_kernel_pch.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
# REQUIRES: peano
"""The precompiled intrinsics header builds the object a plain compile builds."""

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from aie.utils import config
from aie.utils.compile.cache.utils import file_lock

_COMPILE = """
import sys
from aie.utils.compile.utils import compile_cxx_core_function
source, output, *args = sys.argv[1:]
compile_cxx_core_function(source, "aie2p", output, compile_args=args)
"""

_KERNEL = """
extern "C" void fill(int *out) {
#ifdef VALUE
  out[0] = VALUE;
#else
  out[0] = 1;
#endif
}
"""


def _compile(
    tmp_path, cache, name, args=(), source=_KERNEL, pch=True, peano=None, environ=None
):
    """Compile ``source`` in a fresh interpreter whose NPU_CACHE_HOME is ``cache``.

    Each compile has a directory of its own and the same file names, so objects
    built alike are byte-identical.
    """
    work = tmp_path / name
    work.mkdir()
    (work / "include").mkdir()
    (work / "extra.h").write_text("")
    src = work / "kernel.cc"
    src.write_text(source)
    out = work / "kernel.o"
    env = dict(
        os.environ, NPU_CACHE_HOME=str(cache), AIE_KERNEL_PCH="1" if pch else "0"
    )
    if peano is not None:
        env["PEANO_INSTALL_DIR"] = str(peano)
    env.update(environ or {})
    ret = subprocess.run(
        [sys.executable, "-c", _COMPILE, str(src), str(out), *args],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
    )
    return ret, out


def _pchs(cache):
    return sorted((cache / "pch").glob("*.pch"))


def test_pch_compile_builds_the_plain_object(tmp_path):
    cache = tmp_path / "cache"
    ret, with_pch = _compile(tmp_path, cache, "with")
    assert ret.returncode == 0, ret.stderr
    ret, plain = _compile(tmp_path, cache, "plain", pch=False)
    assert ret.returncode == 0, ret.stderr
    assert len(_pchs(cache)) == 1
    assert with_pch.read_bytes() == plain.read_bytes()


def test_macros_the_headers_ignore_share_one_pch(tmp_path):
    cache = tmp_path / "cache"
    for value in ("2", "3"):
        ret, with_pch = _compile(tmp_path, cache, f"with{value}", [f"-DVALUE={value}"])
        assert ret.returncode == 0, ret.stderr
        ret, plain = _compile(
            tmp_path, cache, f"plain{value}", [f"-DVALUE={value}"], pch=False
        )
        assert ret.returncode == 0, ret.stderr
        assert with_pch.read_bytes() == plain.read_bytes()
    assert len(_pchs(cache)) == 1


@pytest.mark.parametrize(
    "args",
    [
        ["-D_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER"],
        ["-D__int_least64_t=long"],
        ["-ffast-math"],
        ["-I", "shadow"],
    ],
    ids=[
        "macro-the-headers-test",
        "macro-a-header-defines",
        "language-flag",
        "shadowing-dir",
    ],
)
def test_command_the_headers_read_differently_gets_its_own_pch(tmp_path, args):
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "stdint.h").write_text("#define SHADOWED 1\n#include_next <stdint.h>\n")
    args = [str(shadow) if a == "shadow" else a for a in args]
    cache = tmp_path / "cache"
    assert _compile(tmp_path, cache, "bare")[0].returncode == 0
    ret, with_pch = _compile(tmp_path, cache, "own", args)
    assert ret.returncode == 0, ret.stderr
    ret, plain = _compile(tmp_path, cache, "plain", args, pch=False)
    assert ret.returncode == 0, ret.stderr
    assert with_pch.read_bytes() == plain.read_bytes()
    assert len(_pchs(cache)) == 2


def test_refused_pch_falls_back_and_is_rebuilt(tmp_path):
    cache = tmp_path / "cache"
    assert _compile(tmp_path, cache, "first")[0].returncode == 0
    (pch,) = _pchs(cache)
    pch.write_bytes(b"not a precompiled header")

    ret, fallback = _compile(tmp_path, cache, "fallback")
    assert ret.returncode == 0, ret.stderr
    assert not pch.exists()
    ret, plain = _compile(tmp_path, cache, "plain", pch=False)
    assert fallback.read_bytes() == plain.read_bytes()

    assert _compile(tmp_path, cache, "rebuilt")[0].returncode == 0
    assert _pchs(cache) == [pch]
    assert pch.read_bytes() != b"not a precompiled header"


def test_kernel_error_is_the_kernels_own_and_keeps_the_pch(tmp_path):
    cache = tmp_path / "cache"
    assert _compile(tmp_path, cache, "first")[0].returncode == 0
    ret, _ = _compile(tmp_path, cache, "broken", source="int broken(;\n")
    assert ret.returncode != 0
    assert "kernel.cc" in ret.stderr
    assert "precompiled" not in ret.stderr.lower()
    assert len(_pchs(cache)) == 1


@pytest.mark.parametrize(
    "args",
    [["-Iinclude"], ["-I", "include"], ["-include", "extra.h"]],
    ids=["relative-I", "relative-I-separate", "caller-include"],
)
def test_command_the_pch_cannot_stand_for_compiles_plainly(tmp_path, args):
    cache = tmp_path / "cache"
    ret, _ = _compile(tmp_path, cache, "kernel", args)
    assert ret.returncode == 0, ret.stderr
    assert _pchs(cache) == []


def test_concurrent_compiles_build_one_pch(tmp_path):
    cache = tmp_path / "cache"
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda i: _compile(tmp_path, cache, f"k{i}"), range(4)))
    assert all(ret.returncode == 0 for ret, _ in results), [
        r.stderr for r, _ in results
    ]
    assert len(_pchs(cache)) == 1
    objects = {out.read_bytes() for _, out in results}
    assert len(objects) == 1


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs privileges")
def test_one_compiler_at_two_paths_shares_its_pch(tmp_path):
    cache = tmp_path / "cache"
    other = tmp_path / "other-peano"
    other.symlink_to(config.peano_install_dir(), target_is_directory=True)
    ret, first = _compile(tmp_path, cache, "first")
    assert ret.returncode == 0, ret.stderr
    ret, second = _compile(tmp_path, cache, "second", peano=other)
    assert ret.returncode == 0, ret.stderr
    assert len(_pchs(cache)) == 1
    assert first.read_bytes() == second.read_bytes()


def test_compile_does_not_wait_out_a_stuck_pch_build(tmp_path):
    cache = tmp_path / "cache"
    assert _compile(tmp_path, cache, "first")[0].returncode == 0
    (pch,) = _pchs(cache)
    pch.unlink()
    with file_lock(str(pch.with_suffix(".lock"))):
        start = time.monotonic()
        ret, plain = _compile(tmp_path, cache, "plain")
        waited = time.monotonic() - start
    assert ret.returncode == 0, ret.stderr
    assert _pchs(cache) == []
    assert waited < 120


_COMPILE_AROUND_A_NEW_HEADER = """
import sys
from pathlib import Path
from aie.utils.compile.utils import compile_cxx_core_function
source, before, after, shadow = sys.argv[1:]
args = ["-I", shadow]
compile_cxx_core_function(source, "aie2p", before, compile_args=args)
Path(shadow, "stdint.h").write_text("#define SHADOWED 1\\n#include_next <stdint.h>\\n")
compile_cxx_core_function(source, "aie2p", after, compile_args=args)
"""

_SHADOW_AWARE_KERNEL = """
extern "C" void fill(int *out) {
#ifdef SHADOWED
  out[0] = 7;
#else
  out[0] = 1;
#endif
}
"""


def test_header_created_in_an_include_dir_reaches_the_next_compile(tmp_path):
    """One process compiles, a header that shadows one the injected headers
    include appears in the kernel's include dir, and the process compiles
    again: the second object is the one a plain compile now builds."""
    cache = tmp_path / "cache"
    work = tmp_path / "work"
    shadow = work / "shadow"
    shadow.mkdir(parents=True)
    source = work / "kernel.cc"
    source.write_text(_SHADOW_AWARE_KERNEL)
    env = dict(os.environ, NPU_CACHE_HOME=str(cache), AIE_KERNEL_PCH="1")
    ret = subprocess.run(
        [
            sys.executable,
            "-c",
            _COMPILE_AROUND_A_NEW_HEADER,
            str(source),
            str(work / "before.o"),
            str(work / "after.o"),
            str(shadow),
        ],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
    )
    assert ret.returncode == 0, ret.stderr

    ret, plain = _compile(
        tmp_path,
        cache,
        "plain-run",
        ["-I", str(shadow)],
        source=_SHADOW_AWARE_KERNEL,
        pch=False,
    )
    assert ret.returncode == 0, ret.stderr
    assert (work / "after.o").read_bytes() == plain.read_bytes()
    assert (work / "before.o").read_bytes() != plain.read_bytes()


def test_environment_search_path_gets_its_own_pch(tmp_path):
    """CPATH adds a search path the command does not show; a compile under one
    that shadows a header the injected headers include must not reuse the PCH
    built without it."""
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "stdint.h").write_text("#define SHADOWED 1\n#include_next <stdint.h>\n")
    cache = tmp_path / "cache"
    cpath = {"CPATH": str(shadow)}
    ret, bare = _compile(tmp_path, cache, "bare", source=_SHADOW_AWARE_KERNEL)
    assert ret.returncode == 0, ret.stderr
    ret, with_pch = _compile(
        tmp_path, cache, "cpath", source=_SHADOW_AWARE_KERNEL, environ=cpath
    )
    assert ret.returncode == 0, ret.stderr
    ret, plain = _compile(
        tmp_path, cache, "plain", source=_SHADOW_AWARE_KERNEL, pch=False, environ=cpath
    )
    assert ret.returncode == 0, ret.stderr
    assert with_pch.read_bytes() == plain.read_bytes()
    assert bare.read_bytes() != plain.read_bytes()
