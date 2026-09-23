# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
# REQUIRES: peano
"""KernelObjectCache against a real toolchain.

Every kernel is compiled by Peano. Reuse is observed on the cache entry itself:
an object that was not rebuilt keeps its inode and mtime. No NPU is required.
"""

import contextlib
import subprocess

import aie.utils.compile.utils as compile_utils
import aie.utils.config as config
import pytest
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit import _manifest
from aie.utils.compile.jit._object_cache import KernelObjectCache

_SOURCE = """
#include "scale.h"
extern "C" void helper_fn(int *p) { *p *= SCALE; }
extern "C" void scale(int *p) { helper_fn(p); }
"""


@pytest.fixture(autouse=True)
def _fresh_registry():
    ExternalFunction._instances.clear()
    yield
    ExternalFunction._instances.clear()


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "scale.h").write_text("#define SCALE 2\n")
    path = src / "scale.cc"
    path.write_text(_SOURCE)
    return path


@pytest.fixture
def cache(tmp_path):
    return KernelObjectCache(tmp_path / "objects", lock_timeout_seconds=600)


def _kernel(source, **kwargs):
    return ExternalFunction("scale", source_file=str(source), **kwargs)


def _design_dir(tmp_path, name):
    path = tmp_path / name
    path.mkdir()
    return path


def _entry_object(cache, kernel):
    (obj,) = cache.root.glob(f"*/{kernel.object_file_name}")
    return obj


def _identity(path):
    st = path.stat()
    return st.st_ino, st.st_mtime_ns


def _symbols(object_path):
    out = subprocess.run(
        [config.nm_path(), "--defined-only", "--extern-only", str(object_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(
        line.split()[-1] for line in out.splitlines() if len(line.split()) >= 3
    )


def test_two_designs_share_one_compile(tmp_path, source, cache):
    kernel = _kernel(source)
    first = _design_dir(tmp_path, "design_a")
    second = _design_dir(tmp_path, "design_b")

    compile_utils.compile_external_kernels([kernel], first, "aie2p", object_cache=cache)
    built = _entry_object(cache, kernel)
    before = _identity(built)

    compile_utils.compile_external_kernels(
        [kernel], second, "aie2p", object_cache=cache
    )
    assert _identity(built) == before
    for design in (first, second):
        linked = design / kernel.object_file_name
        assert linked.read_bytes() == built.read_bytes()
        assert (design / f"{kernel.object_file_name}.d").is_file()


def test_fresh_kernel_instance_hits(tmp_path, source, cache):
    """A fresh ExternalFunction -- as each JIT generation builds -- still hits."""
    compile_utils.compile_external_kernels(
        [_kernel(source)], _design_dir(tmp_path, "a"), "aie2p", object_cache=cache
    )
    ExternalFunction._instances.clear()
    again = _kernel(source)
    before = _identity(_entry_object(cache, again))
    compile_utils.compile_external_kernels(
        [again], _design_dir(tmp_path, "b"), "aie2p", object_cache=cache
    )
    assert _identity(_entry_object(cache, again)) == before


def test_header_edit_rebuilds_entry_and_reaches_design_manifest(
    tmp_path, source, cache
):
    kernel = _kernel(source)
    design = _design_dir(tmp_path, "design")
    compile_utils.compile_external_kernels(
        [kernel], design, "aie2p", object_cache=cache
    )
    _manifest.record(design, [kernel], [])
    old_bytes = _entry_object(cache, kernel).read_bytes()
    assert _manifest.is_valid(design)

    (source.parent / "scale.h").write_text("#define SCALE 3\n")

    # The design entry sees the header through the copied depfile.
    assert not _manifest.is_valid(design)
    later = _design_dir(tmp_path, "later")
    compile_utils.compile_external_kernels([kernel], later, "aie2p", object_cache=cache)
    rebuilt = _entry_object(cache, kernel).read_bytes()
    assert rebuilt != old_bytes
    assert (later / kernel.object_file_name).read_bytes() == rebuilt


def test_distinct_recipes_get_distinct_entries(tmp_path, source, cache):
    plain = _kernel(source, object_file_name="plain.o")
    flagged = _kernel(source, object_file_name="flagged.o", compile_flags=["-O1"])
    design = _design_dir(tmp_path, "design")
    compile_utils.compile_external_kernels(
        [plain, flagged], design, "aie2p", object_cache=cache
    )
    assert len(list(cache.root.iterdir())) == 2


def test_prefixed_object_carries_its_stamp(tmp_path, source, cache):
    kernel = _kernel(source, symbol_prefix="op0")
    first = _design_dir(tmp_path, "a")
    second = _design_dir(tmp_path, "b")
    for design in (first, second):
        compile_utils.compile_external_kernels(
            [kernel], design, "aie2p", object_cache=cache
        )
    linked = second / kernel.object_file_name
    assert _symbols(linked) == ["op0_helper_fn", "op0_scale"]
    assert compile_utils._has_current_symbol_prefix_stamp(str(linked), "op0_")
    # The design directory trusts the copy: a later compile there is a no-op.
    before = _identity(linked)
    compile_utils.compile_external_kernel(kernel, second, "aie2p")
    assert _identity(linked) == before


def test_chess_kernel_is_declined(tmp_path, source, cache):
    kernel = _kernel(source, use_chess=True)
    design = _design_dir(tmp_path, "design")
    assert not cache.fetch(kernel, design, "aie2p", None, False)
    assert not cache.root.exists()
    assert list(design.iterdir()) == []


def test_relative_include_dir_resolves_where_the_kernel_was_declared(tmp_path, cache):
    (tmp_path / "inc").mkdir()
    header = tmp_path / "inc" / "relhdr.h"
    header.write_text("#define SCALE 2\n")
    with contextlib.chdir(tmp_path):
        kernel = ExternalFunction(
            "scale",
            source_string='#include "relhdr.h"\nextern "C" void scale(int *p) { *p *= SCALE; }\n',
            include_dirs=["inc"],
        )
    design = _design_dir(tmp_path, "design")
    compile_utils.compile_external_kernels(
        [kernel], design, "aie2p", object_cache=cache
    )
    assert (design / kernel.object_file_name).is_file()
    assert str(header) in (design / f"{kernel.object_file_name}.d").read_text()


def test_one_source_compiled_per_architecture(tmp_path, source, cache):
    """A generic kernel built for aie2 and aie2p gets two entries, not one."""
    kernel = _kernel(source)
    objects = {}
    for arch in ("aie2", "aie2p"):
        design = _design_dir(tmp_path, arch)
        compile_utils.compile_external_kernels(
            [kernel], design, arch, object_cache=cache
        )
        objects[arch] = (design / kernel.object_file_name).read_bytes()
    assert len(list(cache.root.iterdir())) == 2
    assert objects["aie2"] != objects["aie2p"]
    # Each design links the object built for its own target: ELF32 e_flags differ.
    assert objects["aie2"][36:40] != objects["aie2p"][36:40]
