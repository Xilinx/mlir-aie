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
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import aie.utils.compile.utils as compile_utils
import aie.utils.config as config
import numpy as np
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


def _relocated(tool, root):
    """Copy ``tool`` to a new path under ``root`` and return the working copy.

    Some toolchain binaries find their libraries relative to themselves --
    Peano's llvm-nm loads libLLVM.so through RUNPATH ``$ORIGIN/../lib`` -- so a
    bare copy does not start. For those, the rest of the install prefix is
    linked in beside the copy, so the relative paths still resolve.
    """
    copy = root / tool.parent.name / tool.name
    copy.parent.mkdir(parents=True)
    shutil.copy2(tool, copy)
    if subprocess.run([copy, "--version"], capture_output=True).returncode:
        for entry in tool.parent.parent.iterdir():
            if entry != tool.parent:
                (root / entry.name).symlink_to(entry, entry.is_dir())
        subprocess.run([copy, "--version"], capture_output=True, check=True)
    return copy


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


def _build(tmp_path, cache, kernel, arch="aie2p", include_dirs=None, ir=False):
    design = tmp_path / "designs" / str(len(list(tmp_path.glob("designs/*"))))
    design.mkdir(parents=True)
    compile_utils.compile_external_kernels(
        [kernel],
        design,
        arch,
        include_dirs=include_dirs,
        embed_bitcode=ir,
        object_cache=cache,
    )
    return design / kernel.object_file_name


def _edit_source(source):
    source.write_text(_SOURCE.replace("*p *= SCALE", "*p *= SCALE + 1"))
    return {}


def _extra_include(source):
    extra = source.parent / "extra"
    extra.mkdir()
    return {"include_dirs": [str(extra)]}


# Each changes one input of the object's bytes, keeping its file name.
_KEY_INPUTS = {
    "source": lambda source: (_edit_source(source), {}),
    "compile_flags": lambda source: ({"compile_flags": ["-DUNUSED=1"]}, {}),
    "kernel_include_dirs": lambda source: (_extra_include(source), {}),
    "symbol_prefix": lambda source: ({"symbol_prefix": "op0"}, {}),
    "object_file_name": lambda source: ({"object_file_name": "other.o"}, {}),
    "target_arch": lambda source: ({}, {"arch": "aie2"}),
    "design_include_dirs": lambda source: ({}, {"include_dirs": [str(source.parent)]}),
    "retained_ir": lambda source: ({}, {"ir": True}),
}

# A key input need not change the output (an unused -D, an unread include
# dir); these ones do, so the two entries must hold different objects.
_CHANGES_OUTPUT = {"source", "symbol_prefix", "target_arch", "retained_ir"}

# Each changes something the object's bytes do not depend on.
_OTHER_INPUTS = {
    "fresh_instance": {},
    "arg_types": {"arg_types": [np.ndarray[(16,), np.dtype[np.int32]]]},
    "stack_size_override": {"stack_size_override": 512},
}


@pytest.mark.parametrize("change", sorted(_KEY_INPUTS))
def test_changing_a_key_input_misses(tmp_path, source, cache, change):
    first = _build(tmp_path, cache, _kernel(source))
    ExternalFunction._instances.clear()
    kernel_kwargs, build_kwargs = _KEY_INPUTS[change](source)
    second = _build(tmp_path, cache, _kernel(source, **kernel_kwargs), **build_kwargs)
    assert len(list(cache.root.iterdir())) == 2
    if change in _CHANGES_OUTPUT:
        assert first.read_bytes() != second.read_bytes()


@pytest.mark.parametrize("tool", ["nm", "objcopy"])
@pytest.mark.parametrize("mode", ["prefixed", "bitcode"])
def test_object_tool_changes_miss(tmp_path, source, cache, monkeypatch, tool, mode):
    kernel = _kernel(source, **({"symbol_prefix": "op0"} if mode == "prefixed" else {}))
    build_kwargs = {"ir": mode == "bitcode"}
    first = _build(tmp_path, cache, kernel, **build_kwargs)
    original = _entry_object(cache, kernel)
    before = _identity(original)
    _build(tmp_path, cache, kernel, **build_kwargs)
    assert _identity(original) == before

    resolved = Path(getattr(config, f"{tool}_path")()).resolve()
    selected = _relocated(resolved, tmp_path / "toolchain")
    monkeypatch.setenv(f"AIE_{tool.upper()}_PATH", str(selected))
    second = _build(tmp_path, cache, kernel, **build_kwargs)
    assert len(list(cache.root.iterdir())) == 2

    stat = selected.stat()
    os.utime(selected, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    third = _build(tmp_path, cache, kernel, **build_kwargs)
    assert len(list(cache.root.iterdir())) == 3
    if mode == "prefixed":
        assert first.read_bytes() == second.read_bytes() == third.read_bytes()
    else:
        for obj in (first, second, third):
            assert _symbols(obj) == ["helper_fn", "scale"]
            assert compile_utils._object_has_bitcode(str(obj))


@pytest.mark.parametrize("change", sorted(_OTHER_INPUTS))
def test_changing_another_input_hits(tmp_path, source, cache, change):
    kernel = _kernel(source)
    _build(tmp_path, cache, kernel)
    before = _identity(_entry_object(cache, kernel))
    ExternalFunction._instances.clear()
    again = _kernel(source, **_OTHER_INPUTS[change])
    assert again.object_file_name == kernel.object_file_name
    with contextlib.chdir(source.parent):
        _build(tmp_path, cache, again)
    assert len(list(cache.root.iterdir())) == 1
    assert _identity(_entry_object(cache, kernel)) == before


def test_failed_compile_leaves_nothing_to_hit(tmp_path, source, cache):
    header = source.parent / "scale.h"
    header.write_text("#define SCALE 2 +\n")
    kernel = _kernel(source)
    with pytest.raises(Exception):
        _build(tmp_path, cache, kernel)
    (entry,) = cache.root.iterdir()
    assert not (entry / kernel.object_file_name).exists()
    assert not (entry / _manifest.MANIFEST_NAME).exists()

    header.write_text("#define SCALE 2\n")
    linked = _build(tmp_path, cache, kernel)
    assert _symbols(linked) == ["helper_fn", "scale"]


@pytest.mark.parametrize("lost", ["object", "manifest"])
def test_incomplete_entry_is_rebuilt(tmp_path, source, cache, lost):
    """An entry cut short (e.g. a killed process) is a miss, not a hit."""
    kernel = _kernel(source)
    _build(tmp_path, cache, kernel)
    built = _entry_object(cache, kernel)
    entry = built.parent
    good = built.read_bytes()
    if lost == "object":
        built.unlink()
    else:
        (entry / _manifest.MANIFEST_NAME).unlink()
    before = entry.stat().st_mtime_ns

    linked = _build(tmp_path, cache, kernel)
    assert (entry / _manifest.MANIFEST_NAME).is_file()
    assert _manifest.is_valid(entry)
    assert built.read_bytes() == good == linked.read_bytes()
    assert entry.stat().st_mtime_ns != before


_WORKER = textwrap.dedent("""
    import logging, os, sys, time
    from pathlib import Path
    import aie.utils.compile.utils as compile_utils
    from aie.iron.kernel import ExternalFunction
    from aie.utils.compile.jit._object_cache import KernelObjectCache

    source, root, design, ready, go = sys.argv[1:]
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout, format="%(message)s")
    logging.getLogger("aie.utils.compile.jit._object_cache").setLevel(logging.DEBUG)
    kernel = ExternalFunction("scale", source_file=source)
    Path(design).mkdir()
    Path(ready).touch()
    while not Path(go).exists():
        time.sleep(0.01)
    compile_utils.compile_external_kernels(
        [kernel], design, "aie2p", object_cache=KernelObjectCache(Path(root), 600)
    )
    """)


def test_concurrent_processes_compile_a_shared_kernel_once(tmp_path, source, cache):
    go = tmp_path / "go"
    workers = []
    for name in ("a", "b"):
        ready = tmp_path / f"{name}.ready"
        args = [source, cache.root, tmp_path / name, ready, go]
        proc = subprocess.Popen(
            [sys.executable, "-c", _WORKER, *map(str, args)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        workers.append((proc, ready))
    deadline = time.monotonic() + 120
    while not all(ready.exists() for _, ready in workers):
        assert time.monotonic() < deadline, "workers never became ready"
        assert all(proc.poll() is None for proc, _ in workers)
        time.sleep(0.01)
    go.touch()
    logs = []
    for proc, _ in workers:
        out, _ = proc.communicate(timeout=600)
        assert proc.returncode == 0, out
        logs.append(out)
    assert sum(log.count("cache miss") for log in logs) == 1
    assert sum(log.count("cache hit") for log in logs) == 1
    built = _entry_object(cache, ExternalFunction("scale", source_file=str(source)))
    for name in ("a", "b"):
        assert (tmp_path / name / "scale.o").read_bytes() == built.read_bytes()


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


def test_cached_kernels_sharing_a_name_compile_side_by_side(tmp_path, source, cache):
    """Only a kernel compiled in place stages its source in the design directory.

    So only those wait on another kernel of the same name. Kernels the cache
    builds are grouped by object alone, as a graph's per-op instances are.
    """
    kernels = [_kernel(source, symbol_prefix=prefix) for prefix in ("op0", "op1")]
    assert len(compile_utils._kernel_compile_groups(kernels)) == 1
    groups = compile_utils._kernel_compile_groups(
        kernels, in_place=lambda f: not cache.accepts(f)
    )
    assert groups == [[kernels[0]], [kernels[1]]]

    design = _design_dir(tmp_path, "design")
    compile_utils.compile_external_kernels(kernels, design, "aie2p", object_cache=cache)
    for prefix, kernel in zip(("op0", "op1"), kernels):
        assert _symbols(design / kernel.object_file_name) == [
            f"{prefix}_helper_fn",
            f"{prefix}_scale",
        ]
    assert not list(design.glob("*.cc"))


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
