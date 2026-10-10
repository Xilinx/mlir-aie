# test_device_binding_cache_compile.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# REQUIRES: peano
# RUN: %pytest %s
"""Target-device-sensitive compiles and their cache entries, without an NPU.

Nothing here may start the NPU runtime.
"""

import shutil

import numpy as np
import pytest

import aie.utils as utils
import aie.utils.compile.jit.compilabledesign as compilabledesign_module
from aie.dialects.aiex import npu_address_patch
from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.utils import get_current_device, set_current_device
from aie.utils.compile.jit import _manifest
from aie.utils.compile.jit.compilabledesign import CompilableDesign
from aie.utils.compile.jit.markers import DispatchTime

needs_xclbinutil = pytest.mark.skipif(
    shutil.which("xclbinutil") is None, reason="xclbinutil"
)


@pytest.fixture(autouse=True)
def offline():
    """Run each test with no device bound, and check it never started the runtime."""
    previous = get_current_device(probe_runtime=False)
    set_current_device(None)
    assert utils._DefaultNPURuntime is None
    try:
        yield
    finally:
        set_current_device(previous)
    assert utils._DefaultNPURuntime is None


def copy():
    """A copy through a memtile, 16 words long on NPU1 and 32 elsewhere."""
    device = get_current_device(probe_runtime=False)
    n = 16 if isinstance(device, NPU1Col1) else 32
    ty = np.ndarray[(n,), np.dtype[np.int32]]
    of_in = ObjectFifo(ty)
    of_out = of_in.cons().forward()

    def sequence(a, b, a_in, b_out):
        a_in.fill(a)
        b_out.drain(b, wait=True)

    rt = Runtime(sequence, [ty, ty, of_in.prod(), of_out.cons()])
    return Program(device, rt).resolve_program()


@needs_xclbinutil
def test_cache_hit_refreshes_tensor_metadata_for_the_selected_artifact():
    """Returning to a cached target restores that artifact's validation metadata."""
    cd = CompilableDesign(copy)
    sizes = {}
    for device in (NPU1Col1(), NPU2Col1(), NPU1Col1()):
        set_current_device(device)
        cd.compile()
        sizes.setdefault(type(device), cd.expected_tensor_sizes)
        assert cd.expected_tensor_sizes == sizes[type(device)]
    assert sizes == {NPU1Col1: [16 * 32] * 2, NPU2Col1: [32 * 32] * 2}


@needs_xclbinutil
def test_tensor_sizes_are_read_when_asked():
    """A compile leaves the lowered module unparsed until a caller asks."""
    set_current_device(NPU2Col1())
    cd = CompilableDesign(copy)
    cd.compile()
    assert "expected_tensor_sizes" not in vars(cd)
    assert cd.expected_tensor_sizes == [32 * 32] * 2


@needs_xclbinutil
def test_static_mlir_compile_does_not_bind_a_device(tmp_path):
    """Compiling a written MLIR file takes its target from the file, not the runtime."""
    set_current_device(NPU2Col1())
    mlir_path = tmp_path / "design.mlir"
    mlir_path.write_text(CompilableDesign(copy)._generated.mlir_text)
    set_current_device(None)

    cd = CompilableDesign(mlir_path, use_cache=False)
    xclbin_path = tmp_path / "out.xclbin"
    inst_path = tmp_path / "out.insts"
    assert cd.compile(xclbin_path=xclbin_path, inst_path=inst_path) == (
        xclbin_path.resolve(),
        inst_path.resolve(),
    )
    assert get_current_device(probe_runtime=False) is None


@needs_xclbinutil
def test_compile_mode_switch_replaces_artifact_state(tmp_path, monkeypatch):
    monkeypatch.setattr(compilabledesign_module, "NPU_CACHE_HOME", tmp_path / "cache")
    set_current_device(NPU2Col1())
    mlir_path = tmp_path / "design.mlir"
    mlir_path.write_text(CompilableDesign(copy)._generated.mlir_text)
    out = tmp_path / "out"

    design = CompilableDesign(mlir_path)
    design.compile(xclbin_path=out / "design.xclbin", inst_path=out / "insts.bin")
    assert design.get_artifacts() is not None

    design.compile(full_elf_path=out / "design.elf")
    entry = design.get_cache_entry()
    assert entry.elf == (out / "design.elf").resolve()
    assert entry.xclbin is None and entry.insts is None
    assert design.get_artifacts() is None

    design.compile(xclbin_path=out / "design.xclbin", inst_path=out / "insts.bin")
    entry = design.get_cache_entry()
    assert entry.xclbin == (out / "design.xclbin").resolve()
    assert entry.insts == (out / "insts.bin").resolve()
    assert entry.elf is None
    assert design._full_elf_kernel_name is None


def patch_bar_then_baz(
    *, bar: DispatchTime[np.int32] = 3, baz: DispatchTime[np.int32] = 7
):
    def sequence(baz_value, bar_value):
        npu_address_patch(addr=119300, arg_idx=0, arg_plus=bar_value)
        npu_address_patch(addr=119304, arg_idx=0, arg_plus=baz_value)

    return Program(NPU2Col1(), Runtime(sequence, [baz, bar])).resolve_program()


@needs_xclbinutil
@pytest.mark.parametrize("cache_hit", [False, True])
def test_dispatch_library_selected_once_per_compile(
    tmp_path, monkeypatch, npu2_device, cache_hit
):
    monkeypatch.setattr(compilabledesign_module, "NPU_CACHE_HOME", tmp_path)
    if cache_hit:
        CompilableDesign(patch_bar_then_baz).compile()
    design = CompilableDesign(patch_bar_then_baz)
    design.compile()
    first = design.get_dispatch_lib_path()
    xclbin = design.get_cache_entry().xclbin
    built = xclbin.stat()

    second = first.with_name(f"dispatch-{'b' * 64}{first.suffix}")
    shutil.copy(first, second)
    _manifest._write(first.parent, {}, dispatch_library=second.name)
    # A later publication must not silently change this design's selected ABI.
    assert design.get_dispatch_lib_path() == first
    design.compile()
    assert design.get_dispatch_lib_path() == second
    after = xclbin.stat()
    assert (built.st_ino, built.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)
