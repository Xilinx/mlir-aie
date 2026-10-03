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
from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.utils import get_current_device, set_current_device
from aie.utils.compile.jit.compilabledesign import CompilableDesign

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
        sizes.setdefault(type(device), cd._expected_tensor_sizes)
        assert cd._expected_tensor_sizes == sizes[type(device)]
    assert sizes == {NPU1Col1: [16 * 32] * 2, NPU2Col1: [32 * 32] * 2}


@needs_xclbinutil
def test_static_mlir_compile_does_not_bind_a_device(tmp_path):
    """Compiling a written MLIR file takes its target from the file, not the runtime."""
    set_current_device(NPU2Col1())
    mlir_path = tmp_path / "design.mlir"
    mlir_path.write_text(CompilableDesign(copy)._generated[0])
    set_current_device(None)

    cd = CompilableDesign(mlir_path, use_cache=False)
    xclbin_path = tmp_path / "out.xclbin"
    inst_path = tmp_path / "out.insts"
    assert cd.compile(xclbin_path=xclbin_path, inst_path=inst_path) == (
        xclbin_path.resolve(),
        inst_path.resolve(),
    )
    assert get_current_device(probe_runtime=False) is None
