# test_device_binding_cache_unit.py -*- Python -*-
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Target-device-sensitive generation and cache identity, without an NPU.

Nothing here may start the NPU runtime; test/python/npu/test_device_binding.py
covers binding the device the runtime reports.
"""

import shutil

import numpy as np
import pytest

import aie.utils as utils
from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import NPU1Col1, NPU2Col1, NPU2Col2
from aie.utils import get_current_device, set_current_device
from aie.utils.compile.jit import CompileTime, In, Out
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


def _gemm_gen():
    def gemm(
        a: In,
        b: In,
        c: Out,
        *,
        M: CompileTime[int],
        K: CompileTime[int],
        N: CompileTime[int],
    ):
        pass

    return gemm


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


def test_generated_cache_tracks_active_device():
    """MLIR generation cache entries are keyed by the active IRON device."""
    generated_for = []

    def gen():
        generated_for.append(type(get_current_device(probe_runtime=False)).__name__)

    cd = CompilableDesign(gen)

    set_current_device(NPU1Col1())
    first_aie2 = cd._generated
    second_aie2 = cd._generated

    set_current_device(NPU2Col1())
    first_aie2p = cd._generated
    second_aie2p = cd._generated

    # Real generation runs once per distinct device, then serves from cache.
    assert generated_for == ["NPU1Col1", "NPU2Col1"]
    assert first_aie2 is second_aie2
    assert first_aie2p is second_aie2p
    # One cache entry per device, keyed by device identity.
    assert len(cd._generated_cache) == 2
    keyed_devices = {key[2].rsplit(".", 1)[-1] for key in cd._generated_cache}
    assert keyed_devices == {"NPU1Col1", "NPU2Col1"}


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


def test_compute_hash_changes_when_active_device_width_changes():
    """The filesystem cache key includes the active device identity, not just arch."""
    cd = CompilableDesign(_gemm_gen(), compile_kwargs={"M": 64, "K": 64, "N": 64})

    set_current_device(NPU2Col1())
    h_one_col = cd._compute_cache_hash()

    set_current_device(NPU2Col2())
    h_two_col = cd._compute_cache_hash()

    assert h_one_col != h_two_col


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


def test_cache_hash_does_not_bind_a_device():
    """Inspecting an unbound design's cache key leaves the device unbound."""
    cd = CompilableDesign(_gemm_gen(), compile_kwargs={"M": 64, "K": 64, "N": 64})
    assert cd._compute_cache_hash()
    assert get_current_device(probe_runtime=False) is None


def test_iron_reexports_set_current_device():
    """The documented IRON device-selection entry point is public."""
    import aie.iron as iron

    assert iron.set_current_device is set_current_device


def test_cleanup_npu_runtime_does_not_initialize_default_runtime():
    """Runtime cleanup is a no-op until a default runtime already exists."""
    utils.cleanup_npu_runtime()
