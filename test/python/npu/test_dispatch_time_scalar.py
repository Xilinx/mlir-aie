# test_dispatch_time_scalar.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# RUN: %run_on_npu_hsa% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings || hsa_npu

# End-to-end DispatchTime[T]: one compiled design, called with different
# runtime scalar values. The rolled tile loop produces different instruction
# stream lengths, exercising grow/shrink as well as changing DMA offsets.

import aie.iron as iron
import numpy as np
import pytest
from aie.extras.dialects import arith
from aie.helpers.util import np_dtype_to_mlir_type
from aie.iron import (
    CompileTime,
    DispatchTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_

TILE_SIZE = 256
MAX_TILES = 8


@iron.jit
def dyn_copy(
    a: In,
    b: Out,
    *,
    n_tiles: DispatchTime[np.int32] = 3,
    start_tile: DispatchTime[np.int32] = 0,
    tile_size: CompileTime[int] = TILE_SIZE,
    max_tiles: CompileTime[int] = MAX_TILES,
):
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
    max_ty = np.ndarray[(max_tiles * tile_size,), np.dtype[np.int32]]

    of_in = ObjectFifo(tile_ty, name="of_in", depth=2)
    of_out = ObjectFifo(tile_ty, name="of_out", depth=2)

    def core_fn(in_cons, out_prod):
        elem_in = in_cons.acquire(1)
        elem_out = out_prod.acquire(1)
        for i in range_(tile_size):
            elem_out[i] = elem_in[i]
        in_cons.release(1)
        out_prod.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

    def seq(a_h, b_h, start, n, in_prod, out_cons):
        i32 = np_dtype_to_mlir_type(np.int32)
        i64 = np_dtype_to_mlir_type(np.int64)
        n64 = arith.extsi(i64, n)
        for tile in range_(n64):
            tile_i32 = arith.index_cast(tile, to=i32)
            offset = (start + tile_i32) * arith.constant(tile_size, i32)
            tg = TaskGroup()
            out_cons.drain(
                b_h,
                sizes=[1, 1, 1, tile_size],
                strides=[0, 0, tile_size, 1],
                offset=offset,
                transfer_len=tile_size,
                wait=True,
                group=tg,
            )
            in_prod.fill(
                a_h,
                sizes=[1, 1, 1, tile_size],
                strides=[0, 0, tile_size, 1],
                offset=offset,
                transfer_len=tile_size,
                group=tg,
            )
            tg.finish()

    # Reverse the same-typed dispatch parameters to exercise identity binding.
    rt = Runtime(
        seq, [max_ty, max_ty, start_tile, n_tiles, of_in.prod(), of_out.cons()]
    )
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _random_tiles(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**16, size=(MAX_TILES * TILE_SIZE,), dtype=np.int32)


def _assert_copied_region(a, b, count, start=0):
    begin, end = start * TILE_SIZE, (start + count) * TILE_SIZE
    expected = np.zeros((MAX_TILES * TILE_SIZE,), dtype=np.int32)
    expected[begin:end] = a.numpy()[begin:end]
    assert np.array_equal(b.numpy(), expected)


def test_dispatch_time_scalar_varies_without_recompile():
    """Shrink/grow calls reuse one kernel and leave each fresh output tail intact."""
    design = dyn_copy.specialize()
    first_kernel = None
    first_artifacts = None
    instruction_sizes = {}
    for seed, count in enumerate((1, 6, 2, MAX_TILES, 3, 7, 1)):
        a = iron.tensor(_random_tiles(seed=seed), dtype=np.int32, device="npu")
        b = iron.zeros((MAX_TILES * TILE_SIZE,), dtype=np.int32, device="npu")
        design(a, b, n_tiles=count)
        _assert_copied_region(a, b, count)
        assert len(design._kernel_cache) == 1
        kernel = next(iter(design._kernel_cache.values()))
        artifacts = (kernel.xclbin_path, kernel.dispatch_lib_path)
        if first_kernel is None:
            first_kernel, first_artifacts = kernel, artifacts
        assert kernel is first_kernel
        assert artifacts == first_artifacts
        instructions = kernel._generate_dispatch_insts(
            {"n_tiles": count, "start_tile": 0}
        )
        instruction_sizes[count] = instructions.nbytes
    assert instruction_sizes[1] < instruction_sizes[3] < instruction_sizes[MAX_TILES]


def test_dispatch_time_scalar_repeated_same_value():
    """Reusing a cached kernel must also leave tiles beyond n_tiles untouched."""
    a = iron.tensor(_random_tiles(seed=4), dtype=np.int32, device="npu")
    for _ in range(5):
        b = iron.zeros((MAX_TILES * TILE_SIZE,), dtype=np.int32, device="npu")
        dyn_copy(a, b, n_tiles=3)
        _assert_copied_region(a, b, 3)


@pytest.mark.parametrize("undersized", ["a", "b"])
def test_dynamic_copy_rejects_undersized_buffer_before_dispatch(undersized):
    design = dyn_copy.specialize()
    expected_size = MAX_TILES * TILE_SIZE
    itemsize = np.dtype(np.int32).itemsize
    a_values = _random_tiles(seed=8)
    if undersized == "a":
        a_values = a_values[:-1]
    a = iron.tensor(a_values, dtype=np.int32, device="npu")
    b = iron.zeros((expected_size - (undersized == "b"),), dtype=np.int32, device="npu")
    with pytest.raises(
        RuntimeError,
        match=(
            f"Tensor argument '{undersized}' covers "
            f"{(expected_size - 1) * itemsize} bytes but the kernel was "
            f"compiled for {expected_size * itemsize}"
        ),
    ):
        design(a, b, n_tiles=1)
    assert np.all(b.numpy() == 0)


@pytest.mark.parametrize("static", [False, True])
def test_same_generator_dynamic_default_and_static_specialization(static):
    a = iron.tensor(_random_tiles(seed=3), dtype=np.int32, device="npu")
    b = iron.zeros((MAX_TILES * TILE_SIZE,), dtype=np.int32, device="npu")
    design = (
        dyn_copy.specialize(n_tiles=3, start_tile=0)
        if static
        else dyn_copy.specialize()
    )
    design(a, b)
    _assert_copied_region(a, b, 3)
    if static:
        assert design.compilable.dispatch_params == []
        assert design.compilable.get_dispatch_lib_path() is None
        with pytest.raises(TypeError, match="specialized"):
            design(a, b, n_tiles=6)
    else:
        assert design.compilable.dispatch_params == ["n_tiles", "start_tile"]
        b = iron.zeros((MAX_TILES * TILE_SIZE,), dtype=np.int32, device="npu")
        design(a, b, n_tiles=6)
        _assert_copied_region(a, b, 6)
        assert len(design._kernel_cache) == 1


@pytest.mark.parametrize("bound_name", ["n_tiles", "start_tile"])
def test_mixed_dispatch_specialization(bound_name):
    """Either dispatch scalar can become static while the other stays dynamic."""
    bound_value = 3 if bound_name == "n_tiles" else 1
    design = dyn_copy.specialize(**{bound_name: bound_value})
    active = "start_tile" if bound_name == "n_tiles" else "n_tiles"
    assert design.compilable.dispatch_params == [active]
    values = (0, 4, 1) if active == "start_tile" else (1, 6, 2)
    a = iron.tensor(_random_tiles(seed=6), dtype=np.int32, device="npu")
    for value in values:
        b = iron.zeros((MAX_TILES * TILE_SIZE,), dtype=np.int32, device="npu")
        design(a, b, **{active: value})
        count = bound_value if bound_name == "n_tiles" else value
        start = bound_value if bound_name == "start_tile" else value
        _assert_copied_region(a, b, count, start)
        assert len(design._kernel_cache) == 1


def test_dispatch_explicit_paths_compile_and_execute(tmp_path):
    """Real compilation publishes xclbin/PDI and an immutable dispatch library."""
    from aie.utils.npukernel import NPUKernel

    design = dyn_copy.specialize().compilable
    xclbin = tmp_path / "copy.xclbin"
    pdi = tmp_path / "copy.pdi"
    assert design.compile(xclbin_path=xclbin, pdi_path=pdi) == (xclbin, None)
    library = design.get_dispatch_lib_path()
    assert xclbin.is_file() and pdi.is_file()
    assert library is not None and library.is_file()
    assert library.parent == tmp_path / "copy.prj"
    original_library = library.read_bytes()
    kernel = NPUKernel(
        xclbin,
        None,
        kernel_name="MLIR_AIE",
        dispatch_params=design.dispatch_params,
        dispatch_lib_path=library,
    )
    a = iron.tensor(_random_tiles(seed=5), dtype=np.int32, device="npu")
    for count in (3, 6, 1, 7, 2):
        b = iron.zeros((MAX_TILES * TILE_SIZE,), dtype=np.int32, device="npu")
        kernel(a, b, n_tiles=count, start_tile=0)
        _assert_copied_region(a, b, count)
        assert design.get_dispatch_lib_path() == library
        assert library.read_bytes() == original_library
