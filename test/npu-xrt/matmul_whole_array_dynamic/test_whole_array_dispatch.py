# test_whole_array_dispatch.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Whole-array dynamic i16 GEMM on real hardware, swept over n_aie_cols 1, 2, 4.
# The runtime sequence keeps its scf.for rolled over a runtime trip count, which
# only the dynamic BD free-list pool path supports. M/N remain DispatchTime, and
# K is explicitly specialized to the fixed core reduction depth, so one
# compiled artifact per column count serves every shape below.

import aie.iron as iron
import numpy as np
import pytest
from whole_array_dynamic import whole_array_dynamic

# Compiled maxima. These size the host buffers; the per-call M/K/N ride along as
# runtime scalars and only have to fit. K is fixed at the compiled value.
M_MAX, K_FIX, N_MAX = 768, 512, 1024
m, k, n = 64, 64, 32
N_AIE_ROWS = 4

# M must be a multiple of m * N_AIE_ROWS and N a multiple of n * n_aie_cols, so
# these hold for every column count in the sweep.
SHAPES = [(512, 512, 512), (256, 512, 1024), (768, 512, 768)]


def _design(n_aie_cols):
    return whole_array_dynamic.specialize(
        A_elements=M_MAX * K_FIX,
        B_elements=K_FIX * N_MAX,
        C_elements=M_MAX * N_MAX,
        K=K_FIX,
        m=m,
        k=k,
        n=n,
        n_aie_cols=n_aie_cols,
    )


def _run_shape(design, M_rt, K_rt, N_rt, *, static=False):
    """Dispatch one shape and return (actual, expected) as M_rt x N_rt."""
    # The design reads and writes packed at the RUNTIME dims, so the live data
    # is a dense prefix of the max-sized host buffer, not a sub-block of an
    # M_MAX x K_FIX rectangle.
    rng = np.random.default_rng(1726250518)
    a = np.zeros((M_MAX * K_FIX,), dtype=np.int16)
    b = np.zeros((K_FIX * N_MAX,), dtype=np.int16)
    a[: M_rt * K_rt] = rng.integers(-8, 9, size=M_rt * K_rt, dtype=np.int16)
    b[: K_rt * N_rt] = rng.integers(-8, 9, size=K_rt * N_rt, dtype=np.int16)

    A = iron.tensor(a, dtype=np.int16, device="npu")
    B = iron.tensor(b, dtype=np.int16, device="npu")
    C = iron.zeros((M_MAX * N_MAX,), dtype=np.int16, device="npu")
    if static:
        design.specialize(M=M_rt, N=N_rt)(A, B, C)
    else:
        design(A, B, C, M_rt, N_rt)

    # int32 accumulate then narrow, matching the kernel's i16 output.
    expected = (
        a[: M_rt * K_rt].reshape(M_rt, K_rt).astype(np.int32)
        @ b[: K_rt * N_rt].reshape(K_rt, N_rt).astype(np.int32)
    ).astype(np.int16)
    # .copy(): C.numpy() views the device buffer, which is released when C goes
    # out of scope here -- comparing the view in the caller reads freed memory.
    return C.numpy()[: M_rt * N_rt].reshape(M_rt, N_rt).copy(), expected


@pytest.mark.parametrize("n_aie_cols", [1, 2, 4])
def test_shapes_share_one_compiled_artifact(n_aie_cols):
    """Every shape must be correct, and all of them from a single compile.

    Recompiling per shape would still produce correct results, so correctness
    alone does not show the runtime scalars reached the BDs -- the kernel-cache
    count is what distinguishes DispatchTime[T] from CompileTime[T] here.
    """
    design = _design(n_aie_cols)
    after_first = None
    artifacts = None
    for M_rt, K_rt, N_rt in SHAPES:
        actual, expected = _run_shape(design, M_rt, K_rt, N_rt)
        assert np.array_equal(actual, expected), (
            f"n_aie_cols={n_aie_cols} M={M_rt} K={K_rt} N={N_rt}: "
            f"{np.count_nonzero(actual != expected)} mismatching element(s); "
            f"first at {np.argwhere(actual != expected)[0].tolist()}"
        )
        # Neither the in-process kernel nor the on-disk artifacts may change.
        if after_first is None:
            after_first = len(design._kernel_cache)
            kernel = next(iter(design._kernel_cache.values()))
            artifacts = (kernel.xclbin_path, kernel.dispatch_lib_path)
        kernel = next(iter(design._kernel_cache.values()))
        assert (kernel.xclbin_path, kernel.dispatch_lib_path) == artifacts

    assert len(design._kernel_cache) == after_first == 1


@pytest.mark.parametrize("n_aie_cols", [1, 2, 4])
@pytest.mark.parametrize("shape", SHAPES)
def test_same_generator_static_specialization(n_aie_cols, shape):
    actual, expected = _run_shape(_design(n_aie_cols), *shape, static=True)
    assert np.array_equal(actual, expected)
