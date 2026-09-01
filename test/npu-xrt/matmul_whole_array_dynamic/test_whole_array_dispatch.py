# test_whole_array_dispatch.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# REQUIRES: xrt_python_bindings, peano

# Whole-array dynamic i16 GEMM on real hardware, swept over n_aie_cols 1, 2, 4.
# The runtime sequence keeps its scf.for rolled over a RUNTIME trip count
# derived from M/K/N, which the dynamic BD free-list pool path supports (the
# static allocator would reject it). One compiled artifact per column count
# serves every shape below: `whole_array_dispatch` declares M/K/N as
# DispatchTime[np.int32], so each call rebuilds the instruction stream through
# the dispatch bridge instead of recompiling.

import aie.iron as iron
import numpy as np
import pytest
from whole_array_dynamic import whole_array_dispatch

# Compiled maxima. These size the host buffers; the per-call M/K/N ride along as
# runtime scalars and only have to fit. K is fixed at the compiled value.
M_MAX, K_FIX, N_MAX = 768, 512, 1024
m, k, n = 64, 64, 32
N_AIE_ROWS = 4

# M must be a multiple of m * N_AIE_ROWS and N a multiple of n * n_aie_cols, so
# these hold for every column count in the sweep.
SHAPES = [(512, 512, 512), (256, 512, 1024), (768, 512, 768)]


def _dispatch(A, B, C, M_rt, K_rt, N_rt, n_aie_cols):
    whole_array_dispatch(
        A,
        B,
        C,
        M_rt,
        K_rt,
        N_rt,
        M=M_MAX,
        K=K_FIX,
        N=N_MAX,
        m=m,
        k=k,
        n=n,
        n_aie_cols=n_aie_cols,
        dtype_in_str="i16",
        dtype_out_str="i16",
    )


def _run_shape(M_rt, K_rt, N_rt, n_aie_cols):
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
    _dispatch(A, B, C, M_rt, K_rt, N_rt, n_aie_cols)

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
    after_first = None
    for M_rt, K_rt, N_rt in SHAPES:
        actual, expected = _run_shape(M_rt, K_rt, N_rt, n_aie_cols)
        assert np.array_equal(actual, expected), (
            f"n_aie_cols={n_aie_cols} M={M_rt} K={K_rt} N={N_rt}: "
            f"{np.count_nonzero(actual != expected)} mismatching element(s); "
            f"first at {np.argwhere(actual != expected)[0].tolist()}"
        )
        # The cache is shared across the whole parametrized sweep (one
        # CallableDesign, one entry per column count), so compare against the
        # count after this column count's first shape rather than against 1.
        if after_first is None:
            after_first = len(whole_array_dispatch._kernel_cache)

    assert len(whole_array_dispatch._kernel_cache) == after_first, (
        f"n_aie_cols={n_aie_cols}: the {len(SHAPES)} shapes added "
        f"{len(whole_array_dispatch._kernel_cache) - after_first} kernel(s) -- a "
        "DispatchTime[T] value must not enter the kernel cache key."
    )
