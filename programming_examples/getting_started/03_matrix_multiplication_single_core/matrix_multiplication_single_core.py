# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Single-core matmul, optionally AOT-compiled at known shapes.

The design is decorated with the usual ``@iron.jit`` so callers can use
it the JIT way (compile-on-first-call).  ``main()`` demonstrates the
opt-in AOT path: each desired shape is pre-compiled by building a
``CompilableDesign`` from the JIT generator and calling ``.compile()``
eagerly.  This produces distinct xclbin/insts artifacts on disk before
any kernel runs, so subsequent ``matrix_multiplication_single_core(...)``
calls with matching compile-time kwargs hit the on-disk cache instead
of paying ``aiecc`` time on first call.

Use this pattern when you know your problem sizes in advance and want
to ship pre-compiled binaries — without having to change how the design
is decorated.
"""

import numpy as np

import aie.iron as iron
from aie.iron import (
    Compile,
    In,
    Out,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.utils.verify import assert_pass

# Tile size moved to/from the compute cores via mem tiles.
_TILE_M = _TILE_K = _TILE_N = 64
# AIE kernel intrinsic (r, s, t) used to drive the DMA layout transforms
# below (r*s / s*t / r*t sub-tiles) is read from `matmul_kernel.mac_dims`
# inside the design — see aie_kernels/aie2{,p}/mm.cc for the per-arch
# values (e.g. int16/int16 is 4x4x4 on AIE2 but 4x4x8 on AIE2P, so a
# hardcoded `_R, _S, _T` would break on whichever arch you didn't pin to).


@iron.jit
def matrix_multiplication_single_core(
    input0: In,
    input1: In,
    output: Out,
    *,
    M: Compile[int],
    K: Compile[int],
    N: Compile[int],
    element_type: Compile[type],
):
    m, k, n = _TILE_M, _TILE_K, _TILE_N

    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=element_type,
        output_dtype=element_type,
        vectorized=True,
    )
    # Pull the MMUL geometry from the kernel itself so the DMA layout
    # transforms below match whatever the freshly-compiled binary expects
    # (4x4x4 on AIE2, 4x4x8 on AIE2P for i16/i16 — see _MM_MAC_DIMS in
    # python/iron/kernels/linalg.py).
    r, s, t = matmul_kernel.mac_dims

    A_ty = np.ndarray[(M, K), np.dtype[element_type]]
    B_ty = np.ndarray[(K, N), np.dtype[element_type]]
    C_ty = np.ndarray[(M, N), np.dtype[element_type]]
    # Tile types are flat to match the kernels.mm() ExternalFunction signature.
    a_ty = np.ndarray[(m * k,), np.dtype[element_type]]
    b_ty = np.ndarray[(k * n,), np.dtype[element_type]]
    c_ty = np.ndarray[(m * n,), np.dtype[element_type]]

    # The DMA-level layout transformations rearrange m*k / k*n / m*n tiles
    # into r*s / s*t / r*t sub-tiles for the MMUL intrinsic.  See
    # programming_guide/section-2/section-2c/ for n-D layout transformations.
    fifo_A_L3L2 = ObjectFifo(a_ty, name="A_L3L2")
    tap_A_L2L1 = TensorTiler2D.group_tiler((m, k), (r, s), (m // r, k // s))[0]
    fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
        dims_to_stream=tap_A_L2L1.transformation_dims, name="A_L2L1"
    )

    fifo_B_L3L2 = ObjectFifo(b_ty, name="B_L3L2")
    tap_B_L2L1 = TensorTiler2D.group_tiler((k, n), (s, t), (k // s, n // t))[0]
    fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
        dims_to_stream=tap_B_L2L1.transformation_dims, name="B_L2L1"
    )

    fifo_C_L1L2 = ObjectFifo(c_ty, name="C_L1L2")
    # Inverse tiling that unpacks C from the kernel's r*t sub-tile layout
    # back to row-major.
    tap_C_L1L2 = TensorAccessPattern(
        tensor_dims=(m, n),
        offset=0,
        sizes=[m // r, r, n // t, t],
        strides=[r * n, t, r * t, 1],
    )
    fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
        dims_to_stream=tap_C_L1L2.transformation_dims, name="C_L2L3"
    )

    def core_fn(of_a, of_b, of_c, matmul):
        for _ in range_(M // m * N // n):
            elem_out = of_c.acquire(1)
            for i in range_(m * n):
                elem_out[i] = 0
            for _ in range_(K // k):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn,
        [fifo_A_L2L1.cons(), fifo_B_L2L1.cons(), fifo_C_L1L2.prod(), matmul_kernel],
    )

    # Each task group encompasses all data movement for one row of output
    # tiles. See programming_guide/section-2/section-2f/ for multi-level
    # (L3→L2→L1) data-movement patterns.
    a_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K // k), pattern_repeat=(N // n)
    )
    b_tap = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K // k, N // n), tile_group_col_major=True
    )[0]
    c_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, N // n))

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)
        for tile_row in range(M // m):
            task_group = rt.task_group()
            rt.fill(fifo_A_L3L2.prod(), A, tap=a_taps[tile_row], task_group=task_group)
            rt.fill(fifo_B_L3L2.prod(), B, tap=b_tap, task_group=task_group)
            rt.drain(
                fifo_C_L2L3.cons(),
                C,
                tap=c_taps[tile_row],
                task_group=task_group,
                wait=True,
            )
            rt.finish_task_group(task_group)

    return Program(iron.get_current_device(), rt).resolve_program()


def aot_compile(M: int, K: int, N: int, element_type) -> None:
    """Pre-compile the JIT design for one shape and print the artifact paths."""
    xclbin, insts = matrix_multiplication_single_core.specialize(
        M=M, K=K, N=N, element_type=element_type
    ).compile()
    print(f"  AOT compiled matmul {M}x{K}x{N} {np.dtype(element_type).name}")
    print(f"    xclbin: {xclbin}")
    print(f"    insts:  {insts}")


def run_and_verify(M: int, K: int, N: int, element_type) -> None:
    print(f"\n=== Running matmul {M}x{K}x{N} {np.dtype(element_type).name} ===")
    input0 = iron.randint(0, 256, (M, K), dtype=element_type, device="npu")
    input1 = iron.randint(0, 256, (K, N), dtype=element_type, device="npu")
    output = iron.zeros(M * N, dtype=element_type, device="npu")
    ref = np.matmul(input0.numpy(), input1.numpy())

    # Same JIT-decorated function — because the AOT step above already
    # populated the on-disk cache for these compile_kwargs, this call hits
    # the cache instead of paying aiecc time.
    matrix_multiplication_single_core(
        input0, input1, output, M=M, K=K, N=N, element_type=element_type
    )

    assert_pass(
        output.numpy(),
        ref.flatten(),
        fail_msg=f"matmul {M}x{K}x{N} output does not match np.matmul",
        print_pass=False,
    )
    print("  PASS")


def main():
    shapes = [
        dict(M=256, K=256, N=256, element_type=np.int16),
        dict(M=512, K=512, N=512, element_type=np.int16),
    ]

    # AOT phase: pre-compile both shapes upfront so the artifacts exist on
    # disk before any kernel runs.  This is purely opt-in — the @iron.jit
    # decorator above means callers who skip this step would simply pay the
    # compile cost on first invocation instead.
    print("=== AOT pre-compile phase ===")
    for kw in shapes:
        aot_compile(**kw)

    # Runtime dispatch.  Each call uses the same JIT-decorated function and
    # finds its compiled artifact already in the on-disk cache.
    for kw in shapes:
        run_and_verify(**kw)

    print("\nAll variants PASS!")


if __name__ == "__main__":
    main()
