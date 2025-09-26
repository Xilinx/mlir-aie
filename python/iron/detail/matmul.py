# detail/matmul.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import numpy as np
from typing import Optional
from ..tensor import Tensor
from ..jit import Promise

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    str_to_dtype,
    dtype_to_str,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D
from aie.iron import ExternalFunction
import aie.iron as iron
import aie.utils as utils
import os
from ml_dtypes import bfloat16
import sys

microkernel_mac_dim_map = {
    "npu": {
        "bf16": (4, 8, 4),
        "i8": (4, 8, 8),
        "i16": (4, 4, 4),
    },
    "npu2": {
        "bf16": {
            # emulate_bf16_mmul_with_bfp16
            True: (8, 8, 8),
            False: (4, 8, 8),
        },
        "i8": (8, 8, 8),
        "i16": (4, 4, 8),
    },
}


# Need ceildiv to capture partial tiling patterns
def ceildiv(a, b):
    return (a + b - 1) // b


@iron.jit(is_placed=False)
def matmul(A, B, C):

    m = 64
    k = 64
    n = 32

    b_col_maj = False
    emulate_bf16_mmul_with_bfp16 = False
    trace_size = 0
    generate_taps = False

    dtype_in = A.dtype
    dtype_out = C.dtype
    dtype_in_str = dtype_to_str(dtype_in)
    dtype_out_str = dtype_to_str(dtype_out)

    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]
    assert M % m == 0, f"M % m != 0: {M} % {m} = {M % m}"
    assert K % k == 0, f"K % k != 0: {K} % {k} = {K % k}"
    assert N % n == 0, f"N % n != 0: {N} % {n} = {N % n}"

    dev_ty = iron.get_current_device()
    device_str = "npu" if isinstance(dev_ty, NPU1) else "npu2"

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    mac_dims = microkernel_mac_dim_map[device_str][dtype_in_str]
    if device_str == "npu2" and dtype_in_str == "bf16":
        r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]
    else:
        r, s, t = mac_dims

    assert m % r == 0, f"m % r != 0: {m} % {r} = {m % r}"
    assert k % s == 0, f"k % s != 0: {k} % {s} = {k % s}"
    assert n % t == 0, f"n % t != 0: {n} % {t} = {n % t}"

    vectorized = True
    enable_tracing = True if trace_size > 0 else False

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    # These will hold TensorAccessPattern objects that represent the runtime
    # npu_dma_memcpy_nd operations of this design. They are only used if generate_taps is true
    A_taps = []
    B_taps = []
    C_taps = []

    # Define tensor types
    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    a_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    b_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    c_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    # AIE Core Function declarations
    vectorized = True

    func_type = "" if vectorized else "scalar_"

    root_path = utils.config.root_path()
    kernels_path = os.path.join(
        root_path,
        "..",
        "aie_kernels",
        "aie2" if device_str == "npu" else "aie2p",
    )
    kernel_path = os.path.join(kernels_path, "mm.cc")
    include_dirs = [
        os.path.join(root_path, "..", "include"),
        os.path.join(root_path, "..", "aie_kernels"),
        kernels_path,
    ]

    compile_flags = [
        f"-D{dtype_in_str}___{dtype_out_str}_ONLY",
        f"-DDIM_M={m}",
        f"-DDIM_K={k}",
        f"-DDIM_N={n}",
    ]
    zero_kernel = ExternalFunction(
        f"zero_{func_type}{dtype_out_str}",
        source_file=kernel_path,
        arg_types=[c_ty],
        compile_flags=compile_flags,
        include_dirs=include_dirs,
    )
    matmul_vectorized_func_name = f"matmul_{dtype_in_str}_{dtype_out_str}"
    matmul_kernel = ExternalFunction(
        matmul_vectorized_func_name,
        source_file=kernel_path,
        arg_types=[a_ty, b_ty, c_ty],
        compile_flags=compile_flags,
        include_dirs=include_dirs,
    )

    # AIE-array data movement with object fifos
    # Input A
    inA = ObjectFifo(a_ty, name="inA")
    a_dims = None
    if vectorized:
        a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    # Input B
    inB = ObjectFifo(b_ty, name="inB")
    b_dims = None
    if vectorized:
        if b_col_maj:
            b_dims = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
        else:
            b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    # Output C
    memC = ObjectFifo(c_ty, name="memC")
    c_dims = None
    if vectorized:
        c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    # Task each core will run
    def core_fn(of_a, of_b, of_c, zero, matmul):
        for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547
            elem_out = of_c.acquire(1)
            zero(elem_out)

            # issue #1547
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    # Create worker from task
    worker = Worker(
        core_fn,
        [memA.cons(), memB.cons(), memC.prod(), zero_kernel, matmul_kernel],
        stack_size=0xD00,
    )

    # only do 4 tile rows at a time before synchronizing, so we can reuse BDs
    rows_per_block = 4

    # Define tensor access patterns for inputs/outputs
    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K_div_k), pattern_repeat=N_div_n
    )
    # There is only one access pattern for B - it tiles the entire matrix in (k x n) tiles.
    if b_col_maj:
        b_tap = TensorTiler2D.group_tiler((N, K), (n, k), (N_div_n, K_div_k))[0]
    else:
        b_tap = TensorTiler2D.group_tiler(
            (K, N), (k, n), (K_div_k, N_div_n), tile_group_col_major=True
        )[0]

    C_tiles = TensorTiler2D.group_tiler((M, N), (m, n), (rows_per_block // 2, N_div_n))
    c_index = 0

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.enable_trace(trace_size, workers=[worker])
        rt.start(worker)

        tgs = []
        for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
            # we only sync on half the BDs before reusing them, so the other half can concurrently keep running
            # that's what this loop is for. We can track of this in the task groups for syncing.
            for pingpong in [0, 1]:

                row_base = (
                    tile_row_block * rows_per_block + pingpong * rows_per_block // 2
                )
                num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                if num_tile_rows <= 0:
                    # At the very last iteration, we may not need a 'pong' iteration
                    break
                tgs.append(rt.task_group())
                for tile_row in range(num_tile_rows):
                    # -- A --
                    tile_offset = (row_base + tile_row) % len(A_tiles)
                    rt.fill(inA.prod(), A, tap=A_tiles[tile_offset], task_group=tgs[-1])
                    A_taps.append(A_tiles[tile_offset])

                    # -- B --
                    rt.fill(inB.prod(), B, tap=b_tap, task_group=tgs[-1])
                    B_taps.append(b_tap)

                # -- C --
                rt.drain(
                    outC.cons(), C, tap=C_tiles[c_index], task_group=tgs[-1], wait=True
                )
                C_taps.append(C_tiles[c_index])
                c_index += 1

                if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                    rt.finish_task_group(tgs[-2])
                    del tgs[-2]

        rt.finish_task_group(tgs[-1])
        del tgs[-1]

    if generate_taps:
        # If generate taps is true, return a representation of tensor access patterns
        # representing all the npu_dma_memcpy_nd runtime sequence operations per input/ouput tensor.
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )

    # Create the program from the device type and runtime
    my_program = Program(dev_ty, rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program(SequentialPlacer())
    return module


def matmul_impl(input: Tensor, other: Tensor, out: Optional[Tensor] = None) -> Promise:
    """
    Implementation of matrix product of two tensors, similar to torch.matmul.

    Args:
        input: First input tensor
        other: Second input tensor
        out: Optional output tensor

    Returns:
        Promise: Promise object for asynchronous execution
    """
    # Validate inputs
    if not isinstance(input, Tensor):
        raise TypeError(f"Expected Tensor for input, got {type(input)}")
    if not isinstance(other, Tensor):
        raise TypeError(f"Expected Tensor for other, got {type(other)}")

    if out is None:
        out = Tensor(
            input.shape[:-1] + other.shape[1:], dtype=input.dtype, device=input.device
        )

    # Assert input properties
    assert input.ndim >= 1, f"Input must be at least 1D, got {input.ndim}D"
    assert other.ndim >= 1, f"Other must be at least 1D, got {other.ndim}D"
    assert (
        input.shape[-1] == other.shape[-2]
    ), f"Incompatible shapes: {input.shape} and {other.shape}"

    # Use numpy's matmul which handles all the broadcasting and dimension cases
    # result_data = np.matmul(input.numpy(), other.numpy())
    promise = matmul(input, other, out)

    # Assert result properties
    assert out.ndim >= 1, f"Result should be at least 1D, got {out.ndim}D"
    assert (
        out.shape == input.shape[:-1] + other.shape[1:]
    ), f"Unexpected result shape: {out.shape}"

    assert out.device == input.device, f"Unexpected result device: {out.device}"

    return promise


def matmul_graph_capture_impl(
    input: Tensor, other: Tensor, out: Optional[Tensor] = None
):
    """
    Graph capture implementation of matrix product.

    Captures the operation in the graph without performing actual computation.
    Creates a placeholder output tensor with the correct shape and dtype.

    Args:
        input: First input tensor
        other: Second input tensor
        out: Optional output tensor

    """
    # Validate inputs (same as regular implementation)
    if not isinstance(input, Tensor):
        raise TypeError(f"Expected Tensor for input, got {type(input)}")
    if not isinstance(other, Tensor):
        raise TypeError(f"Expected Tensor for other, got {type(other)}")

    if out is None:
        out = Tensor(
            input.shape[:-1] + other.shape[1:], dtype=input.dtype, device=input.device
        )

    # Assert input properties (same as regular implementation)
    assert input.ndim >= 1, f"Input must be at least 1D, got {input.ndim}D"
    assert other.ndim >= 1, f"Other must be at least 1D, got {other.ndim}D"
    assert (
        input.shape[-1] == other.shape[-2]
    ), f"Incompatible shapes: {input.shape} and {other.shape}"

    # Capture the operation in the graph
    from ..graph import add_to_graph

    add_to_graph(
        operation="matmul",
        func=matmul_impl,  # Reference to the actual implementation function
        inputs=[input, other],
        output=out,
        input_shapes=(input.shape, other.shape),
        output_shape=out.shape,
        input_dtypes=(input.dtype, other.dtype),
        output_dtype=out.dtype,
        has_out_param=out is not None,
    )
