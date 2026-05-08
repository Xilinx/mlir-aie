# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Vector reduce-max across 4 cores in one NPU column (cascade reduction)."""

import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, Compile, ExternalFunction, In, Out
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.helpers.dialects.scf import else_, if_
from aie.helpers.util import np_ndarray_type_get_shape
from aie.utils.config import cxx_header_path


@iron.jit
def vector_reduce_max(
    input0: In,
    output: Out,
    *,
    in_tensor_size: Compile[int],
    element_type: Compile[type],
):
    n_cores = 4
    N = 2048
    elems_per_core = N // n_cores
    num_iter = in_tensor_size // N

    in_ty = np.ndarray[(in_tensor_size,), np.dtype[element_type]]
    mem_ty = np.ndarray[(N,), np.dtype[element_type]]
    op_ty = np.ndarray[(elems_per_core,), np.dtype[element_type]]
    # DMA transfers must be 4-byte aligned; pad to ceil(4/itemsize) elements.
    _itemsize = np.dtype(element_type).itemsize
    out_elems = (4 + _itemsize - 1) // _itemsize
    out_ty = np.ndarray[(out_elems,), np.dtype[element_type]]

    of_in = ObjectFifo(mem_ty, name="of_in")

    if n_cores > 1:
        of_a_offsets = [
            (np.prod(np_ndarray_type_get_shape(mem_ty)) // n_cores) * i
            for i in range(n_cores)
        ]
    else:
        of_a_offsets = [0]

    # split() distributes one ObjectFIFO into n_cores smaller ones at the
    # given offsets. See programming_guide/section-2/section-2b/ for more on
    # ObjectFIFO distribute/join patterns.
    in_fifos = of_in.cons().split(
        of_a_offsets,
        obj_types=[op_ty] * n_cores,
        names=[f"memA{i}" for i in range(n_cores)],
    )

    min_val = np.full(out_elems, bfloat16(float("-inf")), dtype=element_type)
    out_fifos = []
    nextC_buffers = []
    tmp_buffers = []
    for i in range(n_cores):
        out_fifos.append(ObjectFifo(out_ty, name=f"memC{i}"))
        nextC_buffers.append(Buffer(type=out_ty, initial_value=min_val))
        tmp_buffers.append(Buffer(type=out_ty, initial_value=min_val))

    # kernels.reduce_max() can't be used here: its 1-element output is only
    # 2 bytes for bfloat16, which violates the 4-byte DMA alignment requirement.
    # We bind the kernel directly so we can request a multi-element output.
    reduce_max_vector = ExternalFunction(
        "reduce_max_vector_bfloat16",
        source_file=cxx_header_path() + "/aie_kernels/aie2/reduce_max.cc",
        arg_types=[op_ty, out_ty, np.int32],
        include_dirs=[cxx_header_path()],
    )

    # final_core_body runs on the last core in the cascade — it has no
    # downstream neighbor and writes the final maximum to the output ObjectFIFO.
    def final_core_body(of_in, of_out, reduce_fn, nextC_buffer, tmp_buffer):
        elem_out = of_out.acquire(1)
        for _ in range_(num_iter):
            elem_in = of_in.acquire(1)
            reduce_fn(elem_in, tmp_buffer, elems_per_core)
            with if_(nextC_buffer[0] < tmp_buffer[0]) as if_op:
                nextC_buffer[0] = tmp_buffer[0]
            of_in.release(1)
        elem_out[0] = nextC_buffer[0]
        of_out.release(1)

    def core_body(of_in, of_out, in0, reduce_fn, nextC_buffer, tmp_buffer):
        for _ in range_(num_iter):
            elem_in = of_in.acquire(1)
            reduce_fn(elem_in, tmp_buffer, elems_per_core)
            with if_(nextC_buffer[0] < tmp_buffer[0]) as if_op:
                nextC_buffer[0] = tmp_buffer[0]
            of_in.release(1)

        elem_out = of_out.acquire(1)
        elem_in1 = in0.acquire(1)
        with if_(elem_in1[0] > nextC_buffer[0]) as if_op:
            elem_out[0] = elem_in1[0]
        with else_(if_op):
            elem_out[0] = nextC_buffer[0]
        in0.release(1)
        of_out.release(1)

    workers = []
    for i in range(n_cores):
        if i != n_cores - 1:
            workers.append(
                Worker(
                    core_body,
                    fn_args=[
                        in_fifos[i].cons(),
                        out_fifos[i].prod(),
                        out_fifos[i + 1].cons(),
                        reduce_max_vector,
                        nextC_buffers[i],
                        tmp_buffers[i],
                    ],
                    trace=None,
                )
            )
        else:
            workers.append(
                Worker(
                    final_core_body,
                    fn_args=[
                        in_fifos[i].cons(),
                        out_fifos[i].prod(),
                        reduce_max_vector,
                        nextC_buffers[i],
                        tmp_buffers[i],
                    ],
                    trace=None,
                )
            )

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_in, c_out):
        rt.start(*workers)
        rt.fill(of_in.prod(), a_in)
        rt.drain(out_fifos[0].cons(), c_out, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def main():
    in_size = 524288
    element_type = bfloat16
    in_tensor_size = in_size // element_type(0).nbytes

    # Output needs enough elements for 4-byte DMA alignment.
    out_elems = (4 + element_type(0).nbytes - 1) // element_type(0).nbytes
    input0 = iron.arange(in_tensor_size, dtype=element_type, device="npu")
    output = iron.zeros(out_elems, dtype=element_type, device="npu")

    vector_reduce_max(
        input0, output, in_tensor_size=in_tensor_size, element_type=element_type
    )

    # Initialize ref to -inf so all-negative inputs still produce the right max.
    ref_max = bfloat16(float("-inf"))
    for i in input0:
        if i > ref_max:
            ref_max = i

    if output[0] != ref_max:
        print(f"\nFAIL: {output[0]} != {ref_max}")
        sys.exit(1)
    print(f"Correct output: {output} == {ref_max}")
    print("\nPASS!")


if __name__ == "__main__":
    main()
