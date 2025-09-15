#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import argparse

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D


def my_matmul(dev):
    M = 288
    K = 288
    m = 32
    k = 32

    # TODO: increase this
    n_cores = 1
    M_div_n_cores = M // n_cores
    M_div_m_div_n_cores = M // (m * n_cores)
    K_div_k = K // k

    # FIXME vectorized kernel is currently erroneous
    vectorized = False

    # Define types
    dtype_in = np.dtype[np.int16]
    dtype_in_str = "i16"
    dtype_out = np.dtype[np.int32]
    dtype_out_str = "i32"
    A_ty = np.ndarray[(M, K), dtype_in]
    B_ty = np.ndarray[(1, K), dtype_in]
    C_ty = np.ndarray[(1, M), dtype_out]
    inA_ty = np.ndarray[(m, k), dtype_in]
    inB_ty = np.ndarray[(k,), dtype_in]
    outC_ty = np.ndarray[(m,), dtype_out]
    A_ty = np.ndarray[(m, k), dtype_in]

    # AIE Core Function declarations
    func_type = "vectorized" if vectorized else "scalar"
    zero = Kernel(f"zero_{func_type}_{dtype_out_str}", f"mv_{m}x{k}.o", [outC_ty])
    matvec = Kernel(
        f"matvec_{func_type}_{dtype_in_str}_{dtype_out_str}",
        f"mv_{m}x{k}.o",
        [A_ty, inB_ty, outC_ty],
    )

    # Define the work each core will do
    def core_fn(of_a, of_b, of_c, zero, matvec):
        elem_out = of_c.acquire(1)
        zero(elem_out)
        for _ in range_(K_div_k):
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            matvec(elem_in_a, elem_in_b, elem_out)
            of_a.release(1)
            of_b.release(1)
        of_c.release(1)

    # Create object fifos and workers for each core
    memA_fifos = []
    coreA_fifos = []
    outC_fifos = []
    workers = []
    B_fifo = ObjectFifo(inB_ty)
    for i in range(n_cores):
        a_fifo = ObjectFifo(inA_ty, name=f"memA{i}")
        memA_fifos.append(a_fifo)
        coreA_fifos.append(a_fifo.cons().forward())  # TODO: transform if vectorized
        outC_fifos.append(ObjectFifo(outC_ty, name=f"outC{i}"))
        w = Worker(
            core_fn,
            [coreA_fifos[i].cons(), B_fifo.cons(), outC_fifos[i].prod(), zero, matvec],
        )
        workers.append(w)

    # Define the tiling access patterns for input and output tensors
    A_taps = TensorTiler2D.group_tiler((M, K), (m, k), (M_div_m_div_n_cores, K_div_k))
    C_taps = TensorTiler2D.simple_tiler((1, M), (1, M_div_n_cores))
    b_tap = TensorTiler2D.simple_tiler((1, K), pattern_repeat=M_div_m_div_n_cores)[0]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (a_in, b_in, c_out):
        rt.start(*workers)

        # there is only one b tile
        rt.fill(B_fifo.prod(), b_in, b_tap)

        for i, (a_tap, c_tap) in enumerate(zip(A_taps, C_taps)):
            rt.fill(memA_fifos[i].prod(), a_in, a_tap)
            rt.drain(outC_fifos[i].cons(), c_out, c_tap, wait=True)

    # Create the program from the device type and runtime
    if dev == "npu":
        dev_ty = NPU1()
    else:
        dev_ty = NPU2()
    my_program = Program(dev_ty, rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program(SequentialPlacer())

    # Print the generated MLIR
    print(module)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Vector Multiplication MLIR Design",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    args, _ = argparser.parse_known_args()  # <- ignore the rest args in makefile-common
    dev = args.dev
    my_matmul(dev)
