#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
import numpy as np
import sys

from aie.iron import GlobalBuffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1
from aie.iron.controlflow import range_

width = 32
height = 32
in_channels = 64
out_channels = 64

if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])


actIn = width * in_channels  # 32*64 = 2048
bufIn = actIn * 2  # double buffer

weights = in_channels * out_channels

actOut = width * out_channels  # 32*64 = 2048
bufOut = actOut * 2  # double buffer

tensorSize = width * height * in_channels

N_in_bytes = tensorSize  # Number of bytes of output data (1 byte/elem)


def conv2dk1(trace_size: int):
    # Type definitions
    actIn_ty = np.ndarray[(actIn,), np.dtype[np.int8]]
    bufIn_ty = np.ndarray[(bufIn,), np.dtype[np.int8]]

    weights_ty = np.ndarray[(weights,), np.dtype[np.int8]]

    out_ty = np.ndarray[(actOut,), np.dtype[np.int8]]
    bufOut_ty = np.ndarray[(bufOut,), np.dtype[np.int8]]
    tensor_ty = np.ndarray[(tensorSize,), np.dtype[np.int8]]

    # AIE Core Function declarations
    conv2dk1_i8_kernel = Kernel(
        "conv2dk1_i8",
        "conv2dk1_i8.o",
        [
            actIn_ty,
            weights_ty,
            out_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )

    # AIE-array data movement with object fifos
    # Input
    of_inOF_act_L3L2 = ObjectFifo(bufIn_ty, name="inOF_act_L3L2")
    of_act_L2_02 = of_inOF_act_L3L2.cons().forward(obj_type=actIn_ty, name="act_L2_02")

    # wts
    of_inOF_wts_0_L3L2 = ObjectFifo(weights_ty, default_depth=1, name="inOF_wts_0_L3L2")

    # Output
    of_out_02_L2 = ObjectFifo(out_ty, name="out_02_L2")
    of_outOFL2L3 = of_out_02_L2.cons().forward(obj_type=bufOut_ty, name="outOFL2L3")

    # Setup a global buffer to hold runtime parameters
    rtp = GlobalBuffer(
        np.ndarray[(16,), np.dtype[np.int32]],
        name="rtp",
        use_write_rtp=True,
    )

    # Task for the core to perform
    def core_fn(of_wts, of_act, of_out, my_rtp, conv2dk1_i8):
        y_dim = 32
        x_dim = 32
        ci = 64
        co = 64

        elemWts = of_wts.acquire(1)
        scale = my_rtp[0]
        # scale = memref.load(rtpComputeTile2, [0])

        for _ in range_(y_dim):
            elemIn = of_act.acquire(1)
            elemOut0 = of_out.acquire(1)

            conv2dk1_i8(elemIn, elemWts, elemOut0, x_dim, ci, co, scale)
            of_act.release(1)
            of_out.release(1)
        of_wts.release(1)

    # Create a worker to perform the task
    worker = Worker(
        core_fn,
        [
            of_inOF_wts_0_L3L2.cons(),
            of_act_L2_02.cons(),
            of_out_02_L2.prod(),
            rtp,
            conv2dk1_i8_kernel,
        ],
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, weights_ty, tensor_ty) as (I, W, O):
        # Initialize the runtime parameter values
        def set_rtps(my_rtp):
            my_rtp[0] = 10

        rt.inline_ops(set_rtps, [rtp])

        # Start worker
        rt.start(worker)

        # Fill/drain input/output ObjectFifos
        rt.fill(of_inOF_act_L3L2.prod(), I)
        rt.fill(of_inOF_wts_0_L3L2.prod(), W)
        rt.drain(of_outOFL2L3.cons(), O, wait=True)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(NPU1Col1(), rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    trace_size = 0 if (len(sys.argv) != 2) else int(sys.argv[1])
    print(conv2dk1(trace_size=trace_size))
