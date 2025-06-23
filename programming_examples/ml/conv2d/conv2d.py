#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
import numpy as np
import sys

from aie.iron import (
    GlobalBuffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    WorkerRuntimeBarrier,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_


def conv2dk1(
    dev, width: int, height: int, in_channels: int, out_channels: int, trace_size: int
):

    actIn = width * in_channels  # 32*64 = 2048
    bufIn = actIn * 2  # double buffer

    weights = in_channels * out_channels

    actOut = width * out_channels  # 32*64 = 2048
    bufOut = actOut * 2  # double buffer

    tensorInSize = width * height * in_channels
    tensorOutSize = width * height * out_channels

    N_in_bytes = tensorOutSize  # Number of bytes of output data (1 byte/elem)

    # Type definitions
    actIn_ty = np.ndarray[(actIn,), np.dtype[np.int8]]
    bufIn_ty = np.ndarray[(bufIn,), np.dtype[np.int8]]

    weights_ty = np.ndarray[(weights,), np.dtype[np.int8]]

    out_ty = np.ndarray[(actOut,), np.dtype[np.int8]]
    bufOut_ty = np.ndarray[(bufOut,), np.dtype[np.int8]]
    tensorIn_ty = np.ndarray[(tensorInSize,), np.dtype[np.int8]]
    tensorOut_ty = np.ndarray[(tensorOutSize,), np.dtype[np.int8]]

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
    of_inOF_wts_0_L3L2 = ObjectFifo(weights_ty, depth=1, name="inOF_wts_0_L3L2")

    # Output
    of_out_02_L2 = ObjectFifo(out_ty, name="out_02_L2")
    of_outOFL2L3 = of_out_02_L2.cons().forward(obj_type=bufOut_ty, name="outOFL2L3")

    # Setup a global buffer to hold runtime parameters
    rtp = GlobalBuffer(
        np.ndarray[(16,), np.dtype[np.int32]],
        name="rtp",
        use_write_rtp=True,
    )

    rtp_barrier = WorkerRuntimeBarrier()

    # Task for the core to perform
    def core_fn(of_wts, of_act, of_out, my_rtp, conv2dk1_i8, barrier):
        y_dim = height
        x_dim = width
        ci = in_channels
        co = out_channels

        barrier.wait_for_value(1)
        scale = my_rtp[0]

        elemWts = of_wts.acquire(1)

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
            rtp_barrier,
        ],
        stack_size=0x600,
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensorIn_ty, weights_ty, tensorOut_ty) as (I, W, O):
        # Initialize the runtime parameter values
        def set_rtps(my_rtp):
            my_rtp[0] = 10

        rt.inline_ops(set_rtps, [rtp])

        rt.set_barrier(rtp_barrier, 1)

        # Start worker
        rt.start(worker)

        # Fill/drain input/output ObjectFifos
        rt.fill(of_inOF_act_L3L2.prod(), I)
        rt.fill(of_inOF_wts_0_L3L2.prod(), W)
        rt.drain(of_outOFL2L3.cons(), O, wait=True)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = NPU1Col1()
    elif device_name == "npu2":
        dev = NPU2Col1()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
    width = int(sys.argv[2])
    if width % 8 != 0 or width < 8:
        print("Width size must be a multiple of 8 and greater than or equal to 8")
        raise ValueError
    height = int(sys.argv[3])
    if height < 2:
        print("Height needs to be > 1 at the moment (BUG)")
        raise ValueError
    in_channels = int(sys.argv[4])
    if in_channels % 8 != 0 or in_channels < 8:
        print(
            "Input channels size must be a multiple of 8 and greater than or equal to 8"
        )
        raise ValueError
    out_channels = int(sys.argv[5])
    if out_channels % 8 != 0 or out_channels < 8:
        print(
            "Output channel size must be a multiple of 8 and greater than or equal to 8"
        )
        raise ValueError
    trace_size = 0 if (len(sys.argv) != 7) else int(sys.argv[6])
except ValueError:
    print("Argument has inappropriate value")
module = conv2dk1(dev, width, height, in_channels, out_channels, trace_size)
print(module)
