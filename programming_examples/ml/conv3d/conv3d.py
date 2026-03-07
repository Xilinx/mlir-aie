#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
import numpy as np
import sys

from aie.iron import (
    Buffer,
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


def conv3dk3(
    dev, depth: int, width: int, height: int, in_channels: int, out_channels: int, trace_size: int
):

    # Full depth plane size (height * width * channels)
    actIn = height * width * in_channels  # Full plane, not just one line
    bufIn = actIn * 2  # double buffer

    # Weights for 3x3x3 kernel: (in_channels * out_channels * kd * kh * kw)
    weights = in_channels * out_channels * 3 * 3 * 3

    actOut = height * width * out_channels  # Full output plane
    bufOut = actOut * 2  # double buffer

    tensorInSize = depth * width * height * in_channels  # Added depth dimension
    tensorOutSize = depth * width * height * out_channels  # Added depth dimension

    N_in_bytes = tensorOutSize  # Number of bytes of output data (1 byte/elem)

    # Type definitions
    actIn_ty = np.ndarray[(actIn,), np.dtype[np.uint8]]  # uint8 for activations
    bufIn_ty = np.ndarray[(bufIn,), np.dtype[np.uint8]]

    weights_ty = np.ndarray[(weights,), np.dtype[np.int8]]  # int8 for weights

    out_ty = np.ndarray[(actOut,), np.dtype[np.uint8]]  # uint8 for output
    bufOut_ty = np.ndarray[(bufOut,), np.dtype[np.uint8]]
    tensorIn_ty = np.ndarray[(tensorInSize,), np.dtype[np.uint8]]
    tensorOut_ty = np.ndarray[(tensorOutSize,), np.dtype[np.uint8]]

    # AIE Core Function declarations
    conv3dk3_kernel = Kernel(
        "conv3dk3_ui8_scalar",
        "conv3dk3_ui8.o",
        [
            actIn_ty,   # plane0
            actIn_ty,   # plane1
            actIn_ty,   # plane2
            weights_ty, # weights
            out_ty,     # output
            np.int32,   # width
            np.int32,   # height
            np.int32,   # in_channels
            np.int32,   # out_channels
            np.int32,   # kernel_width
            np.int32,   # kernel_height
            np.int32,   # kernel_depth
            np.int32,   # check
            np.int32,   # scale
            np.int32,   # channel_offset
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
    rtp = Buffer(
        np.ndarray[(16,), np.dtype[np.int32]],
        name="rtp",
        use_write_rtp=True,
    )

    rtp_barrier = WorkerRuntimeBarrier()

    # Task for the core to perform
    # NOTE: This is a simplified implementation that processes each depth plane
    # with the same plane passed three times. For a true 3D convolution with
    # sliding window (using planes z-1, z, z+1), would need more complex
    # buffering logic with ObjectFIFO depth > 1 and manual plane management.
    def core_fn(of_wts, of_act, of_out, my_rtp, conv3dk3_kernel, barrier):
        d_dim = depth
        x_dim = width
        h_dim = height
        ci = in_channels
        co = out_channels

        barrier.wait_for_value(1)
        scale = my_rtp[0]

        elemWts = of_wts.acquire(1)

        # Process each output depth plane
        # Simplified: pass same plane 3 times, use check=1 (middle_plane) always
        for d in range_(d_dim):
            plane_curr = of_act.acquire(1)
            elemOut = of_out.acquire(1)

            # Call kernel with same plane repeated 3 times
            # WORKAROUND: Set kernel_depth=1 to make it 2D convolution (3x3x1)
            # This processes only kd=0 depth position, avoiding triple counting
            # TODO: Implement proper 3D sliding window with distinct planes for true 3x3x3
            conv3dk3_kernel(
                plane_curr, plane_curr, plane_curr,
                elemWts, elemOut,
                x_dim, h_dim, ci, co,
                3, 3, 1,  # kernel_width=3, kernel_height=3, kernel_depth=1 (2D not 3D!)
                1,        # check=1 (middle_plane)
                scale,
                0         # channel_offset
            )

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
            conv3dk3_kernel,  # Changed from conv2dk1_i8_kernel
            rtp_barrier,
        ],
        stack_size=0x800,  # Increased for 3D buffering
        while_true=False,   # Run once, not infinite loop (barrier only set once)
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
    depth = int(sys.argv[2])     # Added depth parameter
    width = int(sys.argv[3])     # Shifted from argv[2]
    if width % 8 != 0 or width < 8:
        print("Width size must be a multiple of 8 and greater than or equal to 8")
        raise ValueError
    height = int(sys.argv[4])    # Shifted from argv[3]
    if height < 2:
        print("Height needs to be > 1 at the moment (BUG)")
        raise ValueError
    in_channels = int(sys.argv[5])  # Shifted from argv[4]
    if in_channels % 8 != 0 or in_channels < 8:
        print(
            "Input channels size must be a multiple of 8 and greater than or equal to 8"
        )
        raise ValueError
    out_channels = int(sys.argv[6])  # Shifted from argv[5]
    if out_channels % 8 != 0 or out_channels < 8:
        print(
            "Output channel size must be a multiple of 8 and greater than or equal to 8"
        )
        raise ValueError
    trace_size = 0 if (len(sys.argv) != 8) else int(sys.argv[7])  # Shifted from argv[6]
except ValueError:
    print("Argument has inappropriate value")
module = conv3dk3(dev, depth, width, height, in_channels, out_channels, trace_size)
print(module)
