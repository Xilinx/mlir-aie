#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 AMD Inc.

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
    """
    3D Convolution with 3x3x3 kernel using sliding window approach.

    Data layout:
    - Input: D{C/8}H{C8}W (depth, channel-groups, height, channels-per-group, width)
    - Weights: {O/8}{I/8}KDHW{I8}{O8} where KD=KH=KW=3
    - Output: D{C/8}H{C8}W

    Processing:
    - Each output depth plane requires 3 input depth planes (z-1, z, z+1)
    - Border handling: replicate first/last planes for z=0 and z=depth-1
    """

    # Size of one depth plane
    actIn = width * in_channels  # One line (like conv2d)
    bufIn = actIn * 2  # double buffer

    # Weights: full 3x3x3 kernel
    weights_per_kernel = 3 * 3 * 3  # kd * kh * kw
    weights = in_channels * out_channels * weights_per_kernel

    actOut = width * out_channels
    bufOut = actOut * 2  # double buffer

    # Total tensor sizes
    tensorInSize = depth * width * height * in_channels
    tensorOutSize = depth * width * height * out_channels

    # Type definitions
    actIn_ty = np.ndarray[(actIn,), np.dtype[np.uint8]]
    bufIn_ty = np.ndarray[(bufIn,), np.dtype[np.uint8]]

    weights_ty = np.ndarray[(weights,), np.dtype[np.int8]]

    out_ty = np.ndarray[(actOut,), np.dtype[np.uint8]]
    bufOut_ty = np.ndarray[(bufOut,), np.dtype[np.uint8]]
    tensorIn_ty = np.ndarray[(tensorInSize,), np.dtype[np.uint8]]
    tensorOut_ty = np.ndarray[(tensorOutSize,), np.dtype[np.uint8]]

    # AIE Core Function declarations
    conv3dk3_kernel = Kernel(
        "conv3dk3_ui8_scalar",
        "conv3dk3_ui8.o",
        [
            actIn_ty,   # plane0 (z-1)
            actIn_ty,   # plane1 (z)
            actIn_ty,   # plane2 (z+1)
            weights_ty, # weights
            out_ty,     # output
            np.int32,   # width
            np.int32,   # height
            np.int32,   # in_channels
            np.int32,   # out_channels
            np.int32,   # kernel_width
            np.int32,   # kernel_height
            np.int32,   # kernel_depth
            np.int32,   # check (region: top_plane=0, middle_plane=1, bottom_plane=2)
            np.int32,   # scale
            np.int32,   # channel_offset
        ],
    )

    # AIE-array data movement with object fifos
    # Input: Use depth=3 to create sliding window of planes
    of_inOF_act_L3L2 = ObjectFifo(bufIn_ty, name="inOF_act_L3L2", depth=3)
    of_act_L2_02 = of_inOF_act_L3L2.cons().forward(obj_type=actIn_ty, name="act_L2_02")

    # Weights
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
    def core_fn(of_wts, of_act, of_out, my_rtp, conv3dk3_kernel, barrier):
        d_dim = depth
        x_dim = width
        h_dim = height
        ci = in_channels
        co = out_channels

        barrier.wait_for_value(1)
        scale = my_rtp[0]

        elemWts = of_wts.acquire(1)

        # Process each depth plane
        for d in range_(d_dim):
            elemOut = of_out.acquire(1)

            # Determine which region we're in for border handling
            # region: 0=top_plane, 1=middle_plane, 2=bottom_plane
            if d == 0:
                # Top plane: only use current and next planes
                check = 0  # top_plane
                # For top: use plane[0] for z-1 (replicate), plane[0] for z, plane[1] for z+1
                plane0 = of_act.acquire(1)  # Will be used as replicated z-1
                plane1 = plane0              # Current plane (z=0)
                plane2 = of_act.acquire(1)  # Next plane (z=1)

            elif d == d_dim - 1:
                # Bottom plane: only use previous and current planes
                check = 2  # bottom_plane
                # plane0 = already from previous iteration
                # plane1 = acquire new (last plane)
                # plane2 = same as plane1 (replicate)
                plane0 = of_act.acquire(1)  # Previous plane (already in buffer)
                plane1 = of_act.acquire(1)  # Current/last plane
                plane2 = plane1              # Replicate for z+1

            else:
                # Middle planes: use all three
                check = 1  # middle_plane
                plane0 = of_act.acquire(1)  # z-1
                plane1 = of_act.acquire(1)  # z
                plane2 = of_act.acquire(1)  # z+1

            # Call the kernel with three depth planes
            conv3dk3_kernel(
                plane0, plane1, plane2,  # Three depth planes
                elemWts, elemOut,
                x_dim, h_dim, ci, co,
                3, 3, 3,  # kernel dimensions (kw, kh, kd)
                check,    # region flag
                scale,
                0         # channel_offset
            )

            # Release planes as appropriate
            if d < d_dim - 1:
                of_act.release(1)  # Release consumed plane

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
            conv3dk3_kernel,
            rtp_barrier,
        ],
        stack_size=0x800,  # Increased from 0x600 for 3-plane buffering
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensorIn_ty, weights_ty, tensorOut_ty) as (I, W, O):
        # Initialize the runtime parameter values
        def set_rtps(my_rtp):
            my_rtp[0] = 10  # scale factor

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
    depth = int(sys.argv[2])
    width = int(sys.argv[3])
    if width % 8 != 0 or width < 8:
        print("Width size must be a multiple of 8 and greater than or equal to 8")
        raise ValueError
    height = int(sys.argv[4])
    if height < 2:
        print("Height needs to be > 1 at the moment (BUG)")
        raise ValueError
    in_channels = int(sys.argv[5])
    if in_channels % 8 != 0 or in_channels < 8:
        print(
            "Input channels size must be a multiple of 8 and greater than or equal to 8"
        )
        raise ValueError
    out_channels = int(sys.argv[6])
    if out_channels % 8 != 0 or out_channels < 8:
        print(
            "Output channel size must be a multiple of 8 and greater than or equal to 8"
        )
        raise ValueError
    trace_size = 0 if (len(sys.argv) != 8) else int(sys.argv[7])
except ValueError:
    print("Argument has inappropriate value")
module = conv3dk3(dev, depth, width, height, in_channels, out_channels, trace_size)
print(module)
