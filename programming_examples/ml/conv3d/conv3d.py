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
    dev,
    depth: int,
    height: int,
    width: int,
    in_channels: int,
    out_channels: int,
    trace_size: int,
):
    # Buffer sizes for one depth plane
    actInPlane = height * width * in_channels
    bufInPlanes = actInPlane * 3  # Triple buffer for 3 depth planes

    # Weights: 3x3x3 kernel with in_channels x out_channels
    weights = 3 * 3 * 3 * in_channels * out_channels

    actOutPlane = height * width * out_channels
    bufOutPlanes = actOutPlane * 2  # Double buffer output

    # Total tensor sizes
    tensorInSize = depth * height * width * in_channels
    tensorOutSize = depth * height * width * out_channels

    # Type definitions
    actInPlane_ty = np.ndarray[(actInPlane,), np.dtype[np.uint8]]
    bufInPlanes_ty = np.ndarray[(bufInPlanes,), np.dtype[np.uint8]]

    weights_ty = np.ndarray[(weights,), np.dtype[np.int8]]

    actOutPlane_ty = np.ndarray[(actOutPlane,), np.dtype[np.uint8]]
    bufOutPlanes_ty = np.ndarray[(bufOutPlanes,), np.dtype[np.uint8]]

    tensorIn_ty = np.ndarray[(tensorInSize,), np.dtype[np.uint8]]
    tensorOut_ty = np.ndarray[(tensorOutSize,), np.dtype[np.uint8]]

    # Kernel declaration
    conv3dk3_ui8_kernel = Kernel(
        "conv3dk3_ui8_scalar",
        "conv3dk3_ui8.o",
        [
            actInPlane_ty,  # plane0
            actInPlane_ty,  # plane1
            actInPlane_ty,  # plane2
            weights_ty,     # weights
            actOutPlane_ty, # output
            np.int32,       # input_width
            np.int32,       # input_height
            np.int32,       # input_channels
            np.int32,       # output_channels
            np.int32,       # kernel_width
            np.int32,       # kernel_height
            np.int32,       # kernel_depth
            np.int32,       # check (border handling)
            np.int32,       # scale
            np.int32,       # channel_offset
        ],
    )

    # ObjectFifos for data movement
    # Input: L3 → L2 → L1 (need depth=3 for triple buffering of planes)
    of_inOF_act_L3L2 = ObjectFifo(bufInPlanes_ty, depth=3, name="inOF_act_L3L2")
    of_act_L2 = of_inOF_act_L3L2.cons().forward(
        obj_type=actInPlane_ty, name="act_L2"
    )

    # Weights: L3 → L2 → L1
    of_inOF_wts_L3L2 = ObjectFifo(weights_ty, depth=1, name="inOF_wts_L3L2")

    # Output: L1 → L2 → L3
    of_out_L2 = ObjectFifo(actOutPlane_ty, name="out_L2")
    of_outOFL2L3 = of_out_L2.cons().forward(obj_type=bufOutPlanes_ty, name="outOFL2L3")

    # Runtime parameter buffer (scale factor)
    rtp = Buffer(
        np.ndarray[(16,), np.dtype[np.int32]],
        name="rtp",
        use_write_rtp=True,
    )

    rtp_barrier = WorkerRuntimeBarrier()

    # Core function: processes depth slices of 3D volume
    def core_fn(of_wts, of_act, of_out, my_rtp, conv3dk3, barrier):
        d_dim = depth
        h_dim = height
        w_dim = width
        ci = in_channels
        co = out_channels

        barrier.wait_for_value(1)
        scale = my_rtp[0]

        elemWts = of_wts.acquire(1)

        # Process each depth slice
        for z in range_(d_dim):
            # Acquire 3 input planes for 3x3x3 kernel
            planes = of_act.acquire(3)
            plane0 = planes[0]  # z-1 (or replicate for z=0)
            plane1 = planes[1]  # z
            plane2 = planes[2]  # z+1 (or replicate for z=depth-1)

            elemOut = of_out.acquire(1)

            # Border check: 0=top_plane, 1=middle_plane, 2=bottom_plane
            if z == 0:
                check = 0  # top_plane
            elif z == d_dim - 1:
                check = 2  # bottom_plane
            else:
                check = 1  # middle_plane

            # Call 3D convolution kernel
            conv3dk3(
                plane0,
                plane1,
                plane2,
                elemWts,
                elemOut,
                w_dim,
                h_dim,
                ci,
                co,
                3,      # kernel_width
                3,      # kernel_height
                3,      # kernel_depth
                check,
                scale,
                0,      # channel_offset
            )

            of_act.release(3)
            of_out.release(1)

        of_wts.release(1)

    # Create worker
    worker = Worker(
        core_fn,
        [
            of_inOF_wts_L3L2.cons(),
            of_act_L2.cons(),
            of_out_L2.prod(),
            rtp,
            conv3dk3_ui8_kernel,
            rtp_barrier,
        ],
        stack_size=0x800,  # Larger stack for 3D processing
    )

    # Runtime sequence
    rt = Runtime()
    with rt.sequence(tensorIn_ty, weights_ty, tensorOut_ty) as (I, W, O):
        # Initialize runtime parameters
        def set_rtps(my_rtp):
            my_rtp[0] = 10  # scale factor

        rt.inline_ops(set_rtps, [rtp])
        rt.set_barrier(rtp_barrier, 1)

        # Start worker
        rt.start(worker)

        # Fill/drain ObjectFifos
        rt.fill(of_inOF_act_L3L2.prod(), I)
        rt.fill(of_inOF_wts_L3L2.prod(), W)
        rt.drain(of_outOFL2L3.cons(), O, wait=True)

    # Resolve and generate MLIR
    return Program(dev, rt).resolve_program(SequentialPlacer())


# Command-line parsing
try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = NPU1Col1()
    elif device_name == "npu2":
        dev = NPU2Col1()
    else:
        raise ValueError(f"[ERROR] Device name {device_name} is unknown")

    depth = int(sys.argv[2])
    if depth < 2:
        print("Depth must be >= 2")
        raise ValueError

    height = int(sys.argv[3])
    if height < 2:
        print("Height must be >= 2")
        raise ValueError

    width = int(sys.argv[4])
    if width % 8 != 0 or width < 8:
        print("Width must be a multiple of 8 and >= 8")
        raise ValueError

    in_channels = int(sys.argv[5])
    if in_channels % 8 != 0 or in_channels < 8:
        print("Input channels must be a multiple of 8 and >= 8")
        raise ValueError

    out_channels = int(sys.argv[6])
    if out_channels % 8 != 0 or out_channels < 8:
        print("Output channels must be a multiple of 8 and >= 8")
        raise ValueError

    trace_size = 0 if (len(sys.argv) != 8) else int(sys.argv[7])

except (IndexError, ValueError) as e:
    print(f"Error: {e}")
    print(
        "Usage: python conv3d.py <device> <depth> <height> <width> <in_channels> <out_channels> [trace_size]"
    )
    sys.exit(1)

module = conv3dk3(dev, depth, height, width, in_channels, out_channels, trace_size)
print(module)
