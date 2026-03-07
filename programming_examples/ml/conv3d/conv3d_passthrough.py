#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 AMD Inc.

# Minimal passthrough test for conv3d to isolate NPU timeout issues
# This version just copies data through without any convolution

import numpy as np
import sys

from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_


def conv3d_passthrough(
    dev,
    depth: int,
    height: int,
    width: int,
    channels: int,
    trace_size: int,
):
    # Buffer sizes for one depth plane
    plane_size = height * width * channels

    # Total tensor sizes
    tensor_size = depth * height * width * channels

    # Type definitions
    plane_ty = np.ndarray[(plane_size,), np.dtype[np.uint8]]
    tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.uint8]]

    # Kernel declaration - simple passthrough
    passthrough_kernel = Kernel(
        "passthrough_3d_ui8",
        "passthrough_3d_ui8.o",
        [
            plane_ty,     # input_plane
            plane_ty,     # output_plane
            np.int32,     # input_width
            np.int32,     # input_height
            np.int32,     # input_channels
        ],
    )

    # ObjectFifos for data movement
    # Input: L3 → L2 → L1 (depth=2 for double buffering)
    of_inOF_L3L2 = ObjectFifo(plane_ty, depth=2, name="inOF_L3L2")
    of_in_L2 = of_inOF_L3L2.cons().forward(obj_type=plane_ty, name="in_L2")

    # Output: L1 → L2 → L3 (depth=2 for double buffering)
    of_out_L2 = ObjectFifo(plane_ty, depth=2, name="out_L2")
    of_outOFL2L3 = of_out_L2.cons().forward(obj_type=plane_ty, name="outOFL2L3")

    # Core function: processes depth slices - simple passthrough
    def core_fn(of_in, of_out, passthrough_fn):
        h_dim = height
        w_dim = width
        ch = channels

        # Process each depth slice
        for z in range_(depth):
            # Acquire one input plane and one output plane
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)

            # Call passthrough kernel - just copy data
            passthrough_fn(
                elem_in,   # input_plane
                elem_out,  # output_plane
                w_dim,
                h_dim,
                ch,
            )

            of_in.release(1)
            of_out.release(1)

    # Create worker
    worker = Worker(
        core_fn,
        [
            of_in_L2.cons(),
            of_out_L2.prod(),
            passthrough_kernel,
        ],
        stack_size=0x400,  # Smaller stack needed for simple passthrough
        while_true=False,  # Process once, not infinite loop
    )

    # Runtime sequence
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (I, O):
        # Start worker
        rt.start(worker)

        # Fill/drain ObjectFifos
        rt.fill(of_inOF_L3L2.prod(), I)
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
    if depth < 1:
        print("Depth must be >= 1")
        raise ValueError

    height = int(sys.argv[3])
    if height < 1:
        print("Height must be >= 1")
        raise ValueError

    width = int(sys.argv[4])
    if width < 1:
        print("Width must be >= 1")
        raise ValueError

    channels = int(sys.argv[5])
    if channels < 1:
        print("Channels must be >= 1")
        raise ValueError

    trace_size = 0 if (len(sys.argv) != 7) else int(sys.argv[6])

except (IndexError, ValueError) as e:
    print(f"Error: {e}")
    print(
        "Usage: python conv3d_passthrough.py <device> <depth> <height> <width> <channels> [trace_size]"
    )
    sys.exit(1)

module = conv3d_passthrough(dev, depth, height, width, channels, trace_size)
print(module)
