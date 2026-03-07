#
# Simplified conv3d for debugging - single plane, no 3D sliding window
#

import numpy as np
import sys

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_


def conv3dk3_simple(dev, depth: int, height: int, width: int, in_channels: int, out_channels: int):
    # Buffer sizes for one depth plane
    actInPlane = height * width * in_channels
    actOutPlane = height * width * out_channels
    weights = in_channels * out_channels  # 1x1 conv for now

    # Total tensor sizes
    tensorInSize = depth * height * width * in_channels
    tensorOutSize = depth * height * width * out_channels

    # Type definitions
    actInPlane_ty = np.ndarray[(actInPlane,), np.dtype[np.uint8]]
    actOutPlane_ty = np.ndarray[(actOutPlane,), np.dtype[np.uint8]]
    weights_ty = np.ndarray[(weights,), np.dtype[np.int8]]
    tensorIn_ty = np.ndarray[(tensorInSize,), np.dtype[np.uint8]]
    tensorOut_ty = np.ndarray[(tensorOutSize,), np.dtype[np.uint8]]

    # Kernel declaration - use passthrough for now
    passthrough_kernel = Kernel(
        "passthrough_3d_ui8",
        "passthrough_3d_ui8.o",
        [
            actInPlane_ty,  # input_plane
            actOutPlane_ty, # output_plane
            np.int32,       # width
            np.int32,       # height
            np.int32,       # channels
        ],
    )

    # ObjectFifos - need MemTile links like passthrough!
    of_inOF_L3L2 = ObjectFifo(actInPlane_ty, name="inOF_L3L2")
    of_in_L2 = of_inOF_L3L2.cons().forward(obj_type=actInPlane_ty, name="in_L2")

    of_out_L2 = ObjectFifo(actOutPlane_ty, name="out_L2")
    of_outOFL2L3 = of_out_L2.cons().forward(obj_type=actOutPlane_ty, name="outOFL2L3")

    # Core function - exact passthrough pattern
    def core_fn(of_in_h, of_out_h, kernel):
        w = width
        h = height
        c = in_channels  # Use input channels since passthrough doesn't change

        for _ in range_(8):  # Process 8 depth planes
            inp = of_in_h.acquire(1)
            out = of_out_h.acquire(1)

            kernel(inp, out, w, h, c)

            of_in_h.release(1)
            of_out_h.release(1)

    # Create worker
    worker = Worker(
        core_fn,
        [of_in_L2.cons(), of_out_L2.prod(), passthrough_kernel],
        while_true=False,
    )

    # Runtime sequence - passthrough pattern (no weights)
    rt = Runtime()
    with rt.sequence(tensorIn_ty, tensorOut_ty) as (I, O):
        rt.start(worker)
        rt.fill(of_inOF_L3L2.prod(), I)
        rt.drain(of_outOFL2L3.cons(), O, wait=True)

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
    height = int(sys.argv[3])
    width = int(sys.argv[4])
    in_channels = int(sys.argv[5])
    out_channels = int(sys.argv[6])

except (IndexError, ValueError) as e:
    print(f"Error: {e}")
    print("Usage: python conv3d_simple.py <device> <depth> <height> <width> <in_channels> <out_channels>")
    sys.exit(1)

module = conv3dk3_simple(dev, depth, height, width, in_channels, out_channels)
print(module)
