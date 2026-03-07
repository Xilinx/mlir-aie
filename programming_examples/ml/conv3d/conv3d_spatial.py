#
# Conv3D with spatial parallelism (split by height)
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
from aie.iron.device import NPU2Col1, NPU2Col2, NPU2Col4
from aie.iron.controlflow import range_
from aie.helpers.taplib.tap import TensorAccessPattern


def conv3dk3_spatial(
    dev, depth: int, width: int, height: int, in_channels: int, out_channels: int, n_cores: int = 1
):
    """
    Spatial parallelism: split height dimension across cores.
    Each core processes height//n_cores rows.
    """
    assert height % n_cores == 0, f"height must be divisible by {n_cores}"
    height_per_core = height // n_cores

    # Sizes per core (spatial split)
    actIn_per_core = height_per_core * width * in_channels  # Rows per core
    actOut_per_core = height_per_core * width * out_channels

    # All cores share same weights
    weights = in_channels * out_channels * 3 * 3 * 3

    # Total sizes
    tensorInSize = depth * height * width * in_channels
    tensorOutSize = depth * height * width * out_channels

    # Type definitions
    actIn_ty = np.ndarray[(actIn_per_core,), np.dtype[np.uint8]]
    weights_ty = np.ndarray[(weights,), np.dtype[np.int8]]
    actOut_ty = np.ndarray[(actOut_per_core,), np.dtype[np.uint8]]

    tensorIn_ty = np.ndarray[(tensorInSize,), np.dtype[np.uint8]]
    tensorWts_ty = weights_ty
    tensorOut_ty = np.ndarray[(tensorOutSize,), np.dtype[np.uint8]]

    # Kernel
    conv3dk3_kernel = Kernel(
        "conv3dk3_ui8",
        "conv3dk3_ui8.o",
        [
            actIn_ty, actIn_ty, actIn_ty,  # 3 planes
            weights_ty, actOut_ty,
            np.int32, np.int32, np.int32, np.int32,  # w, h, ci, co
            np.int32, np.int32, np.int32,  # kw, kh, kd
            np.int32, np.int32, np.int32,  # check, scale, channel_offset
        ],
    )

    # ObjectFIFOs
    of_in_fifos = []
    of_wts_fifos = []
    of_out_fifos = []

    for c in range(n_cores):
        # Each core gets its spatial slice of input
        of_in = ObjectFifo(actIn_ty, name=f"inOF_act_{c}", depth=3)
        of_in_fifos.append(of_in)

        # Shared weights (broadcast to all cores)
        of_wts = ObjectFifo(weights_ty, depth=1, name=f"inOF_wts_{c}")
        of_wts_fifos.append(of_wts)

        # Each core produces its spatial slice of output
        of_out = ObjectFifo(actOut_ty, name=f"outOF_{c}")
        of_out_fifos.append(of_out)

    # Core function - simple, no conditionals!
    def core_fn(of_wts, of_in, of_out, kernel):
        elemWts = of_wts.acquire(1)

        for d in range_(depth):
            plane = of_in.acquire(1)
            elemOut = of_out.acquire(1)

            # Each core processes its height slice with 2D conv (kernel_depth=1)
            kernel(
                plane, plane, plane,
                elemWts, elemOut,
                width, height_per_core, in_channels, out_channels,
                3, 3, 1,  # 3x3x1 kernel (2D per plane)
                1, 10, 0  # check=middle, scale=10, no channel_offset
            )

            of_in.release(1)
            of_out.release(1)

        of_wts.release(1)

    # Create workers
    workers = []
    for c in range(n_cores):
        worker = Worker(
            core_fn,
            [
                of_wts_fifos[c].cons(),
                of_in_fifos[c].cons(),
                of_out_fifos[c].prod(),
                conv3dk3_kernel,
            ],
            while_true=False,
        )
        workers.append(worker)

    # Runtime
    rt = Runtime()

    # Create TensorAccessPatterns for spatial slicing
    # Input: split by height (each core gets height_per_core rows)
    in_taps = []
    for c in range(n_cores):
        # Offset: skip to this core's rows
        offset = c * (depth * height_per_core * width * in_channels)
        in_taps.append(TensorAccessPattern(
            (1, tensorInSize),
            offset,
            [1, 1, 1, actIn_per_core * depth],
            [0, 0, 0, 1]
        ))

    # Output: concatenate by height
    out_taps = []
    for c in range(n_cores):
        offset = c * (depth * height_per_core * width * out_channels)
        out_taps.append(TensorAccessPattern(
            (1, tensorOutSize),
            offset,
            [1, 1, 1, actOut_per_core * depth],
            [0, 0, 0, 1]
        ))

    with rt.sequence(tensorIn_ty, tensorWts_ty, tensorOut_ty) as (I, W, O):
        # Start all workers
        for worker in workers:
            rt.start(worker)

        # Fill inputs (spatial slices)
        for c in range(n_cores):
            rt.fill(of_in_fifos[c].prod(), I, in_taps[c])

        # Broadcast weights to all cores
        for c in range(n_cores):
            rt.fill(of_wts_fifos[c].prod(), W)

        # Drain outputs (spatial slices)
        for c in range(n_cores):
            wait = (c == n_cores - 1)
            rt.drain(of_out_fifos[c].cons(), O, out_taps[c], wait=wait)

    return Program(dev, rt).resolve_program(SequentialPlacer())


# Parse args
device_name = sys.argv[1]
if device_name == "npu2":
    dev = NPU2Col1()
    n_cores = 1
elif device_name == "npu2_2col":
    dev = NPU2Col2()
    n_cores = 2
elif device_name == "npu2_4col":
    dev = NPU2Col4()
    n_cores = 4
else:
    raise ValueError(f"Unknown device: {device_name}")

depth = int(sys.argv[2])
width = int(sys.argv[3])
height = int(sys.argv[4])
in_channels = int(sys.argv[5])
out_channels = int(sys.argv[6])

module = conv3dk3_spatial(dev, depth, width, height, in_channels, out_channels, n_cores)
print(module)
