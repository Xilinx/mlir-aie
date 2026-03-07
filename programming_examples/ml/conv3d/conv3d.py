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
from aie.iron.device import NPU1Col1, NPU2Col1, NPU2Col2, NPU2Col4
from aie.iron.controlflow import range_


def conv3dk3(
    dev, depth: int, width: int, height: int, in_channels: int, out_channels: int, trace_size: int, n_cores: int = 1
):

    # Output channel split across cores
    assert out_channels % (n_cores * 8) == 0, f"out_channels must be divisible by {n_cores * 8}"
    out_channels_per_core = out_channels // n_cores

    # Full depth plane size (height * width * channels)
    actIn = height * width * in_channels  # Full plane, not just one line
    bufIn = actIn * 2  # double buffer

    # Weights for 3x3x3 kernel: (in_channels * out_channels_per_core * kd * kh * kw)
    weights_per_core = in_channels * out_channels_per_core * 3 * 3 * 3

    actOut = height * width * out_channels_per_core  # Output per core
    bufOut = actOut * 2  # double buffer

    tensorInSize = depth * width * height * in_channels
    tensorOutSize = depth * width * height * out_channels

    N_in_bytes = tensorOutSize  # Number of bytes of output data (1 byte/elem)

    # Type definitions
    actIn_ty = np.ndarray[(actIn,), np.dtype[np.uint8]]
    bufIn_ty = np.ndarray[(bufIn,), np.dtype[np.uint8]]

    weights_ty = np.ndarray[(weights_per_core,), np.dtype[np.int8]]

    out_ty = np.ndarray[(actOut,), np.dtype[np.uint8]]
    bufOut_ty = np.ndarray[(bufOut,), np.dtype[np.uint8]]

    tensorIn_ty = np.ndarray[(tensorInSize,), np.dtype[np.uint8]]

    # For multi-core: separate weight tensors per core
    if n_cores == 1:
        tensorWts_ty = weights_ty
        tensorOut_ty = np.ndarray[(tensorOutSize,), np.dtype[np.uint8]]
    else:
        tensorWts_ty = [weights_ty for _ in range(n_cores)]
        tensorOut_ty = [np.ndarray[(tensorOutSize // n_cores,), np.dtype[np.uint8]] for _ in range(n_cores)]

    # AIE Core Function declarations
    conv3dk3_kernel = Kernel(
        "conv3dk3_ui8",  # Vectorized kernel
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
    # Multi-core setup: broadcast inputs, distribute outputs

    if n_cores == 1:
        # Single core path
        of_inOF_act_L3L2 = ObjectFifo(bufIn_ty, name="inOF_act_L3L2", depth=3)
        of_act_L2_02 = of_inOF_act_L3L2.cons().forward(obj_type=actIn_ty, name="act_L2_02", depth=3)

        of_inOF_wts_0_L3L2 = ObjectFifo(weights_ty, depth=1, name="inOF_wts_0_L3L2")

        of_out_02_L2 = ObjectFifo(out_ty, name="out_02_L2")
        of_outOFL2L3 = of_out_02_L2.cons().forward(obj_type=bufOut_ty, name="outOFL2L3")

        of_act_fifos = [of_act_L2_02]
        of_wts_fifos = [of_inOF_wts_0_L3L2]
        of_out_fifos = [of_out_02_L2]
        of_out_L3_fifos = [of_outOFL2L3]
    else:
        # Multi-core path: separate streams per core
        # Each core gets its own input/weight/output streams
        of_act_fifos = []
        of_wts_fifos = []
        of_out_fifos = []
        of_out_L3_fifos = []
        of_inOF_act_L3L2_list = []  # Store L3L2 ObjectFIFOs

        for c in range(n_cores):
            # Separate input streams per core (duplicated data)
            of_in = ObjectFifo(bufIn_ty, name=f"inOF_act_{c}_L3L2", depth=3)
            of_inOF_act_L3L2_list.append(of_in)
            of_act = of_in.cons().forward(obj_type=actIn_ty, name=f"act_{c}_L2", depth=3)
            of_act_fifos.append(of_act)

            # Separate weights per core
            of_wts = ObjectFifo(weights_ty, depth=1, name=f"inOF_wts_{c}_L3L2")
            of_wts_fifos.append(of_wts)

            # Separate outputs per core
            of_out = ObjectFifo(out_ty, name=f"out_{c}_L2")
            of_out_fifos.append(of_out)

            of_out_L3 = of_out.cons().forward(obj_type=bufOut_ty, name=f"outOF_{c}_L2L3")
            of_out_L3_fifos.append(of_out_L3)

    # Setup buffers to hold runtime parameters (one per core)
    rtps = []
    rtp_barriers = []
    for c in range(n_cores):
        rtp = Buffer(
            np.ndarray[(16,), np.dtype[np.int32]],
            name=f"rtp_{c}",
            use_write_rtp=True,
        )
        rtps.append(rtp)
        rtp_barriers.append(WorkerRuntimeBarrier())

    # Task for the core to perform
    # True 3D convolution with sliding window over depth dimension
    def core_fn(of_wts, of_act, of_out, my_rtp, conv3dk3_kernel, barrier, ch_offset):
        d_dim = depth
        x_dim = width
        h_dim = height
        ci = in_channels
        co = out_channels_per_core  # Per-core output channels

        barrier.wait_for_value(1)
        scale = my_rtp[0]

        elemWts = of_wts.acquire(1)

        # Handle different depth cases
        if d_dim == 1:
            # Special case: single depth plane
            plane = of_act.acquire(1)
            elemOut = of_out.acquire(1)
            # Use same plane for all 3 positions, but kernel won't use z-1 or z+1
            # Set check to skip both kd=0 and kd=2 (only use kd=1)
            # Since we can't do that with current check values, use check=0 which skips kd=0
            # and kernel_depth=1 to process only middle position
            conv3dk3_kernel(
                plane, plane, plane,
                elemWts, elemOut,
                x_dim, h_dim, ci, co,
                3, 3, 1,  # Use kernel_depth=1 for single plane
                1, scale, 0
            )
            of_act.release(1)
            of_out.release(1)

        elif d_dim == 2:
            # Two depth planes
            plane0 = of_act.acquire(1)
            plane1 = of_act.acquire(1)

            # d=0: first plane
            elemOut = of_out.acquire(1)
            conv3dk3_kernel(
                plane0, plane0, plane1,
                elemWts, elemOut,
                x_dim, h_dim, ci, co,
                3, 3, 3,
                0, scale, 0  # check=0 (skip z-1)
            )
            of_out.release(1)

            # d=1: last plane
            elemOut = of_out.acquire(1)
            conv3dk3_kernel(
                plane0, plane1, plane1,
                elemWts, elemOut,
                x_dim, h_dim, ci, co,
                3, 3, 3,
                2, scale, 0  # check=2 (skip z+1)
            )
            of_out.release(1)

            of_act.release(1)
            of_act.release(1)

        else:
            # General case: 3 or more depth planes with sliding window
            plane0 = of_act.acquire(1)
            plane1 = of_act.acquire(1)

            for d in range_(d_dim):
                elemOut = of_out.acquire(1)

                if d == 0:
                    # First plane: check=0 (skip z-1)
                    conv3dk3_kernel(
                        plane0, plane0, plane1,
                        elemWts, elemOut,
                        x_dim, h_dim, ci, co,
                        3, 3, 3, 0, scale, ch_offset
                    )

                elif d == d_dim - 1:
                    # Last plane: check=2 (skip z+1)
                    # At this point, plane0 and plane1 are the last two planes
                    conv3dk3_kernel(
                        plane0, plane1, plane1,
                        elemWts, elemOut,
                        x_dim, h_dim, ci, co,
                        3, 3, 3, 2, scale, ch_offset
                    )

                else:
                    # Middle planes: sliding window
                    plane2 = of_act.acquire(1)
                    conv3dk3_kernel(
                        plane0, plane1, plane2,
                        elemWts, elemOut,
                        x_dim, h_dim, ci, co,
                        3, 3, 3, 1, scale, ch_offset
                    )
                    # Slide window
                    of_act.release(1)
                    plane0 = plane1
                    plane1 = plane2

                of_out.release(1)

            # Release final two planes
            of_act.release(1)
            of_act.release(1)

        of_wts.release(1)

    # Create workers - one per core
    workers = []
    for c in range(n_cores):
        ch_offset_val = c * out_channels_per_core
        worker = Worker(
            core_fn,
            [
                of_wts_fifos[c].cons(),
                of_act_fifos[c].cons(),
                of_out_fifos[c].prod(),
                rtps[c],
                conv3dk3_kernel,
                rtp_barriers[c],
                ch_offset_val,  # Channel offset for this core
            ],
            stack_size=0x800,
            while_true=False,
        )
        workers.append(worker)

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()

    if n_cores == 1:
        # Single core runtime sequence
        with rt.sequence(tensorIn_ty, tensorWts_ty, tensorOut_ty) as (I, W, O):
            def set_rtps(*my_rtps):
                for rtp in my_rtps:
                    rtp[0] = 10
            rt.inline_ops(set_rtps, rtps)
            for c in range(n_cores):
                rt.set_barrier(rtp_barriers[c], 1)
            rt.start(workers[0])
            rt.fill(of_inOF_act_L3L2.prod(), I)
            rt.fill(of_wts_fifos[0].prod(), W)
            rt.drain(of_out_L3_fifos[0].cons(), O, wait=True)
    else:
        # Multi-core runtime sequence
        # For now: duplicate input I to each core (not true broadcast, but simpler)
        # Weights W is list of weight tensors (one per core)
        # Output O is list of output tensors (one per core)
        seq_inputs = [tensorIn_ty] * n_cores + tensorWts_ty
        seq_outputs = tensorOut_ty
        with rt.sequence(*seq_inputs, *seq_outputs) as args:
            I_cores = args[0:n_cores]  # Duplicated inputs
            W = args[n_cores:n_cores*2]
            O = args[n_cores*2:]

            def set_rtps(*my_rtps):
                for rtp in my_rtps:
                    rtp[0] = 10
            rt.inline_ops(set_rtps, rtps)
            for c in range(n_cores):
                rt.set_barrier(rtp_barriers[c], 1)

            # Start all workers
            for worker in workers:
                rt.start(worker)

            # Fill inputs (duplicated to each core's stream)
            for c in range(n_cores):
                # Each core gets full input (duplicated in software)
                rt.fill(of_inOF_act_L3L2_list[c].prod(), I_cores[c])

            # Fill weights (separate for each core)
            for c in range(n_cores):
                rt.fill(of_wts_fifos[c].prod(), W[c])

            # Drain outputs (separate from each core)
            for c in range(n_cores):
                wait = (c == n_cores - 1)  # Only wait on last core
                rt.drain(of_out_L3_fifos[c].cons(), O[c], wait=wait)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = NPU1Col1()
        n_cores = 1
    elif device_name == "npu2":
        dev = NPU2Col1()
        n_cores = 1
    elif device_name == "npu2_2col":
        dev = NPU2Col2()
        n_cores = 2
    elif device_name == "npu2_4col":
        dev = NPU2Col4()
        n_cores = 4
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
    trace_size = 0 if (len(sys.argv) != 8) else int(sys.argv[7])
except ValueError:
    print("Argument has inappropriate value")
module = conv3dk3(dev, depth, width, height, in_channels, out_channels, trace_size, n_cores)
print(module)
