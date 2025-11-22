#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
import numpy as np
import sys

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2
from aie.iron.controlflow import range_


def edge_detect(dev, width, height):
    heightMinus1 = height - 1
    lineWidth = width
    lineWidthInBytes = width * 4
    tensorSize = width * height * 4  # 4 channels

    # Type definitions
    line_bytes_ty = np.ndarray[(lineWidthInBytes,), np.dtype[np.uint8]]
    line_ty = np.ndarray[(lineWidth,), np.dtype[np.uint8]]
    tensor_3x3_ty = np.ndarray[(3, 3), np.dtype[np.int16]]
    tensor_ty = np.ndarray[(tensorSize,), np.dtype[np.int8]]
    tensor_16x16_ty = np.ndarray[(16, 16), np.dtype[np.int32]]

    # AIE Core Function declarations
    rgba2gray_line_kernel = Kernel(
        "rgba2grayLine", "rgba2gray.cc.o", [line_bytes_ty, line_ty, np.int32]
    )
    filter2d_line_kernel = Kernel(
        "filter2dLine",
        "filter2d.cc.o",
        [line_ty, line_ty, line_ty, line_ty, np.int32, tensor_3x3_ty],
    )
    threshold_line_kernel = Kernel(
        "thresholdLine",
        "threshold.cc.o",
        [line_ty, line_ty, np.int32, np.int16, np.int16, np.int8],
    )
    gray2rgba_line_kernel = Kernel(
        "gray2rgbaLine",
        "combined_gray2rgba_addWeighted.a",
        [line_ty, line_bytes_ty, np.int32],
    )
    add_weighted_line_kernel = Kernel(
        "addWeightedLine",
        "combined_gray2rgba_addWeighted.a",
        [
            line_bytes_ty,
            line_bytes_ty,
            line_bytes_ty,
            np.int32,
            np.int16,
            np.int16,
            np.int8,
        ],
    )

    # AIE-array data movement with object fifos
    # Input
    inOF_L3L2 = ObjectFifo(line_bytes_ty, name="inOF_L3L2")
    inOF_L2L1 = inOF_L3L2.cons(7).forward(depth=7, name="inOF_L2L1")

    # Output
    outOF_L1L2 = ObjectFifo(line_bytes_ty, name="outOF_L1L2")
    outOF_L2L3 = outOF_L1L2.cons().forward(name="outOF_L2L3")

    # Intermediate
    depths = [4, 2, 2]
    of_intermediates = [
        ObjectFifo(line_ty, depth=depths[i], name=f"OF_{i + 2}to{i + 3}")
        for i in range(3)
    ]
    of_local = ObjectFifo(line_bytes_ty, depth=1, name="OF_local")

    workers = []

    # Task for the core to perform
    def rgba2gray_fn(of_in, of_out, rgba2gray_line):
        # inOF_L3L2
        # OF_2to3 -> of_intermediates[0]
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        rgba2gray_line(elem_in, elem_out, lineWidth)
        of_in.release(1)
        of_out.release(1)

    # Worker to run the task
    workers.append(
        Worker(
            rgba2gray_fn,
            [inOF_L3L2.cons(), of_intermediates[0].prod(), rgba2gray_line_kernel],
        )
    )

    filter_kernel_buff = Buffer(
        np.ndarray[(3, 3), np.dtype[np.int16]],
        name="kernel",
        initial_value=np.array(
            [[v0, v1, v0], [v1, v_minus4, v1], [v0, v1, v0]], dtype=np.int16
        ),
    )

    # Task for the core to perform
    def filter_fn(of_in, of_out, kernel, filter2d_line):
        # OF_2to3 -> intermediates[0]
        # OF_3to4 -> intermediates[1]
        v0 = 0
        v1 = 4096
        v_minus4 = -16384

        for _ in range_(sys.maxsize):
            # Preamble : Top Border
            elems_in_pre = of_in.acquire(2)
            elem_pre_out = of_out.acquire(1)
            filter2d_line(
                elems_in_pre[0],
                elems_in_pre[0],
                elems_in_pre[1],
                elem_pre_out,
                lineWidth,
                kernel,
            )
            of_out.release(1)

            # Steady State : Middle
            for _ in range_(1, heightMinus1):
                elems_in = of_in.acquire(3)
                elem_out = of_out.acquire(1)
                filter2d_line(
                    elems_in[0],
                    elems_in[1],
                    elems_in[2],
                    elem_out,
                    lineWidth,
                    kernel,
                )
                of_in.release(1)
                of_out.release(1)

            # Postamble : Bottom Border
            elems_in_post = of_in.acquire(2)
            elem_post_out = of_out.acquire(1)
            filter2d_line(
                elems_in_post[0],
                elems_in_post[1],
                elems_in_post[1],
                elem_post_out,
                lineWidth,
                kernel,
            )
            of_in.release(2)
            of_out.release(1)

    # Worker to run the task
    workers.append(
        Worker(
            filter_fn,
            [
                of_intermediates[0].cons(),
                of_intermediates[1].prod(),
                filter2d_line_kernel,
            ],
            while_true=False,
        )
    )

    # Task for the core to perform
    def threshold_fn(of_in, of_out, threshold_line):
        v_thr = 10
        v_max = 255
        v_typ = 0

        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        threshold_line(elem_in, elem_out, lineWidth, v_thr, v_max, v_typ)
        of_in.release(1)
        of_out.release(1)

    # Worker to run the task
    workers.append(
        Worker(
            threshold_fn,
            [
                of_intermediates[1].cons(),
                of_intermediates[2].prod(),
                threshold_line_kernel,
            ],
        )
    )

    # Task for the core to perform
    def gray2rgba_addWeight_fn(
        of_in,
        of_in2,
        if_out_self,
        of_in_self,
        of_out,
        gray2rgba_line,
        add_weighted_line,
    ):
        elem_in = of_in.acquire(1)
        elem_out = if_out_self.acquire(1)

        gray2rgba_line(elem_in, elem_out, lineWidth)

        of_in.release(1)
        if_out_self.release(1)

        elem_in1 = of_in_self.acquire(1)
        elem_in2 = of_in2.acquire(1)
        elem_out2 = of_out.acquire(1)

        alpha = 16384
        beta = 16384
        gamma = 0

        add_weighted_line(
            elem_in1,
            elem_in2,
            elem_out2,
            lineWidthInBytes,
            alpha,
            beta,
            gamma,
        )

        of_in_self.release(1)
        of_in2.release(1)
        of_out.release(1)

    # Worker to run the task
    workers.append(
        Worker(
            gray2rgba_addWeight_fn,
            [
                of_intermediates[2].cons(),
                inOF_L2L1.cons(),
                of_local.prod(),
                of_local.cons(),
                outOF_L1L2.prod(),
                gray2rgba_line_kernel,
                add_weighted_line_kernel,
            ],
        )
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_16x16_ty, tensor_ty) as (I, _B, O):
        rt.start(*workers)
        rt.fill(inOF_L3L2.prod(), I)
        rt.drain(outOF_L2L3.cons(), O, wait=True)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = NPU1Col1()
    elif device_name == "npu2":
        dev = NPU2()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
    width = 36 if (len(sys.argv) != 4) else int(sys.argv[2])
    height = 64 if (len(sys.argv) != 4) else int(sys.argv[3])
except ValueError:
    print("Argument has inappropriate value")
module = edge_detect(dev, width, height)
print(module)
