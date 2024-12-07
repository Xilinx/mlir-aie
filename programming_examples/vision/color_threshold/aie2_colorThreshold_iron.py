#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
import numpy as np
import sys

from aie.iron.runtime import GlobalBuffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1

from aie.extras.dialects.ext import arith
from aie.helpers.util import np_ndarray_type_get_shape
from aie.dialects.aie import T


def color_threshold(dev, width, height):
    lineWidth = width
    lineWidthChannels = width * 4  # 4 channels
    tensorSize = width * height

    tensor_ty = np.ndarray[(tensorSize,), np.dtype[np.int8]]
    line_channels_ty = np.ndarray[(lineWidthChannels,), np.dtype[np.uint8]]
    line_ty = np.ndarray[(lineWidth,), np.dtype[np.uint8]]
    unused_ty = np.ndarray[(32,), np.dtype[np.int32]]  # not used

    # AIE Core Function declarations
    thresholdLine = Kernel(
        "thresholdLine",
        "threshold.cc.o",
        [line_ty, line_ty, np.int32, np.int16, np.int16, np.int8],
    )

    # AIE-array data movement with object fifos
    # Input RGBA broadcast + memtile for skip
    inOOB_L3L2 = ObjectFifo(line_channels_ty, name="inOOB_L3L2")
    of_offsets = [np.prod(np_ndarray_type_get_shape(line_ty)) * i for i in range(4)]
    in00B_L2L1s = inOOB_L3L2.cons().split(
        of_offsets, obj_types=[line_ty] * 4, names=[f"inOOB_L2L1_{i}" for i in range(4)]
    )

    # Output RGBA
    outOOB_L2L3 = ObjectFifo(line_channels_ty, name="outOOB_L2L3")
    outOOB_L1L2s = outOOB_L2L3.prod().join(
        of_offsets,
        obj_types=[line_ty] * 4,
        names=[f"outOOB_L1L2_{i}" for i in range(4)],
    )

    # Runtime parameters
    rtps = []
    for i in range(4):
        rtps.append(
            GlobalBuffer(
                np.ndarray[(16,), np.dtype[np.int32]],
                name=f"rtp{i}",
                use_write_rtp=True,
            )
        )

    # Compute tile 2
    def core_fn(of_in, of_out, my_rtp, threshold_fn):
        elemIn = of_in.acquire(1)
        elemOut = of_out.acquire(1)

        # RTPs written from the instruction stream must be read right before the kernel
        # after the ObjectFIFO acquires
        thresholdValue = arith.trunci(T.i16(), my_rtp[0])
        maxValue = arith.trunci(T.i16(), my_rtp[1])
        thresholdType = arith.trunci(T.i8(), my_rtp[2])
        threshold_fn(
            elemIn,
            elemOut,
            lineWidth,
            thresholdValue,
            maxValue,
            thresholdType,
        )
        of_in.release(1)
        of_out.release(1)

    workers = []
    for i in range(4):
        workers.append(
            Worker(
                core_fn,
                [in00B_L2L1s[i].cons(), outOOB_L1L2s[i].prod(), rtps[i], thresholdLine],
            )
        )

    rt = Runtime()
    with rt.sequence(tensor_ty, unused_ty, tensor_ty) as (inTensor, _, outTensor):

        def set_rtps(*args):
            for rtp in args:
                rtp[0] = 50
                rtp[1] = 255
                rtp[2] = 0

        rt.inline_ops(set_rtps, rtps)
        rt.start(*workers)
        rt.fill(inOOB_L3L2.prod(), inTensor)
        rt.drain(outOOB_L2L3.cons(), outTensor, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())


try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = NPU1Col1()
    elif device_name == "npu2":
        raise NotImplemented("Not implemented yet")
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
    width = 512 if (len(sys.argv) != 4) else int(sys.argv[2])
    height = 9 if (len(sys.argv) != 4) else int(sys.argv[3])
except ValueError:
    print("Argument has inappropriate value")
module = color_threshold(dev, width, height)
print(module)
