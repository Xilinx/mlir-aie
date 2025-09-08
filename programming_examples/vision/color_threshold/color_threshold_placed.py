#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.dialects.ext import arith
from aie.extras.context import mlir_mod_ctx
from aie.helpers.util import np_ndarray_type_get_shape
from aie.helpers.dialects.ext.scf import _for as range_


def color_threshold(dev, width, height):
    lineWidth = width
    lineWidthChannels = width * 4  # 4 channels
    tensorSize = width * height

    @device(dev)
    def device_body():
        line_channels_ty = np.ndarray[(lineWidthChannels,), np.dtype[np.uint8]]
        line_ty = np.ndarray[(lineWidth,), np.dtype[np.uint8]]

        # AIE Core Function declarations
        thresholdLine = external_func(
            "thresholdLine",
            inputs=[line_ty, line_ty, np.int32, np.int16, np.int16, np.int8],
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTile2 = tile(0, 2)
        ComputeTile3 = tile(0, 3)
        ComputeTile4 = tile(0, 4)
        ComputeTile5 = tile(0, 5)

        # Lock declarations
        lock2 = lock(ComputeTile2, init=0)
        lock3 = lock(ComputeTile3, init=0)
        lock4 = lock(ComputeTile4, init=0)
        lock5 = lock(ComputeTile5, init=0)

        # AIE-array data movement with object fifos

        # Input RGBA broadcast + memtile for skip
        inOOB_L3L2 = object_fifo("inOOB_L3L2", ShimTile, MemTile, 2, line_channels_ty)
        inOOB_L2L1_0 = object_fifo("inOOB_L2L1_0", MemTile, ComputeTile2, 2, line_ty)
        inOOB_L2L1_1 = object_fifo("inOOB_L2L1_1", MemTile, ComputeTile3, 2, line_ty)
        inOOB_L2L1_2 = object_fifo("inOOB_L2L1_2", MemTile, ComputeTile4, 2, line_ty)
        inOOB_L2L1_3 = object_fifo("inOOB_L2L1_3", MemTile, ComputeTile5, 2, line_ty)
        of_offsets = [np.prod(np_ndarray_type_get_shape(line_ty)) * i for i in range(4)]
        object_fifo_link(
            inOOB_L3L2,
            [inOOB_L2L1_0, inOOB_L2L1_1, inOOB_L2L1_2, inOOB_L2L1_3],
            [],
            of_offsets,
        )

        # Output RGBA
        outOOB_L2L3 = object_fifo("outOOB_L2L3", MemTile, ShimTile, 2, line_channels_ty)
        outOOB_L1L2_0 = object_fifo("outOOB_L1L2_0", ComputeTile2, MemTile, 2, line_ty)
        outOOB_L1L2_1 = object_fifo("outOOB_L1L2_1", ComputeTile3, MemTile, 2, line_ty)
        outOOB_L1L2_2 = object_fifo("outOOB_L1L2_2", ComputeTile4, MemTile, 2, line_ty)
        outOOB_L1L2_3 = object_fifo("outOOB_L1L2_3", ComputeTile5, MemTile, 2, line_ty)
        object_fifo_link(
            [outOOB_L1L2_0, outOOB_L1L2_1, outOOB_L1L2_2, outOOB_L1L2_3],
            outOOB_L2L3,
            of_offsets,
            [],
        )

        # Runtime parameters
        rtpComputeTile2 = buffer(
            ComputeTile2,
            np.ndarray[(16,), np.dtype[np.int32]],
            "rtpComputeTile2",
            use_write_rtp=True,
        )
        rtpComputeTile3 = buffer(
            ComputeTile3,
            np.ndarray[(16,), np.dtype[np.int32]],
            "rtpComputeTile3",
            use_write_rtp=True,
        )
        rtpComputeTile4 = buffer(
            ComputeTile4,
            np.ndarray[(16,), np.dtype[np.int32]],
            "rtpComputeTile4",
            use_write_rtp=True,
        )
        rtpComputeTile5 = buffer(
            ComputeTile5,
            np.ndarray[(16,), np.dtype[np.int32]],
            "rtpComputeTile5",
            use_write_rtp=True,
        )

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2, "threshold.cc.o")
        def core_body():
            for _ in range_(sys.maxsize):
                # RTPs written from the instruction stream must be synchronized with the runtime sequence
                # This may be done through the usage of locks
                use_lock(lock2, LockAction.Acquire, value=1)
                thresholdValue = arith.trunci(T.i16(), rtpComputeTile2[0])
                maxValue = arith.trunci(T.i16(), rtpComputeTile2[1])
                thresholdType = arith.trunci(T.i8(), rtpComputeTile2[2])

                elemIn = inOOB_L2L1_0.acquire(ObjectFifoPort.Consume, 1)
                elemOut = outOOB_L1L2_0.acquire(ObjectFifoPort.Produce, 1)

                thresholdLine(
                    elemIn,
                    elemOut,
                    lineWidth,
                    thresholdValue,
                    maxValue,
                    thresholdType,
                )

                inOOB_L2L1_0.release(ObjectFifoPort.Consume, 1)
                outOOB_L1L2_0.release(ObjectFifoPort.Produce, 1)

        # Compute tile 3
        @core(ComputeTile3, "threshold.cc.o")
        def core_body():
            for _ in range_(sys.maxsize):
                # RTPs written from the instruction stream must be synchronized with the runtime sequence
                # This may be done through the usage of locks
                use_lock(lock3, LockAction.Acquire, value=1)
                thresholdValue = arith.trunci(T.i16(), rtpComputeTile3[0])
                maxValue = arith.trunci(T.i16(), rtpComputeTile3[1])
                thresholdType = arith.trunci(T.i8(), rtpComputeTile3[2])

                elemIn = inOOB_L2L1_1.acquire(ObjectFifoPort.Consume, 1)
                elemOut = outOOB_L1L2_1.acquire(ObjectFifoPort.Produce, 1)

                thresholdLine(
                    elemIn,
                    elemOut,
                    lineWidth,
                    thresholdValue,
                    maxValue,
                    thresholdType,
                )

                inOOB_L2L1_1.release(ObjectFifoPort.Consume, 1)
                outOOB_L1L2_1.release(ObjectFifoPort.Produce, 1)

        # Compute tile 4
        @core(ComputeTile4, "threshold.cc.o")
        def core_body():
            for _ in range_(sys.maxsize):
                # RTPs written from the instruction stream must be synchronized with the runtime sequence
                # This may be done explicitly through the usage of locks
                use_lock(lock4, LockAction.Acquire, value=1)
                thresholdValue = arith.trunci(T.i16(), rtpComputeTile4[0])
                maxValue = arith.trunci(T.i16(), rtpComputeTile4[1])
                thresholdType = arith.trunci(T.i8(), rtpComputeTile4[2])

                elemIn = inOOB_L2L1_2.acquire(ObjectFifoPort.Consume, 1)
                elemOut = outOOB_L1L2_2.acquire(ObjectFifoPort.Produce, 1)

                thresholdLine(
                    elemIn,
                    elemOut,
                    lineWidth,
                    thresholdValue,
                    maxValue,
                    thresholdType,
                )

                inOOB_L2L1_2.release(ObjectFifoPort.Consume, 1)
                outOOB_L1L2_2.release(ObjectFifoPort.Produce, 1)

        # Compute tile 5
        @core(ComputeTile5, "threshold.cc.o")
        def core_body():
            for _ in range_(sys.maxsize):
                # RTPs written from the instruction stream must be synchronized with the runtime sequence
                # This may be done explicitly through the usage of locks
                use_lock(lock5, LockAction.Acquire, value=1)
                thresholdValue = arith.trunci(T.i16(), rtpComputeTile5[0])
                maxValue = arith.trunci(T.i16(), rtpComputeTile5[1])
                thresholdType = arith.trunci(T.i8(), rtpComputeTile5[2])

                elemIn = inOOB_L2L1_3.acquire(ObjectFifoPort.Consume, 1)
                elemOut = outOOB_L1L2_3.acquire(ObjectFifoPort.Produce, 1)

                thresholdLine(
                    elemIn,
                    elemOut,
                    lineWidth,
                    thresholdValue,
                    maxValue,
                    thresholdType,
                )

                inOOB_L2L1_3.release(ObjectFifoPort.Consume, 1)
                outOOB_L1L2_3.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(
            np.ndarray[(tensorSize,), np.dtype[np.int8]],
            np.ndarray[(32,), np.dtype[np.int32]],  # not used
            np.ndarray[(tensorSize,), np.dtype[np.int8]],
        )
        def sequence(inTensor, notUsed, outTensor):
            # thresholdValue, maxValue, thresholdType
            rtpComputeTile2[0] = 50
            rtpComputeTile2[1] = 255
            rtpComputeTile2[2] = 0
            set_lock(lock2, 1)

            rtpComputeTile3[0] = 50
            rtpComputeTile3[1] = 255
            rtpComputeTile3[2] = 0
            set_lock(lock3, 1)

            rtpComputeTile4[0] = 50
            rtpComputeTile4[1] = 255
            rtpComputeTile4[2] = 0
            set_lock(lock4, 1)

            rtpComputeTile5[0] = 50
            rtpComputeTile5[1] = 255
            rtpComputeTile5[2] = 0
            set_lock(lock5, 1)

            in_task = shim_dma_single_bd_task(
                inOOB_L3L2, inTensor, sizes=[1, 1, 1, tensorSize], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                outOOB_L2L3,
                outTensor,
                sizes=[1, 1, 1, tensorSize],
                issue_token=True,
            )

            dma_start_task(in_task, out_task)
            dma_await_task(in_task, out_task)


try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = AIEDevice.npu1_1col
    elif device_name == "npu2":
        dev = AIEDevice.npu2
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
    width = 512 if (len(sys.argv) != 4) else int(sys.argv[2])
    height = 9 if (len(sys.argv) != 4) else int(sys.argv[3])
except ValueError:
    print("Argument has inappropriate value")
with mlir_mod_ctx() as ctx:
    # print(ctx.module.operation.verify())
    color_threshold(dev, width, height)
    print(ctx.module)
