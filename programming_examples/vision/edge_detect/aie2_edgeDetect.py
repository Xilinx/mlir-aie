#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.dialects.ext import memref
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext.scf import _for as range_

width = 64
height = 36
if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])

heightMinus1 = height - 1
lineWidth = width
lineWidthInBytes = width * 4

enableTrace = False
traceSizeInBytes = 8192
traceSizeInInt32s = traceSizeInBytes // 4


def edge_detect():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():
            line_bytes_ty = T.memref(lineWidthInBytes, T.ui8())
            line_ty = T.memref(lineWidth, T.ui8())
            memRef_3x3_ty = T.memref(3, 3, T.i16())

            # AIE Core Function declarations
            rgba2gray_line = external_func(
                "rgba2grayLine", inputs=[line_bytes_ty, line_ty, T.i32()]
            )
            filter2d_line = external_func(
                "filter2dLine",
                inputs=[line_ty, line_ty, line_ty, line_ty, T.i32(), memRef_3x3_ty],
            )
            threshold_line = external_func(
                "thresholdLine",
                inputs=[line_ty, line_ty, T.i32(), T.i16(), T.i16(), T.i8()],
            )
            gray2rgba_line = external_func(
                "gray2rgbaLine", inputs=[line_ty, line_bytes_ty, T.i32()]
            )
            add_weighted_line = external_func(
                "addWeightedLine",
                inputs=[
                    line_bytes_ty,
                    line_bytes_ty,
                    line_bytes_ty,
                    T.i32(),
                    T.i16(),
                    T.i16(),
                    T.i8(),
                ],
            )

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)
            ComputeTile3 = tile(0, 3)
            ComputeTile4 = tile(0, 4)
            ComputeTile5 = tile(0, 5)

            # AIE-array data movement with object fifos
            # Input
            inOF_L3L2 = object_fifo(
                "inOF_L3L2",
                ShimTile,
                [ComputeTile2, MemTile],
                [2, 2, 7],
                line_bytes_ty,
            )
            inOF_L2L1 = object_fifo(
                "inOF_L2L1",
                MemTile,
                ComputeTile5,
                7,
                line_bytes_ty,
            )
            object_fifo_link(inOF_L3L2, inOF_L2L1)

            # Output
            outOF_L2L3 = object_fifo(
                "outOF_L2L3",
                MemTile,
                ShimTile,
                2,
                line_bytes_ty,
            )
            outOF_L1L2 = object_fifo(
                "outOF_L1L2",
                ComputeTile5,
                MemTile,
                2,
                line_bytes_ty,
            )
            object_fifo_link(outOF_L1L2, outOF_L2L3)

            # Intermediate
            OF_2to3 = object_fifo(
                "OF_2to3",
                ComputeTile2,
                ComputeTile3,
                4,
                line_ty,
            )
            OF_3to4 = object_fifo(
                "OF_3to4",
                ComputeTile3,
                ComputeTile4,
                2,
                line_ty,
            )
            OF_4to5 = object_fifo(
                "OF_4to5",
                ComputeTile4,
                ComputeTile5,
                2,
                line_ty,
            )
            OF_5to5 = object_fifo(
                "OF_5to5",
                ComputeTile5,
                ComputeTile5,
                1,
                line_bytes_ty,
            )

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2, "rgba2gray.cc.o")
            def core_body():
                for _ in range_(sys.maxsize):
                    elem_in = inOF_L3L2.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = OF_2to3.acquire(ObjectFifoPort.Produce, 1)

                    rgba2gray_line(elem_in, elem_out, lineWidth)

                    inOF_L3L2.release(ObjectFifoPort.Consume, 1)
                    OF_2to3.release(ObjectFifoPort.Produce, 1)

            # Compute tile 3
            @core(ComputeTile3, "filter2d.cc.o")
            def core_body():
                kernel = memref.alloc(3, 3, T.i16())
                v0 = 0
                v1 = 4096
                v_minus4 = -16384
                kernel[0, 0] = v0
                kernel[0, 1] = v1
                kernel[0, 2] = v0
                kernel[1, 0] = v1
                kernel[1, 1] = v_minus4
                kernel[1, 2] = v1
                kernel[2, 0] = v0
                kernel[2, 1] = v1
                kernel[2, 2] = v0

                for _ in range_(sys.maxsize):
                    # Preamble : Top Border
                    elems_in_pre = OF_2to3.acquire(ObjectFifoPort.Consume, 2)
                    elem_pre_out = OF_3to4.acquire(ObjectFifoPort.Produce, 1)
                    filter2d_line(
                        elems_in_pre[0],
                        elems_in_pre[0],
                        elems_in_pre[1],
                        elem_pre_out,
                        lineWidth,
                        kernel,
                    )
                    OF_3to4.release(ObjectFifoPort.Produce, 1)

                    # Steady State : Middle
                    for _ in range_(1, heightMinus1):
                        elems_in = OF_2to3.acquire(ObjectFifoPort.Consume, 3)
                        elem_out = OF_3to4.acquire(ObjectFifoPort.Produce, 1)
                        filter2d_line(
                            elems_in[0],
                            elems_in[1],
                            elems_in[2],
                            elem_out,
                            lineWidth,
                            kernel,
                        )
                        OF_2to3.release(ObjectFifoPort.Consume, 1)
                        OF_3to4.release(ObjectFifoPort.Produce, 1)

                    # Postamble : Bottom Border
                    elems_in_post = OF_2to3.acquire(ObjectFifoPort.Consume, 2)
                    elem_post_out = OF_3to4.acquire(ObjectFifoPort.Produce, 1)
                    filter2d_line(
                        elems_in_post[0],
                        elems_in_post[1],
                        elems_in_post[1],
                        elem_post_out,
                        lineWidth,
                        kernel,
                    )
                    OF_2to3.release(ObjectFifoPort.Consume, 2)
                    OF_3to4.release(ObjectFifoPort.Produce, 1)

            # Compute tile 4
            @core(ComputeTile4, "threshold.cc.o")
            def core_body():
                v_thr = 10
                v_max = 255
                v_typ = 0

                for _ in range_(sys.maxsize):
                    elem_in = OF_3to4.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = OF_4to5.acquire(ObjectFifoPort.Produce, 1)

                    threshold_line(elem_in, elem_out, lineWidth, v_thr, v_max, v_typ)

                    OF_3to4.release(ObjectFifoPort.Consume, 1)
                    OF_4to5.release(ObjectFifoPort.Produce, 1)

            # Compute tile 5
            @core(ComputeTile5, "combined_gray2rgba_addWeighted.a")
            def core_body():
                for _ in range_(sys.maxsize):
                    elem_in = OF_4to5.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = OF_5to5.acquire(ObjectFifoPort.Produce, 1)

                    gray2rgba_line(elem_in, elem_out, lineWidth)

                    OF_4to5.release(ObjectFifoPort.Consume, 1)
                    OF_5to5.release(ObjectFifoPort.Produce, 1)

                    elem_in1 = OF_5to5.acquire(ObjectFifoPort.Consume, 1)
                    elem_in2 = inOF_L2L1.acquire(ObjectFifoPort.Consume, 1)
                    elem_out2 = outOF_L1L2.acquire(ObjectFifoPort.Produce, 1)

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

                    OF_5to5.release(ObjectFifoPort.Consume, 1)
                    inOF_L2L1.release(ObjectFifoPort.Consume, 1)
                    outOF_L1L2.release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement

            tensorSize = width * height * 4  # 4 channels
            tensor_ty = T.memref(tensorSize, T.i8())
            memRef_16x16_ty = T.memref(16, 16, T.i32())

            @runtime_sequence(tensor_ty, memRef_16x16_ty, tensor_ty)
            def sequence(I, B, O):
                npu_dma_memcpy_nd(
                    metadata=inOF_L3L2,
                    bd_id=1,
                    mem=I,
                    sizes=[1, 1, 1, tensorSize],
                )
                npu_dma_memcpy_nd(
                    metadata=outOF_L2L3,
                    bd_id=0,
                    mem=O,
                    sizes=[1, 1, 1, tensorSize],
                )
                # outOF_L2L3 will only complete after inOF_L3L2 completes, so we just wait on outOF_L2L3 instead of all
                dma_wait(outOF_L2L3)

    #    print(ctx.module.operation.verify())
    print(ctx.module)


edge_detect()
