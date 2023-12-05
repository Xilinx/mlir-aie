#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

import sys

from aie.ir import *
from aie.dialects.func import *
from aie.dialects.scf import *
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.extras import memref, arith
from aie.util import mlir_mod_ctx

width = 64
height = 36
if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])

heightMinus1 = height - 1
lineWidth = width
lineWidthInBytes = width * 4
lineWidthInInt32s = lineWidthInBytes // 4

enableTrace = False
traceSizeInBytes = 8192
traceSizeInInt32s = traceSizeInBytes // 4


def edge_detect():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
        def device_body():
            line_bytes_ty = T.memref(lineWidthInBytes, T.ui8())
            line_ty = T.memref(lineWidth, T.ui8())
            memRef_3x3_ty = T.memref(3, 3, T.i16())

            ofifo_line_bytes_ty = TypeAttr.get(ObjectFifoType.get(line_bytes_ty))
            ofifo_line_ty = TypeAttr.get(ObjectFifoType.get(line_ty))

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
            objectfifo(
                "inOF_L3L2",
                ShimTile,
                [MemTile],
                2,
                ofifo_line_bytes_ty,
                [],
                [],
            )
            objectfifo(
                "inOF_L2L1",
                MemTile,
                [ComputeTile2, ComputeTile5],
                [2, 2, 7],
                ofifo_line_bytes_ty,
                [],
                [],
            )
            objectfifo_link(["inOF_L3L2"], ["inOF_L2L1"])

            # Output
            objectfifo(
                "outOF_L2L3",
                MemTile,
                [ShimTile],
                2,
                ofifo_line_bytes_ty,
                [],
                [],
            )
            objectfifo(
                "outOF_L1L2",
                ComputeTile5,
                [MemTile],
                2,
                ofifo_line_bytes_ty,
                [],
                [],
            )
            objectfifo_link(["outOF_L1L2"], ["outOF_L2L3"])

            # Intermediate
            objectfifo(
                "OF_2to3",
                ComputeTile2,
                [ComputeTile3],
                4,
                ofifo_line_ty,
                [],
                [],
            )
            objectfifo(
                "OF_3to4",
                ComputeTile3,
                [ComputeTile4],
                2,
                ofifo_line_ty,
                [],
                [],
            )
            objectfifo(
                "OF_4to5",
                ComputeTile4,
                [ComputeTile5],
                2,
                ofifo_line_ty,
                [],
                [],
            )
            objectfifo(
                "OF_5to5",
                ComputeTile5,
                [ComputeTile5],
                1,
                ofifo_line_bytes_ty,
                [],
                [],
            )

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2, "rgba2gray.cc.o")
            def core_body():
                for _ in for_(4294967295):
                    # for _ in for_(36):
                    elem_in = acquire(
                        ObjectFifoPort.Consume, "inOF_L2L1", 1, line_bytes_ty
                    ).acquired_elem()
                    elem_out = acquire(
                        ObjectFifoPort.Produce, "OF_2to3", 1, line_ty
                    ).acquired_elem()

                    Call(rgba2gray_line, [elem_in, elem_out, arith.constant(lineWidth)])

                    objectfifo_release(ObjectFifoPort.Consume, "inOF_L2L1", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "OF_2to3", 1)
                    yield_([])

            # Compute tile 3
            @core(ComputeTile3, "filter2d.cc.o")
            def core_body():
                kernel = memref.alloc([3, 3], T.i16())
                v0 = arith.constant(0, T.i16())
                v1 = arith.constant(4096, T.i16())
                v_minus4 = arith.constant(-16384, T.i16())
                memref.store(v0, kernel, [0, 0])
                memref.store(v1, kernel, [0, 1])
                memref.store(v0, kernel, [0, 2])
                memref.store(v1, kernel, [1, 0])
                memref.store(v_minus4, kernel, [1, 1])
                memref.store(v1, kernel, [1, 2])
                memref.store(v0, kernel, [2, 0])
                memref.store(v1, kernel, [2, 1])
                memref.store(v0, kernel, [2, 2])

                # Preamble : Top Border
                elems_in_pre = acquire(
                    ObjectFifoPort.Consume, "OF_2to3", 2, line_ty
                ).acquired_elem()
                elem_pre_out = acquire(
                    ObjectFifoPort.Produce, "OF_3to4", 1, line_ty
                ).acquired_elem()
                Call(
                    filter2d_line,
                    [
                        elems_in_pre[0],
                        elems_in_pre[0],
                        elems_in_pre[1],
                        elem_pre_out,
                        arith.constant(lineWidth),
                        kernel,
                    ],
                )
                objectfifo_release(ObjectFifoPort.Produce, "OF_3to4", 1)

                # Steady State : Middle
                for _ in for_(1, heightMinus1):
                    elems_in = acquire(
                        ObjectFifoPort.Consume, "OF_2to3", 3, line_ty
                    ).acquired_elem()
                    elem_out = acquire(
                        ObjectFifoPort.Produce, "OF_3to4", 1, line_ty
                    ).acquired_elem()
                    Call(
                        filter2d_line,
                        [
                            elems_in[0],
                            elems_in[1],
                            elems_in[2],
                            elem_out,
                            arith.constant(lineWidth),
                            kernel,
                        ],
                    )
                    objectfifo_release(ObjectFifoPort.Consume, "OF_2to3", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "OF_3to4", 1)
                    yield_([])

                # Postamble : Bottom Border
                elems_in_post = acquire(
                    ObjectFifoPort.Consume, "OF_2to3", 2, line_ty
                ).acquired_elem()
                elem_post_out = acquire(
                    ObjectFifoPort.Produce, "OF_3to4", 1, line_ty
                ).acquired_elem()
                Call(
                    filter2d_line,
                    [
                        elems_in_post[0],
                        elems_in_post[1],
                        elems_in_post[1],
                        elem_post_out,
                        arith.constant(lineWidth),
                        kernel,
                    ],
                )
                objectfifo_release(ObjectFifoPort.Consume, "OF_2to3", 2)
                objectfifo_release(ObjectFifoPort.Produce, "OF_3to4", 1)

            # Compute tile 4
            @core(ComputeTile4, "threshold.cc.o")
            def core_body():
                v_thr = arith.constant(10, T.i16())
                v_max = arith.constant(255, T.i16())
                v_typ = arith.constant(0, T.i8())

                for _ in for_(36):
                    elem_in = acquire(
                        ObjectFifoPort.Consume, "OF_3to4", 1, line_ty
                    ).acquired_elem()
                    elem_out = acquire(
                        ObjectFifoPort.Produce, "OF_4to5", 1, line_ty
                    ).acquired_elem()

                    Call(
                        threshold_line,
                        [
                            elem_in,
                            elem_out,
                            arith.constant(lineWidth),
                            v_thr,
                            v_max,
                            v_typ,
                        ],
                    )

                    objectfifo_release(ObjectFifoPort.Consume, "OF_3to4", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "OF_4to5", 1)
                    yield_([])

            # Compute tile 5
            @core(ComputeTile5, "combined_gray2rgba_addWeighted.a")
            def core_body():
                for _ in for_(36):
                    elem_in = acquire(
                        ObjectFifoPort.Consume, "OF_4to5", 1, line_ty
                    ).acquired_elem()
                    elem_out = acquire(
                        ObjectFifoPort.Produce, "OF_5to5", 1, line_bytes_ty
                    ).acquired_elem()

                    Call(gray2rgba_line, [elem_in, elem_out, arith.constant(lineWidth)])

                    objectfifo_release(ObjectFifoPort.Consume, "OF_4to5", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "OF_5to5", 1)

                    elem_in1 = acquire(
                        ObjectFifoPort.Consume, "OF_5to5", 1, line_bytes_ty
                    ).acquired_elem()
                    elem_in2 = acquire(
                        ObjectFifoPort.Consume, "inOF_L2L1", 1, line_bytes_ty
                    ).acquired_elem()
                    elem_out2 = acquire(
                        ObjectFifoPort.Produce, "outOF_L1L2", 1, line_bytes_ty
                    ).acquired_elem()

                    alpha = arith.constant(16384, T.i16())
                    beta = arith.constant(16384, T.i16())
                    gamma = arith.constant(0, T.i8())

                    Call(
                        add_weighted_line,
                        [
                            elem_in1,
                            elem_in2,
                            elem_out2,
                            arith.constant(lineWidthInBytes),
                            alpha,
                            beta,
                            gamma,
                        ],
                    )

                    objectfifo_release(ObjectFifoPort.Consume, "OF_5to5", 1)
                    objectfifo_release(ObjectFifoPort.Consume, "inOF_L2L1", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "outOF_L1L2", 1)
                    yield_([])

            # To/from AIE-array data movement

            tensorSize = width * height * 4  # 4 channels
            tensorSizeInInt32s = tensorSize // 4
            tensor_ty = T.memref(tensorSizeInInt32s, T.i32())
            memRef_16x16_ty = T.memref(16, 16, T.i32())

            @FuncOp.from_py_func(tensor_ty, memRef_16x16_ty, tensor_ty)
            def sequence(I, B, O):
                ipu_dma_memcpy_nd(
                    metadata="outOF_L2L3",
                    bd_id=0,
                    mem=O,
                    lengths=[1, 1, 1, tensorSizeInInt32s],
                )
                ipu_dma_memcpy_nd(
                    metadata="inOF_L3L2",
                    bd_id=1,
                    mem=I,
                    lengths=[1, 1, 1, tensorSizeInInt32s],
                )
                ipu_sync(column=0, row=0, direction=0, channel=0)

    #    print(ctx.module.operation.verify())
    print(ctx.module)


edge_detect()
