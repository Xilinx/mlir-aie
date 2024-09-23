#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext.scf import _for as range_

width = 32
height = 32
in_channels = 64
out_channels = 64

if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])


actIn = width * in_channels  # 32*64 = 2048
bufIn = actIn * 2  # double buffer

weights = in_channels * out_channels

actOut = width * out_channels  # 32*64 = 2048
bufOut = actOut * 2  # double buffer


def conv2dk1():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():

            actIn_ty = T.memref(actIn, T.i8())
            bufIn_ty = T.memref(bufIn, T.i8())

            weights_ty = T.memref(weights, T.i8())

            out_ty = T.memref(actOut, T.i8())
            bufOut_ty = T.memref(bufOut, T.i8())

            # memRef_3x3_ty = T.memref(3, 3, T.i16())

            ofifo_actIn_ty = TypeAttr.get(ObjectFifoType.get(actIn_ty))
            ofifo_bufIn_ty = TypeAttr.get(ObjectFifoType.get(bufIn_ty))

            ofifo_weights_ty = TypeAttr.get(ObjectFifoType.get(weights_ty))

            ofifo_out_ty = TypeAttr.get(ObjectFifoType.get(out_ty))
            ofifo_bufOut_ty = TypeAttr.get(ObjectFifoType.get(bufOut_ty))

            # AIE Core Function declarations
            conv2dk1_i8 = external_func(
                "conv2dk1_i8",
                inputs=[
                    actIn_ty,
                    weights_ty,
                    out_ty,
                    T.i32(),
                    T.i32(),
                    T.i32(),
                    T.i32(),
                ],
            )

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)
            compute_tile2_col, compute_tile2_row = 0, 2

            # AIE-array data movement with object fifos
            # Input
            of_inOF_act_L3L2 = object_fifo(
                "inOF_act_L3L2", ShimTile, MemTile, 2, bufIn_ty
            )
            of_act_L2_02 = object_fifo("act_L2_02", MemTile, ComputeTile2, 2, actIn_ty)
            object_fifo_link(of_inOF_act_L3L2, of_act_L2_02)

            # wts
            of_inOF_wts_0_L3L2 = object_fifo(
                "inOF_wts_0_L3L2", ShimTile, [ComputeTile2], 1, weights_ty
            )

            # Output
            of_out_02_L2 = object_fifo("out_02_L2", ComputeTile2, [MemTile], 2, out_ty)
            of_outOFL2L3 = object_fifo("outOFL2L3", MemTile, [ShimTile], 2, bufOut_ty)
            object_fifo_link(of_out_02_L2, of_outOFL2L3)

            # Set up compute tiles

            rtp2 = Buffer(ComputeTile2, [16], T.i32(), "rtp2")

            # Compute tile 2
            @core(ComputeTile2, "conv2dk1_i8.o")
            def core_body():
                y_dim = 32
                x_dim = 32
                ci = 64
                co = 64

                for _ in range_(0xFFFFFFFF):
                    elemWts = of_inOF_wts_0_L3L2.acquire(ObjectFifoPort.Consume, 1)

                    scale = memref.load(rtp2, [0])
                    # scale = memref.load(rtpComputeTile2, [0])

                    for _ in range_(y_dim):
                        elemIn = of_act_L2_02.acquire(ObjectFifoPort.Consume, 1)
                        elemOut0 = of_out_02_L2.acquire(ObjectFifoPort.Produce, 1)

                        call(
                            conv2dk1_i8,
                            [
                                elemIn,
                                elemWts,
                                elemOut0,
                                arith.constant(x_dim),
                                arith.constant(ci),
                                arith.constant(co),
                                scale,
                            ],
                        )

                        objectfifo_release(ObjectFifoPort.Consume, "act_L2_02", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "out_02_L2", 1)
                    objectfifo_release(ObjectFifoPort.Consume, "inOF_wts_0_L3L2", 1)

            # To/from AIE-array data movement

            tensorSize = width * height * in_channels
            tensor_ty = T.memref(tensorSize, T.i8())
            memRef_wts_ty = T.memref(weights, T.i8())
            # memRef_16x16_ty = T.memref(16, 16, T.i32())

            @runtime_sequence(tensor_ty, memRef_wts_ty, tensor_ty)
            def sequence(I, W, O):
                NpuWriteRTPOp("rtp2", index=0, value=10)

                npu_dma_memcpy_nd(
                    metadata="inOF_act_L3L2",
                    bd_id=0,
                    mem=I,
                    sizes=[1, 1, 1, tensorSize],
                )
                npu_dma_memcpy_nd(
                    metadata="outOFL2L3",
                    bd_id=2,
                    mem=O,
                    sizes=[1, 1, 1, tensorSize],
                )
                npu_dma_memcpy_nd(
                    metadata="inOF_wts_0_L3L2",
                    bd_id=2,
                    mem=W,
                    sizes=[1, 1, 1, weights],
                )
                npu_sync(column=0, row=0, direction=0, channel=0)

    #    print(ctx.module.operation.verify())
    print(ctx.module)


conv2dk1()
