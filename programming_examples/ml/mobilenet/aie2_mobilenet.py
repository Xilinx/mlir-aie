#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

import argparse
import sys

# from bottleneck_A.aie2_bottleneckA import bottleneckACore
from bottleneck_A.aie2_bottleneckAStatic import bottleneckACoreStatic

# from bottleneck_A.aie2_bottleneckFusedAStatic import bottleneckAFused
# from aie2_bottleneckA_TEST import bottleneckACoreTEST
# from bottleneck_A.aie2_bottleneckFusedA import bottleneckAFused
# from bottleneck_A.aie2_bottleneck0 import bottleneckBN0
from bottleneck_A.aie2_bottleneck0Static import bottleneckBN0Static
from bottleneck_A.aie2_bottleneck8And9Static import bottleneckAFused_8and9

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext import *
from aie.extras.dialects.ext.memref import view as memref_view
import math
import aie.utils.trace as trace_utils

import json


def read_scale_factors(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Read the existing scale factors
file_path = "scale_factors_final.json"
weights_path = "weights/"
scale_factors = read_scale_factors(file_path)


class initConv:
    def __init__(
        self,
        _computeTile,
        _act_buf_in,
        _wts_buf,
        _act_buf_out,
        _init_func,
        _in_W,
        _in_H,
        in_C,
        _out_W,
        _out_H,
        _out_C,
        _init_scaleFactor,
    ):
        self.computeTile = _computeTile
        self.act_buf_in = _act_buf_in
        self.act_buf_out = _act_buf_out
        self.wts_buf = _wts_buf
        self.init_func = _init_func
        self.in_W = _in_W
        self.in_H = _in_H
        self.in_C = in_C
        self.out_W = _out_W
        self.out_H = _out_H
        self.out_C = _out_C
        self.scaleFactor = _init_scaleFactor

        @core(self.computeTile, "init_conv2dk3.o")
        def core_body():
            scale = self.scaleFactor
            for _ in for_(sys.maxsize):

                # acquire weights and rtps once
                # element0Weights = self.wts_buf
                # scale = memref.load(rtpComputeTile3, 0)

                # pre-amble: top row
                elementActivactionsIn = self.act_buf_in.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = self.act_buf_out.acquire(
                    ObjectFifoPort.Produce, 1
                )
                res = call(
                    self.init_func,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        self.wts_buf,
                        element0ActivactionsOut,
                        self.in_W,
                        self.in_C,
                        self.out_C,
                        3,
                        3,
                        0,
                        scale,
                        0,
                        0,
                    ],
                )

                self.act_buf_out.release(ObjectFifoPort.Produce, 1)
                self.act_buf_in.release(ObjectFifoPort.Consume, 1)

                # middle
                for _ in for_(self.out_H - 1):
                    elementActivactionsIn = self.act_buf_in.acquire(
                        ObjectFifoPort.Consume, 3
                    )
                    element0ActivactionsOut = self.act_buf_out.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    res = call(
                        self.init_func,
                        [
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[2],
                            self.wts_buf,
                            element0ActivactionsOut,
                            self.in_W,
                            self.in_C,
                            self.out_C,
                            3,
                            3,
                            1,
                            scale,
                            0,
                            0,
                        ],
                    )
                    self.act_buf_in.release(ObjectFifoPort.Consume, 2)
                    self.act_buf_out.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                self.act_buf_in.release(ObjectFifoPort.Consume, 1)
                # self.wts_buf.release(ObjectFifoPort.Consume,1)
                yield_([])


class postBlock:
    def __init__(
        self,
        _computeTile,
        _act_buf_in,
        # _wts_buf,
        post_L1_tile_prod_lock,
        post_L1_tile_cons_lock,
        post_L1_tile_buff,
        _act_buf_out,
        _post_func,
        _in_W,
        _in_H,
        in_C,
        _out_W,
        _out_H,
        _out_C,
        _out_C_padd,
        _post_scaleFactor,
    ):
        self.computeTile = _computeTile
        self.act_buf_in = _act_buf_in
        self.act_buf_out = _act_buf_out
        # self.wts_buf=_wts_buf
        self.post_func = _post_func
        self.in_W = _in_W
        self.in_H = _in_H
        self.in_C = in_C
        self.out_W = _out_W
        self.out_H = _out_H
        self.out_C = _out_C
        self.out_C_padd = _out_C_padd
        self.scaleFactor = _post_scaleFactor
        PostOutputSplit = 8  # split output channels based on your preference

        @core(self.computeTile, "post_conv2dk1_relu_xy_pool_padded_i8_ui8.o", True)
        def core_body():
            for _ in for_(0xFFFFFFFE):

                scale = self.scaleFactor
                # scale = memref.load(rtpComputeTile2, [0])
                elemOut0 = self.act_buf_out.acquire(ObjectFifoPort.Produce, 1)
                for yIndex in for_(0, self.in_H):
                    yIndex_cast = arith.IndexCastOp(T.i32(), yIndex)
                    elemIn = self.act_buf_in.acquire(ObjectFifoPort.Consume, 1)
                    for WeightIndex in for_(0, PostOutputSplit):
                        WeightIndex_cast = arith.IndexCastOp(T.i32(), WeightIndex)
                        # elemWts = self.wts_buf.acquire(ObjectFifoPort.Consume, 1)
                        use_lock(post_L1_tile_cons_lock, LockAction.AcquireGreaterEqual)
                        call(
                            self.post_func,
                            [
                                elemIn,
                                post_L1_tile_buff,
                                elemOut0,
                                arith.constant(self.in_W),
                                arith.constant(self.in_C),
                                arith.constant(self.out_C),
                                arith.constant(self.out_C_padd),
                                scale,
                                yIndex_cast,
                                PostOutputSplit,
                                WeightIndex_cast,
                            ],
                        )

                        # self.wts_buf.release(ObjectFifoPort.Consume,1)
                        use_lock(post_L1_tile_prod_lock, LockAction.Release)
                        yield_([])
                    self.act_buf_in.release(ObjectFifoPort.Consume, 1)
                    yield_([])
                self.act_buf_out.release(ObjectFifoPort.Produce, 1)

                yield_([])


class postBlockL2:
    def __init__(
        self,
        _computeTile1,
        _computeTile2,
        _computeTile3,
        _computeTile4,
        _act_buf_in,
        _wts_buf1,
        _wts_buf2,
        _wts_buf3,
        _wts_buf4,
        _act_buf_out1,
        _act_buf_out2,
        _act_buf_out3,
        _act_buf_out4,
        _post_func,
        _in_W,
        _in_H,
        in_C,
        in_C_pad,
        _out_W,
        _out_H,
        _out_C,
        _post_scaleFactorFC1,
        _post_scaleFactorFC2,
        PostL2Tile_1_cons_prod_lock,
        PostL2Tile_2_cons_prod_lock,
        PostL2Tile_3_cons_prod_lock,
        PostL2Tile_4_cons_prod_lock,
        PostL2Tile_1_cons_cons_lock,
        PostL2Tile_2_cons_cons_lock,
        PostL2Tile_3_cons_cons_lock,
        PostL2Tile_4_cons_cons_lock,
    ):
        self.computeTile1 = _computeTile1
        self.computeTile2 = _computeTile2
        self.computeTile3 = _computeTile3
        self.computeTile4 = _computeTile4
        self.act_buf_in = _act_buf_in
        self.act_buf_out1 = _act_buf_out1
        self.act_buf_out2 = _act_buf_out2
        self.act_buf_out3 = _act_buf_out3
        self.act_buf_out4 = _act_buf_out4
        self.wts_buf1 = _wts_buf1
        self.wts_buf2 = _wts_buf2
        self.wts_buf3 = _wts_buf3
        self.wts_buf4 = _wts_buf4
        self.post_func = _post_func
        self.in_W = _in_W
        self.in_H = _in_H
        self.in_C = in_C
        self.in_C_pad = in_C_pad
        self.out_W = _out_W
        self.out_H = _out_H
        self.out_C = _out_C
        self.scaleFactorFC1 = _post_scaleFactorFC1
        self.scaleFactorFC2 = _post_scaleFactorFC2
        OutputSplit = 40  # split output channels based on your preference
        post_L2_n_core = 4
        co = self.out_C // (OutputSplit * post_L2_n_core)

        @core(self.computeTile1, "post_L2_conv2dk1_relu_ui16_ui16_pad.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):

                scale = self.scaleFactorFC1
                elemIn = self.act_buf_in.acquire(ObjectFifoPort.Consume, 1)
                for WeightIndex in for_(0, OutputSplit):
                    WeightIndex_cast = arith.IndexCastOp(T.i32(), WeightIndex)
                    elemOut0 = self.act_buf_out1.acquire(ObjectFifoPort.Produce, 1)
                    # elemWts = self.wts_buf1.acquire(ObjectFifoPort.Consume, 1)
                    use_lock(
                        PostL2Tile_1_cons_cons_lock, LockAction.AcquireGreaterEqual
                    )
                    call(
                        self.post_func,
                        [
                            elemIn,
                            self.wts_buf1,
                            elemOut0,
                            arith.constant(self.in_W),
                            arith.constant(self.in_C),
                            arith.constant(self.in_C_pad),
                            arith.constant(co),
                            scale,
                        ],
                    )
                    use_lock(PostL2Tile_1_cons_prod_lock, LockAction.Release)
                    # self.wts_buf1.release(ObjectFifoPort.Consume,1)
                    self.act_buf_out1.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                self.act_buf_in.release(ObjectFifoPort.Consume, 1)

                scale = self.scaleFactorFC2
                elemIn = self.act_buf_in.acquire(ObjectFifoPort.Consume, 1)
                for WeightIndex in for_(0, OutputSplit):
                    WeightIndex_cast = arith.IndexCastOp(T.i32(), WeightIndex)
                    elemOut0 = self.act_buf_out1.acquire(ObjectFifoPort.Produce, 1)
                    # elemWts = self.wts_buf1.acquire(ObjectFifoPort.Consume, 1)
                    use_lock(
                        PostL2Tile_1_cons_cons_lock, LockAction.AcquireGreaterEqual
                    )
                    call(
                        self.post_func,
                        [
                            elemIn,
                            self.wts_buf1,
                            elemOut0,
                            arith.constant(self.in_W),
                            arith.constant(self.in_C_pad),
                            arith.constant(self.in_C_pad),
                            arith.constant(co),
                            scale,
                        ],
                    )
                    use_lock(PostL2Tile_1_cons_prod_lock, LockAction.Release)
                    # self.wts_buf1.release(ObjectFifoPort.Consume,1)
                    self.act_buf_out1.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                self.act_buf_in.release(ObjectFifoPort.Consume, 1)

                yield_([])

        @core(self.computeTile2, "post_L2_conv2dk1_relu_ui16_ui16_pad.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):

                scale = self.scaleFactorFC1
                elemIn = self.act_buf_in.acquire(ObjectFifoPort.Consume, 1)
                for WeightIndex in for_(0, OutputSplit):
                    WeightIndex_cast = arith.IndexCastOp(T.i32(), WeightIndex)
                    elemOut0 = self.act_buf_out2.acquire(ObjectFifoPort.Produce, 1)
                    # elemWts = self.wts_buf2.acquire(ObjectFifoPort.Consume, 1)
                    use_lock(
                        PostL2Tile_2_cons_cons_lock, LockAction.AcquireGreaterEqual
                    )
                    call(
                        self.post_func,
                        [
                            elemIn,
                            self.wts_buf2,
                            elemOut0,
                            arith.constant(self.in_W),
                            arith.constant(self.in_C),
                            arith.constant(self.in_C_pad),
                            arith.constant(co),
                            scale,
                        ],
                    )
                    # self.wts_buf2.release(ObjectFifoPort.Consume,1)
                    use_lock(PostL2Tile_2_cons_prod_lock, LockAction.Release)
                    self.act_buf_out2.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                self.act_buf_in.release(ObjectFifoPort.Consume, 1)

                scale = self.scaleFactorFC2
                elemIn = self.act_buf_in.acquire(ObjectFifoPort.Consume, 1)
                for WeightIndex in for_(0, OutputSplit):
                    WeightIndex_cast = arith.IndexCastOp(T.i32(), WeightIndex)
                    elemOut0 = self.act_buf_out2.acquire(ObjectFifoPort.Produce, 1)
                    # elemWts = self.wts_buf2.acquire(ObjectFifoPort.Consume, 1)
                    use_lock(
                        PostL2Tile_2_cons_cons_lock, LockAction.AcquireGreaterEqual
                    )
                    call(
                        self.post_func,
                        [
                            elemIn,
                            self.wts_buf2,
                            elemOut0,
                            arith.constant(self.in_W),
                            arith.constant(self.in_C_pad),
                            arith.constant(self.in_C_pad),
                            arith.constant(co),
                            scale,
                        ],
                    )
                    # self.wts_buf2.release(ObjectFifoPort.Consume,1)
                    use_lock(PostL2Tile_2_cons_prod_lock, LockAction.Release)
                    self.act_buf_out2.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                self.act_buf_in.release(ObjectFifoPort.Consume, 1)

                yield_([])

        @core(self.computeTile3, "post_L2_conv2dk1_relu_ui16_ui16_pad.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):

                scale = self.scaleFactorFC1
                elemIn = self.act_buf_in.acquire(ObjectFifoPort.Consume, 1)
                for WeightIndex in for_(0, OutputSplit):
                    WeightIndex_cast = arith.IndexCastOp(T.i32(), WeightIndex)
                    elemOut0 = self.act_buf_out3.acquire(ObjectFifoPort.Produce, 1)
                    # elemWts = self.wts_buf3.acquire(ObjectFifoPort.Consume, 1)
                    use_lock(
                        PostL2Tile_3_cons_cons_lock, LockAction.AcquireGreaterEqual
                    )
                    call(
                        self.post_func,
                        [
                            elemIn,
                            self.wts_buf3,
                            elemOut0,
                            arith.constant(self.in_W),
                            arith.constant(self.in_C),
                            arith.constant(self.in_C_pad),
                            arith.constant(co),
                            scale,
                        ],
                    )
                    # self.wts_buf3.release(ObjectFifoPort.Consume,1)
                    use_lock(PostL2Tile_3_cons_prod_lock, LockAction.Release)
                    self.act_buf_out3.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                self.act_buf_in.release(ObjectFifoPort.Consume, 1)

                scale = self.scaleFactorFC2
                elemIn = self.act_buf_in.acquire(ObjectFifoPort.Consume, 1)
                for WeightIndex in for_(0, OutputSplit):
                    WeightIndex_cast = arith.IndexCastOp(T.i32(), WeightIndex)
                    elemOut0 = self.act_buf_out3.acquire(ObjectFifoPort.Produce, 1)
                    # elemWts = self.wts_buf3.acquire(ObjectFifoPort.Consume, 1)
                    use_lock(
                        PostL2Tile_3_cons_cons_lock, LockAction.AcquireGreaterEqual
                    )
                    call(
                        self.post_func,
                        [
                            elemIn,
                            self.wts_buf3,
                            elemOut0,
                            arith.constant(self.in_W),
                            arith.constant(self.in_C_pad),
                            arith.constant(self.in_C_pad),
                            arith.constant(co),
                            scale,
                        ],
                    )
                    # self.wts_buf3.release(ObjectFifoPort.Consume,1)
                    use_lock(PostL2Tile_3_cons_prod_lock, LockAction.Release)
                    self.act_buf_out3.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                self.act_buf_in.release(ObjectFifoPort.Consume, 1)
                yield_([])

        @core(self.computeTile4, "post_L2_conv2dk1_relu_ui16_ui16_pad.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):

                scale = self.scaleFactorFC1
                elemIn = self.act_buf_in.acquire(ObjectFifoPort.Consume, 1)
                for WeightIndex in for_(0, OutputSplit):
                    WeightIndex_cast = arith.IndexCastOp(T.i32(), WeightIndex)
                    elemOut0 = self.act_buf_out4.acquire(ObjectFifoPort.Produce, 1)
                    # elemWts = self.wts_buf4.acquire(ObjectFifoPort.Consume, 1)
                    use_lock(
                        PostL2Tile_4_cons_cons_lock, LockAction.AcquireGreaterEqual
                    )
                    call(
                        self.post_func,
                        [
                            elemIn,
                            self.wts_buf4,
                            elemOut0,
                            arith.constant(self.in_W),
                            arith.constant(self.in_C),
                            arith.constant(self.in_C_pad),
                            arith.constant(co),
                            scale,
                        ],
                    )
                    # self.wts_buf4.release(ObjectFifoPort.Consume,1)
                    use_lock(PostL2Tile_4_cons_prod_lock, LockAction.Release)
                    self.act_buf_out4.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                self.act_buf_in.release(ObjectFifoPort.Consume, 1)

                scale = self.scaleFactorFC2
                elemIn = self.act_buf_in.acquire(ObjectFifoPort.Consume, 1)
                for WeightIndex in for_(0, OutputSplit):
                    WeightIndex_cast = arith.IndexCastOp(T.i32(), WeightIndex)
                    elemOut0 = self.act_buf_out4.acquire(ObjectFifoPort.Produce, 1)
                    # elemWts = self.wts_buf4.acquire(ObjectFifoPort.Consume, 1)
                    use_lock(
                        PostL2Tile_4_cons_cons_lock, LockAction.AcquireGreaterEqual
                    )
                    call(
                        self.post_func,
                        [
                            elemIn,
                            self.wts_buf4,
                            elemOut0,
                            arith.constant(self.in_W),
                            arith.constant(self.in_C_pad),
                            arith.constant(self.in_C_pad),
                            arith.constant(co),
                            scale,
                        ],
                    )
                    # self.wts_buf4.release(ObjectFifoPort.Consume,1)
                    use_lock(PostL2Tile_4_cons_prod_lock, LockAction.Release)
                    self.act_buf_out4.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                self.act_buf_in.release(ObjectFifoPort.Consume, 1)
                yield_([])


class BottleneckBCore:
    def __init__(
        self,
        _bottleneckName,
        _computeTileBN10_1,
        _computeTileBN10_2,
        _computeTileBN10_3,
        _computeTileBN11_1,
        _computeTileBN11_2,
        _computeTileBN11_3,
        _computeTileBN12_1,
        _computeTileBN12_2,
        _weightsInBN10_1,
        _weightsInBN10_2,
        _weightsInBN10_3,
        _weightsInBN11_1,
        _weightsInBN11_2,
        _weightsInBN11_3,
        _weightsInBN12_1,
        _weightsInBN12_2_3,
        _tensorLayer12_2Out_ty,
        _rtpBN10_1,
        _rtpBN10_2,
        _rtpBN10_3,
        _rtpBN11_1,
        _rtpBN11_2,
        _rtpBN11_3,
        _rtpBN12_1,
        _rtpBN12_2,
        _skipMemTile,
        _actIn,
        _actOut,
        bn10_scaleFactor1,
        bn10_scaleFactor2,
        bn10_scaleFactor3,
        bn11_scaleFactor1,
        bn11_scaleFactor2,
        bn11_scaleFactor3,
        bn11_scaleFactorAdd,
        bn12_scaleFactor1,
        bn12_scaleFactor2,
        bn12_scaleFactor3,
    ):

        self.bottleneckName = _bottleneckName
        self.computeTileBN10_1 = _computeTileBN10_1
        self.computeTileBN10_2 = _computeTileBN10_2
        self.computeTileBN10_3 = _computeTileBN10_3

        self.computeTileBN11_1 = _computeTileBN11_1
        self.computeTileBN11_2 = _computeTileBN11_2
        self.computeTileBN11_3 = _computeTileBN11_3

        self.computeTileBN12_1 = _computeTileBN12_1
        self.computeTileBN12_2 = _computeTileBN12_2
        # self.computeTileBN12_3 = _computeTileBN12_3

        self.weightsInBN10_layer1 = _weightsInBN10_1
        self.weightsInBN10_layer2 = _weightsInBN10_2
        self.weightsInBN10_layer3 = _weightsInBN10_3

        self.weightsInBN11_layer1 = _weightsInBN11_1
        self.weightsInBN11_layer2 = _weightsInBN11_2
        self.weightsInBN11_layer3 = _weightsInBN11_3

        self.weightsInBN12_layer1 = _weightsInBN12_1
        self.weightsInBN12_layer2_3 = _weightsInBN12_2_3
        # self.weightsInBN12_layer2 = _weightsInBN12_2
        # self.weightsInBN12_layer3 = _weightsInBN12_3

        self.tensorLayer12_2Out_ty = _tensorLayer12_2Out_ty

        self.rtpBN10_layer1 = _rtpBN10_1
        self.rtpBN10_layer2 = _rtpBN10_2
        self.rtpBN10_layer3 = _rtpBN10_3

        self.rtpBN11_layer1 = _rtpBN11_1
        self.rtpBN11_layer2 = _rtpBN11_2
        self.rtpBN11_layer3 = _rtpBN11_3

        self.rtpBN12_layer1 = _rtpBN12_1
        self.rtpBN12_layer2 = _rtpBN12_2
        # self.rtpBN12_layer3 = _rtpBN12_3

        self.skipMemTile = _skipMemTile
        self.actIn = _actIn
        self.actOut = _actOut

        b10_InW1 = 14
        b10_InH1 = 14
        b10_InC1 = 80
        b10_OutC1 = 480

        b10_InW2 = 14
        b10_InH2 = 14
        b10_OutC2 = b10_OutC1

        b10_InW3 = 14
        b10_InH3 = 14
        b10_OutC3 = 112

        b11_OutC1 = 336
        b11_OutC2 = 336
        b11_OutC3 = 112

        b12_OutC1 = 336
        b12_OutC2 = 336
        b12_InW2 = 7
        b12_InH2 = 7
        b12_OutC3 = 80

        enableTrace = False
        trace_size = 16384
        traceSizeInInt32s = trace_size // 4

        # define types
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int32_ty = IntegerType.get_signless(32)
        uint32_ty = IntegerType.get_unsigned(32)
        # ************************ bneck10 ************************
        b10_layer1_in = MemRefType.get(
            (
                b10_InW1,
                1,
                b10_InC1,
            ),
            int8_ty,
        )
        b10_layer2_in = MemRefType.get(
            (
                b10_InW2,
                1,
                b10_OutC1,
            ),
            uint8_ty,
        )
        b10_layer3_in = MemRefType.get(
            (
                b10_InW3,
                1,
                b10_OutC2,
            ),
            uint8_ty,
        )

        # define wts
        b10_layer1_wts = MemRefType.get((b10_InC1 * b10_OutC1,), int8_ty)
        b10_layer2_wts = MemRefType.get((3 * 3 * b10_OutC2 * 1,), int8_ty)
        b10_layer3_wts = MemRefType.get((b10_OutC2 * b10_OutC3,), int8_ty)
        # b10_all_wts= MemRefType.get((b10_InC1 * b10_OutC1 + 3 * 3 * b10_OutC2 * 1 + b10_OutC2 * b10_OutC3, ), int8_ty, )
        # b10_all_wts= MemRefType.get(( b10_OutC2 * b10_OutC3, ), int8_ty, )
        # output
        b10_layer1_out = MemRefType.get(
            (
                b10_InW2,
                1,
                b10_OutC1,
            ),
            uint8_ty,
        )
        b10_layer2_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b10_OutC2,
            ),
            uint8_ty,
        )
        b10_layer3_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b10_OutC3,
            ),
            int8_ty,
        )
        # ************************ bneck11 ************************
        # input
        b11_layer1_in = MemRefType.get(
            (
                b10_InW3,
                1,
                b10_OutC3,
            ),
            int8_ty,
        )
        b11_layer2_in = MemRefType.get(
            (
                b10_InW3,
                1,
                b11_OutC1,
            ),
            uint8_ty,
        )
        b11_layer3_in = MemRefType.get(
            (
                b10_InW3,
                1,
                b11_OutC2,
            ),
            uint8_ty,
        )

        # define wts
        b11_layer1_wts = MemRefType.get((b10_OutC3 * b11_OutC1,), int8_ty)
        b11_layer2_wts = MemRefType.get((3 * 3 * b11_OutC2 * 1,), int8_ty)
        b11_layer3_wts = MemRefType.get((b11_OutC2 * b11_OutC3,), int8_ty)
        # b11_all_wts= MemRefType.get((b10_OutC3 * b11_OutC1 + 3 * 3 * b11_OutC2 * 1 + b11_OutC2 * b11_OutC3, ), int8_ty, )
        # output
        b11_layer1_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b11_OutC1,
            ),
            uint8_ty,
        )
        b11_layer2_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b11_OutC2,
            ),
            uint8_ty,
        )
        b11_layer3_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b11_OutC3,
            ),
            int8_ty,
        )
        # ************************ bneck12 ************************
        b12_layer1_in = MemRefType.get(
            (
                b10_InW1,
                1,
                b11_OutC3,
            ),
            int8_ty,
        )
        b12_layer2_in = MemRefType.get(
            (
                b10_InW1,
                1,
                b12_OutC1,
            ),
            uint8_ty,
        )
        b12_layer3_in = MemRefType.get(
            (
                b12_InW2,
                1,
                b12_OutC2,
            ),
            uint8_ty,
        )
        # define wts
        b12_layer1_wts = MemRefType.get((b11_OutC3 * b12_OutC1,), int8_ty)
        b12_layer2_wts = MemRefType.get((3 * 3 * b12_OutC2 * 1,), int8_ty)
        b12_layer3_wts = MemRefType.get((b12_OutC2 * b12_OutC3,), int8_ty)
        # b12_all_wts= MemRefType.get((b11_OutC3 * b12_OutC1 + 3 * 3 * b12_OutC2 * 1 + b12_OutC2 * b12_OutC3, ), int8_ty, )
        # output
        b12_layer1_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b12_OutC1,
            ),
            uint8_ty,
        )
        b12_layer2_out = MemRefType.get(
            (
                b12_InW2,
                1,
                b12_OutC2,
            ),
            uint8_ty,
        )
        b12_layer3_out = MemRefType.get(
            (
                b12_InW2,
                1,
                b12_OutC3,
            ),
            int8_ty,
        )
        # AIE Core Function declarations
        # ************************ bneck10 ************************
        bn10_conv2dk1_fused_relu = external_func(
            "bn10_conv2dk1_relu_i8_ui8",
            inputs=[
                b10_layer1_in,
                b10_layer1_wts,
                b10_layer1_out,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn10_conv2dk3_dw = external_func(
            "bn10_conv2dk3_dw_stride1_relu_ui8_ui8",
            inputs=[
                b10_layer2_in,
                b10_layer2_in,
                b10_layer2_in,
                b10_layer2_wts,
                b10_layer2_out,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn10_conv2dk1_ui8 = external_func(
            "bn10_conv2dk1_ui8_i8",
            inputs=[
                b10_layer3_in,
                b10_layer3_wts,
                b10_layer3_out,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        # ************************ bneck11 ************************
        bn11_conv2dk1_fused_relu = external_func(
            "bn11_conv2dk1_relu_i8_ui8",
            inputs=[
                b11_layer1_in,
                b11_layer1_wts,
                b11_layer1_out,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn11_conv2dk3_dw = external_func(
            "bn11_conv2dk3_dw_stride1_relu_ui8_ui8",
            inputs=[
                b11_layer2_in,
                b11_layer2_in,
                b11_layer2_in,
                b11_layer2_wts,
                b11_layer2_out,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn11_conv2dk1_skip = external_func(
            "bn11_conv2dk1_skip_ui8_i8_i8",
            inputs=[
                b11_layer3_in,
                b11_layer3_wts,
                b11_layer3_out,
                b11_layer1_in,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        # # ************************ bneck12 ************************
        bn12_conv2dk1_fused_relu = external_func(
            "bn12_conv2dk1_relu_i8_ui8",
            inputs=[
                b12_layer1_in,
                b12_layer1_wts,
                b12_layer1_out,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn12_conv2dk3_dw = external_func(
            "bn12_conv2dk3_dw_stride2_relu_ui8_ui8",
            inputs=[
                b12_layer2_in,
                b12_layer2_in,
                b12_layer2_in,
                b12_layer2_wts,
                b12_layer2_out,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn12_conv2dk1_ui8 = external_func(
            "bn12_conv2dk1_ui8_i8",
            inputs=[
                b12_layer3_in,
                b12_layer3_wts,
                b12_layer3_out,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        # AIE-array data movement with object fifos
        # ************************ bneck10 ************************
        # Input
        # OF_inOF_act_L3L2 = object_fifo("inOF_act_L3L2", ShimTile00, MemTile01, 2, b10_layer1_in )
        # self.actIn = object_fifo("self.actIn", MemTile01, self.computeTileBN10_1, 2, b10_layer1_in)
        # object_fifo_link(OF_inOF_act_L3L2, self.actIn)
        # wts

        # Output
        # OF_b10_act_layer1_layer2 = object_fifo("OF_b10_act_layer1_layer2", self.computeTileBN10_1, [self.computeTileBN10_2], 4,b10_layer2_in,via_DMA=True)
        OF_b10_act_layer1_layer2 = object_fifo(
            "OF_b10_act_layer1_layer2",
            self.computeTileBN10_1,
            [self.computeTileBN10_2],
            4,
            b10_layer2_in,
        )
        OF_b10_act_layer1_layer2.set_via_shared_mem(ObjectFifoPort.Consume)
        OF_b10_act_layer2_layer3 = object_fifo(
            "OF_b10_act_layer2_layer3",
            self.computeTileBN10_2,
            [self.computeTileBN10_3],
            2,
            b10_layer3_in,
        )
        # ************************ bneck11 ************************
        OF_b10_layer3_bn_11_layer1 = object_fifo(
            "OF_b10_layer3_bn_11_layer1",
            self.computeTileBN10_3,
            [self.computeTileBN11_1, self.skipMemTile],
            [2, 2, 6],
            b11_layer1_in,
        )
        OF_b11_skip = object_fifo(
            "OF_b11_skip", self.skipMemTile, [self.computeTileBN11_3], 2, b11_layer1_in
        )
        object_fifo_link(OF_b10_layer3_bn_11_layer1, OF_b11_skip)
        # OF_b11_act_layer1_layer2 = object_fifo("OF_b11_act_layer1_layer2", self.computeTileBN11_1, [self.computeTileBN11_2], 4,b11_layer2_in,via_DMA=True)

        OF_b11_act_layer1_layer2 = object_fifo(
            "OF_b11_act_layer1_layer2",
            self.computeTileBN11_1,
            [self.computeTileBN11_2],
            4,
            b11_layer2_in,
        )
        OF_b11_act_layer1_layer2.set_via_shared_mem(ObjectFifoPort.Consume)
        OF_b11_act_layer2_layer3 = object_fifo(
            "OF_b11_act_layer2_layer3",
            self.computeTileBN11_2,
            [self.computeTileBN11_3],
            2,
            b11_layer3_in,
        )  #
        # ************************ bneck12 ************************
        OF_b11_layer3_bn_12_layer1 = object_fifo(
            "OF_b11_layer3_bn_12_layer1",
            self.computeTileBN11_3,
            [self.computeTileBN12_1],
            2,
            b12_layer1_in,
        )
        OF_b12_act_layer1_layer2 = object_fifo(
            "OF_b12_act_layer1_layer2",
            self.computeTileBN12_1,
            [self.computeTileBN12_2],
            4,
            b12_layer1_out,
            via_DMA=True,
        )
        # OF_b12_act_layer2_layer3 = object_fifo("OF_b12_act_layer2_layer3", self.computeTileBN12_2, [self.computeTileBN12_3], 2,b12_layer2_out)
        self.of_act_bn12_2_3 = object_fifo(
            "act_bn12_2_3",
            self.computeTileBN12_2,
            self.computeTileBN12_2,
            1,
            self.tensorLayer12_2Out_ty,
        )
        # self.actOut = object_fifo("self.actOut", self.computeTileBN12_3, [MemTile21], 2, b12_layer3_out)
        # OF_outOFL2L3 = object_fifo("outOFL2L3", MemTile21, [ShimTile10], 2, b12_layer3_out)
        # object_fifo_link(self.actOut, OF_outOFL2L3)
        # Set up compute tiles

        objectArchiveName = "fused_bn12_layer2_3.a"

        # ************************ bneck10 ************************
        # 1x1 conv2d
        @core(self.computeTileBN10_1, "bn10_conv2dk1_fused_relu.o")
        def core_body():
            for _ in for_(sys.maxsize):

                # acquire weights once
                # elementWeight10_1 = _weightsInBN10_1
                # scale = memref.load(self.rtpBN10_layer1, [0])
                scale = bn10_scaleFactor1
                for _ in for_(b10_InH1):
                    element0ActivactionsIn = self.actIn.acquire(
                        ObjectFifoPort.Consume, 1
                    )
                    element0ActivactionsOut = OF_b10_act_layer1_layer2.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    call(
                        bn10_conv2dk1_fused_relu,
                        [
                            element0ActivactionsIn,
                            self.weightsInBN10_layer1,
                            element0ActivactionsOut,
                            b10_InW1,
                            b10_InC1,
                            b10_OutC1,
                            scale,
                        ],
                    )
                    self.actIn.release(ObjectFifoPort.Consume, 1)
                    OF_b10_act_layer1_layer2.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                # self.weightsInBN10_layer1.release(ObjectFifoPort.Consume,1)
                yield_([])

        # # # Compute tile 3
        @core(self.computeTileBN10_2, "bn10_conv2dk3_dw.o")
        def core_body():
            scale = bn10_scaleFactor2
            for _ in for_(sys.maxsize):

                # acquire weights and rtps once
                # element0Weights = self.weightsInBN10_layer2.acquire(ObjectFifoPort.Consume, 1)
                # scale = memref.load(rtpself.computeTileBN10_2, 0)

                # pre-amble: top row
                elementActivactionsIn = OF_b10_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = OF_b10_act_layer2_layer3.acquire(
                    ObjectFifoPort.Produce, 1
                )
                res = call(
                    bn10_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        self.weightsInBN10_layer2,
                        element0ActivactionsOut,
                        b10_InW2,
                        1,
                        b10_OutC2,
                        3,
                        3,
                        0,
                        scale,
                        0,
                    ],
                )
                OF_b10_act_layer2_layer3.release(ObjectFifoPort.Produce, 1)

                # middle
                for _ in for_(b10_InH2 - 2):
                    elementActivactionsIn = OF_b10_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 3
                    )
                    element0ActivactionsOut = OF_b10_act_layer2_layer3.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    res = call(
                        bn10_conv2dk3_dw,
                        [
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[2],
                            self.weightsInBN10_layer2,
                            element0ActivactionsOut,
                            b10_InW2,
                            1,
                            b10_OutC2,
                            3,
                            3,
                            1,
                            scale,
                            0,
                        ],
                    )

                    OF_b10_act_layer1_layer2.release(ObjectFifoPort.Consume, 1)
                    OF_b10_act_layer2_layer3.release(ObjectFifoPort.Produce, 1)
                    yield_([])

                # last part
                elementActivactionsIn = OF_b10_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = OF_b10_act_layer2_layer3.acquire(
                    ObjectFifoPort.Produce, 1
                )
                res = call(
                    bn10_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        elementActivactionsIn[1],
                        self.weightsInBN10_layer2,
                        element0ActivactionsOut,
                        b10_InW2,
                        1,
                        b10_OutC2,
                        3,
                        3,
                        2,
                        scale,
                        0,
                    ],
                )

                OF_b10_act_layer1_layer2.release(ObjectFifoPort.Consume, 2)
                OF_b10_act_layer2_layer3.release(ObjectFifoPort.Produce, 1)
                # self.weightsInBN10_layer2.release(ObjectFifoPort.Consume,1)

                yield_([])

        # Compute tile 4
        @core(self.computeTileBN10_3, "bn10_conv2dk1_ui8.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):
                # elemWts = self.weightsInBN10_layer3.acquire(ObjectFifoPort.Consume, 1)

                # scale = memref.load(self.rtpBN10_layer3, [0])
                scale = bn10_scaleFactor3
                # scale = memref.load(rtpself.computeTileBN10_1, [0])

                for _ in for_(b10_InH3):
                    elemIn = OF_b10_act_layer2_layer3.acquire(ObjectFifoPort.Consume, 1)
                    elemOut0 = OF_b10_layer3_bn_11_layer1.acquire(
                        ObjectFifoPort.Produce, 1
                    )

                    call(
                        bn10_conv2dk1_ui8,
                        [
                            elemIn,
                            self.weightsInBN10_layer3,
                            elemOut0,
                            b10_InW3,
                            b10_OutC2,
                            b10_OutC3,
                            scale,
                        ],
                    )
                    OF_b10_act_layer2_layer3.release(ObjectFifoPort.Consume, 1)
                    OF_b10_layer3_bn_11_layer1.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                # self.weightsInBN10_layer3.release(ObjectFifoPort.Consume,1)
                yield_([])

        # # # ************************ bneck11 ************************
        # #     #     # 1x1 conv2d
        @core(self.computeTileBN11_1, "bn11_conv2dk1_fused_relu.o")
        def core_body():
            for _ in for_(sys.maxsize):

                # acquire weights once
                # element0Weights = self.weightsInBN11_layer1.acquire(ObjectFifoPort.Consume, 1)
                # scale = memref.load(self.rtpBN11_layer1, [0])
                scale = bn11_scaleFactor1
                for _ in for_(b10_InH1):
                    element0ActivactionsIn = OF_b10_layer3_bn_11_layer1.acquire(
                        ObjectFifoPort.Consume, 1
                    )
                    element0ActivactionsOut = OF_b11_act_layer1_layer2.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    res = call(
                        bn11_conv2dk1_fused_relu,
                        [
                            element0ActivactionsIn,
                            self.weightsInBN11_layer1,
                            element0ActivactionsOut,
                            b10_InW1,
                            b10_OutC3,
                            b11_OutC1,
                            scale,
                        ],
                    )
                    OF_b10_layer3_bn_11_layer1.release(ObjectFifoPort.Consume, 1)
                    OF_b11_act_layer1_layer2.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                # self.weightsInBN11_layer1.release(ObjectFifoPort.Consume,1)
                yield_([])

        # # # # # # Compute tile 3
        @core(self.computeTileBN11_2, "bn11_conv2dk3_dw.o")
        def core_body():
            scale = bn11_scaleFactor2
            for _ in for_(sys.maxsize):

                # acquire weights and rtps once
                # element0Weights = self.weightsInBN11_layer2.acquire(ObjectFifoPort.Consume, 1)
                # scale = memref.load(rtpself.computeTileBN10_2, 0)

                # pre-amble: top row
                elementActivactionsIn = OF_b11_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = OF_b11_act_layer2_layer3.acquire(
                    ObjectFifoPort.Produce, 1
                )
                res = call(
                    bn11_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        self.weightsInBN11_layer2,
                        element0ActivactionsOut,
                        b10_InW2,
                        1,
                        b11_OutC2,
                        3,
                        3,
                        0,
                        scale,
                        0,
                    ],
                )
                OF_b11_act_layer2_layer3.release(ObjectFifoPort.Produce, 1)

                # middle
                for _ in for_(b10_InH2 - 2):
                    elementActivactionsIn = OF_b11_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 3
                    )
                    element0ActivactionsOut = OF_b11_act_layer2_layer3.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    res = call(
                        bn11_conv2dk3_dw,
                        [
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[2],
                            self.weightsInBN11_layer2,
                            element0ActivactionsOut,
                            b10_InW2,
                            1,
                            b11_OutC2,
                            3,
                            3,
                            1,
                            scale,
                            0,
                        ],
                    )
                    OF_b11_act_layer1_layer2.release(ObjectFifoPort.Consume, 1)
                    OF_b11_act_layer2_layer3.release(ObjectFifoPort.Produce, 1)
                    yield_([])

                # last part
                elementActivactionsIn = OF_b11_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = OF_b11_act_layer2_layer3.acquire(
                    ObjectFifoPort.Produce, 1
                )
                res = call(
                    bn11_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        elementActivactionsIn[1],
                        self.weightsInBN11_layer2,
                        element0ActivactionsOut,
                        b10_InW2,
                        1,
                        b11_OutC2,
                        3,
                        3,
                        2,
                        scale,
                        0,
                    ],
                )
                OF_b11_act_layer1_layer2.release(ObjectFifoPort.Consume, 2)
                OF_b11_act_layer2_layer3.release(ObjectFifoPort.Produce, 1)

                # self.weightsInBN11_layer2.release(ObjectFifoPort.Consume,1)

                yield_([])

        # # Compute tile 4
        @core(self.computeTileBN11_3, "bn11_conv2dk1_skip.o")
        def core_body():

            for _ in for_(0xFFFFFFFF):
                # elemWts = self.weightsInBN11_layer3.acquire(ObjectFifoPort.Consume, 1)
                scale = bn11_scaleFactor3
                skipScale = bn11_scaleFactorAdd
                # scale = memref.load(self.rtpBN11_layer3 , [0])
                # skipScale = memref.load(self.rtpBN11_layer3 , [1])

                for _ in for_(b10_InH3):
                    elemIn = OF_b11_act_layer2_layer3.acquire(ObjectFifoPort.Consume, 1)
                    elemOut0 = OF_b11_layer3_bn_12_layer1.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    elementSkipsIn = OF_b11_skip.acquire(ObjectFifoPort.Consume, 1)

                    call(
                        bn11_conv2dk1_skip,
                        [
                            elemIn,
                            self.weightsInBN11_layer3,
                            elemOut0,
                            elementSkipsIn,
                            b10_InW3,
                            b11_OutC2,
                            b11_OutC3,
                            scale,
                            skipScale,
                        ],
                    )

                    OF_b11_act_layer2_layer3.release(ObjectFifoPort.Consume, 1)
                    OF_b11_layer3_bn_12_layer1.release(ObjectFifoPort.Produce, 1)
                    OF_b11_skip.release(ObjectFifoPort.Consume, 1)
                    yield_([])
                # self.weightsInBN11_layer3.release(ObjectFifoPort.Consume,1)
                yield_([])

        # # # ************************ bneck12 ************************
        #     # 1x1 conv2d
        @core(self.computeTileBN12_1, "bn12_conv2dk1_fused_relu.o")
        def core_body():
            for _ in for_(sys.maxsize):

                # acquire weights once
                # element0Weights = self.weightsInBN12_layer1.acquire(ObjectFifoPort.Consume, 1)
                # scale = memref.load(self.rtpBN12_layer1, [0])
                scale = bn12_scaleFactor1
                for _ in for_(b10_InH1):
                    element0ActivactionsIn = OF_b11_layer3_bn_12_layer1.acquire(
                        ObjectFifoPort.Consume, 1
                    )
                    element0ActivactionsOut = OF_b12_act_layer1_layer2.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    res = call(
                        bn12_conv2dk1_fused_relu,
                        [
                            element0ActivactionsIn,
                            self.weightsInBN12_layer1,
                            element0ActivactionsOut,
                            b10_InW1,
                            b11_OutC3,
                            b12_OutC1,
                            scale,
                        ],
                    )
                    OF_b11_layer3_bn_12_layer1.release(ObjectFifoPort.Consume, 1)
                    OF_b12_act_layer1_layer2.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                # self.weightsInBN12_layer1.release(ObjectFifoPort.Consume,1)
                yield_([])

        @core(self.computeTileBN12_2, objectArchiveName)
        def core_body():
            scale2 = bn12_scaleFactor2
            scale3 = bn12_scaleFactor3
            for _ in for_(sys.maxsize):
                weightsInBN12_layer2 = memref_view(
                    self.weightsInBN12_layer2_3, [3 * 3 * b12_OutC2 * 1], shift=0
                )
                weightsInBN12_layer3 = memref_view(
                    self.weightsInBN12_layer2_3,
                    [b12_OutC2 * b12_OutC3],
                    shift=3 * 3 * b12_OutC2 * 1,
                )
                # acquire weights and rtps once
                # element0Weights = self.weightsInBN12_layer2.acquire(ObjectFifoPort.Consume, 1)
                # scale = memref.load(rtpComputeTile3, 0)

                # pre-amble: top row
                elementActivactionsIn = OF_b12_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = self.of_act_bn12_2_3.acquire(
                    ObjectFifoPort.Produce, 1
                )
                res = call(
                    bn12_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        weightsInBN12_layer2,
                        element0ActivactionsOut,
                        b10_InW3,
                        1,
                        b12_OutC2,
                        3,
                        3,
                        0,
                        scale2,
                        0,
                    ],
                )
                self.of_act_bn12_2_3.release(ObjectFifoPort.Produce, 1)
                OF_b12_act_layer1_layer2.release(ObjectFifoPort.Consume, 1)

                elemIn = self.of_act_bn12_2_3.acquire(ObjectFifoPort.Consume, 1)
                elemOut0 = self.actOut.acquire(ObjectFifoPort.Produce, 1)
                call(
                    bn12_conv2dk1_ui8,
                    [
                        elemIn,
                        weightsInBN12_layer3,
                        elemOut0,
                        b12_InW2,
                        b12_OutC2,
                        b12_OutC3,
                        scale3,
                    ],
                )
                self.of_act_bn12_2_3.release(ObjectFifoPort.Consume, 1)
                self.actOut.release(ObjectFifoPort.Produce, 1)

                # middle
                for _ in for_(b12_InH2 - 2):
                    elementActivactionsIn = OF_b12_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 3
                    )
                    element0ActivactionsOut = self.of_act_bn12_2_3.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    res = call(
                        bn12_conv2dk3_dw,
                        [
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[2],
                            weightsInBN12_layer2,
                            element0ActivactionsOut,
                            b10_InW3,
                            1,
                            b12_OutC2,
                            3,
                            3,
                            1,
                            scale2,
                            0,
                        ],
                    )
                    OF_b12_act_layer1_layer2.release(ObjectFifoPort.Consume, 2)
                    self.of_act_bn12_2_3.release(ObjectFifoPort.Produce, 1)

                    elemIn = self.of_act_bn12_2_3.acquire(ObjectFifoPort.Consume, 1)
                    elemOut0 = self.actOut.acquire(ObjectFifoPort.Produce, 1)
                    call(
                        bn12_conv2dk1_ui8,
                        [
                            elemIn,
                            weightsInBN12_layer3,
                            elemOut0,
                            b12_InW2,
                            b12_OutC2,
                            b12_OutC3,
                            scale3,
                        ],
                    )
                    self.of_act_bn12_2_3.release(ObjectFifoPort.Consume, 1)
                    self.actOut.release(ObjectFifoPort.Produce, 1)

                    yield_([])

                elementActivactionsIn = OF_b12_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 3
                )
                element0ActivactionsOut = self.of_act_bn12_2_3.acquire(
                    ObjectFifoPort.Produce, 1
                )
                res = call(
                    bn12_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        elementActivactionsIn[2],
                        weightsInBN12_layer2,
                        element0ActivactionsOut,
                        b10_InW3,
                        1,
                        b12_OutC2,
                        3,
                        3,
                        1,
                        scale2,
                        0,
                    ],
                )
                OF_b12_act_layer1_layer2.release(ObjectFifoPort.Consume, 3)
                self.of_act_bn12_2_3.release(ObjectFifoPort.Produce, 1)

                elemIn = self.of_act_bn12_2_3.acquire(ObjectFifoPort.Consume, 1)
                elemOut0 = self.actOut.acquire(ObjectFifoPort.Produce, 1)
                call(
                    bn12_conv2dk1_ui8,
                    [
                        elemIn,
                        weightsInBN12_layer3,
                        elemOut0,
                        b12_InW2,
                        b12_OutC2,
                        b12_OutC3,
                        scale3,
                    ],
                )
                self.of_act_bn12_2_3.release(ObjectFifoPort.Consume, 1)
                self.actOut.release(ObjectFifoPort.Produce, 1)
                yield_([])


class bottleneckCCore:
    def __init__(
        self,
        _computeTileBN13_1,
        _computeTileBN13_2,
        _computeTileBN13_3,
        _computeTileBN13_4,
        _computeTileBN13_5,
        _computeTileBN14_1,
        _computeTileBN14_2,
        _computeTileBN14_3,
        _computeTileBN14_4,
        _computeTileBN14_5,
        _weightsInBN13_1,
        _weightsInBN13_2,
        _weightsInBN13_3,
        _weightsInBN13_4,
        _weightsInBN13_5,
        _weightsInBN14_1,
        _weightsInBN14_2,
        _weightsInBN14_3,
        _weightsInBN14_4,
        _weightsInBN14_5,
        _rtp_bn13_tile_layer1_get,
        _rtp_bn13_tile_layer3_get,
        _bn13_scaleFactor1,
        _bn13_scaleFactor2,
        _bn13_scaleFactor3,
        _bn13_scaleFactorAdd,
        _bn14_scaleFactor1,
        _bn14_scaleFactor2,
        _bn14_scaleFactor3,
        _bn14_scaleFactorAdd,
        _skipMemTile,
        _actIn,
        _actOut,
        _bn13_skip,
    ):

        self.computeTileBN13_layer1_put = _computeTileBN13_1
        self.computeTileBN13_layer1_get = _computeTileBN13_2
        self.computeTileBN13_layer2 = _computeTileBN13_3
        self.computeTileBN13_layer3_put = _computeTileBN13_4
        self.computeTileBN13_layer3_get = _computeTileBN13_5

        self.computeTileBN14_layer1_put = _computeTileBN14_1
        self.computeTileBN14_layer1_get = _computeTileBN14_2
        self.computeTileBN14_layer2 = _computeTileBN14_3
        self.computeTileBN14_layer3_put = _computeTileBN14_4
        self.computeTileBN14_layer3_get = _computeTileBN14_5

        # wts

        self.weightsInBN13_layer1_put = _weightsInBN13_1
        self.weightsInBN13_layer1_get = _weightsInBN13_2
        self.weightsInBN13_layer2 = _weightsInBN13_3
        self.weightsInBN13_layer3_put = _weightsInBN13_4
        self.weightsInBN13_layer3_get = _weightsInBN13_5

        self.weightsInBN14_layer1_put = _weightsInBN14_1
        self.weightsInBN14_layer1_get = _weightsInBN14_2
        self.weightsInBN14_layer2 = _weightsInBN14_3
        self.weightsInBN14_layer3_put = _weightsInBN14_4
        self.weightsInBN14_layer3_get = _weightsInBN14_5

        self.skipMemTile = _skipMemTile

        self.actIn = _actIn
        self.actOut = _actOut
        self.bn13_skip = _bn13_skip

        bneck_13_InW1 = 7
        bneck_13_InH1 = 7
        bneck_13_InC1 = 80
        bneck_13_OutC1 = 960
        InputSplit = 2
        OutputSplit = 2  # split output channels based on your preference --> 4
        OC8 = bneck_13_OutC1 // (8 * OutputSplit)  # how many loops of OC8

        bneck_13_InW2 = bneck_13_InW1
        bneck_13_InH2 = bneck_13_InH1
        bneck_13_OutC2 = bneck_13_OutC1

        bneck_13_InW3 = bneck_13_InW2
        bneck_13_InH3 = bneck_13_InH2
        bneck_13_OutC3 = 80
        OutputSplit2 = (
            2  # calculate 8 OCs at a time, should bneck_13_InC1rease to more --> 4
        )
        OC8_out = bneck_13_OutC3 // (8 * OutputSplit2)  # how many loops of OC8

        # second block
        bneck_14_InW1 = bneck_13_InW1
        bneck_14_InH1 = bneck_13_InH1
        bneck_14_InC1 = bneck_13_OutC3
        bneck_14_OutC1 = 960

        bneck_14_InW2 = bneck_14_InW1
        bneck_14_InH2 = bneck_14_InH1
        bneck_14_OutC2 = bneck_14_OutC1

        bneck_14_InW3 = bneck_14_InW2
        bneck_14_InH3 = bneck_14_InH2
        bneck_14_OutC3 = 80

        self.rtp_bn13_tile_layer1_get = _rtp_bn13_tile_layer1_get
        self.rtp_bn13_tile_layer3_get = _rtp_bn13_tile_layer3_get

        self.bn13_scaleFactor1 = _bn13_scaleFactor1
        self.bn13_scaleFactor2 = _bn13_scaleFactor2
        self.bn13_scaleFactor3 = _bn13_scaleFactor3
        self.bn13_scaleFactorAdd = _bn13_scaleFactorAdd

        self.bn14_scaleFactor1 = _bn14_scaleFactor1
        self.bn14_scaleFactor2 = _bn14_scaleFactor2
        self.bn14_scaleFactor3 = _bn14_scaleFactor3
        self.bn14_scaleFactorAdd = _bn14_scaleFactorAdd

        # define types
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int32_ty = IntegerType.get_signless(32)
        uint32_ty = IntegerType.get_unsigned(32)

        # ************************ bneck13 ************************

        ty_bneck_13_layer1_in = MemRefType.get(
            (
                bneck_13_InW1,
                1,
                bneck_13_InC1,
            ),
            int8_ty,
        )
        ty_bneck_13_layer2_in = MemRefType.get(
            (
                bneck_13_InW2,
                1,
                bneck_13_OutC1,
            ),
            uint8_ty,
        )

        # define wts
        # layer1
        ty_bneck_13_layer1_wts_split = MemRefType.get(
            ((bneck_13_InC1 // InputSplit) * (bneck_13_OutC1 // OutputSplit),), int8_ty
        )
        ty_bneck_13_layer1_wts_full = MemRefType.get(
            (bneck_13_InC1 * bneck_13_OutC1,),
            int8_ty,
        )
        # layer2
        b13_layer2_wts_size = 3 * 3 * bneck_13_OutC2 * 1
        ty_bneck_13_layer2_wts = MemRefType.get((3 * 3 * bneck_13_OutC2 * 1,), int8_ty)
        # layer3
        ty_bneck_13_layer3_wts_split = MemRefType.get(
            ((bneck_13_OutC2 // InputSplit) * (bneck_13_OutC3 // OutputSplit2),),
            int8_ty,
        )
        ty_bneck_13_layer3_wts_full = MemRefType.get(
            (bneck_13_OutC2 * bneck_13_OutC3,),
            int8_ty,
        )

        # OUTPUT
        ty_bneck_13_layer1_out = MemRefType.get(
            (
                bneck_13_InW1,
                1,
                bneck_13_OutC1,
            ),
            uint8_ty,
        )
        ty_bneck_13_layer2_out = MemRefType.get(
            (
                bneck_13_InW3,
                1,
                bneck_13_OutC2,
            ),
            uint8_ty,
        )
        ty_bneck_13_layer2_out_split = MemRefType.get(
            (
                bneck_13_InW3,
                1,
                bneck_13_OutC2 // InputSplit,
            ),
            uint8_ty,
        )
        # layer3
        ty_bneck_13_layer3_out = MemRefType.get(
            (
                bneck_13_InW3,
                1,
                bneck_13_OutC3,
            ),
            int8_ty,
        )

        # HERE

        # ************************ bneck14 ************************

        ty_bneck_14_layer1_in = MemRefType.get(
            (
                bneck_14_InW1,
                1,
                bneck_14_InC1,
            ),
            int8_ty,
        )
        ty_bneck_14_layer2_in = MemRefType.get(
            (
                bneck_14_InW2,
                1,
                bneck_14_OutC1,
            ),
            uint8_ty,
        )

        # define wts
        # layer1
        ty_bneck_14_layer1_wts_split = MemRefType.get(
            ((bneck_14_InC1 // InputSplit) * (bneck_14_OutC1 // OutputSplit),), int8_ty
        )
        ty_bneck_14_layer1_wts_full = MemRefType.get(
            (bneck_14_InC1 * bneck_14_OutC1,),
            int8_ty,
        )
        # layer2
        b14_layer2_wts_size = 3 * 3 * bneck_14_OutC2 * 1
        ty_bneck_14_layer2_wts = MemRefType.get((3 * 3 * bneck_14_OutC2 * 1,), int8_ty)
        # layer3
        ty_bneck_14_layer3_wts_split = MemRefType.get(
            ((bneck_14_OutC2 // InputSplit) * (bneck_14_OutC3 // OutputSplit2),),
            int8_ty,
        )
        ty_bneck_14_layer3_wts_full = MemRefType.get(
            (bneck_14_OutC2 * bneck_14_OutC3,),
            int8_ty,
        )

        # OUTPUT
        ty_bneck_14_layer1_out = MemRefType.get(
            (
                bneck_14_InW1,
                1,
                bneck_14_OutC1,
            ),
            uint8_ty,
        )
        ty_bneck_14_layer2_out = MemRefType.get(
            (
                bneck_14_InW3,
                1,
                bneck_14_OutC2,
            ),
            uint8_ty,
        )
        ty_bneck_14_layer2_out_split = MemRefType.get(
            (
                bneck_14_InW3,
                1,
                bneck_14_OutC2 // InputSplit,
            ),
            uint8_ty,
        )
        # layer3
        ty_bneck_14_layer3_out = MemRefType.get(
            (
                bneck_14_InW3,
                1,
                bneck_14_OutC3,
            ),
            int8_ty,
        )

        bn13_conv2dk1_fused_relu_get = external_func(
            "bn13_1_conv2dk1_i8_ui8_partial_width_get_new",
            inputs=[
                ty_bneck_13_layer1_in,
                ty_bneck_13_layer1_wts_split,
                ty_bneck_13_layer1_out,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn13_conv2dk1_fused_relu_put = external_func(
            "bn13_1_conv2dk1_i8_ui8_partial_width_put_new",
            inputs=[
                ty_bneck_13_layer1_in,
                ty_bneck_13_layer1_wts_split,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn13_conv2dk3_dw = external_func(
            "bn13_conv2dk3_ui8_out_split",
            inputs=[
                ty_bneck_13_layer2_in,
                ty_bneck_13_layer2_in,
                ty_bneck_13_layer2_in,
                ty_bneck_13_layer2_wts,
                ty_bneck_13_layer2_out_split,
                ty_bneck_13_layer2_out_split,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        bn13_layer3_conv2dk1_put = external_func(
            "bn13_1_conv2dk1_ui8_ui8_input_split_partial_width_put_new",
            inputs=[
                ty_bneck_13_layer2_out_split,
                ty_bneck_13_layer3_wts_split,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        bn13_layer3_conv2dk1_skip_get = external_func(
            "bn_13_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new",
            inputs=[
                ty_bneck_13_layer2_out_split,
                ty_bneck_13_layer3_wts_split,
                ty_bneck_13_layer3_out,
                ty_bneck_13_layer1_in,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn14_conv2dk1_fused_relu_get = external_func(
            "bn14_1_conv2dk1_i8_ui8_partial_width_get_new",
            inputs=[
                ty_bneck_14_layer1_in,
                ty_bneck_14_layer1_wts_split,
                ty_bneck_14_layer1_out,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn14_conv2dk1_fused_relu_put = external_func(
            "bn14_1_conv2dk1_i8_ui8_partial_width_put_new",
            inputs=[
                ty_bneck_14_layer1_in,
                ty_bneck_14_layer1_wts_split,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn14_conv2dk3_dw = external_func(
            "bn14_conv2dk3_ui8_out_split",
            inputs=[
                ty_bneck_14_layer2_in,
                ty_bneck_14_layer2_in,
                ty_bneck_14_layer2_in,
                ty_bneck_14_layer2_wts,
                ty_bneck_14_layer2_out_split,
                ty_bneck_14_layer2_out_split,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn14_layer3_conv2dk1_put = external_func(
            "bn14_1_conv2dk1_ui8_ui8_input_split_partial_width_put_new",
            inputs=[
                ty_bneck_14_layer2_out_split,
                ty_bneck_14_layer3_wts_split,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        bn14_layer3_conv2dk1_skip_get = external_func(
            "bn_14_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new",
            inputs=[
                ty_bneck_14_layer2_out_split,
                ty_bneck_14_layer3_wts_split,
                ty_bneck_14_layer3_out,
                ty_bneck_14_layer1_in,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        # AIE-array data movement with object fifos
        # ************************ bneck13 ************************

        # OUTPUT
        bn13_act_layer1_layer2 = object_fifo(
            "bn13_act_layer1_layer2",
            self.computeTileBN13_layer1_get,
            [self.computeTileBN13_layer2],
            4,
            ty_bneck_13_layer2_in,
            via_DMA=True,
        )
        # bn13_act_layer1_layer2 = object_fifo("bn13_act_layer1_layer2", self.computeTileBN13_layer1_get, [self.computeTileBN13_layer2], 4,ty_bneck_13_layer2_in)
        # bn13_act_layer1_layer2.set_via_shared_mem(ObjectFifoPort.Consume)

        bn13_act_layer2_layer3_first = object_fifo(
            "bn13_act_layer2_layer3_first",
            self.computeTileBN13_layer2,
            [self.computeTileBN13_layer3_put],
            2,
            ty_bneck_13_layer2_out_split,
        )
        bn13_act_layer2_layer3_second = object_fifo(
            "bn13_act_layer2_layer3_second",
            self.computeTileBN13_layer2,
            [self.computeTileBN13_layer3_get],
            2,
            ty_bneck_13_layer2_out_split,
        )

        # ************************ bneck14 ************************
        # Input

        bn13_act_layer3_bn_14_layer1 = object_fifo(
            "bn13_act_layer3_bn_14_layer1",
            self.computeTileBN13_layer3_get,
            [
                self.computeTileBN14_layer1_put,
                self.computeTileBN14_layer1_get,
                self.skipMemTile,
            ],
            [2, 2, 2, 6],
            ty_bneck_13_layer3_out,
        )
        bn14_skip = object_fifo(
            "bn14_skip",
            self.skipMemTile,
            self.computeTileBN14_layer3_get,
            2,
            ty_bneck_13_layer3_out,
        )
        object_fifo_link(bn13_act_layer3_bn_14_layer1, bn14_skip)

        # Object FIFO for b14 block results
        bn14_act_layer1_layer2 = object_fifo(
            "bn14_act_layer1_layer2",
            self.computeTileBN14_layer1_get,
            self.computeTileBN14_layer2,
            4,
            ty_bneck_14_layer1_out,
            via_DMA=True,
        )
        # bn14_act_layer1_layer2 = object_fifo("bn14_act_layer1_layer2", self.computeTileBN14_layer1_get, self.computeTileBN14_layer2, 4, ty_bneck_14_layer1_out)
        # bn14_act_layer1_layer2.set_via_shared_mem(ObjectFifoPort.Consume)

        bn14_act_layer2_layer3_first = object_fifo(
            "bn14_act_layer2_layer3_first",
            self.computeTileBN14_layer2,
            self.computeTileBN14_layer3_put,
            2,
            ty_bneck_14_layer2_out_split,
        )
        bn14_act_layer2_layer3_second = object_fifo(
            "bn14_act_layer2_layer3_second",
            self.computeTileBN14_layer2,
            [self.computeTileBN14_layer3_get],
            2,
            ty_bneck_14_layer2_out_split,
        )

        # object_fifo_link(bn13_act_layer2_layer3_first, [OF_outOFL2L3],[],[0])
        # object_fifo_link([bn13_act_layer2_layer3_first,bn13_act_layer2_layer3_second],[OF_outOFL2L3],[0,(bneck_13_InW3 *  bneck_13_OutC2//2)])

        # ************************ bneck13 ************************
        # conv1x1_first put
        @core(self.computeTileBN13_layer1_put, "bn13_1_conv2dk1_put.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):
                for _ in for_(bneck_13_InH1):
                    elemIn = self.actIn.acquire(ObjectFifoPort.Consume, 1)
                    # for oc in range(0,OutputSplit):
                    for WeightIndex in for_(
                        0, OutputSplit
                    ):  # how many input channel splits, 1 in case InputSplit is 2
                        WeightIndex_cast = arith.IndexCastOp(T.i32(), WeightIndex)
                        elemWts = self.weightsInBN13_layer1_put.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        for oc in for_(0, OC8):
                            oc_cast = arith.IndexCastOp(T.i32(), oc)
                            x_start = 0
                            call(
                                bn13_conv2dk1_fused_relu_put,
                                [
                                    elemIn,
                                    elemWts,
                                    arith.constant(bneck_13_InW1),
                                    arith.constant(bneck_13_InC1),
                                    arith.constant(bneck_13_OutC1),
                                    InputSplit,
                                    WeightIndex_cast,
                                    x_start,
                                    oc_cast,
                                ],
                            )
                            yield_([])
                        self.weightsInBN13_layer1_put.release(ObjectFifoPort.Consume, 1)
                        yield_([])
                    self.actIn.release(ObjectFifoPort.Consume, 1)
                    yield_([])
                yield_([])

        # conv1x1_first get
        @core(self.computeTileBN13_layer1_get, "bn13_1_conv2dk1_get.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):
                for _ in for_(bneck_13_InH1):
                    elemIn = self.actIn.acquire(ObjectFifoPort.Consume, 1)
                    elemOut0 = bn13_act_layer1_layer2.acquire(ObjectFifoPort.Produce, 1)

                    # scale = memref.load(rtp_bn1, [0])
                    scale = self.bn13_scaleFactor1
                    # for oc in range(0,OutputSplit):
                    for WeightIndex in for_(0, OutputSplit):
                        WeightIndex_cast = arith.IndexCastOp(T.i32(), WeightIndex)
                        elemWts = self.weightsInBN13_layer1_get.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        for oc in for_(0, OC8):
                            oc_cast = arith.IndexCastOp(T.i32(), oc)
                            x_start = 0
                            call(
                                bn13_conv2dk1_fused_relu_get,
                                [
                                    elemIn,
                                    elemWts,
                                    elemOut0,
                                    arith.constant(bneck_13_InW1),
                                    arith.constant(bneck_13_InC1),
                                    arith.constant(bneck_13_OutC1),
                                    scale,
                                    InputSplit,
                                    OutputSplit,
                                    WeightIndex_cast,
                                    x_start,
                                    oc_cast,
                                ],
                            )
                            yield_([])
                        self.weightsInBN13_layer1_get.release(ObjectFifoPort.Consume, 1)
                        yield_([])
                    self.actIn.release(ObjectFifoPort.Consume, 1)
                    bn13_act_layer1_layer2.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                yield_([])

        # conv3x3
        @core(self.computeTileBN13_layer2, "bn13_conv2dk3_dw.o")
        def core_body():
            scale = self.bn13_scaleFactor2
            for _ in for_(sys.maxsize):

                # acquire weights and rtps once
                # element0Weights = self.weightsInBN13_layer2.acquire(ObjectFifoPort.Consume, 1)
                # scale = memref.load(rtpself.computeTileBN13_layer1_get, 0)

                # pre-amble: top row
                elementActivactionsIn = bn13_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = bn13_act_layer2_layer3_first.acquire(
                    ObjectFifoPort.Produce, 1
                )
                element1ActivactionsOut = bn13_act_layer2_layer3_second.acquire(
                    ObjectFifoPort.Produce, 1
                )
                res = call(
                    bn13_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        self.weightsInBN13_layer2,
                        element0ActivactionsOut,
                        element1ActivactionsOut,
                        bneck_13_InW2,
                        1,
                        bneck_13_OutC2,
                        3,
                        3,
                        0,
                        scale,
                        0,
                    ],
                )
                bn13_act_layer2_layer3_first.release(ObjectFifoPort.Produce, 1)
                bn13_act_layer2_layer3_second.release(ObjectFifoPort.Produce, 1)
                # middle
                for _ in for_(bneck_13_InH2 - 2):
                    elementActivactionsIn = bn13_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 3
                    )
                    element0ActivactionsOut = bn13_act_layer2_layer3_first.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    element1ActivactionsOut = bn13_act_layer2_layer3_second.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    res = call(
                        bn13_conv2dk3_dw,
                        [
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[2],
                            self.weightsInBN13_layer2,
                            element0ActivactionsOut,
                            element1ActivactionsOut,
                            bneck_13_InW2,
                            1,
                            bneck_13_OutC2,
                            3,
                            3,
                            1,
                            scale,
                            0,
                        ],
                    )
                    bn13_act_layer1_layer2.release(ObjectFifoPort.Consume, 1)
                    bn13_act_layer2_layer3_first.release(ObjectFifoPort.Produce, 1)
                    bn13_act_layer2_layer3_second.release(ObjectFifoPort.Produce, 1)
                    yield_([])

                # last part
                elementActivactionsIn = bn13_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = bn13_act_layer2_layer3_first.acquire(
                    ObjectFifoPort.Produce, 1
                )
                element1ActivactionsOut = bn13_act_layer2_layer3_second.acquire(
                    ObjectFifoPort.Produce, 1
                )
                res = call(
                    bn13_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        elementActivactionsIn[1],
                        self.weightsInBN13_layer2,
                        element0ActivactionsOut,
                        element1ActivactionsOut,
                        bneck_13_InW2,
                        1,
                        bneck_13_OutC2,
                        3,
                        3,
                        2,
                        scale,
                        0,
                    ],
                )
                bn13_act_layer1_layer2.release(ObjectFifoPort.Consume, 2)
                bn13_act_layer2_layer3_first.release(ObjectFifoPort.Produce, 1)
                bn13_act_layer2_layer3_second.release(ObjectFifoPort.Produce, 1)
                # self.weightsInBN13_layer2.release(ObjectFifoPort.Consume,1)
                yield_([])

        # conv1x1_second put
        @core(self.computeTileBN13_layer3_put, "bn13_conv2dk1_put.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):

                for _ in for_(bneck_13_InH3):
                    elemIn = bn13_act_layer2_layer3_first.acquire(
                        ObjectFifoPort.Consume, 1
                    )
                    # for oc in range(0,OutputSplit):

                    # for WeightIndex in range (0,InputSplit//2):
                    for WeightIndex in for_(
                        0, OutputSplit2
                    ):  # how many input channel splits, 1 in case InputSplit is 2
                        WeightIndex_cast = arith.IndexCastOp(T.i32(), WeightIndex)
                        elemWts = self.weightsInBN13_layer3_put.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        for oc in for_(0, OC8_out):
                            oc_cast = arith.IndexCastOp(T.i32(), oc)
                            x_start = 0
                            call(
                                bn13_layer3_conv2dk1_put,
                                [
                                    elemIn,
                                    elemWts,
                                    arith.constant(bneck_13_InW3),
                                    arith.constant(bneck_13_OutC2),
                                    arith.constant(bneck_13_OutC3),
                                    InputSplit,
                                    WeightIndex_cast,
                                    x_start,
                                    oc_cast,
                                ],
                            )
                            yield_([])
                        self.weightsInBN13_layer3_put.release(ObjectFifoPort.Consume, 1)
                        yield_([])
                    bn13_act_layer2_layer3_first.release(ObjectFifoPort.Consume, 1)
                    yield_([])
                yield_([])

        # conv1x1_second get
        @core(self.computeTileBN13_layer3_get, "bn13_conv2dk1_skip_get.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):

                for _ in for_(bneck_13_InH3):

                    elemIn = bn13_act_layer2_layer3_second.acquire(
                        ObjectFifoPort.Consume, 1
                    )
                    elemOut0 = bn13_act_layer3_bn_14_layer1.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    elementSkipsIn = self.bn13_skip.acquire(ObjectFifoPort.Consume, 1)

                    scale = self.bn13_scaleFactor3
                    scale_skip = self.bn13_scaleFactorAdd
                    # scale = memref.load(rtp_bn1, [0])
                    # for oc in range(0,OutputSplit):

                    for WeightIndex in for_(0, OutputSplit2):
                        WeightIndex_cast = arith.IndexCastOp(T.i32(), WeightIndex)
                        elemWts = self.weightsInBN13_layer3_get.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        for oc in for_(0, OC8_out):
                            oc_cast = arith.IndexCastOp(T.i32(), oc)
                            x_start = 0

                            call(
                                bn13_layer3_conv2dk1_skip_get,
                                [
                                    elemIn,
                                    elemWts,
                                    elemOut0,
                                    elementSkipsIn,
                                    arith.constant(bneck_13_InW3),
                                    arith.constant(bneck_13_OutC2),
                                    arith.constant(bneck_13_OutC3),
                                    scale,
                                    scale_skip,
                                    InputSplit,
                                    OutputSplit2,
                                    WeightIndex_cast,
                                    x_start,
                                    oc_cast,
                                ],
                            )

                            yield_([])
                        self.weightsInBN13_layer3_get.release(ObjectFifoPort.Consume, 1)
                        yield_([])
                    bn13_act_layer2_layer3_second.release(ObjectFifoPort.Consume, 1)
                    bn13_act_layer3_bn_14_layer1.release(ObjectFifoPort.Produce, 1)
                    self.bn13_skip.release(ObjectFifoPort.Consume, 1)
                    yield_([])
                yield_([])

        # ************************ bneck14 ************************
        # conv1x1_first put
        @core(self.computeTileBN14_layer1_put, "bn14_1_conv2dk1_put.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):
                for _ in for_(bneck_13_InH1):
                    elemIn = bn13_act_layer3_bn_14_layer1.acquire(
                        ObjectFifoPort.Consume, 1
                    )
                    # for oc in range(0,OutputSplit):
                    for WeightIndex in for_(0, OutputSplit):
                        WeightIndex_cast = arith.IndexCastOp(T.i32(), WeightIndex)
                        elemWts = self.weightsInBN14_layer1_put.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        for oc in for_(0, OC8):
                            oc_cast = arith.IndexCastOp(T.i32(), oc)
                            x_start = 0
                            call(
                                bn14_conv2dk1_fused_relu_put,
                                [
                                    elemIn,
                                    elemWts,
                                    arith.constant(bneck_14_InW1),
                                    arith.constant(bneck_14_InC1),
                                    arith.constant(bneck_14_OutC1),
                                    InputSplit,
                                    WeightIndex_cast,
                                    x_start,
                                    oc_cast,
                                ],
                            )
                            yield_([])
                        self.weightsInBN14_layer1_put.release(ObjectFifoPort.Consume, 1)
                        yield_([])
                    bn13_act_layer3_bn_14_layer1.release(ObjectFifoPort.Consume, 1)
                    yield_([])
                yield_([])

        # conv1x1_first get
        @core(self.computeTileBN14_layer1_get, "bn14_1_conv2dk1_get.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):

                for _ in for_(bneck_13_InH1):
                    elemIn = bn13_act_layer3_bn_14_layer1.acquire(
                        ObjectFifoPort.Consume, 1
                    )
                    elemOut0 = bn14_act_layer1_layer2.acquire(ObjectFifoPort.Produce, 1)

                    # scale = memref.load(rtp_bn1, [0])
                    scale = self.bn14_scaleFactor1
                    # for oc in range(0,OutputSplit):
                    for WeightIndex in for_(0, OutputSplit):
                        WeightIndex_cast = arith.IndexCastOp(T.i32(), WeightIndex)
                        elemWts = self.weightsInBN14_layer1_get.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        for oc in for_(0, OC8):
                            oc_cast = arith.IndexCastOp(T.i32(), oc)
                            x_start = 0
                            call(
                                bn14_conv2dk1_fused_relu_get,
                                [
                                    elemIn,
                                    elemWts,
                                    elemOut0,
                                    arith.constant(bneck_14_InW1),
                                    arith.constant(bneck_14_InC1),
                                    arith.constant(bneck_14_OutC1),
                                    scale,
                                    InputSplit,
                                    OutputSplit,
                                    WeightIndex_cast,
                                    x_start,
                                    oc_cast,
                                ],
                            )
                            yield_([])
                        self.weightsInBN14_layer1_get.release(ObjectFifoPort.Consume, 1)
                        yield_([])
                    bn13_act_layer3_bn_14_layer1.release(ObjectFifoPort.Consume, 1)
                    bn14_act_layer1_layer2.release(ObjectFifoPort.Produce, 1)

                    yield_([])
                yield_([])

        # conv3x3
        @core(self.computeTileBN14_layer2, "bn14_conv2dk3_dw.o")
        def core_body():
            scale = self.bn14_scaleFactor2
            for _ in for_(sys.maxsize):

                # acquire weights and rtps once
                # element0Weights = self.weightsInBN14_layer2.acquire(ObjectFifoPort.Consume, 1)
                # scale = memref.load(rtpself.computeTileBN13_layer1_get, 0)

                # pre-amble: top row
                elementActivactionsIn = bn14_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = bn14_act_layer2_layer3_first.acquire(
                    ObjectFifoPort.Produce, 1
                )
                element1ActivactionsOut = bn14_act_layer2_layer3_second.acquire(
                    ObjectFifoPort.Produce, 1
                )
                res = call(
                    bn14_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        self.weightsInBN14_layer2,
                        element0ActivactionsOut,
                        element1ActivactionsOut,
                        bneck_14_InW2,
                        1,
                        bneck_14_OutC2,
                        3,
                        3,
                        0,
                        scale,
                        0,
                    ],
                )
                bn14_act_layer2_layer3_first.release(ObjectFifoPort.Produce, 1)
                bn14_act_layer2_layer3_second.release(ObjectFifoPort.Produce, 1)

                # middle
                for _ in for_(bneck_14_InH2 - 2):
                    elementActivactionsIn = bn14_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 3
                    )
                    element0ActivactionsOut = bn14_act_layer2_layer3_first.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    element1ActivactionsOut = bn14_act_layer2_layer3_second.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    res = call(
                        bn14_conv2dk3_dw,
                        [
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[2],
                            self.weightsInBN14_layer2,
                            element0ActivactionsOut,
                            element1ActivactionsOut,
                            bneck_14_InW2,
                            1,
                            bneck_14_OutC2,
                            3,
                            3,
                            1,
                            scale,
                            0,
                        ],
                    )
                    bn14_act_layer1_layer2.release(ObjectFifoPort.Consume, 1)
                    bn14_act_layer2_layer3_first.release(ObjectFifoPort.Produce, 1)
                    bn14_act_layer2_layer3_second.release(ObjectFifoPort.Produce, 1)

                    yield_([])

                # last part
                elementActivactionsIn = bn14_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = bn14_act_layer2_layer3_first.acquire(
                    ObjectFifoPort.Produce, 1
                )
                element1ActivactionsOut = bn14_act_layer2_layer3_second.acquire(
                    ObjectFifoPort.Produce, 1
                )
                res = call(
                    bn14_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        elementActivactionsIn[1],
                        self.weightsInBN14_layer2,
                        element0ActivactionsOut,
                        element1ActivactionsOut,
                        bneck_14_InW2,
                        1,
                        bneck_14_OutC2,
                        3,
                        3,
                        2,
                        scale,
                        0,
                    ],
                )

                bn14_act_layer1_layer2.release(ObjectFifoPort.Consume, 2)
                bn14_act_layer2_layer3_first.release(ObjectFifoPort.Produce, 1)
                bn14_act_layer2_layer3_second.release(ObjectFifoPort.Produce, 1)
                # self.weightsInBN14_layer2.release(ObjectFifoPort.Consume,1)
                yield_([])

        # conv1x1_second put
        @core(self.computeTileBN14_layer3_put, "bn14_conv2dk1_put.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):

                for _ in for_(bneck_14_InH3):
                    elemIn = bn14_act_layer2_layer3_first.acquire(
                        ObjectFifoPort.Consume, 1
                    )
                    # for oc in range(0,OutputSplit):
                    for WeightIndex in for_(
                        0, OutputSplit2
                    ):  # how many input channel splits, 1 in case InputSplit is 2
                        WeightIndex_cast = arith.IndexCastOp(T.i32(), WeightIndex)
                        elemWts = self.weightsInBN14_layer3_put.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        for oc in for_(0, OC8_out):
                            oc_cast = arith.IndexCastOp(T.i32(), oc)
                            x_start = 0
                            call(
                                bn14_layer3_conv2dk1_put,
                                [
                                    elemIn,
                                    elemWts,
                                    arith.constant(bneck_14_InW3),
                                    arith.constant(bneck_14_OutC2),
                                    arith.constant(bneck_14_OutC3),
                                    InputSplit,
                                    WeightIndex_cast,
                                    x_start,
                                    oc_cast,
                                ],
                            )
                            yield_([])
                        self.weightsInBN14_layer3_put.release(ObjectFifoPort.Consume, 1)
                        yield_([])
                    bn14_act_layer2_layer3_first.release(ObjectFifoPort.Consume, 1)
                    yield_([])
                yield_([])

        # conv1x1_second get
        @core(self.computeTileBN14_layer3_get, "bn14_conv2dk1_skip_get.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):

                for _ in for_(bneck_14_InH3):

                    elemIn = bn14_act_layer2_layer3_second.acquire(
                        ObjectFifoPort.Consume, 1
                    )
                    elemOut0 = self.actOut.acquire(ObjectFifoPort.Produce, 1)
                    elementSkipsIn = bn14_skip.acquire(ObjectFifoPort.Consume, 1)

                    scale = self.bn14_scaleFactor3
                    scale_skip = self.bn14_scaleFactorAdd

                    for WeightIndex in for_(0, OutputSplit2):
                        WeightIndex_cast = arith.IndexCastOp(T.i32(), WeightIndex)
                        elemWts = self.weightsInBN14_layer3_get.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        for oc in for_(0, OC8_out):
                            oc_cast = arith.IndexCastOp(T.i32(), oc)
                            x_start = 0

                            call(
                                bn14_layer3_conv2dk1_skip_get,
                                [
                                    elemIn,
                                    elemWts,
                                    elemOut0,
                                    elementSkipsIn,
                                    arith.constant(bneck_14_InW3),
                                    arith.constant(bneck_14_OutC2),
                                    arith.constant(bneck_14_OutC3),
                                    scale,
                                    scale_skip,
                                    InputSplit,
                                    OutputSplit2,
                                    WeightIndex_cast,
                                    x_start,
                                    oc_cast,
                                ],
                            )
                            yield_([])
                        self.weightsInBN14_layer3_get.release(ObjectFifoPort.Consume, 1)
                        yield_([])
                    bn14_act_layer2_layer3_second.release(ObjectFifoPort.Consume, 1)
                    self.actOut.release(ObjectFifoPort.Produce, 1)
                    bn14_skip.release(ObjectFifoPort.Consume, 1)
                    yield_([])
                yield_([])


def select_b_cores(start_col, start_row):
    # Initialize the list to store the selected cores
    selected_cores = []

    # Current position
    current_col = start_col
    current_row = start_row

    # Direction flag for snake-like pattern
    downward = True

    # Loop to select the next 9 cores
    for _ in range(9):
        # Add the current core to the list
        selected_cores.append((current_col, current_row))

        # Move to the next core based on the direction
        if downward:
            current_row += 1
            if current_row > 5:  # If we reach the bottom boundary
                current_row = 5
                current_col += 1
                downward = False  # Change direction
        else:
            current_row -= 1
            if current_row < 2:  # If we reach the top boundary
                current_row = 2
                current_col += 1
                downward = True  # Change direction

        # If the column index exceeds the limit, break the loop
        if current_col > 4:
            break

    return selected_cores


def mobilenetV3_A_B(
    tileColIndex=0,
    b_start_col=0,
    b_start_row=2,
    tensorInW=224,
    tensorInH=224,
    tensorInC=8,
    init_tensorOutC=16,
    init_scaleFactor=8,
    bn0_scaleFactor2=9,
    bn0_scaleFactor3=8,
    bn0_scaleFactorAdd=2,
    bn0_depthWiseStride=1,
    bn0_withSkip=True,
    bn1_depthWiseStride=2,
    bn1_depthWiseChannels=64,
    bn1_withSkip=False,
    bn1_tensorOutC=24,
    bn1_scaleFactor1=8,
    bn1_scaleFactor2=8,
    bn1_scaleFactor3=11,
    bn1_scaleFactorAdd=0,
    bn2_depthWiseStride=1,
    bn2_depthWiseChannels=72,
    bn2_withSkip=True,
    bn2_tensorOutC=24,
    bn2_scaleFactor1=8,
    bn2_scaleFactor2=8,
    bn2_scaleFactor3=11,
    bn2_scaleFactorAdd=0,
    bn3_depthWiseStride=2,
    bn3_depthWiseChannels=72,
    bn3_withSkip=False,
    bn3_tensorOutC=40,
    bn3_scaleFactor1=8,
    bn3_scaleFactor2=8,
    bn3_scaleFactor3=11,
    bn3_scaleFactorAdd=0,
    bn4_depthWiseStride=1,
    bn4_depthWiseChannels=120,
    bn4_withSkip=True,
    bn4_tensorOutC=40,
    bn4_scaleFactor1=8,
    bn4_scaleFactor2=8,
    bn4_scaleFactor3=11,
    bn4_scaleFactorAdd=0,
    bn5_depthWiseStride=1,
    bn5_depthWiseChannels=120,
    bn5_withSkip=True,
    bn5_tensorOutC=80,
    bn5_scaleFactor1=8,
    bn5_scaleFactor2=8,
    bn5_scaleFactor3=11,
    bn5_scaleFactorAdd=0,
    bn6_depthWiseStride=2,
    bn6_depthWiseChannels=240,
    bn6_withSkip=False,
    bn6_tensorOutC=80,
    bn6_scaleFactor1=8,
    bn6_scaleFactor2=8,
    bn6_scaleFactor3=11,
    bn6_scaleFactorAdd=0,
    bn7_depthWiseStride=1,
    bn7_depthWiseChannels=200,
    bn7_withSkip=True,
    bn7_tensorOutC=80,
    bn7_scaleFactor1=9,
    bn7_scaleFactor2=8,
    bn7_scaleFactor3=11,
    bn7_scaleFactorAdd=0,
    bn8_depthWiseStride=1,
    bn8_depthWiseChannels=184,
    bn8_withSkip=True,
    bn8_tensorOutC=80,
    bn8_scaleFactor1=9,
    bn8_scaleFactor2=8,
    bn8_scaleFactor3=11,
    bn8_scaleFactorAdd=0,
    bn9_depthWiseStride=1,
    bn9_depthWiseChannels=184,
    bn9_withSkip=True,
    bn9_tensorOutC=80,
    bn9_scaleFactor1=9,
    bn9_scaleFactor2=8,
    bn9_scaleFactor3=11,
    bn9_scaleFactorAdd=0,
    enableTrace=False,
    trace_size=16384,
    traceSizeInInt32s=4096,
    bn10_scaleFactor1=10,
    bn10_scaleFactor2=7,
    bn10_scaleFactor3=9,
    bn11_scaleFactor1=9,
    bn11_scaleFactor2=8,
    bn11_scaleFactor3=12,
    bn11_scaleFactorAdd=1,
    bn12_scaleFactor1=8,
    bn12_scaleFactor2=8,
    bn12_scaleFactor3=9,
    bn13_scaleFactor1=10,
    bn13_scaleFactor2=7,
    bn13_scaleFactor3=9,
    bn13_scaleFactorAdd=1,
    bn14_scaleFactor1=9,
    bn14_scaleFactor2=8,
    bn14_scaleFactor3=12,
    bn14_scaleFactorAdd=1,
    post_scaleFactor=8,
    post_FC1_scaleFactor=9,
    post_FC2_scaleFactor=9,
):

    # init conv
    tensor_init_InC = tensorInC
    tensor_init_InW = tensorInW
    tensor_init_InH = tensorInH

    tensor_init_OutC = init_tensorOutC
    tensor_init_OutH = tensor_init_InH // 2
    tensor_init_OutW = tensor_init_InW // 2

    # bn0
    tensorL0_2InC = tensor_init_OutC
    tensorL0_2InW = tensor_init_OutW
    tensorL0_2InH = tensor_init_OutH
    tensorL0_2OutC = tensorL0_2InC

    tensorL0_3InC = tensorL0_2OutC
    tensorL0_3InW = tensorL0_2InW
    tensorL0_3InH = tensorL0_2InH
    tensorL0_3OutC = tensorL0_3InC

    # bn1
    tensorL1_1InC = tensorL0_3OutC
    tensorL1_1InW = tensorL0_3InW
    tensorL1_1InH = tensorL0_3InH
    tensorL1_1OutC = bn1_depthWiseChannels

    tensorL1_2InC = tensorL1_1OutC
    tensorL1_2InW = tensorL1_1InW
    tensorL1_2InH = tensorL1_1InH
    tensorL1_2OutC = tensorL1_2InC

    tensorL1_3InC = tensorL1_2OutC
    tensorL1_3InW = tensorL1_2InW // bn1_depthWiseStride
    tensorL1_3InH = tensorL1_2InH // bn1_depthWiseStride
    tensorL1_3OutC = bn1_tensorOutC
    #

    tensorL2_1InC = tensorL1_3OutC
    tensorL2_1InW = tensorL1_3InW
    tensorL2_1InH = tensorL1_3InH

    tensorL2_2InC = bn2_depthWiseChannels
    tensorL2_2InW = tensorL2_1InW
    tensorL2_2InH = tensorL2_1InH

    tensorL2_3InC = tensorL2_2InC
    tensorL2_3InW = tensorL2_2InW // bn2_depthWiseStride
    tensorL2_3InH = tensorL2_2InH // bn2_depthWiseStride
    tensorL2_3OutC = bn2_tensorOutC

    #

    tensorL3_1InC = tensorL2_3OutC
    tensorL3_1InW = tensorL2_3InW
    tensorL3_1InH = tensorL2_3InH

    tensorL3_2InC = bn3_depthWiseChannels
    tensorL3_2InW = tensorL3_1InW
    tensorL3_2InH = tensorL3_1InH

    tensorL3_3InC = tensorL3_2InC
    tensorL3_3InW = tensorL3_2InW // bn3_depthWiseStride
    tensorL3_3InH = tensorL3_2InH // bn3_depthWiseStride
    tensorL3_3OutC = bn3_tensorOutC

    #

    tensorL4_1InC = tensorL3_3OutC
    tensorL4_1InW = tensorL3_3InW
    tensorL4_1InH = tensorL3_3InH

    tensorL4_2InC = bn4_depthWiseChannels
    tensorL4_2InW = tensorL4_1InW
    tensorL4_2InH = tensorL4_1InH

    tensorL4_3InC = tensorL4_2InC
    tensorL4_3InW = tensorL4_2InW // bn4_depthWiseStride
    tensorL4_3InH = tensorL4_2InH // bn4_depthWiseStride
    tensorL4_3OutC = bn4_tensorOutC

    #

    tensorL5_1InC = tensorL4_3OutC
    tensorL5_1InW = tensorL4_3InW
    tensorL5_1InH = tensorL4_3InH

    tensorL5_2InC = bn5_depthWiseChannels
    tensorL5_2InW = tensorL5_1InW
    tensorL5_2InH = tensorL5_1InH

    tensorL5_3InC = tensorL5_2InC
    tensorL5_3InW = tensorL5_2InW // bn5_depthWiseStride
    tensorL5_3InH = tensorL5_2InH // bn5_depthWiseStride
    tensorL5_3OutC = bn5_tensorOutC
    #
    tensorL6_1InC = tensorL5_3OutC
    tensorL6_1InW = tensorL5_3InW
    tensorL6_1InH = tensorL5_3InH

    tensorL6_2InC = bn6_depthWiseChannels
    tensorL6_2InW = tensorL6_1InW
    tensorL6_2InH = tensorL6_1InH

    tensorL6_3InC = tensorL6_2InC
    tensorL6_3InW = tensorL6_2InW // bn6_depthWiseStride
    tensorL6_3InH = tensorL6_2InH // bn6_depthWiseStride
    tensorL6_3OutC = bn6_tensorOutC
    #
    tensorL7_1InC = tensorL6_3OutC
    tensorL7_1InW = tensorL6_3InW
    tensorL7_1InH = tensorL6_3InH

    tensorL7_2InC = bn7_depthWiseChannels
    tensorL7_2InW = tensorL7_1InW
    tensorL7_2InH = tensorL7_1InH

    tensorL7_3InC = tensorL7_2InC
    tensorL7_3InW = tensorL7_2InW // bn7_depthWiseStride
    tensorL7_3InH = tensorL7_2InH // bn7_depthWiseStride
    tensorL7_3OutC = bn7_tensorOutC
    #
    tensorL8_1InC = tensorL7_3OutC
    tensorL8_1InW = tensorL7_3InW
    tensorL8_1InH = tensorL7_3InH

    tensorL8_2InC = bn8_depthWiseChannels
    tensorL8_2InW = tensorL8_1InW
    tensorL8_2InH = tensorL8_1InH

    tensorL8_3InC = tensorL8_2InC
    tensorL8_3InW = tensorL8_2InW // bn8_depthWiseStride
    tensorL8_3InH = tensorL8_2InH // bn8_depthWiseStride
    tensorL8_3OutC = bn8_tensorOutC
    #
    tensorL9_1InC = tensorL8_3OutC
    tensorL9_1InW = tensorL8_3InW
    tensorL9_1InH = tensorL8_3InH

    tensorL9_2InC = bn9_depthWiseChannels
    tensorL9_2InW = tensorL9_1InW
    tensorL9_2InH = tensorL9_1InH

    tensorL9_3InC = tensorL9_2InC
    tensorL9_3InW = tensorL9_2InW // bn9_depthWiseStride
    tensorL9_3InH = tensorL9_2InH // bn9_depthWiseStride
    tensorL9_3OutC = bn9_tensorOutC

    b10_InW1 = 14
    b10_InH1 = 14
    b10_InC1 = 80
    b10_OutC1 = 480

    b10_InW2 = 14
    b10_InH2 = 14
    b10_OutC2 = b10_OutC1

    b10_InW3 = 14
    b10_InH3 = 14
    b10_OutC3 = 112

    b11_OutC1 = 336
    b11_OutC2 = 336
    b11_OutC3 = 112
    b11_InW2 = 14
    b11_InH2 = 14

    b12_OutC1 = 336
    b12_OutC2 = 336
    b12_InW2 = 7
    b12_InH2 = 7
    b12_OutC3 = 80

    # C mappings

    bneck_13_InW1 = 7
    bneck_13_InH1 = 7
    bneck_13_InC1 = 80
    bneck_13_OutC1 = 960
    InputSplit = 2
    OutputSplit = 2  # split output channels based on your preference

    RepeatChannels = math.floor(bneck_13_InH1)

    bneck_13_InW2 = bneck_13_InW1
    bneck_13_InH2 = bneck_13_InH1
    bneck_13_OutC2 = bneck_13_OutC1

    bneck_13_InW3 = bneck_13_InW2
    bneck_13_InH3 = bneck_13_InH2
    bneck_13_OutC3 = 80

    # second block
    bneck_14_InW1 = bneck_13_InW1
    bneck_14_InH1 = bneck_13_InH1
    bneck_14_InC1 = bneck_13_OutC3
    bneck_14_OutC1 = 960

    OutputSplit2 = 2  # split output channels based on your preference

    bneck_14_InW2 = bneck_14_InW1
    bneck_14_InH2 = bneck_14_InH1
    bneck_14_OutC2 = bneck_14_OutC1

    bneck_14_InW3 = bneck_14_InW2
    bneck_14_InH3 = bneck_14_InH2
    bneck_14_OutC3 = 80

    post_L1_InW = bneck_14_InW3
    post_L1_InH = bneck_14_InH3
    post_L1_InC = bneck_14_OutC3

    post_L1_OutW = 1
    post_L1_OutH = 1
    post_L1_OutC = 960

    post_L1_OutC_padd = 1280  # added for padding

    post_L2_OutW = 1
    post_L2_OutH = 1
    post_L2_OutC = 1280

    tensorOutW = post_L1_OutW
    tensorOutH = post_L1_OutH
    tensorOutC = post_L1_OutC_padd

    selected_cores = select_b_cores(b_start_col, b_start_row)

    @device(AIEDevice.npu2)
    def device_body():

        # define types
        uint8_ty = IntegerType.get_unsigned(8)
        uint16_ty = IntegerType.get_unsigned(16)
        int8_ty = IntegerType.get_signless(8)
        int16_ty = IntegerType.get_signless(16)
        int32_ty = IntegerType.get_signless(32)

        tensorLayerIn_ty = MemRefType.get((tensorInW, 1, tensorInC), int8_ty)
        tensorLayerOut_ty = MemRefType.get((tensorOutW, 1, tensorOutC), int8_ty)

        init_weights_size = 3 * 3 * tensor_init_InC * tensor_init_OutC
        init_weights_ty = MemRefType.get((init_weights_size,), int8_ty)
        # setup all the weights here
        bn0_weights_size = (
            3 * 3 * tensorL0_3InC * 1 + 1 * 1 * tensorL0_3InC * tensorL0_3OutC
        )
        bn0_weightsAllLayers_ty = MemRefType.get((bn0_weights_size,), int8_ty)
        bn1_weights_size = (
            1 * 1 * tensorL1_1InC * tensorL1_2InC
            + 3 * 3 * tensorL1_3InC * 1
            + 1 * 1 * tensorL1_3InC * tensorL1_3OutC
        )
        bn1_weightsAllLayers_ty = MemRefType.get((bn1_weights_size,), int8_ty)
        # bn0_1_weights_size=bn0_weights_size+bn1_weights_size

        bn2_weights_size = (
            1 * 1 * tensorL2_1InC * tensorL2_2InC
            + 3 * 3 * tensorL2_3InC * 1
            + 1 * 1 * tensorL2_3InC * tensorL2_3OutC
        )
        bn2_weightsAllLayers_ty = MemRefType.get((bn2_weights_size,), int8_ty)
        bn3_weights_size = (
            1 * 1 * tensorL3_1InC * tensorL3_2InC
            + 3 * 3 * tensorL3_3InC * 1
            + 1 * 1 * tensorL3_3InC * tensorL3_3OutC
        )
        bn3_weightsAllLayers_ty = MemRefType.get((bn3_weights_size,), int8_ty)
        bn4_weights_size = (
            1 * 1 * tensorL4_1InC * tensorL4_2InC
            + 3 * 3 * tensorL4_3InC * 1
            + 1 * 1 * tensorL4_3InC * tensorL4_3OutC
        )
        bn4_weightsAllLayers_ty = MemRefType.get((bn4_weights_size,), int8_ty)
        bn5_weights_size = (
            1 * 1 * tensorL5_1InC * tensorL5_2InC
            + 3 * 3 * tensorL5_3InC * 1
            + 1 * 1 * tensorL5_3InC * tensorL5_3OutC
        )
        bn5_weightsAllLayers_ty = MemRefType.get((bn5_weights_size,), int8_ty)
        bn6_weights_size = (
            1 * 1 * tensorL6_1InC * tensorL6_2InC
            + 3 * 3 * tensorL6_3InC * 1
            + 1 * 1 * tensorL6_3InC * tensorL6_3OutC
        )
        bn6_weightsAllLayers_ty = MemRefType.get((bn6_weights_size,), int8_ty)
        bn7_weights_size = (
            1 * 1 * tensorL7_1InC * tensorL7_2InC
            + 3 * 3 * tensorL7_3InC * 1
            + 1 * 1 * tensorL7_3InC * tensorL7_3OutC
        )
        bn7_weightsAllLayers_ty = MemRefType.get((bn7_weights_size,), int8_ty)
        bn8_weights_size = (
            1 * 1 * tensorL8_1InC * tensorL8_2InC
            + 3 * 3 * tensorL8_3InC * 1
            + 1 * 1 * tensorL8_3InC * tensorL8_3OutC
        )
        bn8_weightsAllLayers_ty = MemRefType.get((bn8_weights_size,), int8_ty)

        bn9_weights_size = (
            1 * 1 * tensorL9_1InC * tensorL9_2InC
            + 3 * 3 * tensorL9_3InC * 1
            + 1 * 1 * tensorL9_3InC * tensorL9_3OutC
        )
        bn9_weightsAllLayers_ty = MemRefType.get((bn9_weights_size,), int8_ty)
        bn4_5_weights_size = bn4_weights_size + bn5_weights_size
        bn4_5_weightsAllLayers_ty = MemRefType.get((bn4_5_weights_size,), int8_ty)

        bn8_9_weights_size = bn8_weights_size + bn9_weights_size
        bn8_9_weightsAllLayers_ty = MemRefType.get((bn8_9_weights_size,), int8_ty)

        # memtile_01_wts=bn2_weights_size+bn3_weights_size+bn4_weights_size
        # memtile_01_wts_ty = MemRefType.get((memtile_01_wts,), int8_ty)
        memtile_01_wts = 0
        memtile_11_wts = 0
        memtile_11_wts_ty = MemRefType.get((memtile_11_wts,), int8_ty)

        total_weights = memtile_01_wts + memtile_11_wts
        total_weights_ty = MemRefType.get((total_weights,), int8_ty)

        ShimTile00 = tile(0, 0)
        ShimTile10 = tile(1, 0)
        ShimTile20 = tile(2, 0)
        ShimTile30 = tile(3, 0)
        ShimTile40 = tile(4, 0)
        ShimTile50 = tile(5, 0)
        ShimTile60 = tile(6, 0)
        ShimTile70 = tile(7, 0)

        MemTile01 = tile(0, 1)
        MemTile11 = tile(1, 1)
        MemTile21 = tile(2, 1)
        MemTile31 = tile(3, 1)
        MemTile41 = tile(4, 1)
        MemTile51 = tile(5, 1)
        MemTile61 = tile(6, 1)
        MemTile71 = tile(7, 1)

        # column 0
        bn2_tile = tile(tileColIndex + 0, 5)  # bn2
        bn1_tile = tile(tileColIndex + 0, 4)  # bn0
        bn0_tile = tile(tileColIndex + 0, 3)  # bn0
        init_tile = tile(tileColIndex + 0, 2)  # init

        # column 1
        bn10_tile_1 = tile(tileColIndex + 1, 5)
        bn6_tile = tile(tileColIndex + 1, 4)  # bn6
        bn3_tile = tile(tileColIndex + 1, 3)  # bn3
        bn4_5_tile = tile(tileColIndex + 1, 2)  # bn0

        # column 2
        bn10_tile_3 = tile(tileColIndex + 2, 5)
        bn10_tile_2 = tile(tileColIndex + 2, 4)
        bn7_tile = tile(tileColIndex + 2, 3)  # bn7
        bn11_tile_3 = tile(tileColIndex + 2, 2)

        # column 3
        bn12_tile_1 = tile(tileColIndex + 3, 5)
        bn11_tile_2 = tile(tileColIndex + 3, 4)
        bn8_bn9_tile = tile(tileColIndex + 3, 3)  # bn8+bn9
        bn11_tile_1 = tile(tileColIndex + 3, 2)

        # column 4
        bn13_tile_layer1_put = tile(4, 5)
        bn12_tile_2 = tile(tileColIndex + 4, 4)
        bn13_tile_layer3_put = tile(4, 3)
        bn14_tile_layer3_put = tile(4, 2)  # put

        # column 5
        bn13_tile_layer1_get = tile(5, 5)
        bn13_tile_layer2 = tile(5, 4)
        bn13_tile_layer3_get = tile(5, 3)
        bn14_tile_layer3_get = tile(5, 2)  # get

        # column 6
        bn14_tile_layer1_put = tile(6, 5)  # put
        PostL1Tile = tile(6, 4)
        PostL2Tile_1 = tile(6, 3)  # post L2
        bn14_tile_layer2 = tile(6, 2)

        # column 7
        bn14_tile_layer1_get = tile(7, 5)  # get
        PostL2Tile_2 = tile(7, 4)  # post
        PostL2Tile_3 = tile(7, 3)  # post
        PostL2Tile_4 = tile(7, 2)  # post
        # ******************************************************************* C block *******************************************************************
        cascade_flow(bn13_tile_layer1_put, bn13_tile_layer1_get)
        cascade_flow(bn13_tile_layer3_put, bn13_tile_layer3_get)
        cascade_flow(bn14_tile_layer1_put, bn14_tile_layer1_get)
        cascade_flow(bn14_tile_layer3_put, bn14_tile_layer3_get)

        b10_layer1_in = MemRefType.get(
            (
                b10_InW1,
                1,
                b10_InC1,
            ),
            int8_ty,
        )
        b12_layer3_out = MemRefType.get(
            (
                b12_InW2,
                1,
                b12_OutC3,
            ),
            int8_ty,
        )
        # define wts
        b10_layer1_wts_size = b10_InC1 * b10_OutC1
        b10_layer2_wts_size = 3 * 3 * b10_OutC2 * 1
        b10_layer3_wts_size = b10_OutC2 * b10_OutC3

        b10_layer1_wts = MemRefType.get((b10_InC1 * b10_OutC1,), int8_ty)
        b10_layer2_wts = MemRefType.get((3 * 3 * b10_OutC2 * 1,), int8_ty)
        b10_layer3_wts = MemRefType.get((b10_OutC2 * b10_OutC3,), int8_ty)
        # b10_all_wts= MemRefType.get((b10_InC1 * b10_OutC1 + 3 * 3 * b10_OutC2 * 1 + b10_OutC2 * b10_OutC3, ), int8_ty, )
        b10_all_wts = MemRefType.get(
            (b10_OutC2 * b10_OutC3,),
            int8_ty,
        )
        # output
        b10_layer1_out = MemRefType.get(
            (
                b10_InW2,
                1,
                b10_OutC1,
            ),
            uint8_ty,
        )
        b10_layer2_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b10_OutC2,
            ),
            uint8_ty,
        )
        b10_layer3_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b10_OutC3,
            ),
            int8_ty,
        )
        # ************************ bneck11 ************************
        # input
        b11_layer1_in = MemRefType.get(
            (
                b10_InW3,
                1,
                b10_OutC3,
            ),
            int8_ty,
        )
        b11_layer2_in = MemRefType.get(
            (
                b10_InW3,
                1,
                b11_OutC1,
            ),
            uint8_ty,
        )
        b11_layer3_in = MemRefType.get(
            (
                b10_InW3,
                1,
                b11_OutC2,
            ),
            uint8_ty,
        )

        # define wts
        b11_layer1_wts_size = b10_OutC3 * b11_OutC1
        b11_layer2_wts_size = 3 * 3 * b11_OutC2 * 1
        b11_layer3_wts_size = b11_OutC2 * b11_OutC3

        b11_layer1_wts = MemRefType.get((b10_OutC3 * b11_OutC1,), int8_ty)
        b11_layer2_wts = MemRefType.get((3 * 3 * b11_OutC2 * 1,), int8_ty)
        b11_layer3_wts = MemRefType.get((b11_OutC2 * b11_OutC3,), int8_ty)
        # b11_all_wts= MemRefType.get(( b11_OutC2 * b11_OutC3, ), int8_ty, )
        # output
        b11_layer1_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b11_OutC1,
            ),
            uint8_ty,
        )
        b11_layer2_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b11_OutC2,
            ),
            uint8_ty,
        )
        b11_layer3_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b11_OutC3,
            ),
            int8_ty,
        )
        # ************************ bneck12 ************************
        b12_layer1_in = MemRefType.get(
            (
                b10_InW1,
                1,
                b11_OutC3,
            ),
            int8_ty,
        )
        b12_layer2_in = MemRefType.get(
            (
                b10_InW1,
                1,
                b12_OutC1,
            ),
            uint8_ty,
        )
        b12_layer3_in = MemRefType.get(
            (
                b12_InW2,
                1,
                b12_OutC2,
            ),
            uint8_ty,
        )
        # define wts
        b12_layer1_wts_size = b11_OutC3 * b12_OutC1
        b12_layer2_wts_size = 3 * 3 * b12_OutC2 * 1
        b12_layer3_wts_size = b12_OutC2 * b12_OutC3

        b12_layer2_3_wts_size = b12_layer2_wts_size + b12_layer3_wts_size
        b12_layer1_wts = MemRefType.get((b11_OutC3 * b12_OutC1,), int8_ty)
        b12_layer2_wts = MemRefType.get((3 * 3 * b12_OutC2 * 1,), int8_ty)
        b12_layer3_wts = MemRefType.get((b12_OutC2 * b12_OutC3,), int8_ty)
        b12_all_wts = MemRefType.get(
            (b12_OutC2 * b12_OutC3,),
            int8_ty,
        )
        # output
        b12_layer1_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b12_OutC1,
            ),
            uint8_ty,
        )
        b12_layer2_out = MemRefType.get(
            (
                b12_InW2,
                1,
                b12_OutC2,
            ),
            uint8_ty,
        )
        b12_layer3_out = MemRefType.get(
            (
                b12_InW2,
                1,
                b12_OutC3,
            ),
            int8_ty,
        )
        # Input
        # ******************************************************************* Post block *******************************************************************
        PostOutputSplit = 8  # split output channels based on your preference
        # OC8=post_L1_OutC//(8*PostOutputSplit) # how many loops of OC8
        PostRepeatChannels = math.floor(post_L1_InH)
        #  ±±±±±±±±±±±± layer 1 (Avg pool) ±±±±±±±±±±±±
        post_act_in_Layer1_all = 1 * post_L1_InW * post_L1_InC
        post_act_in_Layer1_L2L1 = post_L1_InW * post_L1_InC
        post_act_out_Layer1 = 1 * post_L1_OutC_padd
        post_wts_Layer1_all = post_L1_InC * post_L1_OutC
        post_wts_Layer1_L2L1 = post_L1_InC * post_L1_OutC // PostOutputSplit
        post_L1_wts_size = post_L1_InC * post_L1_OutC
        post_L1_wts_size32b = post_L1_wts_size // 4

        ty_post_act_Layer1_L2L1 = MemRefType.get(
            (
                post_L1_InW,
                1,
                post_L1_InC,
            ),
            int8_ty,
        )
        ty_post_wts_Layer1_all = MemRefType.get((post_wts_Layer1_all,), int8_ty)
        ty_post_wts_Layer1_L2L1 = MemRefType.get((post_wts_Layer1_L2L1,), int8_ty)
        ty_post_Layer1_out = MemRefType.get((post_act_out_Layer1,), uint16_ty)

        #  ±±±±±±±±±±±± layer 2 (FC) ±±±±±±±±±±±±
        PostOutputSplitL2 = 40  # split output channels based on your preference
        post_L2_n_core = 4
        post_act_in_Layer2_all = 1 * 1 * post_L1_OutC_padd

        post_act_out_Layer2 = 1 * post_L2_OutC
        post_act_out_Layer2_split = (
            1 * post_L2_OutC // (PostOutputSplitL2 * post_L2_n_core)
        )

        post_wts_Layer2_all = post_L1_OutC_padd * post_L2_OutC

        post_wts_Layer2_split = post_wts_Layer2_all // (post_L2_n_core)
        post_wts_Layer2_split_2 = post_wts_Layer2_split // 2
        post_wts_Layer2_split_L1 = post_wts_Layer2_all // (
            PostOutputSplitL2 * post_L2_n_core
        )

        post_L2_wts_size = post_L1_OutC_padd * post_L2_OutC

        post_L2_wts_size32b = post_L2_wts_size // 4

        post_wts_Layer2_split_32b = post_wts_Layer2_split // 4
        post_wts_Layer2_all_32b = post_wts_Layer2_all // 4

        ty_post_act_Layer2_all = MemRefType.get((post_act_in_Layer2_all,), uint16_ty)
        ty_post_wts_Layer2_all = MemRefType.get((post_wts_Layer2_all,), int8_ty)
        ty_post_wts_Layer2_split = MemRefType.get((post_wts_Layer2_split,), int8_ty)
        ty_post_wts_Layer2_split_2 = MemRefType.get((post_wts_Layer2_split_2,), int8_ty)
        ty_post_wts_Layer2_split_L1 = MemRefType.get(
            (post_wts_Layer2_split_L1,), int8_ty
        )
        ty_post_Layer2_out_all = MemRefType.get((post_act_out_Layer2,), uint16_ty)
        ty_post_Layer2_out_split = MemRefType.get(
            (post_act_out_Layer2_split,), uint16_ty
        )

        # ******************************************************************* WTS B block *******************************************************************

        bn10_1_wts_ary = np.fromfile(
            weights_path + "bn10_1_chain.txt", sep=",", dtype=np.int8
        )
        bn10_2_wts_ary = np.fromfile(
            weights_path + "bn10_2_chain.txt", sep=",", dtype=np.int8
        )
        bn10_3_wts_ary = np.fromfile(
            weights_path + "bn10_3_chain.txt", sep=",", dtype=np.int8
        )

        bn11_1_wts_ary = np.fromfile(
            weights_path + "bn11_1_chain.txt", sep=",", dtype=np.int8
        )
        bn11_2_wts_ary = np.fromfile(
            weights_path + "bn11_2_chain.txt", sep=",", dtype=np.int8
        )
        bn11_3_wts_ary = np.fromfile(
            weights_path + "bn11_3_chain.txt", sep=",", dtype=np.int8
        )

        bn12_1_wts_ary = np.fromfile(
            weights_path + "bn12_1_chain.txt", sep=",", dtype=np.int8
        )
        # bn12_2_wts_ary=np.fromfile(weights_path+"bn12_2_chain.txt", sep=",", dtype=np.int8)
        # bn12_3_wts_ary=np.fromfile(weights_path+"bn12_3_chain.txt", sep=",", dtype=np.int8)
        bn12_2_3_wts_ary = np.fromfile(
            weights_path + "bn12_2_3_chain.txt", sep=",", dtype=np.int8
        )

        bn10_1_wts_static = buffer(
            bn10_tile_1,
            np.ndarray[(b10_layer1_wts_size,), np.dtype[np.int8]],
            "bn10_1_wts_static",
            initial_value=bn10_1_wts_ary,
        )
        bn10_2_wts_static = buffer(
            bn10_tile_2,
            np.ndarray[(b10_layer2_wts_size,), np.dtype[np.int8]],
            "bn10_2_wts_static",
            initial_value=bn10_2_wts_ary,
        )
        bn10_3_wts_static = buffer(
            bn10_tile_3,
            np.ndarray[(b10_layer3_wts_size,), np.dtype[np.int8]],
            "bn10_3_wts_static",
            initial_value=bn10_3_wts_ary,
        )

        bn11_1_wts_static = buffer(
            bn11_tile_1,
            np.ndarray[(b11_layer1_wts_size,), np.dtype[np.int8]],
            "bn11_1_wts_static",
            initial_value=bn11_1_wts_ary,
        )
        bn11_2_wts_static = buffer(
            bn11_tile_2,
            np.ndarray[(b11_layer2_wts_size,), np.dtype[np.int8]],
            "bn11_2_wts_static",
            initial_value=bn11_2_wts_ary,
        )
        bn11_3_wts_static = buffer(
            bn11_tile_3,
            np.ndarray[(b11_layer3_wts_size,), np.dtype[np.int8]],
            "bn11_3_wts_static",
            initial_value=bn11_3_wts_ary,
        )

        bn12_1_wts_static = buffer(
            bn12_tile_1,
            np.ndarray[(b12_layer1_wts_size,), np.dtype[np.int8]],
            "bn12_1_wts_static",
            initial_value=bn12_1_wts_ary,
        )
        bn12_2_3_wts_static = buffer(
            bn12_tile_2,
            np.ndarray[(b12_layer2_3_wts_size,), np.dtype[np.int8]],
            "bn12_2_3_wts_static",
            initial_value=bn12_2_3_wts_ary,
        )
        # bn12_2_wts_static = buffer(bn12_tile_2, (b12_layer2_wts_size,), int8_ty, "bn12_2_wts_static", initial_value=bn12_2_wts_ary)
        # bn12_3_wts_static = buffer(bn12_tile_3, (b12_layer3_wts_size,), int8_ty, "bn12_3_wts_static", initial_value=bn12_3_wts_ary)

        bn10_1_rtp = buffer(
            bn10_tile_1, np.ndarray[(16,), np.dtype[np.int32]], "bn10_1_rtp"
        )
        bn10_2_rtp = buffer(
            bn10_tile_2, np.ndarray[(16,), np.dtype[np.int32]], "bn10_2_rtp"
        )
        bn10_3_rtp = buffer(
            bn10_tile_3, np.ndarray[(16,), np.dtype[np.int32]], "bn10_3_rtp"
        )

        bn11_1_rtp = buffer(
            bn11_tile_1, np.ndarray[(16,), np.dtype[np.int32]], "bn11_1_rtp"
        )
        bn11_2_rtp = buffer(
            bn11_tile_2, np.ndarray[(16,), np.dtype[np.int32]], "bn11_2_rtp"
        )
        bn11_3_rtp = buffer(
            bn11_tile_3, np.ndarray[(16,), np.dtype[np.int32]], "bn11_3_rtp"
        )

        bn12_1_rtp = buffer(
            bn12_tile_1, np.ndarray[(16,), np.dtype[np.int32]], "bn12_1_rtp"
        )
        bn12_2_rtp = buffer(
            bn12_tile_2, np.ndarray[(16,), np.dtype[np.int32]], "bn12_2_rtp"
        )
        # bn12_3_rtp = buffer(bn12_tile_3, np.ndarray[(16,), np.dtype[np.int32]], "bn12_3_rtp")
        ##End

        # ************************ bneck13 ************************

        ty_bneck_13_layer1_in = MemRefType.get(
            (
                bneck_13_InW1,
                1,
                bneck_13_InC1,
            ),
            int8_ty,
        )
        ty_bneck_13_layer2_in = MemRefType.get(
            (
                bneck_13_InW2,
                1,
                bneck_13_OutC1,
            ),
            uint8_ty,
        )

        # define wts
        # layer1
        ty_bneck_13_layer1_wts_split = MemRefType.get(
            ((bneck_13_InC1 // InputSplit) * (bneck_13_OutC1 // OutputSplit),), int8_ty
        )
        ty_bneck_13_layer1_wts_full = MemRefType.get(
            (bneck_13_InC1 * bneck_13_OutC1,),
            int8_ty,
        )
        # layer2
        b13_layer2_wts_size = 3 * 3 * bneck_13_OutC2 * 1
        ty_bneck_13_layer2_wts = MemRefType.get((3 * 3 * bneck_13_OutC2 * 1,), int8_ty)
        # layer3
        ty_bneck_13_layer3_wts_split = MemRefType.get(
            ((bneck_13_OutC2 // InputSplit) * (bneck_13_OutC3 // OutputSplit2),),
            int8_ty,
        )
        ty_bneck_13_layer3_wts_full = MemRefType.get(
            (bneck_13_OutC2 * bneck_13_OutC3,),
            int8_ty,
        )

        # OUTPUT
        ty_bneck_13_layer1_out = MemRefType.get(
            (
                bneck_13_InW1,
                1,
                bneck_13_OutC1,
            ),
            uint8_ty,
        )
        ty_bneck_13_layer2_out = MemRefType.get(
            (
                bneck_13_InW3,
                1,
                bneck_13_OutC2,
            ),
            uint8_ty,
        )
        ty_bneck_13_layer2_out_split = MemRefType.get(
            (
                bneck_13_InW3,
                1,
                bneck_13_OutC2 // InputSplit,
            ),
            uint8_ty,
        )
        # layer3
        ty_bneck_13_layer3_out = MemRefType.get(
            (
                bneck_13_InW3,
                1,
                bneck_13_OutC3,
            ),
            int8_ty,
        )

        # HERE

        # ************************ bneck14 ************************

        ty_bneck_14_layer1_in = MemRefType.get(
            (
                bneck_14_InW1,
                1,
                bneck_14_InC1,
            ),
            int8_ty,
        )
        ty_bneck_14_layer2_in = MemRefType.get(
            (
                bneck_14_InW2,
                1,
                bneck_14_OutC1,
            ),
            uint8_ty,
        )

        # define wts
        # layer1
        ty_bneck_14_layer1_wts_split = MemRefType.get(
            ((bneck_14_InC1 // InputSplit) * (bneck_14_OutC1 // OutputSplit),), int8_ty
        )
        ty_bneck_14_layer1_wts_full = MemRefType.get(
            (bneck_14_InC1 * bneck_14_OutC1,),
            int8_ty,
        )
        # layer2
        b14_layer2_wts_size = 3 * 3 * bneck_14_OutC2 * 1
        ty_bneck_14_layer2_wts = MemRefType.get((3 * 3 * bneck_14_OutC2 * 1,), int8_ty)
        # layer3
        ty_bneck_14_layer3_wts_split = MemRefType.get(
            ((bneck_14_OutC2 // InputSplit) * (bneck_14_OutC3 // OutputSplit2),),
            int8_ty,
        )
        ty_bneck_14_layer3_wts_full = MemRefType.get(
            (bneck_14_OutC2 * bneck_14_OutC3,),
            int8_ty,
        )

        # OUTPUT
        ty_bneck_14_layer1_out = MemRefType.get(
            (
                bneck_14_InW1,
                1,
                bneck_14_OutC1,
            ),
            uint8_ty,
        )
        ty_bneck_14_layer2_out = MemRefType.get(
            (
                bneck_14_InW3,
                1,
                bneck_14_OutC2,
            ),
            uint8_ty,
        )
        ty_bneck_14_layer2_out_split = MemRefType.get(
            (
                bneck_14_InW3,
                1,
                bneck_14_OutC2 // InputSplit,
            ),
            uint8_ty,
        )
        # layer3
        ty_bneck_14_layer3_out = MemRefType.get(
            (
                bneck_14_InW3,
                1,
                bneck_14_OutC3,
            ),
            int8_ty,
        )

        # ************ wts block 13 ************
        # LAYER1
        bn13_wts_L3L2_layer1 = object_fifo(
            "bn13_wts_L3L2_layer1",
            ShimTile40,
            MemTile01,
            1,
            ty_bneck_13_layer1_wts_full,
        )
        bn13_wts_memtile_layer1_put = object_fifo(
            "bn13_wts_memtile_layer1_put",
            MemTile01,
            bn13_tile_layer1_put,
            [1, 1],
            ty_bneck_13_layer1_wts_split,
        )
        bn13_wts_memtile_layer1_get = object_fifo(
            "bn13_wts_memtile_layer1_get",
            MemTile01,
            bn13_tile_layer1_get,
            [1, 1],
            ty_bneck_13_layer1_wts_split,
        )
        object_fifo_link(
            bn13_wts_L3L2_layer1,
            [bn13_wts_memtile_layer1_put, bn13_wts_memtile_layer1_get],
            [],
            [0, (bneck_13_InC1 * bneck_13_OutC1) // 2],
        )
        bn13_wts_memtile_layer1_put.set_repeat_count(RepeatChannels)
        bn13_wts_memtile_layer1_get.set_repeat_count(RepeatChannels)
        # LAYER2
        bn13_2_wts_ary = np.fromfile(
            weights_path + "bn13_2_chain.txt", sep=",", dtype=np.int8
        )
        bn13_2_wts_static = buffer(
            bn13_tile_layer2,
            np.ndarray[(b13_layer2_wts_size,), np.dtype[np.int8]],
            "bn13_2_wts_static",
            initial_value=bn13_2_wts_ary,
        )

        # LAYER3
        bn13_wts_L3L2_layer3 = object_fifo(
            "bn13_wts_L3L2_layer3",
            ShimTile50,
            MemTile11,
            1,
            ty_bneck_13_layer3_wts_full,
        )
        bn13_wts_memtile_layer3_put = object_fifo(
            "bn13_wts_memtile_layer3_put",
            MemTile11,
            bn13_tile_layer3_put,
            1,
            ty_bneck_13_layer3_wts_split,
        )
        bn13_wts_memtile_layer3_get = object_fifo(
            "bn13_wts_memtile_layer3_get",
            MemTile11,
            bn13_tile_layer3_get,
            1,
            ty_bneck_13_layer3_wts_split,
        )
        object_fifo_link(
            bn13_wts_L3L2_layer3,
            [bn13_wts_memtile_layer3_put, bn13_wts_memtile_layer3_get],
            [],
            [0, (bneck_13_OutC2 * bneck_13_OutC3) // 2],
        )
        bn13_wts_memtile_layer3_put.set_repeat_count(RepeatChannels)
        bn13_wts_memtile_layer3_get.set_repeat_count(RepeatChannels)

        # ************ wts block 14 ************
        # wts for new block
        bn14_wts_L3L2_layer1 = object_fifo(
            "bn14_wts_L3L2_layer1",
            ShimTile60,
            MemTile21,
            1,
            ty_bneck_14_layer1_wts_full,
        )
        bn14_wts_memtile_layer1_put = object_fifo(
            "bn14_wts_memtile_layer1_put",
            MemTile21,
            bn14_tile_layer1_put,
            [1, 1],
            ty_bneck_14_layer1_wts_split,
        )
        bn14_wts_memtile_layer1_get = object_fifo(
            "bn14_wts_memtile_layer1_get",
            MemTile21,
            bn14_tile_layer1_get,
            [1, 1],
            ty_bneck_14_layer1_wts_split,
        )
        object_fifo_link(
            bn14_wts_L3L2_layer1,
            [bn14_wts_memtile_layer1_put, bn14_wts_memtile_layer1_get],
            [],
            [0, (bneck_14_InC1 * bneck_14_OutC1) // 2],
        )
        bn14_wts_memtile_layer1_put.set_repeat_count(RepeatChannels)
        bn14_wts_memtile_layer1_get.set_repeat_count(RepeatChannels)
        # LAYER2
        bn14_2_wts_ary = np.fromfile(
            weights_path + "bn14_2_chain.txt", sep=",", dtype=np.int8
        )
        bn14_2_wts_static = buffer(
            bn14_tile_layer2,
            np.ndarray[(b14_layer2_wts_size,), np.dtype[np.int8]],
            "bn14_2_wts_static",
            initial_value=bn14_2_wts_ary,
        )
        # LAYER3
        bn14_wts_L3L2_layer3 = object_fifo(
            "bn14_wts_L3L2_layer3",
            ShimTile70,
            MemTile31,
            1,
            ty_bneck_14_layer3_wts_full,
        )
        bn14_wts_memtile_layer3_put = object_fifo(
            "bn14_wts_memtile_layer3_put",
            MemTile31,
            bn14_tile_layer3_put,
            [1, 1],
            ty_bneck_14_layer3_wts_split,
        )
        bn14_wts_memtile_layer3_get = object_fifo(
            "bn14_wts_memtile_layer3_get",
            MemTile31,
            bn14_tile_layer3_get,
            [1, 1],
            ty_bneck_14_layer3_wts_split,
        )
        object_fifo_link(
            bn14_wts_L3L2_layer3,
            [bn14_wts_memtile_layer3_put, bn14_wts_memtile_layer3_get],
            [],
            [0, (bneck_14_OutC2 * bneck_14_OutC3) // 2],
        )
        bn14_wts_memtile_layer3_put.set_repeat_count(RepeatChannels)
        bn14_wts_memtile_layer3_get.set_repeat_count(RepeatChannels)

        # Set up compute tiles
        rtp_bn13_tile_layer1_get = buffer(
            bn13_tile_layer1_get,
            np.ndarray[(16,), np.dtype[np.int32]],
            "rtp_bn13_tile_layer1_get",
        )
        rtp_bn13_tile_layer2 = buffer(
            bn13_tile_layer2,
            np.ndarray[(16,), np.dtype[np.int32]],
            "rtp_bn13_tile_layer2",
        )
        rtp_bn13_tile_layer3_get = buffer(
            bn13_tile_layer3_get,
            np.ndarray[(16,), np.dtype[np.int32]],
            "rtp_bn13_tile_layer3_get",
        )

        rtp_bn14_tile_layer1_get = buffer(
            bn14_tile_layer1_get,
            np.ndarray[(16,), np.dtype[np.int32]],
            "rtp_bn14_tile_layer1_get",
        )
        rtp_bn14_tile_layer2 = buffer(
            bn14_tile_layer2,
            np.ndarray[(16,), np.dtype[np.int32]],
            "rtp_bn14_tile_layer2",
        )
        rtp_bn14_tile_layer3_get = buffer(
            bn14_tile_layer3_get,
            np.ndarray[(16,), np.dtype[np.int32]],
            "rtp_bn14_tile_layer3_get",
        )

        # ************ wts POST block ************
        # post_L1_wts_L3_L2 = object_fifo("post_L1_wts_L3_L2", ShimTile70, MemTile41, 1, ty_post_wts_Layer1_all )
        # post_L1_wts_L2_L1 = object_fifo("post_L1_wts_L2_L1", MemTile41, PostL1Tile, 2, ty_post_wts_Layer1_L2L1)
        # object_fifo_link(post_L1_wts_L3_L2, [post_L1_wts_L2_L1],[],[0])
        # post_L1_wts_L2_L1.set_repeat_count(PostRepeatChannels)
        post_L1_wts_ary = np.fromfile(
            weights_path + "post_conv_chain.txt", sep=",", dtype=np.int8
        )
        post_L1_wts_prod_lock = lock(MemTile41, lock_id=2, init=0)
        post_L1_wts_cons_lock = lock(MemTile41, lock_id=3, init=PostRepeatChannels)
        post_L1_wts_buff = buffer(
            MemTile41,
            np.ndarray[(post_wts_Layer1_all,), np.dtype[np.int8]],
            "post_L1_wts_buff",
            initial_value=post_L1_wts_ary,
        )

        post_L1_tile_prod_lock = lock(PostL1Tile, lock_id=0, init=1)
        post_L1_tile_cons_lock = lock(PostL1Tile, lock_id=1, init=0)
        post_L1_tile_buff = buffer(
            PostL1Tile,
            np.ndarray[(post_wts_Layer1_L2L1,), np.dtype[np.int8]],
            "post_L1_tile_buff",
        )

        rtp_post_L1 = buffer(
            PostL1Tile, np.ndarray[(16,), np.dtype[np.int32]], "rtp_post_L1"
        )
        rtp_post_L2_C1 = buffer(
            PostL2Tile_1, np.ndarray[(16,), np.dtype[np.int32]], "rtp_post_L2_C1"
        )
        rtp_post_L2_C2 = buffer(
            PostL2Tile_2, np.ndarray[(16,), np.dtype[np.int32]], "rtp_post_L2_C2"
        )

        post_L2_wts_L2L1_ty = MemRefType.get((post_wts_Layer2_split,), int8_ty)
        post_L2_in_ty = MemRefType.get(
            (
                1,
                1,
                post_L1_OutC,
            ),
            int8_ty,
        )
        post_L2_out_ty = MemRefType.get(
            (
                1,
                1,
                post_L2_OutC,
            ),
            uint8_ty,
        )

        wts_ary_01 = np.fromfile(
            weights_path + "FC1_0_chain.txt", sep=",", dtype=np.int8
        )
        wts_ary_21 = np.fromfile(
            weights_path + "FC1_1_chain.txt", sep=",", dtype=np.int8
        )
        wts_ary_41 = np.fromfile(
            weights_path + "FC1_2_chain.txt", sep=",", dtype=np.int8
        )
        wts_ary_61 = np.fromfile(
            weights_path + "FC1_3_chain.txt", sep=",", dtype=np.int8
        )

        wts_ary_11 = np.fromfile(
            weights_path + "FC2_0_chain.txt", sep=",", dtype=np.int8
        )
        wts_ary_31 = np.fromfile(
            weights_path + "FC2_1_chain.txt", sep=",", dtype=np.int8
        )
        wts_ary_51 = np.fromfile(
            weights_path + "FC2_2_chain.txt", sep=",", dtype=np.int8
        )
        wts_ary_71 = np.fromfile(
            weights_path + "FC2_3_chain.txt", sep=",", dtype=np.int8
        )

        PostL2Tile_1_cons_prod_lock = lock(PostL2Tile_1, lock_id=2, init=1)
        PostL2Tile_1_cons_cons_lock = lock(PostL2Tile_1, lock_id=3, init=0)
        mem_L2_wts_core1 = buffer(
            PostL2Tile_1,
            np.ndarray[(post_wts_Layer2_split_L1,), np.dtype[np.int8]],
            name="mem_L2_wts_core1",
        )

        PostL2Tile_2_cons_prod_lock = lock(PostL2Tile_2, lock_id=2, init=1)
        PostL2Tile_2_cons_cons_lock = lock(PostL2Tile_2, lock_id=3, init=0)
        mem_L2_wts_core2 = buffer(
            PostL2Tile_2,
            np.ndarray[(post_wts_Layer2_split_L1,), np.dtype[np.int8]],
            name="mem_L2_wts_core2",
        )

        PostL2Tile_3_cons_prod_lock = lock(PostL2Tile_3, lock_id=2, init=1)
        PostL2Tile_3_cons_cons_lock = lock(PostL2Tile_3, lock_id=3, init=0)
        mem_L2_wts_core3 = buffer(
            PostL2Tile_3,
            np.ndarray[(post_wts_Layer2_split_L1,), np.dtype[np.int8]],
            name="mem_L2_wts_core3",
        )

        PostL2Tile_4_cons_prod_lock = lock(PostL2Tile_4, lock_id=2, init=1)
        PostL2Tile_4_cons_cons_lock = lock(PostL2Tile_4, lock_id=3, init=0)
        mem_L2_wts_core4 = buffer(
            PostL2Tile_4,
            np.ndarray[(post_wts_Layer2_split_L1,), np.dtype[np.int8]],
            name="mem_L2_wts_core4",
        )

        mem_01_prod_lock = lock(MemTile01, lock_id=0, init=0)
        mem_01_cons_lock = lock(MemTile01, lock_id=1, init=1)
        mem_01_buff = buffer(
            MemTile01,
            np.ndarray[(post_wts_Layer2_split,), np.dtype[np.int8]],
            "mem_01_buff",
            initial_value=wts_ary_01,
        )
        mem_11_prod_lock = lock(MemTile11, lock_id=0, init=0)
        mem_11_cons_lock = lock(MemTile11, lock_id=1, init=1)
        mem_11_buff = buffer(
            MemTile11,
            np.ndarray[(post_wts_Layer2_split,), np.dtype[np.int8]],
            "mem_11_buff",
            initial_value=wts_ary_11,
        )
        mem_21_prod_lock = lock(MemTile21, lock_id=0, init=0)
        mem_21_cons_lock = lock(MemTile21, lock_id=1, init=1)
        mem_21_buff = buffer(
            MemTile21,
            np.ndarray[(post_wts_Layer2_split,), np.dtype[np.int8]],
            "mem_21_buff",
            initial_value=wts_ary_21,
        )
        mem_31_prod_lock = lock(MemTile31, lock_id=0, init=0)
        mem_31_cons_lock = lock(MemTile31, lock_id=1, init=1)
        mem_31_buff = buffer(
            MemTile31,
            np.ndarray[(post_wts_Layer2_split,), np.dtype[np.int8]],
            "mem_31_buff",
            initial_value=wts_ary_31,
        )
        mem_41_prod_lock = lock(MemTile41, lock_id=0, init=0)
        mem_41_cons_lock = lock(MemTile41, lock_id=1, init=1)
        mem_41_buff = buffer(
            MemTile41,
            np.ndarray[(post_wts_Layer2_split,), np.dtype[np.int8]],
            "mem_41_buff",
            initial_value=wts_ary_41,
        )
        mem_51_prod_lock = lock(MemTile51, lock_id=0, init=0)
        mem_51_cons_lock = lock(MemTile51, lock_id=1, init=1)
        mem_51_buff = buffer(
            MemTile51,
            np.ndarray[(post_wts_Layer2_split,), np.dtype[np.int8]],
            "mem_51_buff",
            initial_value=wts_ary_51,
        )
        mem_61_prod_lock = lock(MemTile61, lock_id=0, init=0)
        mem_61_cons_lock = lock(MemTile61, lock_id=1, init=1)
        mem_61_buff = buffer(
            MemTile61,
            np.ndarray[(post_wts_Layer2_split,), np.dtype[np.int8]],
            "mem_61_buff",
            initial_value=wts_ary_61,
        )
        mem_71_prod_lock = lock(MemTile71, lock_id=0, init=0)
        mem_71_cons_lock = lock(MemTile71, lock_id=1, init=1)
        mem_71_buff = buffer(
            MemTile71,
            np.ndarray[(post_wts_Layer2_split,), np.dtype[np.int8]],
            "mem_71_buff",
            initial_value=wts_ary_71,
        )

        # AIE-array data movement
        flow(MemTile11, WireBundle.DMA, 0, PostL2Tile_1, WireBundle.DMA, 1)
        flow(MemTile31, WireBundle.DMA, 0, PostL2Tile_2, WireBundle.DMA, 1)
        flow(MemTile51, WireBundle.DMA, 0, PostL2Tile_3, WireBundle.DMA, 1)
        flow(MemTile71, WireBundle.DMA, 0, PostL2Tile_4, WireBundle.DMA, 1)

        flow(MemTile41, WireBundle.DMA, 0, PostL1Tile, WireBundle.DMA, 0)

        @memtile_dma(MemTile11)
        def m(block):
            s0 = dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
            with block[1]:
                use_lock(mem_01_cons_lock, LockAction.AcquireGreaterEqual)
                dma_bd(mem_01_buff)
                use_lock(mem_01_prod_lock, LockAction.Release)
                next_bd(block[2])
            with block[2]:
                use_lock(mem_11_cons_lock, LockAction.AcquireGreaterEqual)
                dma_bd(mem_11_buff)
                use_lock(mem_11_prod_lock, LockAction.Release)
                next_bd(block[1])
            with block[3]:
                EndOp()

        @memtile_dma(MemTile31)
        def m(block):
            s0 = dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
            with block[1]:
                use_lock(mem_21_cons_lock, LockAction.AcquireGreaterEqual)
                dma_bd(mem_21_buff)
                use_lock(mem_21_prod_lock, LockAction.Release)
                next_bd(block[2])
            with block[2]:
                use_lock(mem_31_cons_lock, LockAction.AcquireGreaterEqual)
                dma_bd(mem_31_buff)
                use_lock(mem_31_prod_lock, LockAction.Release)
                next_bd(block[1])
            with block[3]:
                EndOp()

        @memtile_dma(MemTile41)
        def m(block):
            s0 = dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(post_L1_wts_cons_lock, LockAction.AcquireGreaterEqual)
                dma_bd(post_L1_wts_buff)
                use_lock(post_L1_wts_prod_lock, LockAction.Release)
                next_bd(block[1])
            with block[2]:
                EndOp()

        @memtile_dma(MemTile51)
        def m(block):
            s0 = dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
            with block[1]:
                use_lock(mem_41_cons_lock, LockAction.AcquireGreaterEqual)
                dma_bd(mem_41_buff)
                use_lock(mem_41_prod_lock, LockAction.Release)
                next_bd(block[2])
            with block[2]:
                use_lock(mem_51_cons_lock, LockAction.AcquireGreaterEqual)
                dma_bd(mem_51_buff)
                use_lock(mem_51_prod_lock, LockAction.Release)
                next_bd(block[1])
            with block[3]:
                EndOp()

        @memtile_dma(MemTile71)
        def m(block):
            s0 = dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
            with block[1]:
                use_lock(mem_61_cons_lock, LockAction.AcquireGreaterEqual)
                dma_bd(mem_61_buff)
                use_lock(mem_61_prod_lock, LockAction.Release)
                next_bd(block[2])
            with block[2]:
                use_lock(mem_71_cons_lock, LockAction.AcquireGreaterEqual)
                dma_bd(mem_71_buff)
                use_lock(mem_71_prod_lock, LockAction.Release)
                next_bd(block[1])
            with block[3]:
                EndOp()

        @mem(PostL1Tile)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(post_L1_tile_prod_lock, LockAction.AcquireGreaterEqual)
                dma_bd(post_L1_tile_buff)
                use_lock(post_L1_tile_cons_lock, LockAction.Release)
                next_bd(block[1])
            with block[2]:
                EndOp()

        @mem(PostL2Tile_1)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 1, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(PostL2Tile_1_cons_prod_lock, LockAction.AcquireGreaterEqual)
                dma_bd(mem_L2_wts_core1)
                use_lock(PostL2Tile_1_cons_cons_lock, LockAction.Release)
                next_bd(block[1])
            with block[2]:
                EndOp()

        @mem(PostL2Tile_2)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 1, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(PostL2Tile_2_cons_prod_lock, LockAction.AcquireGreaterEqual)
                dma_bd(mem_L2_wts_core2)
                use_lock(PostL2Tile_2_cons_cons_lock, LockAction.Release)
                next_bd(block[1])
            with block[2]:
                EndOp()

        @mem(PostL2Tile_3)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 1, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(PostL2Tile_3_cons_prod_lock, LockAction.AcquireGreaterEqual)
                dma_bd(mem_L2_wts_core3)
                use_lock(PostL2Tile_3_cons_cons_lock, LockAction.Release)
                next_bd(block[1])
            with block[2]:
                EndOp()

        @mem(PostL2Tile_4)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 1, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(PostL2Tile_4_cons_prod_lock, LockAction.AcquireGreaterEqual)
                dma_bd(mem_L2_wts_core4)
                use_lock(PostL2Tile_4_cons_cons_lock, LockAction.Release)
                next_bd(block[1])
            with block[2]:
                EndOp()

        ##End
        # *************************************************************************************************************************************
        # Set up compute tiles
        rtpinit_tile = buffer(
            init_tile, np.ndarray[(16,), np.dtype[np.int32]], "rtp_init"
        )  # init
        rtpbn0_tile = buffer(
            bn0_tile, np.ndarray[(16,), np.dtype[np.int32]], "rtp_bn0"
        )  # bn0
        rtpbn1_tile = buffer(
            bn1_tile, np.ndarray[(16,), np.dtype[np.int32]], "rtp_bn1"
        )  # bn1
        rtpbn2_tile = buffer(
            bn2_tile, np.ndarray[(16,), np.dtype[np.int32]], "rtp_bn2"
        )  # bn2
        rtpbn3_tile = buffer(
            bn3_tile, np.ndarray[(16,), np.dtype[np.int32]], "rtp_bn3"
        )  # bn3
        rtp_bn4_5_tile = buffer(
            bn4_5_tile, np.ndarray[(16,), np.dtype[np.int32]], "rtp_bn4_5_tile"
        )  # bn0
        # rtpbn4_tile = buffer(bn4_tile, np.ndarray[(16,), np.dtype[np.int32]], "rtp_bn4") #bn4
        # rtpbn5_tile = buffer(bn5_tile, np.ndarray[(16,), np.dtype[np.int32]], "rtp_bn5") #bn5
        rtpbn6_tile = buffer(
            bn6_tile, np.ndarray[(16,), np.dtype[np.int32]], "rtp_bn6"
        )  # bn6
        rtpbn7_tile = buffer(
            bn7_tile, np.ndarray[(16,), np.dtype[np.int32]], "rtp_bn7"
        )  # bn7
        rtpbn8_bn9_tile = buffer(
            bn8_bn9_tile, np.ndarray[(16,), np.dtype[np.int32]], "rtp_bn8_bn9"
        )  # bn8+bn9
        # rtpComputeTile24 = buffer(ComputeTile24, np.ndarray[(16,), np.dtype[np.int32]], "rtp24") #bn9
        # AIE-array data movement with object fifos

        # Input
        act_in = object_fifo("act_in", ShimTile00, init_tile, [1, 5], tensorLayerIn_ty)

        init_wts_ary = np.fromfile(
            weights_path + "init_chain.txt", sep=",", dtype=np.int8
        )
        bn0_wts_ary = np.fromfile(
            weights_path + "bn0_chain.txt", sep=",", dtype=np.int8
        )
        bn1_wts_ary = np.fromfile(
            weights_path + "bn1_chain.txt", sep=",", dtype=np.int8
        )
        # bn0_1_wts_ary=np.fromfile(weights_path+"bn0_1_chain.txt", sep=",", dtype=np.int8)
        bn2_wts_ary = np.fromfile(
            weights_path + "bn2_chain.txt", sep=",", dtype=np.int8
        )
        bn3_wts_ary = np.fromfile(
            weights_path + "bn3_chain.txt", sep=",", dtype=np.int8
        )
        # bn4_wts_ary=np.fromfile(weights_path+"bn4_chain.txt", sep=",", dtype=np.int8)
        # bn5_wts_ary=np.fromfile(weights_path+"bn5_chain.txt", sep=",", dtype=np.int8)
        bn4_5_wts_ary = np.fromfile(
            weights_path + "bn4_5_chain.txt", sep=",", dtype=np.int8
        )
        bn6_wts_ary = np.fromfile(
            weights_path + "bn6_chain.txt", sep=",", dtype=np.int8
        )
        bn7_wts_ary = np.fromfile(
            weights_path + "bn7_chain.txt", sep=",", dtype=np.int8
        )
        bn8_9_wts_ary = np.fromfile(
            weights_path + "bn8_9_chain.txt", sep=",", dtype=np.int8
        )

        init_wts_static = buffer(
            init_tile,
            np.ndarray[(init_weights_size,), np.dtype[np.int8]],
            "init_wts_static",
            initial_value=init_wts_ary,
        )
        bn0_wts_static = buffer(
            bn0_tile,
            np.ndarray[(bn0_weights_size,), np.dtype[np.int8]],
            "bn0_wts_static",
            initial_value=bn0_wts_ary,
        )
        bn1_wts_static = buffer(
            bn1_tile,
            np.ndarray[(bn1_weights_size,), np.dtype[np.int8]],
            "bn1_wts_static",
            initial_value=bn1_wts_ary,
        )
        # bn0_1_wts_static = buffer(bn0_tile, (bn0_1_weights_size,), int8_ty, "bn0_1_wts_static", initial_value=bn0_1_wts_ary)

        bn2_wts_static = buffer(
            bn2_tile,
            np.ndarray[(bn2_weights_size,), np.dtype[np.int8]],
            "bn2_wts_static",
            initial_value=bn2_wts_ary,
        )
        bn3_wts_static = buffer(
            bn3_tile,
            np.ndarray[(bn3_weights_size,), np.dtype[np.int8]],
            "bn3_wts_static",
            initial_value=bn3_wts_ary,
        )
        bn4_5_wts_static = buffer(
            bn4_5_tile,
            np.ndarray[(bn4_5_weights_size,), np.dtype[np.int8]],
            "bn4_5_wts_static",
            initial_value=bn4_5_wts_ary,
        )
        # bn4_wts_static = buffer(bn6_tile, (bn4_weights_size,), int8_ty, "bn4_wts_static", initial_value=bn4_wts_ary)
        # bn5_wts_static = buffer(bn7_tile, (bn5_weights_size,), int8_ty, "bn5_wts_static", initial_value=bn5_wts_ary)
        bn6_wts_static = buffer(
            bn6_tile,
            np.ndarray[(bn6_weights_size,), np.dtype[np.int8]],
            "bn6_wts_static",
            initial_value=bn6_wts_ary,
        )
        bn7_wts_static = buffer(
            bn7_tile,
            np.ndarray[(bn7_weights_size,), np.dtype[np.int8]],
            "bn7_wts_static",
            initial_value=bn7_wts_ary,
        )
        bn8_9_wts_static = buffer(
            bn8_bn9_tile,
            np.ndarray[(bn8_9_weights_size,), np.dtype[np.int8]],
            "bn8_9_wts_static",
            initial_value=bn8_9_wts_ary,
        )

        # # ******************************************************************init_conv*****************************************************************
        bn0_tensorLayer2In_ty = MemRefType.get(
            (tensorL0_2InW, 1, tensorL0_2InC), uint8_ty
        )
        bn0_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL0_3InC * 1,), int8_ty)
        bn0_tensorLayer2Out_ty = MemRefType.get(
            (tensorL0_3InW, 1, tensorL0_3InC), uint8_ty
        )

        bn0_tensorLayer3In_ty = bn0_tensorLayer2Out_ty
        bn0_weightsLayer3_ty = MemRefType.get(
            (1 * 1 * tensorL0_3InC * tensorL0_3OutC,), int8_ty
        )
        bn0_tensorLayer3Out_ty = MemRefType.get(
            (tensorL0_3InW, 1, tensorL0_3OutC), int8_ty
        )

        # AIE Core Function declarations
        init_conv2dk3_stride2 = external_func(
            "conv2dk3_stride2_i8",
            inputs=[
                tensorLayerIn_ty,
                tensorLayerIn_ty,
                tensorLayerIn_ty,
                init_weights_ty,
                bn0_tensorLayer2In_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        act_init_bn0 = object_fifo(
            "act_init_bn0", init_tile, bn0_tile, [5, 3], bn0_tensorLayer2In_ty
        )
        # act_init_bn01 = object_fifo("act_out", init_tile,ShimTile30, [3, 3], bn0_tensorLayer2In_ty)

        initConv(
            init_tile,
            act_in,
            init_wts_static,
            act_init_bn0,
            init_conv2dk3_stride2,
            tensor_init_InW,
            tensor_init_InH,
            tensor_init_InC,
            tensor_init_OutW,
            tensor_init_OutH,
            tensor_init_OutC,
            init_scaleFactor,
        )

        # # ******************************************************************bn0******************************************************************
        # temporary types for tensor to enable intial test

        bn0_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func(
            "bn0_conv2dk3_dw_stride1_relu_ui8_ui8",
            inputs=[
                bn0_tensorLayer2In_ty,
                bn0_tensorLayer2In_ty,
                bn0_tensorLayer2In_ty,
                bn0_weightsLayer2_ty,
                bn0_tensorLayer2Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        bn0_conv2dk1_skip_ui8_ui8_i8 = external_func(
            "bn0_conv2dk1_skip_ui8_ui8_i8",
            inputs=[
                bn0_tensorLayer3In_ty,
                bn0_weightsLayer3_ty,
                bn0_tensorLayer3Out_ty,
                bn0_tensorLayer2In_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        # Compute tile bn0_combined_conv2dk3dwstride1_conv2dk1skipui8
        bn0_objectArchiveName = "bn0_combined_conv2dk3dwstride1_conv2dk1skipui8.a"

        bn0_tensorLayer2Out_ty = MemRefType.get(
            (tensorL0_3InW, 1, tensorL0_3InC), uint8_ty
        )
        bn0_tensorLayer3Out_ty = MemRefType.get(
            (tensorL0_3InW, 1, tensorL0_3OutC), int8_ty
        )

        # between compute tiles
        act_bn0_bn1 = object_fifo(
            "act_bn0_bn1", bn0_tile, bn1_tile, 2, bn0_tensorLayer3Out_ty
        )
        # act_bn0_bn1 = object_fifo("act_out", bn0_tile, ShimTile10, 2, bn0_tensorLayer3Out_ty)

        bottleneckBN0Static(
            "bn0",
            bn0_tile,
            act_init_bn0,
            bn0_wts_static,
            act_bn0_bn1,
            rtpbn0_tile,
            bn0_objectArchiveName,
            bn0_conv2dk3_dw_stride1_relu_ui8_ui8,
            bn0_conv2dk1_skip_ui8_ui8_i8,
            bn0_tensorLayer2Out_ty,
            tensorL0_2InW,
            tensorL0_2InH,
            tensorL0_2InC,
            bn0_depthWiseStride,
            tensorL0_3OutC,
            bn0_withSkip,
            bn0_scaleFactor2,
            bn0_scaleFactor3,
            bn0_scaleFactorAdd,
        )

        # # # # # ******************************************************************bn2******************************************************************
        #  # temporary types for tensor to enable intial test
        bn1_tensorLayer1In_ty = MemRefType.get(
            (tensorL1_1InW, 1, tensorL1_1InC), int8_ty
        )
        bn1_weightsLayer1_ty = MemRefType.get(
            (1 * 1 * tensorL1_1InC * tensorL1_2InC,), int8_ty
        )
        bn1_tensorLayer1Out_ty = MemRefType.get(
            (tensorL1_2InW, 1, tensorL1_2InC), uint8_ty
        )

        bn1_tensorLayer2In_ty = bn1_tensorLayer1Out_ty
        bn1_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL1_3InC * 1,), int8_ty)
        bn1_tensorLayer2Out_ty = MemRefType.get(
            (tensorL1_3InW, 1, tensorL1_3InC), uint8_ty
        )

        bn1_tensorLayer3In_ty = bn1_tensorLayer2Out_ty
        bn1_weightsLayer3_ty = MemRefType.get(
            (1 * 1 * tensorL1_3InC * tensorL1_3OutC,), int8_ty
        )
        bn1_tensorLayer3Out_ty = MemRefType.get(
            (tensorL1_3InW, 1, tensorL1_3OutC), int8_ty
        )

        # AIE Core Function declarations
        bn1_conv2dk1_relu_i8_ui8 = external_func(
            "bn1_conv2dk1_relu_i8_ui8",
            inputs=[
                bn1_tensorLayer1In_ty,
                bn1_weightsLayer1_ty,
                bn1_tensorLayer1Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn1_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func(
            "bn1_conv2dk3_dw_stride2_relu_ui8_ui8",
            inputs=[
                bn1_tensorLayer2In_ty,
                bn1_tensorLayer2In_ty,
                bn1_tensorLayer2In_ty,
                bn1_weightsLayer2_ty,
                bn1_tensorLayer2Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn1_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func(
            "bn1_conv2dk3_dw_stride1_relu_ui8_ui8",
            inputs=[
                bn1_tensorLayer2In_ty,
                bn1_tensorLayer2In_ty,
                bn1_tensorLayer2In_ty,
                bn1_weightsLayer2_ty,
                bn1_tensorLayer2Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn1_conv2dk1_skip_ui8_i8_i8 = external_func(
            "bn1_conv2dk1_skip_ui8_i8_i8",
            inputs=[
                bn1_tensorLayer3In_ty,
                bn1_weightsLayer3_ty,
                bn1_tensorLayer3Out_ty,
                bn1_tensorLayer3Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        bn1_conv2dk1_ui8_i8 = external_func(
            "bn1_conv2dk1_ui8_i8",
            inputs=[
                bn1_tensorLayer3In_ty,
                bn1_weightsLayer3_ty,
                bn1_tensorLayer3Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        # Compute tile
        bn1_objectArchiveName = (
            "bn1_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a"
            % (bn1_depthWiseStride, "skip" if (bn1_withSkip) else "")
        )
        bn1_tensorLayer1Out_ty = MemRefType.get(
            (tensorL1_2InW, 1, tensorL1_2InC), uint8_ty
        )
        bn1_tensorLayer2Out_ty = MemRefType.get(
            (tensorL1_3InW, 1, tensorL1_3InC), uint8_ty
        )
        bn1_tensorLayer3Out_ty = MemRefType.get(
            (tensorL1_3InW, 1, tensorL1_3OutC), int8_ty
        )

        # between compute tiles
        act_bn1_bn2 = object_fifo(
            "act_bn1_bn2", bn1_tile, bn2_tile, 2, bn1_tensorLayer3Out_ty
        )
        # act_bn1_bn2 = object_fifo("act_out", bn1_tile, ShimTile10, 2, bn1_tensorLayer3Out_ty)

        bottleneckACoreStatic(
            "bn1",
            bn1_tile,
            act_bn0_bn1,
            bn1_wts_static,
            act_bn1_bn2,
            rtpbn1_tile,
            bn1_objectArchiveName,
            bn1_conv2dk1_relu_i8_ui8,
            bn1_conv2dk3_dw_stride1_relu_ui8_ui8,
            bn1_conv2dk3_dw_stride2_relu_ui8_ui8,
            bn1_conv2dk1_ui8_i8,
            bn1_conv2dk1_skip_ui8_i8_i8,
            bn1_tensorLayer1Out_ty,
            bn1_tensorLayer2Out_ty,
            tensorL1_1InW,
            tensorL1_1InH,
            tensorL1_1InC,
            bn1_depthWiseStride,
            bn1_depthWiseChannels,
            tensorL1_3OutC,
            bn1_withSkip,
            bn1_scaleFactor1,
            bn1_scaleFactor2,
            bn1_scaleFactor3,
            bn1_scaleFactorAdd,
        )

        # Compute tile
        #     bn01_objectArchiveName = "fused_bn0_bn1.a"
        #     bn0_tensorLayer0_2Out_ty = MemRefType.get((tensorL0_2InW, 1, tensorL0_2OutC),uint8_ty)
        #     bn0_tensorLayer0_3Out_ty = MemRefType.get((tensorL0_3InW, 1, tensorL0_3OutC),int8_ty)
        #     bn1_tensorLayer1_1Out_ty = MemRefType.get((tensorL1_1InW, 1, tensorL1_1OutC),uint8_ty)
        #     bn1_tensorLayer1_2Out_ty = MemRefType.get((tensorL1_3InW, 1, tensorL1_2OutC),uint8_ty)
        #     bn1_tensorLayer1_3Out_ty = MemRefType.get((tensorL1_3InW, 1, tensorL1_3OutC),int8_ty)

        #    # between compute tiles
        #     act_bn01_bn2 = object_fifo("act_bn01_bn2", bn0_tile, bn2_tile, 2, bn1_tensorLayer1_3Out_ty)
        #     # act_out = object_fifo("act_out", bn0_tile, ShimTile10, 1, bn1_tensorLayer1_3Out_ty)
        #     bottleneckAFused("bn01", bn0_tile, act_init_bn0, bn0_1_wts_static, act_bn01_bn2, rtpbn0_tile, bn01_objectArchiveName,
        #                      bn0_conv2dk3_dw_stride1_relu_ui8_ui8, bn0_conv2dk1_skip_ui8_ui8_i8, bn1_conv2dk1_relu_i8_ui8, bn1_conv2dk3_dw_stride2_relu_ui8_ui8, bn1_conv2dk1_ui8_i8,
        #                      bn0_tensorLayer0_2Out_ty, bn0_tensorLayer0_3Out_ty,bn1_tensorLayer1_1Out_ty,bn1_tensorLayer1_2Out_ty, tensorL0_2InW, tensorL0_2InH, tensorL0_2InC,  bn1_depthWiseStride, bn1_depthWiseChannels, tensorL1_3OutC,
        #                      bn0_scaleFactor2, bn0_scaleFactor3,  bn0_scaleFactorAdd,bn1_scaleFactor1, bn1_scaleFactor2, bn1_scaleFactor3)

        # # # # # ******************************************************************bn2******************************************************************
        #  # temporary types for tensor to enable intial test
        bn2_tensorLayer1In_ty = MemRefType.get(
            (tensorL2_1InW, 1, tensorL2_1InC), int8_ty
        )
        bn2_weightsLayer1_ty = MemRefType.get(
            (1 * 1 * tensorL2_1InC * tensorL2_2InC,), int8_ty
        )
        bn2_tensorLayer1Out_ty = MemRefType.get(
            (tensorL2_2InW, 1, tensorL2_2InC), uint8_ty
        )

        bn2_tensorLayer2In_ty = bn2_tensorLayer1Out_ty
        bn2_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL2_3InC * 1,), int8_ty)
        bn2_tensorLayer2Out_ty = MemRefType.get(
            (tensorL2_3InW, 1, tensorL2_3InC), uint8_ty
        )

        bn2_tensorLayer3In_ty = bn2_tensorLayer2Out_ty
        bn2_weightsLayer3_ty = MemRefType.get(
            (1 * 1 * tensorL2_3InC * tensorL2_3OutC,), int8_ty
        )
        bn2_tensorLayer3Out_ty = MemRefType.get(
            (tensorL2_3InW, 1, tensorL2_3OutC), int8_ty
        )

        # AIE Core Function declarations
        bn2_conv2dk1_relu_i8_ui8 = external_func(
            "bn2_conv2dk1_relu_i8_ui8",
            inputs=[
                bn2_tensorLayer1In_ty,
                bn2_weightsLayer1_ty,
                bn2_tensorLayer1Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn2_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func(
            "bn2_conv2dk3_dw_stride2_relu_ui8_ui8",
            inputs=[
                bn2_tensorLayer2In_ty,
                bn2_tensorLayer2In_ty,
                bn2_tensorLayer2In_ty,
                bn2_weightsLayer2_ty,
                bn2_tensorLayer2Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn2_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func(
            "bn2_conv2dk3_dw_stride1_relu_ui8_ui8",
            inputs=[
                bn2_tensorLayer2In_ty,
                bn2_tensorLayer2In_ty,
                bn2_tensorLayer2In_ty,
                bn2_weightsLayer2_ty,
                bn2_tensorLayer2Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn2_conv2dk1_skip_ui8_i8_i8 = external_func(
            "bn2_conv2dk1_skip_ui8_i8_i8",
            inputs=[
                bn2_tensorLayer3In_ty,
                bn2_weightsLayer3_ty,
                bn2_tensorLayer3Out_ty,
                bn2_tensorLayer3Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        bn2_conv2dk1_ui8_i8 = external_func(
            "bn2_conv2dk1_ui8_i8",
            inputs=[
                bn2_tensorLayer3In_ty,
                bn2_weightsLayer3_ty,
                bn2_tensorLayer3Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        # Compute tile
        bn2_objectArchiveName = (
            "bn2_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a"
            % (bn2_depthWiseStride, "skip" if (bn2_withSkip) else "")
        )
        bn2_tensorLayer1Out_ty = MemRefType.get(
            (tensorL2_2InW, 1, tensorL2_2InC), uint8_ty
        )
        bn2_tensorLayer2Out_ty = MemRefType.get(
            (tensorL2_3InW, 1, tensorL2_3InC), uint8_ty
        )
        bn2_tensorLayer3Out_ty = MemRefType.get(
            (tensorL2_3InW, 1, tensorL2_3OutC), int8_ty
        )

        # between compute tiles
        act_bn2_bn3 = object_fifo(
            "act_bn2_bn3", bn2_tile, bn3_tile, 2, bn2_tensorLayer3Out_ty
        )
        # act_out = object_fifo("act_out", bn2_tile, [ShimTile10], 1, bn2_tensorLayer3Out_ty)

        bottleneckACoreStatic(
            "bn2",
            bn2_tile,
            act_bn1_bn2,
            bn2_wts_static,
            act_bn2_bn3,
            rtpbn2_tile,
            bn2_objectArchiveName,
            bn2_conv2dk1_relu_i8_ui8,
            bn2_conv2dk3_dw_stride1_relu_ui8_ui8,
            bn2_conv2dk3_dw_stride2_relu_ui8_ui8,
            bn2_conv2dk1_ui8_i8,
            bn2_conv2dk1_skip_ui8_i8_i8,
            bn2_tensorLayer1Out_ty,
            bn2_tensorLayer2Out_ty,
            tensorL2_1InW,
            tensorL2_1InH,
            tensorL2_1InC,
            bn2_depthWiseStride,
            bn2_depthWiseChannels,
            tensorL2_3OutC,
            bn2_withSkip,
            bn2_scaleFactor1,
            bn2_scaleFactor2,
            bn2_scaleFactor3,
            bn2_scaleFactorAdd,
        )

        # # # # # ******************************************************************bn3******************************************************************
        # #  # temporary types for tensor to enable intial test
        bn3_tensorLayer1In_ty = MemRefType.get(
            (tensorL3_1InW, 1, tensorL3_1InC), int8_ty
        )
        bn3_weightsLayer1_ty = MemRefType.get(
            (1 * 1 * tensorL3_1InC * tensorL3_2InC,), int8_ty
        )
        bn3_tensorLayer1Out_ty = MemRefType.get(
            (tensorL3_2InW, 1, tensorL3_2InC), uint8_ty
        )

        bn3_tensorLayer2In_ty = bn3_tensorLayer1Out_ty
        bn3_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL3_3InC * 1,), int8_ty)
        bn3_tensorLayer2Out_ty = MemRefType.get(
            (tensorL3_3InW, 1, tensorL3_3InC), uint8_ty
        )

        bn3_tensorLayer3In_ty = bn3_tensorLayer2Out_ty
        bn3_weightsLayer3_ty = MemRefType.get(
            (1 * 1 * tensorL3_3InC * tensorL3_3OutC,), int8_ty
        )
        bn3_tensorLayer3Out_ty = MemRefType.get(
            (tensorL3_3InW, 1, tensorL3_3OutC), int8_ty
        )

        # AIE Core Function declarations
        bn3_conv2dk1_relu_i8_ui8 = external_func(
            "bn3_conv2dk1_relu_i8_ui8",
            inputs=[
                bn3_tensorLayer1In_ty,
                bn3_weightsLayer1_ty,
                bn3_tensorLayer1Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn3_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func(
            "bn3_conv2dk3_dw_stride2_relu_ui8_ui8",
            inputs=[
                bn3_tensorLayer2In_ty,
                bn3_tensorLayer2In_ty,
                bn3_tensorLayer2In_ty,
                bn3_weightsLayer2_ty,
                bn3_tensorLayer2Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn3_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func(
            "bn3_conv2dk3_dw_stride1_relu_ui8_ui8",
            inputs=[
                bn3_tensorLayer2In_ty,
                bn3_tensorLayer2In_ty,
                bn3_tensorLayer2In_ty,
                bn3_weightsLayer2_ty,
                bn3_tensorLayer2Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn3_conv2dk1_skip_ui8_i8_i8 = external_func(
            "bn3_conv2dk1_skip_ui8_i8_i8",
            inputs=[
                bn3_tensorLayer3In_ty,
                bn3_weightsLayer3_ty,
                bn3_tensorLayer3Out_ty,
                bn3_tensorLayer3Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn3_conv2dk1_ui8_i8 = external_func(
            "bn3_conv2dk1_ui8_i8",
            inputs=[
                bn3_tensorLayer3In_ty,
                bn3_weightsLayer3_ty,
                bn3_tensorLayer3Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        # Compute tile
        bn3_objectArchiveName = (
            "bn3_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a"
            % (bn3_depthWiseStride, "skip" if (bn3_withSkip) else "")
        )
        bn3_tensorLayer1Out_ty = MemRefType.get(
            (tensorL3_2InW, 1, tensorL3_2InC), uint8_ty
        )
        bn3_tensorLayer2Out_ty = MemRefType.get(
            (tensorL3_3InW, 1, tensorL3_3InC), uint8_ty
        )
        bn3_tensorLayer3Out_ty = MemRefType.get(
            (tensorL3_3InW, 1, tensorL3_3OutC), int8_ty
        )

        # # # between compute tiles
        act_bn3_bn4 = object_fifo(
            "act_bn3_bn4", bn3_tile, bn4_5_tile, 2, bn3_tensorLayer3Out_ty
        )
        # act_out = object_fifo("act_out", bn3_tile, [ShimTile10], 1, bn3_tensorLayer3Out_ty)
        bottleneckACoreStatic(
            "bn3",
            bn3_tile,
            act_bn2_bn3,
            bn3_wts_static,
            act_bn3_bn4,
            rtpbn3_tile,
            bn3_objectArchiveName,
            bn3_conv2dk1_relu_i8_ui8,
            bn3_conv2dk3_dw_stride1_relu_ui8_ui8,
            bn3_conv2dk3_dw_stride2_relu_ui8_ui8,
            bn3_conv2dk1_ui8_i8,
            bn3_conv2dk1_skip_ui8_i8_i8,
            bn3_tensorLayer1Out_ty,
            bn3_tensorLayer2Out_ty,
            tensorL3_1InW,
            tensorL3_1InH,
            tensorL3_1InC,
            bn3_depthWiseStride,
            bn3_depthWiseChannels,
            tensorL3_3OutC,
            bn3_withSkip,
            bn3_scaleFactor1,
            bn3_scaleFactor2,
            bn3_scaleFactor3,
            bn3_scaleFactorAdd,
        )

        # # # # # ******************************************************************bn4******************************************************************

        # temporary types for tensor to enable intial test
        bn4_tensorLayer1In_ty = MemRefType.get(
            (tensorL4_1InW, 1, tensorL4_1InC), int8_ty
        )
        bn4_weightsLayer1_ty = MemRefType.get(
            (1 * 1 * tensorL4_1InC * tensorL4_2InC,), int8_ty
        )
        bn4_tensorLayer1Out_ty = MemRefType.get(
            (tensorL4_2InW, 1, tensorL4_2InC), uint8_ty
        )

        bn4_tensorLayer2In_ty = bn4_tensorLayer1Out_ty
        bn4_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL4_3InC * 1,), int8_ty)
        bn4_tensorLayer2Out_ty = MemRefType.get(
            (tensorL4_3InW, 1, tensorL4_3InC), uint8_ty
        )

        bn4_tensorLayer3In_ty = bn4_tensorLayer2Out_ty
        bn4_weightsLayer3_ty = MemRefType.get(
            (1 * 1 * tensorL4_3InC * tensorL4_3OutC,), int8_ty
        )
        bn4_tensorLayer3Out_ty = MemRefType.get(
            (tensorL4_3InW, 1, tensorL4_3OutC), int8_ty
        )

        # AIE Core Function declarations
        bn4_conv2dk1_relu_i8_ui8 = external_func(
            "bn4_conv2dk1_relu_i8_ui8",
            inputs=[
                bn4_tensorLayer1In_ty,
                bn4_weightsLayer1_ty,
                bn4_tensorLayer1Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn4_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func(
            "bn4_conv2dk3_dw_stride2_relu_ui8_ui8",
            inputs=[
                bn4_tensorLayer2In_ty,
                bn4_tensorLayer2In_ty,
                bn4_tensorLayer2In_ty,
                bn4_weightsLayer2_ty,
                bn4_tensorLayer2Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn4_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func(
            "bn4_conv2dk3_dw_stride1_relu_ui8_ui8",
            inputs=[
                bn4_tensorLayer2In_ty,
                bn4_tensorLayer2In_ty,
                bn4_tensorLayer2In_ty,
                bn4_weightsLayer2_ty,
                bn4_tensorLayer2Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn4_conv2dk1_skip_ui8_i8_i8 = external_func(
            "bn4_conv2dk1_skip_ui8_i8_i8",
            inputs=[
                bn4_tensorLayer3In_ty,
                bn4_weightsLayer3_ty,
                bn4_tensorLayer3Out_ty,
                bn4_tensorLayer3Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn4_conv2dk1_ui8_i8 = external_func(
            "bn4_conv2dk1_ui8_i8",
            inputs=[
                bn4_tensorLayer3In_ty,
                bn4_weightsLayer3_ty,
                bn4_tensorLayer3Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        # Compute tile 6
        bn4_objectArchiveName = (
            "bn4_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a"
            % (bn4_depthWiseStride, "skip" if (bn4_withSkip) else "")
        )
        bn4_tensorLayer1Out_ty = MemRefType.get(
            (tensorL4_2InW, 1, tensorL4_2InC), uint8_ty
        )
        bn4_tensorLayer2Out_ty = MemRefType.get(
            (tensorL4_3InW, 1, tensorL4_3InC), uint8_ty
        )
        bn4_tensorLayer3Out_ty = MemRefType.get(
            (tensorL4_3InW, 1, tensorL4_3OutC), int8_ty
        )

        # # # between compute tiles
        # act_bn4_bn5 = object_fifo("act_bn4_bn5", bn6_tile, bn7_tile, 2, bn4_tensorLayer3Out_ty)
        # # act_out = object_fifo("act_out", bn6_tile, [ShimTile10], 1, bn4_tensorLayer3Out_ty)
        # bottleneckACoreStatic("bn4", bn6_tile, act_bn3_bn4, bn4_wts_static, act_bn4_bn5, rtp_bn6_tile, bn4_objectArchiveName,
        #                  bn4_conv2dk1_relu_i8_ui8, bn4_conv2dk3_dw_stride1_relu_ui8_ui8, bn4_conv2dk3_dw_stride2_relu_ui8_ui8, bn4_conv2dk1_ui8_i8, bn4_conv2dk1_skip_ui8_i8_i8,
        #                    bn4_tensorLayer1Out_ty, bn4_tensorLayer2Out_ty, tensorL4_1InW, tensorL4_1InH, tensorL4_1InC,  bn4_depthWiseStride, bn4_depthWiseChannels, tensorL4_3OutC, bn4_withSkip,
        #                    bn4_scaleFactor1, bn4_scaleFactor2, bn4_scaleFactor3,  bn4_scaleFactorAdd)

        # # # # # # ******************************************************************bn5******************************************************************

        # temporary types for tensor to enable intial test
        bn5_tensorLayer1In_ty = MemRefType.get(
            (tensorL5_1InW, 1, tensorL5_1InC), int8_ty
        )
        bn5_weightsLayer1_ty = MemRefType.get(
            (1 * 1 * tensorL5_1InC * tensorL5_2InC,), int8_ty
        )
        bn5_tensorLayer1Out_ty = MemRefType.get(
            (tensorL5_2InW, 1, tensorL5_2InC), uint8_ty
        )

        bn5_tensorLayer2In_ty = bn5_tensorLayer1Out_ty
        bn5_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL5_3InC * 1,), int8_ty)
        bn5_tensorLayer2Out_ty = MemRefType.get(
            (tensorL5_3InW, 1, tensorL5_3InC), uint8_ty
        )

        bn5_tensorLayer3In_ty = bn5_tensorLayer2Out_ty
        bn5_weightsLayer3_ty = MemRefType.get(
            (1 * 1 * tensorL5_3InC * tensorL5_3OutC,), int8_ty
        )
        bn5_tensorLayer3Out_ty = MemRefType.get(
            (tensorL5_3InW, 1, tensorL5_3OutC), int8_ty
        )

        # AIE Core Function declarations
        bn5_conv2dk1_relu_i8_ui8 = external_func(
            "bn5_conv2dk1_relu_i8_ui8",
            inputs=[
                bn5_tensorLayer1In_ty,
                bn5_weightsLayer1_ty,
                bn5_tensorLayer1Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn5_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func(
            "bn5_conv2dk3_dw_stride2_relu_ui8_ui8",
            inputs=[
                bn5_tensorLayer2In_ty,
                bn5_tensorLayer2In_ty,
                bn5_tensorLayer2In_ty,
                bn5_weightsLayer2_ty,
                bn5_tensorLayer2Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn5_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func(
            "bn5_conv2dk3_dw_stride1_relu_ui8_ui8",
            inputs=[
                bn5_tensorLayer2In_ty,
                bn5_tensorLayer2In_ty,
                bn5_tensorLayer2In_ty,
                bn5_weightsLayer2_ty,
                bn5_tensorLayer2Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn5_conv2dk1_skip_ui8_i8_i8 = external_func(
            "bn5_conv2dk1_skip_ui8_i8_i8",
            inputs=[
                bn5_tensorLayer3In_ty,
                bn5_weightsLayer3_ty,
                bn5_tensorLayer3Out_ty,
                bn5_tensorLayer3Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn5_conv2dk1_ui8_i8 = external_func(
            "bn5_conv2dk1_ui8_i8",
            inputs=[
                bn5_tensorLayer3In_ty,
                bn5_weightsLayer3_ty,
                bn5_tensorLayer3Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        # Compute tile 6
        bn5_objectArchiveName = (
            "bn5_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a"
            % (bn5_depthWiseStride, "skip" if (bn5_withSkip) else "")
        )
        bn5_tensorLayer1Out_ty = MemRefType.get(
            (tensorL5_2InW, 1, tensorL5_2InC), uint8_ty
        )
        bn5_tensorLayer2Out_ty = MemRefType.get(
            (tensorL5_3InW, 1, tensorL5_3InC), uint8_ty
        )
        bn5_tensorLayer3Out_ty = MemRefType.get(
            (tensorL5_3InW, 1, tensorL5_3OutC), int8_ty
        )

        # between compute tiles
        # act_bn4_bn5 = object_fifo("act_bn4_bn5", bn3_tile, bn7_tile, 2, bn4_tensorLayer3Out_ty)
        act_bn5_bn6 = object_fifo(
            "act_bn5_bn6", bn4_5_tile, bn6_tile, 2, bn5_tensorLayer3Out_ty
        )
        # # act_out = object_fifo("act_out", bn7_tile, [ShimTile10], 1, bn5_tensorLayer3Out_ty)
        # bottleneckACoreStatic("bn5", bn7_tile, act_bn4_bn5, bn5_wts_static, act_bn5_bn6, rtp_bn7_tile, bn5_objectArchiveName,
        #                  bn5_conv2dk1_relu_i8_ui8, bn5_conv2dk3_dw_stride1_relu_ui8_ui8, bn5_conv2dk3_dw_stride2_relu_ui8_ui8, bn5_conv2dk1_ui8_i8, bn5_conv2dk1_skip_ui8_i8_i8,
        #                    bn5_tensorLayer1Out_ty, bn5_tensorLayer2Out_ty, tensorL5_1InW, tensorL5_1InH, tensorL5_1InC,  bn5_depthWiseStride, bn5_depthWiseChannels, tensorL5_3OutC, bn5_withSkip,
        #                    bn5_scaleFactor1, bn5_scaleFactor2, bn5_scaleFactor3,  bn5_scaleFactorAdd)

        bottleneckAFused_8and9(
            "bn4_bn5",
            bn4_5_tile,
            init_tile,
            act_bn3_bn4,
            bn4_5_wts_static,
            act_bn5_bn6,
            rtp_bn4_5_tile,
            "combined_bn_4_5.a",
            bn4_conv2dk1_relu_i8_ui8,
            bn4_conv2dk3_dw_stride1_relu_ui8_ui8,
            bn4_conv2dk1_skip_ui8_i8_i8,
            bn5_conv2dk1_relu_i8_ui8,
            bn5_conv2dk3_dw_stride1_relu_ui8_ui8,
            bn5_conv2dk1_skip_ui8_i8_i8,
            bn4_tensorLayer1Out_ty,
            bn4_tensorLayer2Out_ty,
            bn4_tensorLayer3Out_ty,
            bn5_tensorLayer1Out_ty,
            bn5_tensorLayer2Out_ty,
            tensorL4_1InW,
            tensorL4_1InH,
            tensorL4_1InC,
            bn4_depthWiseStride,
            bn4_depthWiseChannels,
            bn5_depthWiseStride,
            bn5_depthWiseChannels,
            tensorL5_3OutC,
            bn4_scaleFactor1,
            bn4_scaleFactor2,
            bn4_scaleFactor3,
            bn4_scaleFactorAdd,
            bn5_scaleFactor1,
            bn5_scaleFactor2,
            bn5_scaleFactor3,
            bn5_scaleFactorAdd,
        )

        # # temporary types for tensor to enable intial test
        bn6_tensorLayer1In_ty = MemRefType.get(
            (tensorL6_1InW, 1, tensorL6_1InC), int8_ty
        )
        bn6_weightsLayer1_ty = MemRefType.get(
            (1 * 1 * tensorL6_1InC * tensorL6_2InC,), int8_ty
        )
        bn6_tensorLayer2In_ty = MemRefType.get(
            (tensorL6_2InW, 1, tensorL6_2InC), uint8_ty
        )
        bn6_tensorLayer1Out_ty = bn6_tensorLayer2In_ty
        bn6_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL6_3InC * 1,), int8_ty)
        bn6_tensorLayer3In_ty = MemRefType.get(
            (tensorL6_3InW, 1, tensorL6_3InC), uint8_ty
        )
        bn6_tensorLayer2Out_ty = bn6_tensorLayer3In_ty
        bn6_weightsLayer3_ty = MemRefType.get(
            (1 * 1 * tensorL6_3InC * tensorL6_3OutC,), int8_ty
        )
        bn6_tensorLayer3Out_ty = MemRefType.get(
            (tensorL6_3InW, 1, tensorL6_3OutC), int8_ty
        )

        # AIE Core Function declarations
        bn6_conv2dk1_relu_i8_ui8 = external_func(
            "bn6_conv2dk1_relu_i8_ui8",
            inputs=[
                bn6_tensorLayer1In_ty,
                bn6_weightsLayer1_ty,
                bn6_tensorLayer1Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn6_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func(
            "bn6_conv2dk3_dw_stride2_relu_ui8_ui8",
            inputs=[
                bn6_tensorLayer2In_ty,
                bn6_tensorLayer2In_ty,
                bn6_tensorLayer2In_ty,
                bn6_weightsLayer2_ty,
                bn6_tensorLayer2Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn6_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func(
            "bn6_conv2dk3_dw_stride1_relu_ui8_ui8",
            inputs=[
                bn6_tensorLayer2In_ty,
                bn6_tensorLayer2In_ty,
                bn6_tensorLayer2In_ty,
                bn6_weightsLayer2_ty,
                bn6_tensorLayer2Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn6_conv2dk1_skip_ui8_i8_i8 = external_func(
            "bn6_conv2dk1_skip_ui8_i8_i8",
            inputs=[
                bn6_tensorLayer3In_ty,
                bn6_weightsLayer3_ty,
                bn6_tensorLayer3Out_ty,
                bn6_tensorLayer3Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn6_conv2dk1_ui8_i8 = external_func(
            "bn6_conv2dk1_ui8_i8",
            inputs=[
                bn6_tensorLayer3In_ty,
                bn6_weightsLayer3_ty,
                bn6_tensorLayer3Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        # # Compute tile 6
        bn6_objectArchiveName = (
            "bn6_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a"
            % (bn6_depthWiseStride, "skip" if (bn6_withSkip) else "")
        )
        bn6_tensorLayer1Out_ty = MemRefType.get(
            (tensorL6_2InW, 1, tensorL6_2InC), uint8_ty
        )
        bn6_tensorLayer2Out_ty = MemRefType.get(
            (tensorL6_3InW, 1, tensorL6_3InC), uint8_ty
        )
        bn6_tensorLayer3Out_ty = MemRefType.get(
            (tensorL6_3InW, 1, tensorL6_3OutC), int8_ty
        )

        # between compute tiles
        act_bn6_bn7 = object_fifo(
            "act_bn6_bn7", bn6_tile, bn7_tile, 2, bn6_tensorLayer3Out_ty
        )

        # act_out = object_fifo("act_out", bn6_tile, [ShimTile10], 1, bn6_tensorLayer3Out_ty)
        bottleneckACoreStatic(
            "bn6",
            bn6_tile,
            act_bn5_bn6,
            bn6_wts_static,
            act_bn6_bn7,
            rtpbn6_tile,
            bn6_objectArchiveName,
            bn6_conv2dk1_relu_i8_ui8,
            bn6_conv2dk3_dw_stride1_relu_ui8_ui8,
            bn6_conv2dk3_dw_stride2_relu_ui8_ui8,
            bn6_conv2dk1_ui8_i8,
            bn6_conv2dk1_skip_ui8_i8_i8,
            bn6_tensorLayer1Out_ty,
            bn6_tensorLayer2Out_ty,
            tensorL6_1InW,
            tensorL6_1InH,
            tensorL6_1InC,
            bn6_depthWiseStride,
            bn6_depthWiseChannels,
            tensorL6_3OutC,
            bn6_withSkip,
            bn6_scaleFactor1,
            bn6_scaleFactor2,
            bn6_scaleFactor3,
            bn6_scaleFactorAdd,
        )

        # ##### ******************************************************************************************************************************
        bn7_tensorLayer1In_ty = MemRefType.get(
            (tensorL7_1InW, 1, tensorL7_1InC), int8_ty
        )
        bn7_weightsLayer1_ty = MemRefType.get(
            (1 * 1 * tensorL7_1InC * tensorL7_2InC,), int8_ty
        )
        bn7_tensorLayer2In_ty = MemRefType.get(
            (tensorL7_2InW, 1, tensorL7_2InC), uint8_ty
        )
        bn7_tensorLayer1Out_ty = bn7_tensorLayer2In_ty
        bn7_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL7_3InC * 1,), int8_ty)
        bn7_tensorLayer3In_ty = MemRefType.get(
            (tensorL7_3InW, 1, tensorL7_3InC), uint8_ty
        )
        bn7_tensorLayer2Out_ty = bn7_tensorLayer3In_ty
        bn7_weightsLayer3_ty = MemRefType.get(
            (1 * 1 * tensorL7_3InC * tensorL7_3OutC,), int8_ty
        )
        bn7_tensorLayer3Out_ty = MemRefType.get(
            (tensorL7_3InW, 1, tensorL7_3OutC), int8_ty
        )

        # AIE Core Function declarations
        bn7_conv2dk1_relu_i8_ui8 = external_func(
            "bn7_conv2dk1_relu_i8_ui8",
            inputs=[
                bn7_tensorLayer1In_ty,
                bn7_weightsLayer1_ty,
                bn7_tensorLayer1Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn7_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func(
            "bn7_conv2dk3_dw_stride2_relu_ui8_ui8",
            inputs=[
                bn7_tensorLayer2In_ty,
                bn7_tensorLayer2In_ty,
                bn7_tensorLayer2In_ty,
                bn7_weightsLayer2_ty,
                bn7_tensorLayer2Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn7_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func(
            "bn7_conv2dk3_dw_stride1_relu_ui8_ui8",
            inputs=[
                bn7_tensorLayer2In_ty,
                bn7_tensorLayer2In_ty,
                bn7_tensorLayer2In_ty,
                bn7_weightsLayer2_ty,
                bn7_tensorLayer2Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn7_conv2dk1_skip_ui8_i8_i8 = external_func(
            "bn7_conv2dk1_skip_ui8_i8_i8",
            inputs=[
                bn7_tensorLayer3In_ty,
                bn7_weightsLayer3_ty,
                bn7_tensorLayer3Out_ty,
                bn7_tensorLayer3Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn7_conv2dk1_ui8_i8 = external_func(
            "bn7_conv2dk1_ui8_i8",
            inputs=[
                bn7_tensorLayer3In_ty,
                bn7_weightsLayer3_ty,
                bn7_tensorLayer3Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        bn7_objectArchiveName = (
            "bn7_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a"
            % (bn7_depthWiseStride, "skip" if (bn7_withSkip) else "")
        )

        # between compute tiles
        act_bn7_bn8 = object_fifo(
            "act_bn7_bn8", bn7_tile, bn8_bn9_tile, 2, bn7_tensorLayer3Out_ty
        )
        # act_out = object_fifo("act_out", bn7_tile, [ShimTile10], 1, bn7_tensorLayer3Out_ty)
        bottleneckACoreStatic(
            "bn7",
            bn7_tile,
            act_bn6_bn7,
            bn7_wts_static,
            act_bn7_bn8,
            rtpbn7_tile,
            bn7_objectArchiveName,
            bn7_conv2dk1_relu_i8_ui8,
            bn7_conv2dk3_dw_stride1_relu_ui8_ui8,
            bn7_conv2dk3_dw_stride2_relu_ui8_ui8,
            bn7_conv2dk1_ui8_i8,
            bn7_conv2dk1_skip_ui8_i8_i8,
            bn7_tensorLayer1Out_ty,
            bn7_tensorLayer2Out_ty,
            tensorL7_1InW,
            tensorL7_1InH,
            tensorL7_1InC,
            bn7_depthWiseStride,
            bn7_depthWiseChannels,
            tensorL7_3OutC,
            bn7_withSkip,
            bn7_scaleFactor1,
            bn7_scaleFactor2,
            bn7_scaleFactor3,
            bn7_scaleFactorAdd,
        )

        # ##### ******************************************************************************************************************************
        bn8_tensorLayer1In_ty = MemRefType.get(
            (tensorL8_1InW, 1, tensorL8_1InC), int8_ty
        )
        bn8_weightsLayer1_ty = MemRefType.get(
            (1 * 1 * tensorL8_1InC * tensorL8_2InC,), int8_ty
        )
        bn8_tensorLayer2In_ty = MemRefType.get(
            (tensorL8_2InW, 1, tensorL8_2InC), uint8_ty
        )
        bn8_tensorLayer1Out_ty = bn8_tensorLayer2In_ty
        bn8_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL8_3InC * 1,), int8_ty)
        bn8_tensorLayer3In_ty = MemRefType.get(
            (tensorL8_3InW, 1, tensorL8_3InC), uint8_ty
        )
        bn8_tensorLayer2Out_ty = bn8_tensorLayer3In_ty
        bn8_weightsLayer3_ty = MemRefType.get(
            (1 * 1 * tensorL8_3InC * tensorL8_3OutC,), int8_ty
        )
        bn8_tensorLayer3Out_ty = MemRefType.get(
            (tensorL8_3InW, 1, tensorL8_3OutC), int8_ty
        )

        bn9_tensorLayer1In_ty = MemRefType.get(
            (tensorL9_1InW, 1, tensorL9_1InC), int8_ty
        )
        bn9_weightsLayer1_ty = MemRefType.get(
            (1 * 1 * tensorL9_1InC * tensorL9_2InC,), int8_ty
        )
        bn9_tensorLayer2In_ty = MemRefType.get(
            (tensorL9_2InW, 1, tensorL9_2InC), uint8_ty
        )
        bn9_tensorLayer1Out_ty = bn9_tensorLayer2In_ty
        bn9_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL9_3InC * 1,), int8_ty)
        bn9_tensorLayer3In_ty = MemRefType.get(
            (tensorL9_3InW, 1, tensorL9_3InC), uint8_ty
        )
        bn9_tensorLayer2Out_ty = bn9_tensorLayer3In_ty
        bn9_weightsLayer3_ty = MemRefType.get(
            (1 * 1 * tensorL9_3InC * tensorL9_3OutC,), int8_ty
        )
        bn9_tensorLayer3Out_ty = MemRefType.get(
            (tensorL9_3InW, 1, tensorL9_3OutC), int8_ty
        )

        # # AIE Core Function declarations
        bn8_conv2dk1_relu_i8_ui8 = external_func(
            "bn8_conv2dk1_relu_i8_ui8",
            inputs=[
                bn8_tensorLayer1In_ty,
                bn8_weightsLayer1_ty,
                bn8_tensorLayer1Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn8_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func(
            "bn8_conv2dk3_dw_stride1_relu_ui8_ui8",
            inputs=[
                bn8_tensorLayer2In_ty,
                bn8_tensorLayer2In_ty,
                bn8_tensorLayer2In_ty,
                bn8_weightsLayer2_ty,
                bn8_tensorLayer2Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn8_conv2dk1_skip_ui8_i8_i8 = external_func(
            "bn8_conv2dk1_skip_ui8_i8_i8",
            inputs=[
                bn8_tensorLayer3In_ty,
                bn8_weightsLayer3_ty,
                bn8_tensorLayer3Out_ty,
                bn8_tensorLayer3Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        bn9_conv2dk1_relu_i8_ui8 = external_func(
            "bn9_conv2dk1_relu_i8_ui8",
            inputs=[
                bn9_tensorLayer1In_ty,
                bn9_weightsLayer1_ty,
                bn9_tensorLayer1Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn9_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func(
            "bn9_conv2dk3_dw_stride1_relu_ui8_ui8",
            inputs=[
                bn9_tensorLayer2In_ty,
                bn9_tensorLayer2In_ty,
                bn9_tensorLayer2In_ty,
                bn9_weightsLayer2_ty,
                bn9_tensorLayer2Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn9_conv2dk1_skip_ui8_i8_i8 = external_func(
            "bn9_conv2dk1_skip_ui8_i8_i8",
            inputs=[
                bn9_tensorLayer3In_ty,
                bn9_weightsLayer3_ty,
                bn9_tensorLayer3Out_ty,
                bn9_tensorLayer3Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        act_bn9_bn10 = object_fifo(
            "act_bn9_bn10", bn8_bn9_tile, bn10_tile_1, [1, 2], bn8_tensorLayer3Out_ty
        )  #  GAGAN: make it double buffered $
        act_bn9_bn10.set_via_shared_mem(ObjectFifoPort.Consume)
        # act_bn9_bn10 = object_fifo("act_bn9_bn10", bn8_bn9_tile, bn10_tile_1, 1, bn8_tensorLayer3Out_ty)  # make it double buffered $

        # act_out = object_fifo("act_out", bn8_bn9_tile, [ShimTile10], 1, bn8_tensorLayer3Out_ty)

        bottleneckAFused_8and9(
            "bn8_bn9",
            bn8_bn9_tile,
            bn11_tile_2,
            act_bn7_bn8,
            bn8_9_wts_static,
            act_bn9_bn10,
            rtpbn8_bn9_tile,
            "combined_bn_8_9.a",
            bn8_conv2dk1_relu_i8_ui8,
            bn8_conv2dk3_dw_stride1_relu_ui8_ui8,
            bn8_conv2dk1_skip_ui8_i8_i8,
            bn9_conv2dk1_relu_i8_ui8,
            bn9_conv2dk3_dw_stride1_relu_ui8_ui8,
            bn9_conv2dk1_skip_ui8_i8_i8,
            bn8_tensorLayer1Out_ty,
            bn8_tensorLayer2Out_ty,
            bn8_tensorLayer3Out_ty,
            bn9_tensorLayer1Out_ty,
            bn9_tensorLayer2Out_ty,
            tensorL8_1InW,
            tensorL8_1InH,
            tensorL8_1InC,
            bn8_depthWiseStride,
            bn8_depthWiseChannels,
            bn9_depthWiseStride,
            bn9_depthWiseChannels,
            tensorL9_3OutC,
            bn8_scaleFactor1,
            bn8_scaleFactor2,
            bn8_scaleFactor3,
            bn8_scaleFactorAdd,
            bn9_scaleFactor1,
            bn9_scaleFactor2,
            bn9_scaleFactor3,
            bn9_scaleFactorAdd,
        )

        # # ##### ******************************************************************************************************************************
        act_B_C = object_fifo(
            "act_B_C",
            bn12_tile_2,
            [bn13_tile_layer1_put, bn13_tile_layer1_get, MemTile51],
            [2, 2, 2, 6],
            ty_bneck_13_layer1_in,
        )
        bn13_skip = object_fifo(
            "bn13_skip", MemTile51, bn13_tile_layer3_get, 2, ty_bneck_13_layer1_in
        )
        object_fifo_link(act_B_C, bn13_skip)

        BottleneckBCore(
            "B",
            bn10_tile_1,
            bn10_tile_2,
            bn10_tile_3,
            bn11_tile_1,
            bn11_tile_2,
            bn11_tile_3,
            bn12_tile_1,
            bn12_tile_2,
            bn10_1_wts_static,
            bn10_2_wts_static,
            bn10_3_wts_static,
            bn11_1_wts_static,
            bn11_2_wts_static,
            bn11_3_wts_static,
            bn12_1_wts_static,
            bn12_2_3_wts_static,
            b12_layer2_out,
            bn10_1_rtp,
            bn10_2_rtp,
            bn10_3_rtp,
            bn11_1_rtp,
            bn11_2_rtp,
            bn11_3_rtp,
            bn12_1_rtp,
            bn12_2_rtp,
            MemTile21,
            act_bn9_bn10,
            act_B_C,
            bn10_scaleFactor1,
            bn10_scaleFactor2,
            bn10_scaleFactor3,
            bn11_scaleFactor1,
            bn11_scaleFactor2,
            bn11_scaleFactor3,
            bn11_scaleFactorAdd,
            bn12_scaleFactor1,
            bn12_scaleFactor2,
            bn12_scaleFactor3,
        )

        # act_out = object_fifo("act_out", bn14_tile_layer3_get, ShimTile70, 2, ty_bneck_14_layer3_out)
        # act_in_C_post = object_fifo("act_out", bn14_tile_layer3_get, ShimTile70, 2, ty_bneck_14_layer3_out)
        act_in_C_post = object_fifo(
            "act_in_C_post", bn14_tile_layer3_get, PostL1Tile, 2, ty_bneck_14_layer3_out
        )

        bottleneckCCore(
            bn13_tile_layer1_put,
            bn13_tile_layer1_get,
            bn13_tile_layer2,
            bn13_tile_layer3_put,
            bn13_tile_layer3_get,
            bn14_tile_layer1_put,
            bn14_tile_layer1_get,
            bn14_tile_layer2,
            bn14_tile_layer3_put,
            bn14_tile_layer3_get,
            bn13_wts_memtile_layer1_put,
            bn13_wts_memtile_layer1_get,
            bn13_2_wts_static,
            bn13_wts_memtile_layer3_put,
            bn13_wts_memtile_layer3_get,
            bn14_wts_memtile_layer1_put,
            bn14_wts_memtile_layer1_get,
            bn14_2_wts_static,
            bn14_wts_memtile_layer3_put,
            bn14_wts_memtile_layer3_get,
            rtp_bn13_tile_layer1_get,
            rtp_bn13_tile_layer3_get,
            bn13_scaleFactor1,
            bn13_scaleFactor2,
            bn13_scaleFactor3,
            bn13_scaleFactorAdd,
            bn14_scaleFactor1,
            bn14_scaleFactor2,
            bn14_scaleFactor3,
            bn14_scaleFactorAdd,
            MemTile71,
            act_B_C,
            act_in_C_post,
            bn13_skip,
        )

        # # AIE Core Function declarations
        post_fused_conv2dk1_i8_avg_pool = external_func(
            "conv2dk1_xy_pool_fused_relu_large_padded_i8_ui8",
            inputs=[
                ty_post_act_Layer1_L2L1,
                ty_post_wts_Layer1_L2L1,
                ty_post_Layer1_out,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        # act_in_post_L1_L1 = object_fifo("act_in_post_L1_L1", PostL1Tile, [PostL2Tile_1,PostL2Tile_2], [2,2,2], ty_post_Layer1_out)
        # act_out = object_fifo("act_out", PostL1Tile, ShimTile70, 2, ty_post_Layer1_out)
        act_out_post_avgpool_shim = object_fifo(
            "act_out_post_avgpool_shim", PostL1Tile, ShimTile30, 2, ty_post_Layer1_out
        )
        postBlock(
            PostL1Tile,
            act_in_C_post,
            # post_L1_wts_L2_L1,
            post_L1_tile_prod_lock,
            post_L1_tile_cons_lock,
            post_L1_tile_buff,
            act_out_post_avgpool_shim,
            post_fused_conv2dk1_i8_avg_pool,
            post_L1_InW,
            post_L1_InH,
            post_L1_InC,
            post_L1_OutW,
            post_L1_OutH,
            post_L1_OutC,
            post_L1_OutC_padd,
            post_scaleFactor,
        )

        act_out_post_shim_FC = object_fifo(
            "act_out_post_shim_FC",
            ShimTile40,
            [PostL2Tile_1, PostL2Tile_2, PostL2Tile_3, PostL2Tile_4],
            [2, 2, 2, 2, 2],
            ty_post_Layer1_out,
        )
        # AIE Core Function declarations
        post_L2_fused_conv2dk1 = external_func(
            "post_L2_conv2dk1_relu_i16_ui16_pad",
            inputs=[
                ty_post_act_Layer2_all,
                ty_post_wts_Layer2_split_L1,
                ty_post_Layer2_out_split,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        post_L2_out_core1 = object_fifo(
            "post_L2_out_core1", PostL2Tile_1, [MemTile61], 2, ty_post_Layer2_out_split
        )
        post_L2_out_core2 = object_fifo(
            "post_L2_out_core2", PostL2Tile_2, [MemTile61], 2, ty_post_Layer2_out_split
        )
        post_L2_out_core3 = object_fifo(
            "post_L2_out_core3", PostL2Tile_3, [MemTile61], 2, ty_post_Layer2_out_split
        )
        post_L2_out_core4 = object_fifo(
            "post_L2_out_core4", PostL2Tile_4, [MemTile61], 2, ty_post_Layer2_out_split
        )
        act_out = object_fifo(
            "act_out", MemTile61, ShimTile70, 2, ty_post_Layer2_out_all
        )
        object_fifo_link(
            [
                post_L2_out_core1,
                post_L2_out_core2,
                post_L2_out_core3,
                post_L2_out_core4,
            ],
            [act_out],
            [
                0,
                post_act_out_Layer2 // 4,
                post_act_out_Layer2 // 2,
                3 * post_act_out_Layer2 // 4,
            ],
        )

        # act_out = object_fifo("act_out", PostL2Tile, ShimTile70, 2, ty_post_Layer2_out_split)

        postBlockL2(
            PostL2Tile_1,
            PostL2Tile_2,
            PostL2Tile_3,
            PostL2Tile_4,
            act_out_post_shim_FC,
            mem_L2_wts_core1,
            mem_L2_wts_core2,
            mem_L2_wts_core3,
            mem_L2_wts_core4,
            post_L2_out_core1,
            post_L2_out_core2,
            post_L2_out_core3,
            post_L2_out_core4,
            post_L2_fused_conv2dk1,
            post_L2_OutW,
            post_L2_OutH,
            post_L1_OutC,
            post_L1_OutC_padd,
            post_L2_OutW,
            post_L2_OutH,
            post_L2_OutC,
            post_FC1_scaleFactor,
            post_FC2_scaleFactor,
            PostL2Tile_1_cons_prod_lock,
            PostL2Tile_2_cons_prod_lock,
            PostL2Tile_3_cons_prod_lock,
            PostL2Tile_4_cons_prod_lock,
            PostL2Tile_1_cons_cons_lock,
            PostL2Tile_2_cons_cons_lock,
            PostL2Tile_3_cons_cons_lock,
            PostL2Tile_4_cons_cons_lock,
        )

        # instruction stream generation
        activationsInSize32b = (tensorInW * tensorInH * tensorInC) // 4
        # activationsOutSize32b = (tensorOutW * tensorOutH * tensorOutC) // 2
        activationsOutSize32b = (tensorOutW * tensorOutH * tensorOutC) // 2
        activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
        # weightsInL3_ty = MemRefType.get((total_weights//4,), int32_ty)
        activationsOutL3_ty = MemRefType.get((activationsOutSize32b,), int32_ty)

        memtile_01_wts32b = (memtile_01_wts) // 4
        memtile_11_wts32b = (memtile_11_wts) // 4

        memtile_21_wts32b = 0

        memtile_31_wts32b = 0
        memtile_41_wts32b = 0

        memtile_21_offset = memtile_01_wts32b + memtile_11_wts32b
        memtile_31_offset = memtile_21_offset + memtile_21_wts32b
        memtile_41_offset = memtile_31_offset + memtile_31_wts32b

        B_block_totalWeightsSize32b = (
            memtile_21_wts32b + memtile_31_wts32b + memtile_41_wts32b
        )
        #  ***************************** C block

        memtile_51_wts32b_layer1 = (bneck_13_InC1 * bneck_13_OutC1) // 4
        memtile_51_offset = memtile_41_offset + memtile_41_wts32b

        memtile_61_wts32b_layer2 = 0

        memtile_61_offset = memtile_51_offset + memtile_51_wts32b_layer1

        memtile_51_wts32b_layer3_2 = (bneck_13_OutC2 * bneck_13_OutC3) // 4

        memtile_51_offset_2 = memtile_61_offset + memtile_61_wts32b_layer2

        # bn14
        memtile_71_offset = memtile_51_offset_2 + memtile_51_wts32b_layer3_2

        memtile_61_offset_2 = memtile_71_offset + memtile_51_wts32b_layer1

        memtile_71_offset_2 = memtile_61_offset_2 + memtile_61_wts32b_layer2

        C_block_13_wts_size32b = (
            memtile_51_wts32b_layer1
            + memtile_61_wts32b_layer2
            + memtile_51_wts32b_layer3_2
        )  # bn13
        C_total_wts_size32b = 2 * C_block_13_wts_size32b
        # post weights

        post_memtile_71_offset_3 = memtile_71_offset_2 + memtile_51_wts32b_layer3_2

        post_L2_wts_32b = post_wts_Layer2_all // 4
        post_layer2_memtile_71_offset_3 = post_memtile_71_offset_3 + post_L1_wts_size32b
        post_layer2_memtile_71_offset_4 = (
            post_layer2_memtile_71_offset_3 + post_wts_Layer2_split_32b
        )
        post_layer2_memtile_71_offset_5 = (
            post_layer2_memtile_71_offset_4 + post_wts_Layer2_split_32b
        )
        post_layer2_memtile_71_offset_6 = (
            post_layer2_memtile_71_offset_5 + post_wts_Layer2_split_32b
        )

        post_layer3_memtile_71_offset_1 = (
            post_layer2_memtile_71_offset_6 + post_wts_Layer2_split_32b
        )
        post_layer3_memtile_71_offset_2 = (
            post_layer3_memtile_71_offset_1 + post_wts_Layer2_split_32b
        )
        post_layer3_memtile_71_offset_3 = (
            post_layer3_memtile_71_offset_2 + post_wts_Layer2_split_32b
        )
        post_layer3_memtile_71_offset_4 = (
            post_layer3_memtile_71_offset_3 + post_wts_Layer2_split_32b
        )

        weightsInL3_ty = MemRefType.get(
            (
                total_weights // 4
                + B_block_totalWeightsSize32b
                + C_total_wts_size32b
                + post_L1_wts_size32b,
            ),
            int32_ty,
        )

        #  instruction stream generation
        # tiles_to_trace = [bn1_tile, bn2_tile, bn3_tile, bn4_tile,
        #                  bn5_tile, bn6_tile, bn7_tile, bn8_bn9_tile, ComputeTile24]

        tiles_to_trace = [
            init_tile,  # initconv,     tile(1,2)
            #    bn0_tile, #bn0      tile(0,2)
            #    bn1_tile, #bn1,     tile(0,3)
            #    bn2_tile, #bn2,     tile(0,4)
            #    bn3_tile, #bn3,     tile(0,5)
            #    bn4_5_tile,
            #    bn4_tile, #bn4,     tile(1,5)
            #    bn5_tile, #bn5,     tile(1,4)
            #    bn6_tile, #bn6,     tile(1,3)
            #    bn7_tile, #bn7,     tile(2,3)
            #    bn8_bn9_tile, #bn8+9,     tile(2,2)
            #    bn10_tile_1,           # tile(3,2)
            #    bn10_tile_2,           # tile(3,3)
            #    bn10_tile_3,           # tile(3,4)
            #    bn11_tile_1,           # tile(2,4)
            #    bn11_tile_2,           # tile(2,5)
            #    bn11_tile_3,           # tile(3,5)
            #    bn12_tile_1,           # tile(4,2)
            #    bn12_tile_2,           # tile(4,4)
            #    bn12_tile_3,           # tile(6,4)
            #    bn13_tile_layer1_put,  # tile(4,5)
            #    bn13_tile_layer1_get,  # tile(5,5)
            #    bn13_tile_layer2,      # tile(5,4)
            #    bn13_tile_layer3_put,  # tile(4,3)
            #    bn13_tile_layer3_get,  # tile(5,3)
            #    bn14_tile_layer1_put,  # tile(6,5)
            #    bn14_tile_layer1_get,  # tile(7,5)
            #    bn14_tile_layer2,      # tile(6,3)
            #    bn14_tile_layer3_put,  # tile(5,2)
            #    bn14_tile_layer3_get,  # tile(6,2)
            PostL1Tile,  # tile (7,4)
            PostL2Tile_4,  # tile (7,3)
            #    PostL2Tile_2,          # tile (7,2)
        ]

        # Set up a packet-switched flow from core to shim for tracing information
        if opts.trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile00)  # 13
            # packetflow(13, MemTile01, WireBundle.Trace, 0, ShimTile00, WireBundle.DMA, 1, keep_pkt_header=True)
            # packetflow(14, MemTile11, WireBundle.Trace, 0, ShimTile00, WireBundle.DMA, 1, keep_pkt_header=True)
            # packetflow(11, MemTile31, WireBundle.Trace, 0, ShimTile00, WireBundle.DMA, 1, keep_pkt_header=True)
            # packetflow(12, MemTile41, WireBundle.Trace, 0, ShimTile00, WireBundle.DMA, 1, keep_pkt_header=True)
            # packetflow(13, MemTile61, WireBundle.Trace, 0, ShimTile00, WireBundle.DMA, 1, keep_pkt_header=True)
            # packetflow(14, MemTile71, WireBundle.Trace, 0, ShimTile00, WireBundle.DMA, 1, keep_pkt_header=True)

        @runtime_sequence(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
        def sequence(inputFromL3, weightsFromL3, outputToL3):
            # init
            NpuWriteRTPOp("rtp_init", index=0, value=init_scaleFactor)

            # # bn0
            NpuWriteRTPOp("rtp_bn0", index=0, value=bn0_scaleFactor2)
            NpuWriteRTPOp("rtp_bn0", index=1, value=bn0_scaleFactor3)
            NpuWriteRTPOp("rtp_bn0", index=2, value=bn0_scaleFactorAdd)
            # # bn1
            NpuWriteRTPOp("rtp_bn0", index=3, value=bn1_scaleFactor1)
            NpuWriteRTPOp("rtp_bn0", index=4, value=bn1_scaleFactor2)
            NpuWriteRTPOp("rtp_bn0", index=5, value=bn1_scaleFactor3)

            # bn2
            NpuWriteRTPOp("rtp_bn2", index=0, value=bn2_scaleFactor1)
            NpuWriteRTPOp("rtp_bn2", index=1, value=bn2_scaleFactor2)
            NpuWriteRTPOp("rtp_bn2", index=2, value=bn2_scaleFactor3)
            NpuWriteRTPOp("rtp_bn2", index=3, value=bn2_scaleFactorAdd)

            # # bn3
            NpuWriteRTPOp("rtp_bn3", index=0, value=bn3_scaleFactor1)
            NpuWriteRTPOp("rtp_bn3", index=1, value=bn3_scaleFactor2)
            NpuWriteRTPOp("rtp_bn3", index=2, value=bn3_scaleFactor3)
            NpuWriteRTPOp("rtp_bn3", index=3, value=bn3_scaleFactorAdd)

            # bn4
            NpuWriteRTPOp("rtp_bn4_5_tile", index=0, value=bn4_scaleFactor1)
            NpuWriteRTPOp("rtp_bn4_5_tile", index=1, value=bn4_scaleFactor2)
            NpuWriteRTPOp("rtp_bn4_5_tile", index=2, value=bn4_scaleFactor3)
            NpuWriteRTPOp("rtp_bn4_5_tile", index=3, value=bn4_scaleFactorAdd)

            # # bn5
            NpuWriteRTPOp("rtp_bn4_5_tile", index=4, value=bn5_scaleFactor1)
            NpuWriteRTPOp("rtp_bn4_5_tile", index=5, value=bn5_scaleFactor2)
            NpuWriteRTPOp("rtp_bn4_5_tile", index=6, value=bn5_scaleFactor3)
            NpuWriteRTPOp("rtp_bn4_5_tile", index=7, value=bn5_scaleFactorAdd)

            NpuWriteRTPOp("rtp_bn6", index=0, value=bn6_scaleFactor1)
            NpuWriteRTPOp("rtp_bn6", index=1, value=bn6_scaleFactor2)
            NpuWriteRTPOp("rtp_bn6", index=2, value=bn6_scaleFactor3)
            NpuWriteRTPOp("rtp_bn6", index=3, value=bn6_scaleFactorAdd)

            NpuWriteRTPOp("rtp_bn7", index=0, value=bn7_scaleFactor1)
            NpuWriteRTPOp("rtp_bn7", index=1, value=bn7_scaleFactor2)
            NpuWriteRTPOp("rtp_bn7", index=2, value=bn7_scaleFactor3)
            NpuWriteRTPOp("rtp_bn7", index=3, value=bn7_scaleFactorAdd)

            NpuWriteRTPOp("rtp_bn8_bn9", index=0, value=bn8_scaleFactor1)
            NpuWriteRTPOp("rtp_bn8_bn9", index=1, value=bn8_scaleFactor2)
            NpuWriteRTPOp("rtp_bn8_bn9", index=2, value=bn8_scaleFactor3)
            NpuWriteRTPOp("rtp_bn8_bn9", index=3, value=bn8_scaleFactorAdd)

            NpuWriteRTPOp("rtp_bn8_bn9", index=4, value=bn9_scaleFactor1)
            NpuWriteRTPOp("rtp_bn8_bn9", index=5, value=bn9_scaleFactor2)
            NpuWriteRTPOp("rtp_bn8_bn9", index=6, value=bn9_scaleFactor3)
            NpuWriteRTPOp("rtp_bn8_bn9", index=7, value=bn9_scaleFactorAdd)

            NpuWriteRTPOp("bn10_1_rtp", index=0, value=bn10_scaleFactor1)
            NpuWriteRTPOp("bn10_2_rtp", index=0, value=bn10_scaleFactor2)
            NpuWriteRTPOp("bn10_3_rtp", index=0, value=bn10_scaleFactor3)

            NpuWriteRTPOp("bn11_1_rtp", index=0, value=bn11_scaleFactor1)
            NpuWriteRTPOp("bn11_2_rtp", index=0, value=bn11_scaleFactor2)
            NpuWriteRTPOp("bn11_3_rtp", index=0, value=bn11_scaleFactor3)
            NpuWriteRTPOp("bn11_3_rtp", index=1, value=bn11_scaleFactorAdd)

            NpuWriteRTPOp("bn12_1_rtp", index=0, value=bn12_scaleFactor1)
            NpuWriteRTPOp("bn12_2_rtp", index=0, value=bn12_scaleFactor2)
            NpuWriteRTPOp("bn12_2_rtp", index=1, value=bn12_scaleFactor3)

            NpuWriteRTPOp("rtp_bn13_tile_layer1_get", index=0, value=bn13_scaleFactor1)
            NpuWriteRTPOp("rtp_bn13_tile_layer2", index=0, value=bn13_scaleFactor2)
            NpuWriteRTPOp("rtp_bn13_tile_layer3_get", index=0, value=bn13_scaleFactor3)
            NpuWriteRTPOp(
                "rtp_bn13_tile_layer3_get", index=1, value=bn13_scaleFactorAdd
            )

            NpuWriteRTPOp("rtp_bn14_tile_layer1_get", index=0, value=bn14_scaleFactor1)
            NpuWriteRTPOp("rtp_bn14_tile_layer2", index=0, value=bn14_scaleFactor2)
            NpuWriteRTPOp("rtp_bn14_tile_layer3_get", index=0, value=bn14_scaleFactor3)
            NpuWriteRTPOp(
                "rtp_bn14_tile_layer3_get", index=1, value=bn14_scaleFactorAdd
            )

            NpuWriteRTPOp("rtp_post_L1", index=0, value=post_scaleFactor)
            NpuWriteRTPOp("rtp_post_L2_C1", index=0, value=post_FC1_scaleFactor)
            NpuWriteRTPOp("rtp_post_L2_C2", index=0, value=post_FC2_scaleFactor)

            N_in_bytes = (
                tensorOutW * tensorOutH * tensorOutC * 2
            )  # x2 since output is uin16
            # N_in_bytes = tensorOutW * tensorOutH * tensorOutC

            if opts.trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace, ShimTile00, opts.trace_size, N_in_bytes
                )

                # trace_utils.configure_memtile_packet_tracing_aie2(MemTile01, ShimTile00, 13, 3, opts.trace_size, N_in_bytes)
                # trace_utils.configure_memtile_packet_tracing_aie2(MemTile11, ShimTile00, 14, 2, opts.trace_size, N_in_bytes)
                # trace_utils.configure_memtile_packet_tracing_aie2(MemTile31, ShimTile00, 11, 5, opts.trace_size, N_in_bytes)
                # trace_utils.configure_memtile_packet_tracing_aie2(MemTile41, ShimTile00, 12, 4, opts.trace_size, N_in_bytes)
                # trace_utils.configure_memtile_packet_tracing_aie2(MemTile61, ShimTile00, 13, 3, opts.trace_size, N_in_bytes)
                # trace_utils.configure_memtile_packet_tracing_aie2(MemTile71, ShimTile00, 14, 2, opts.trace_size, N_in_bytes)
                trace_utils.configure_shim_packet_tracing_aie2(ShimTile00)

            npu_dma_memcpy_nd(
                metadata="act_in",
                bd_id=0,
                mem=inputFromL3,
                sizes=[1, 1, 1, activationsInSize32b],
            )

            # C mappings
            npu_dma_memcpy_nd(
                metadata="bn13_wts_L3L2_layer1",
                bd_id=1,
                mem=weightsFromL3,
                offsets=[0, 0, 0, memtile_51_offset],
                sizes=[1, 1, 1, memtile_51_wts32b_layer1],
            )

            npu_dma_memcpy_nd(
                metadata="bn13_wts_L3L2_layer3",
                bd_id=2,
                mem=weightsFromL3,
                offsets=[0, 0, 0, memtile_51_offset_2],
                sizes=[1, 1, 1, memtile_51_wts32b_layer3_2],
            )

            # bn 14

            npu_dma_memcpy_nd(
                metadata="bn14_wts_L3L2_layer1",
                bd_id=1,
                mem=weightsFromL3,
                offsets=[0, 0, 0, memtile_71_offset],
                sizes=[1, 1, 1, memtile_51_wts32b_layer1],
            )

            npu_dma_memcpy_nd(
                metadata="bn14_wts_L3L2_layer3",
                bd_id=1,
                mem=weightsFromL3,
                offsets=[0, 0, 0, memtile_71_offset_2],
                sizes=[1, 1, 1, memtile_51_wts32b_layer3_2],
            )

            # npu_dma_memcpy_nd(
            #     metadata="post_L1_wts_L3_L2",
            #     bd_id=2,
            #     mem=weightsFromL3,
            #     offsets=[0, 0, 0, post_memtile_71_offset_3],
            #     sizes=[1, 1, 1, post_L1_wts_size32b],
            # )

            # # ****************** write avg pool out to shim and read from shim *****************
            npu_dma_memcpy_nd(
                metadata="act_out_post_avgpool_shim",
                bd_id=1,
                mem=inputFromL3,
                offsets=[0, 0, 0, activationsOutSize32b],
                sizes=[1, 1, 1, activationsOutSize32b],
            )
            # npu_sync(column=2, row=0, direction=0, channel=0)
            dma_wait("act_out_post_avgpool_shim")

            npu_dma_memcpy_nd(
                metadata="act_out_post_shim_FC",
                bd_id=2,
                mem=inputFromL3,
                offsets=[0, 0, 0, activationsOutSize32b],
                sizes=[1, 1, 1, activationsOutSize32b],
            )

            # # ****************** begin time multiplex ******************
            npu_dma_memcpy_nd(
                metadata="act_out",
                bd_id=3,
                mem=inputFromL3,
                offsets=[0, 0, 0, 2 * activationsOutSize32b],
                sizes=[1, 1, 1, activationsOutSize32b],
            )

            # npu_sync(column=3, row=0, direction=0, channel=0)
            dma_wait("act_out")

            npu_dma_memcpy_nd(
                metadata="act_out_post_shim_FC",
                bd_id=2,
                mem=inputFromL3,
                offsets=[0, 0, 0, 2 * activationsOutSize32b],
                sizes=[1, 1, 1, activationsOutSize32b],
            )

            # # ****************** end time multiplexing *****************
            npu_dma_memcpy_nd(
                metadata="act_out",
                bd_id=3,
                mem=outputToL3,
                sizes=[1, 1, 1, activationsOutSize32b],
            )
            dma_wait("act_out")
            # npu_sync(column=7, row=0, direction=0, channel=0)


with mlir_mod_ctx() as ctx:
    p = argparse.ArgumentParser()
    p.add_argument(
        "-t",
        "--trace_sz",
        dest="trace_size",
        default=0,
        type=int,
        help="trace size in bytes",
    )
    opts = p.parse_args(sys.argv[1:])

    mobilenetV3_A_B(
        tileColIndex=0,
        b_start_col=2,
        b_start_row=4,
        tensorInW=224,
        tensorInH=224,
        tensorInC=8,
        init_tensorOutC=16,
        init_scaleFactor=scale_factors["INIT"]["conv3x3"],
        bn0_scaleFactor2=scale_factors["BN0"]["conv3x3"],
        bn0_scaleFactor3=scale_factors["BN0"]["conv1x1_2"],
        bn0_scaleFactorAdd=scale_factors["BN0"]["skip_add"],
        bn1_depthWiseStride=2,
        bn1_depthWiseChannels=64,
        bn1_withSkip=False,
        bn1_tensorOutC=24,
        bn1_scaleFactor1=scale_factors["BN1"]["conv1x1_1"],
        bn1_scaleFactor2=scale_factors["BN1"]["conv3x3"],
        bn1_scaleFactor3=scale_factors["BN1"]["conv1x1_2"],
        bn1_scaleFactorAdd=scale_factors["BN1"]["skip_add"],
        bn2_depthWiseStride=1,
        bn2_depthWiseChannels=72,
        bn2_withSkip=True,
        bn2_tensorOutC=24,
        bn2_scaleFactor1=scale_factors["BN2"]["conv1x1_1"],
        bn2_scaleFactor2=scale_factors["BN2"]["conv3x3"],
        bn2_scaleFactor3=scale_factors["BN2"]["conv1x1_2"],
        bn2_scaleFactorAdd=scale_factors["BN2"]["skip_add"],
        bn3_depthWiseStride=2,
        bn3_depthWiseChannels=72,
        bn3_withSkip=False,
        bn3_tensorOutC=40,
        bn3_scaleFactor1=scale_factors["BN3"]["conv1x1_1"],
        bn3_scaleFactor2=scale_factors["BN3"]["conv3x3"],
        bn3_scaleFactor3=scale_factors["BN3"]["conv1x1_2"],
        bn3_scaleFactorAdd=scale_factors["BN3"]["skip_add"],
        bn4_depthWiseStride=1,
        bn4_depthWiseChannels=120,
        bn4_withSkip=True,
        bn4_tensorOutC=40,
        bn4_scaleFactor1=scale_factors["BN4"]["conv1x1_1"],
        bn4_scaleFactor2=scale_factors["BN4"]["conv3x3"],
        bn4_scaleFactor3=scale_factors["BN4"]["conv1x1_2"],
        bn4_scaleFactorAdd=scale_factors["BN4"]["skip_add"],
        bn5_depthWiseStride=1,
        bn5_depthWiseChannels=120,
        bn5_withSkip=False,
        bn5_tensorOutC=40,
        bn5_scaleFactor1=scale_factors["BN5"]["conv1x1_1"],
        bn5_scaleFactor2=scale_factors["BN5"]["conv3x3"],
        bn5_scaleFactor3=scale_factors["BN5"]["conv1x1_2"],
        bn5_scaleFactorAdd=scale_factors["BN5"]["skip_add"],
        bn6_depthWiseStride=2,
        bn6_depthWiseChannels=240,
        bn6_withSkip=False,
        bn6_tensorOutC=80,
        bn6_scaleFactor1=scale_factors["BN6"]["conv1x1_1"],
        bn6_scaleFactor2=scale_factors["BN6"]["conv3x3"],
        bn6_scaleFactor3=scale_factors["BN6"]["conv1x1_2"],
        bn6_scaleFactorAdd=scale_factors["BN6"]["skip_add"],
        bn7_depthWiseStride=1,
        bn7_depthWiseChannels=200,
        bn7_withSkip=True,
        bn7_tensorOutC=80,
        bn7_scaleFactor1=scale_factors["BN7"]["conv1x1_1"],
        bn7_scaleFactor2=scale_factors["BN7"]["conv3x3"],
        bn7_scaleFactor3=scale_factors["BN7"]["conv1x1_2"],
        bn7_scaleFactorAdd=scale_factors["BN7"]["skip_add"],
        bn8_depthWiseStride=1,
        bn8_depthWiseChannels=184,
        bn8_withSkip=True,
        bn8_tensorOutC=80,
        bn8_scaleFactor1=scale_factors["BN8"]["conv1x1_1"],
        bn8_scaleFactor2=scale_factors["BN8"]["conv3x3"],
        bn8_scaleFactor3=scale_factors["BN8"]["conv1x1_2"],
        bn8_scaleFactorAdd=scale_factors["BN8"]["skip_add"],
        bn9_depthWiseStride=1,
        bn9_depthWiseChannels=184,
        bn9_withSkip=True,
        bn9_tensorOutC=80,
        bn9_scaleFactor1=scale_factors["BN9"]["conv1x1_1"],
        bn9_scaleFactor2=scale_factors["BN9"]["conv3x3"],
        bn9_scaleFactor3=scale_factors["BN9"]["conv1x1_2"],
        bn9_scaleFactorAdd=scale_factors["BN9"]["skip_add"],
        enableTrace=False,
        trace_size=16384,
        traceSizeInInt32s=4096,
        bn10_scaleFactor1=scale_factors["BN10"]["conv1x1_1"],
        bn10_scaleFactor2=scale_factors["BN10"]["conv3x3"],
        bn10_scaleFactor3=scale_factors["BN10"]["conv1x1_2"],
        bn11_scaleFactor1=scale_factors["BN11"]["conv1x1_1"],
        bn11_scaleFactor2=scale_factors["BN11"]["conv3x3"],
        bn11_scaleFactor3=scale_factors["BN11"]["conv1x1_2"],
        bn11_scaleFactorAdd=scale_factors["BN11"]["skip_add"],
        bn12_scaleFactor1=scale_factors["BN12"]["conv1x1_1"],
        bn12_scaleFactor2=scale_factors["BN12"]["conv3x3"],
        bn12_scaleFactor3=scale_factors["BN12"]["conv1x1_2"],
        bn13_scaleFactor1=scale_factors["BN13"]["conv1x1_1"],
        bn13_scaleFactor2=scale_factors["BN13"]["conv3x3"],
        bn13_scaleFactor3=scale_factors["BN13"]["conv1x1_2"],
        bn13_scaleFactorAdd=scale_factors["BN13"]["skip_add"],
        bn14_scaleFactor1=scale_factors["BN14"]["conv1x1_1"],
        bn14_scaleFactor2=scale_factors["BN14"]["conv3x3"],
        bn14_scaleFactor3=scale_factors["BN14"]["conv1x1_2"],
        bn14_scaleFactorAdd=scale_factors["BN14"]["skip_add"],
        post_scaleFactor=scale_factors["POST"]["conv1x1_1"],
        post_FC1_scaleFactor=scale_factors["POST"]["FC1"],
        post_FC2_scaleFactor=scale_factors["POST"]["FC2"],
    )
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
