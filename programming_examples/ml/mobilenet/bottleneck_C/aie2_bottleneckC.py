#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, bneck_13_InC1.

import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects import memref, arith
from aie.extras.context import mlir_mod_ctx
import math

import aie.utils.trace as trace_utils


import json


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
        OutputSplit = 2  # split output channels based on your preference
        OC8 = bneck_13_OutC1 // (8 * OutputSplit)  # how many loops of OC8

        bneck_13_InW2 = bneck_13_InW1
        bneck_13_InH2 = bneck_13_InH1
        bneck_13_OutC2 = bneck_13_OutC1

        bneck_13_InW3 = bneck_13_InW2
        bneck_13_InH3 = bneck_13_InH2
        bneck_13_OutC3 = 80
        OutputSplit2 = 2  # calculate 8 OCs at a time, should bneck_13_InC1rease to more
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
            link_with="bn13_1_conv2dk1_get.o",
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
            link_with="bn13_1_conv2dk1_put.o",
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
            link_with="bn13_conv2dk3_dw.o",
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
            link_with="bn13_conv2dk1_put.o",
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
            link_with="bn13_conv2dk1_skip_get.o",
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
            link_with="bn14_1_conv2dk1_get.o",
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
            link_with="bn14_1_conv2dk1_put.o",
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
            link_with="bn14_conv2dk3_dw.o",
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
            link_with="bn14_conv2dk1_put.o",
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
            link_with="bn14_conv2dk1_skip_get.o",
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
        @core(self.computeTileBN13_layer1_put)
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
        @core(self.computeTileBN13_layer1_get)
        def core_body():
            for _ in for_(0xFFFFFFFF):
                for _ in for_(bneck_13_InH1):
                    elemIn = self.actIn.acquire(ObjectFifoPort.Consume, 1)
                    elemOut0 = bn13_act_layer1_layer2.acquire(ObjectFifoPort.Produce, 1)

                    # scale = memref.load(rtp04, [0])
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
        @core(self.computeTileBN13_layer2)
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
        @core(self.computeTileBN13_layer3_put)
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
        @core(self.computeTileBN13_layer3_get)
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
                    # scale = memref.load(rtp04, [0])
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
        @core(self.computeTileBN14_layer1_put)
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
        @core(self.computeTileBN14_layer1_get)
        def core_body():
            for _ in for_(0xFFFFFFFF):

                for _ in for_(bneck_13_InH1):
                    elemIn = bn13_act_layer3_bn_14_layer1.acquire(
                        ObjectFifoPort.Consume, 1
                    )
                    elemOut0 = bn14_act_layer1_layer2.acquire(ObjectFifoPort.Produce, 1)

                    # scale = memref.load(rtp04, [0])
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
        @core(self.computeTileBN14_layer2)
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
        @core(self.computeTileBN14_layer3_put)
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
        @core(
            self.computeTileBN14_layer3_get,
        )
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
