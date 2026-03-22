#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

import argparse
import sys

from aie2_bottleneckA_subblockStatic import bottleneckASubblockStatic
from aie2_bottleneckA_subblock_fused2Static import bottleneckASubblockFused2Static
from aie2_bottleneckA_subblock0Static import bottleneckASubblockBN0Static

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects import *
from aie.extras.dialects.memref import view as memref_view

import aie.utils.trace as trace_utils

import json


class bottleneckACoreStatic:
    def __init__(
        self,
        _bottleneckName,
        _computeTileBN0,
        _computeTileBN1,
        _computeTileBN2,
        _computeTileBN3,
        _computeTileBN45,
        _computeTileBN6,
        _computeTileBN7,
        _computeTileBN89,
        _L1_tile_for_bn4_5,
        _L1_tile_for_bn8_9,
        _bn0_wts_static,
        _bn1_wts_static,
        _bn2_wts_static,
        _bn3_wts_static,
        _bn4_5_wts_static,
        _bn6_wts_static,
        _bn7_wts_static,
        _bn8_9_wts_static,
        _rtp_bn0_tile,
        _rtp_bn1_tile,
        _rtp_bn2_tile,
        _rtp_bn3_tile,
        _rtp_bn4_5_tile,
        _rtp_bn6_tile,
        _rtp_bn7_tile,
        _rtp_bn8_9_tile,
        _actIn,
        _actOut,
        bn0_scaleFactor2,
        bn0_scaleFactor3,
        bn0_scaleFactorAdd,
        bn1_scaleFactor1,
        bn1_scaleFactor2,
        bn1_scaleFactor3,
        bn1_scaleFactorAdd,
        bn2_scaleFactor1,
        bn2_scaleFactor2,
        bn2_scaleFactor3,
        bn2_scaleFactorAdd,
        bn3_scaleFactor1,
        bn3_scaleFactor2,
        bn3_scaleFactor3,
        bn3_scaleFactorAdd,
        bn4_scaleFactor1,
        bn4_scaleFactor2,
        bn4_scaleFactor3,
        bn4_scaleFactorAdd,
        bn5_scaleFactor1,
        bn5_scaleFactor2,
        bn5_scaleFactor3,
        bn5_scaleFactorAdd,
        bn6_scaleFactor1,
        bn6_scaleFactor2,
        bn6_scaleFactor3,
        bn6_scaleFactorAdd,
        bn7_scaleFactor1,
        bn7_scaleFactor2,
        bn7_scaleFactor3,
        bn7_scaleFactorAdd,
        bn8_scaleFactor1,
        bn8_scaleFactor2,
        bn8_scaleFactor3,
        bn8_scaleFactorAdd,
        bn9_scaleFactor1,
        bn9_scaleFactor2,
        bn9_scaleFactor3,
        bn9_scaleFactorAdd,
    ):
        self.bottleneckName = _bottleneckName
        self.bn0_tile = _computeTileBN0
        self.bn1_tile = _computeTileBN1
        self.bn2_tile = _computeTileBN2
        self.bn3_tile = _computeTileBN3
        self.bn4_5_tile = _computeTileBN45
        self.bn6_tile = _computeTileBN6
        self.bn7_tile = _computeTileBN7
        self.bn8_9_tile = _computeTileBN89

        self.L1_tile_for_bn4_5 = _L1_tile_for_bn4_5
        self.L1_tile_for_bn8_9 = _L1_tile_for_bn8_9

        self.bn0_wts_static = _bn0_wts_static
        self.bn1_wts_static = _bn1_wts_static
        self.bn2_wts_static = _bn2_wts_static
        self.bn3_wts_static = _bn3_wts_static
        self.bn4_5_wts_static = _bn4_5_wts_static
        self.bn6_wts_static = _bn6_wts_static
        self.bn7_wts_static = _bn7_wts_static
        self.bn8_9_wts_static = _bn8_9_wts_static

        self.bn0_wts_static = _bn0_wts_static
        self.bn1_wts_static = _bn1_wts_static
        self.bn2_wts_static = _bn2_wts_static
        self.bn3_wts_static = _bn3_wts_static
        self.bn4_5_wts_static = _bn4_5_wts_static
        self.bn6_wts_static = _bn6_wts_static
        self.bn7_wts_static = _bn7_wts_static
        self.bn8_9_wts_static = _bn8_9_wts_static

        self.rtp_bn0_tile = _rtp_bn0_tile
        self.rtp_bn1_tile = _rtp_bn1_tile
        self.rtp_bn2_tile = _rtp_bn2_tile
        self.rtp_bn3_tile = _rtp_bn3_tile
        self.rtp_bn4_5_tile = _rtp_bn4_5_tile
        self.rtp_bn6_tile = _rtp_bn6_tile
        self.rtp_bn7_tile = _rtp_bn7_tile
        self.rtp_bn8_9_tile = _rtp_bn8_9_tile

        self.act_in = _actIn
        self.act_out = _actOut

        tensorInC = 16
        tensorInW = 112
        tensorInH = 112

        bn0_depthWiseStride = 1
        bn0_withSkip = True

        bn1_depthWiseStride = 2
        bn1_depthWiseChannels = 64
        bn1_withSkip = False
        bn1_tensorOutC = 24

        bn2_depthWiseStride = 1
        bn2_depthWiseChannels = 72
        bn2_withSkip = True
        bn2_tensorOutC = 24

        bn3_depthWiseStride = 2
        bn3_depthWiseChannels = 72
        bn3_withSkip = False
        bn3_tensorOutC = 40

        bn4_depthWiseStride = 1
        bn4_depthWiseChannels = 120
        bn4_withSkip = True
        bn4_tensorOutC = 40

        bn5_depthWiseStride = 1
        bn5_depthWiseChannels = 120
        bn5_withSkip = False
        bn5_tensorOutC = 40

        bn6_depthWiseStride = 2
        bn6_depthWiseChannels = 240
        bn6_withSkip = False
        bn6_tensorOutC = 80

        bn7_depthWiseStride = 1
        bn7_depthWiseChannels = 200
        bn7_withSkip = True
        bn7_tensorOutC = 80

        bn8_depthWiseStride = 1
        bn8_depthWiseChannels = 184
        bn8_withSkip = True
        bn8_tensorOutC = 80

        bn9_depthWiseStride = 1
        bn9_depthWiseChannels = 184
        # bn9_withSkip=True
        bn9_tensorOutC = 80

        # bn0
        tensorL0_2InC = tensorInC
        tensorL0_2InW = tensorInW
        tensorL0_2InH = tensorInH
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
        # final output

        # tensorInC=tensorL0_1InC
        # tensorInW=tensorL0_1InW
        # tensorInH=tensorL0_1InH

        tensorOutW = tensorL9_3InW
        tensorOutH = tensorL9_3InH
        tensorOutC = tensorL9_3OutC

        # define types
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int16_ty = IntegerType.get_signless(16)
        int32_ty = IntegerType.get_signless(32)

        # # # ******************************************************************bn0******************************************************************
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
            link_with="bn0_conv2dk3_dw_stride1.o",
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
            link_with="bn0_conv2dk1_skipui8.o",
        )

        # Compute tile bn0_combined_conv2dk3dwstride1_conv2dk1skipui8
        # bn0_objectArchiveName = "bn0_combined_conv2dk3dwstride1_conv2dk1skipui8.a"

        bn0_tensorLayer2Out_ty = MemRefType.get(
            (tensorL0_3InW, 1, tensorL0_3InC), uint8_ty
        )
        bn0_tensorLayer3Out_ty = MemRefType.get(
            (tensorL0_3InW, 1, tensorL0_3OutC), int8_ty
        )

        # between compute tiles
        act_bn0_bn1 = object_fifo(
            "act_bn0_bn1", self.bn0_tile, self.bn1_tile, 2, bn0_tensorLayer3Out_ty
        )
        # act_bn0_bn1 = object_fifo("act_out", bn0_tile, ShimTile10, 2, bn0_tensorLayer3Out_ty)

        bottleneckASubblockBN0Static(
            "bn0",
            self.bn0_tile,
            self.act_in,
            # bn0_wts_OF_L3L1,
            self.bn0_wts_static,
            act_bn0_bn1,
            self.rtp_bn0_tile,
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
            link_with="bn1_conv2dk1_fused_relu.o",
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
            link_with="bn1_conv2dk3_dw_stride2.o",
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
            link_with="bn1_conv2dk3_dw_stride1.o",
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
            link_with="bn1_conv2dk1_skip.o",
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
            link_with="bn1_conv2dk1_i8.o",
        )

        # Compute tile
        # bn1_objectArchiveName = (
        #     "bn1_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a"
        #     % (bn1_depthWiseStride, "skip" if (bn1_withSkip) else "")
        # )
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
            "act_bn1_bn2", self.bn1_tile, self.bn2_tile, 2, bn1_tensorLayer3Out_ty
        )
        # act_bn1_bn2 = object_fifo("act_out", bn1_tile, ShimTile10, 2, bn1_tensorLayer3Out_ty)

        bottleneckASubblockStatic(
            "bn1",
            self.bn1_tile,
            act_bn0_bn1,
            # bn1_wts_OF_L3L1,
            self.bn1_wts_static,
            act_bn1_bn2,
            self.rtp_bn1_tile,
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
            link_with="bn2_conv2dk1_fused_relu.o",
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
            link_with="bn2_conv2dk3_dw_stride2.o",
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
            link_with="bn2_conv2dk3_dw_stride1.o",
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
            link_with="bn2_conv2dk1_skip.o",
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
            link_with="bn2_conv2dk1_i8.o",
        )

        # Compute tile
        # bn2_objectArchiveName = (
        #     "bn2_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a"
        #     % (bn2_depthWiseStride, "skip" if (bn2_withSkip) else "")
        # )
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
            "act_bn2_bn3", self.bn2_tile, self.bn3_tile, 2, bn2_tensorLayer3Out_ty
        )
        # act_out = object_fifo("act_out", bn2_tile, [ShimTile10], 1, bn2_tensorLayer3Out_ty)

        bottleneckASubblockStatic(
            "bn2",
            self.bn2_tile,
            act_bn1_bn2,
            # bn2_wts_OF_L3L1,
            self.bn2_wts_static,
            act_bn2_bn3,
            self.rtp_bn2_tile,
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
            link_with="bn3_conv2dk1_fused_relu.o",
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
            link_with="bn3_conv2dk3_dw_stride2.o",
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
            link_with="bn3_conv2dk3_dw_stride1.o",
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
            link_with="bn3_conv2dk1_skip.o",
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
            link_with="bn3_conv2dk1_i8.o",
        )

        # Compute tile
        # bn3_objectArchiveName = (
        #     "bn3_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a"
        #     % (bn3_depthWiseStride, "skip" if (bn3_withSkip) else "")
        # )
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
            "act_bn3_bn4", self.bn3_tile, self.bn4_5_tile, 2, bn3_tensorLayer3Out_ty
        )
        # act_out = object_fifo("act_out", bn3_tile, [ShimTile10], 1, bn3_tensorLayer3Out_ty)
        bottleneckASubblockStatic(
            "bn3",
            self.bn3_tile,
            act_bn2_bn3,
            # bn3_wts_OF_L3L1,
            self.bn3_wts_static,
            act_bn3_bn4,
            self.rtp_bn3_tile,
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
            link_with="bn4_conv2dk1_fused_relu.o",
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
            link_with="bn4_conv2dk3_dw_stride2.o",
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
            link_with="bn4_conv2dk3_dw_stride1.o",
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
            link_with="bn4_conv2dk1_skip.o",
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
            link_with="bn4_conv2dk1_i8.o",
        )

        # Compute tile 6
        # bn4_objectArchiveName = (
        #     "bn4_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a"
        #     % (bn4_depthWiseStride, "skip" if (bn4_withSkip) else "")
        # )
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
        # act_bn4_bn5 = object_fifo(
        #     "act_bn4_bn5", bn4_tile, bn5_tile, 2, bn4_tensorLayer3Out_ty
        # )
        # # act_out = object_fifo("act_out", bn4_tile, [ShimTile10], 1, bn4_tensorLayer3Out_ty)
        # bottleneckACore(
        #     "bn4",
        #     bn4_tile,
        #     act_bn3_bn4,
        #     bn4_wts_OF_L3L1,
        #     act_bn4_bn5,
        #     rtpbn4_tile,
        #     bn4_objectArchiveName,
        #     bn4_conv2dk1_relu_i8_ui8,
        #     bn4_conv2dk3_dw_stride1_relu_ui8_ui8,
        #     bn4_conv2dk3_dw_stride2_relu_ui8_ui8,
        #     bn4_conv2dk1_ui8_i8,
        #     bn4_conv2dk1_skip_ui8_i8_i8,
        #     bn4_tensorLayer1Out_ty,
        #     bn4_tensorLayer2Out_ty,
        #     tensorL4_1InW,
        #     tensorL4_1InH,
        #     tensorL4_1InC,
        #     bn4_depthWiseStride,
        #     bn4_depthWiseChannels,
        #     tensorL4_3OutC,
        #     bn4_withSkip,
        # )

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
            link_with="bn5_conv2dk1_fused_relu.o",
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
            link_with="bn5_conv2dk3_dw_stride2.o",
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
            link_with="bn5_conv2dk3_dw_stride1.o",
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
            link_with="bn5_conv2dk1_skip.o",
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
            link_with="bn5_conv2dk1_i8.o",
        )

        # Compute tile 6
        # bn5_objectArchiveName = (
        #     "bn5_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a"
        #     % (bn5_depthWiseStride, "skip" if (bn5_withSkip) else "")
        # )
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
        act_bn5_bn6 = object_fifo(
            "act_bn5_bn6", self.bn4_5_tile, self.bn6_tile, 2, bn5_tensorLayer3Out_ty
        )
        # act_out = object_fifo("act_out", bn5_tile, [ShimTile10], 1, bn5_tensorLayer3Out_ty)
        bottleneckASubblockFused2Static(  # TODO Static?
            "bn4_bn5",
            self.bn4_5_tile,
            self.L1_tile_for_bn4_5,
            act_bn3_bn4,
            self.bn4_5_wts_static,
            act_bn5_bn6,
            self.rtp_bn4_5_tile,
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
            link_with="bn6_conv2dk1_fused_relu.o",
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
            link_with="bn6_conv2dk3_dw_stride2.o",
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
            link_with="bn6_conv2dk3_dw_stride1.o",
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
            link_with="bn6_conv2dk1_skip.o",
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
            link_with="bn6_conv2dk1_i8.o",
        )

        # # Compute tile 6
        # bn6_objectArchiveName = (
        #     "bn6_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a"
        #     % (bn6_depthWiseStride, "skip" if (bn6_withSkip) else "")
        # )
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
            "act_bn6_bn7", self.bn6_tile, self.bn7_tile, 2, bn6_tensorLayer3Out_ty
        )

        # act_out = object_fifo("act_out", bn6_tile, [ShimTile10], 1, bn6_tensorLayer3Out_ty)
        bottleneckASubblockStatic(
            "bn6",
            self.bn6_tile,
            act_bn5_bn6,
            # bn6_wts_OF_L3L1,
            self.bn6_wts_static,
            act_bn6_bn7,
            self.rtp_bn6_tile,
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
            link_with="bn7_conv2dk1_fused_relu.o",
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
            link_with="bn7_conv2dk3_dw_stride2.o",
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
            link_with="bn7_conv2dk3_dw_stride1.o",
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
            link_with="bn7_conv2dk1_skip.o",
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
            link_with="bn7_conv2dk1_i8.o",
        )

        # bn7_objectArchiveName = (
        #     "bn7_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a"
        #     % (bn7_depthWiseStride, "skip" if (bn7_withSkip) else "")
        # )

        # between compute tiles
        act_bn7_bn8 = object_fifo(
            "act_bn7_bn8", self.bn7_tile, self.bn8_9_tile, 2, bn7_tensorLayer3Out_ty
        )
        # act_bn7_bn8 = object_fifo("act_out", bn7_tile, [ShimTile10], 1, bn7_tensorLayer3Out_ty)
        bottleneckASubblockStatic(
            "bn7",
            self.bn7_tile,
            act_bn6_bn7,
            # bn7_wts_OF_L3L1,
            self.bn7_wts_static,
            act_bn7_bn8,
            self.rtp_bn7_tile,
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

        # # ##### ******************************************************************************************************************************

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
            link_with="bn8_conv2dk1_fused_relu.o",
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
            link_with="bn8_conv2dk3_dw_stride1.o",
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
            link_with="bn8_conv2dk1_skip.o",
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
            link_with="bn9_conv2dk1_fused_relu.o",
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
            link_with="bn9_conv2dk3_dw_stride1.o",
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
            link_with="bn9_conv2dk1_skip.o",
        )

        # TODO
        # act_out = object_fifo("act_out", bn8_9_tile, [ShimTile10], 1, tensorLayerOut_ty)

        bottleneckASubblockFused2Static(
            "bn8_bn9",
            self.bn8_9_tile,
            self.L1_tile_for_bn8_9,
            act_bn7_bn8,
            # bn8_9_wts_OF_L3L1,
            self.bn8_9_wts_static,
            self.act_out,
            self.rtp_bn8_9_tile,
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
