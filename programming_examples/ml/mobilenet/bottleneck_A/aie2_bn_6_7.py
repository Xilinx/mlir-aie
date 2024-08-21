#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

from aie2_bottleneckA import bottleneckACore

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext import *
from aie.extras.dialects.ext.memref import view as memref_view

def mobilenetV3_bn_6_7(tileColIndex = 0,tensorInW = 112, tensorInH = 112, tensorInC = 16, bn6_depthWiseStride = 2, bn6_depthWiseChannels = 240, bn6_withSkip = False, bn6_tensorOutC = 80, bn6_scaleFactor1 = 8, bn6_scaleFactor2 = 9, bn6_scaleFactor3 = 11, enableTrace = False, trace_size = 16384, traceSizeInInt32s = 4096):

    tensorL6_1InC = tensorInC
    tensorL6_1InW = tensorInW
    tensorL6_1InH = tensorInH

    tensorL6_2InC = bn6_depthWiseChannels
    tensorL6_2InW = tensorL6_1InW
    tensorL6_2InH = tensorL6_1InH

    tensorL6_3InC = tensorL6_2InC
    tensorL6_3InW = tensorL6_2InW // bn6_depthWiseStride
    tensorL6_3InH = tensorL6_2InH // bn6_depthWiseStride
    tensorL6_3OutC = bn6_tensorOutC

    tensorOutW = tensorL6_3InW
    tensorOutH = tensorL6_3InH
    tensorOutC = tensorL6_3OutC

    @device(AIEDevice.npu1_1col)
    def device_body():
        
        # define types
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int16_ty = IntegerType.get_signless(16)
        int32_ty = IntegerType.get_signless(32)

        tensorLayerIn_ty = MemRefType.get((tensorInW, 1, tensorInC), int8_ty)
        tensorLayerOut_ty = MemRefType.get((tensorOutW, 1, tensorOutC), int8_ty)
        weightsAllLayers_ty = MemRefType.get((1 * 1 * tensorL6_1InC * tensorL6_2InC + 3 * 3 * tensorL6_3InC * 1 + 1 * 1 * tensorL6_3InC * tensorL6_3OutC,), int8_ty)

        # temporary types for tensor to enable intial test
        tensorLayer1In_ty = MemRefType.get((tensorInW, 1, tensorL6_1InC), int8_ty)
        weightsLayer1_ty = MemRefType.get((1 * 1 * tensorL6_1InC * tensorL6_2InC,), int8_ty)
        tensorLayer2In_ty = MemRefType.get((tensorInW, 1, tensorL6_2InC), uint8_ty)
        tensorLayer1Out_ty = tensorLayer2In_ty
        weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL6_3InC * 1,), int8_ty)
        tensorLayer3In_ty = MemRefType.get((tensorOutW, 1, tensorL6_3InC), uint8_ty)
        tensorLayer2Out_ty = tensorLayer3In_ty
        weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL6_3InC * tensorL6_3OutC,), int8_ty)
        tensorLayer3Out_ty = MemRefType.get((tensorOutW, 1, tensorL6_3OutC),int8_ty)
        
        # AIE Core Function declarations
        conv2dk1_relu_i8_ui8 = external_func("conv2dk1_relu_i8_ui8",inputs=[tensorLayer1In_ty, weightsLayer1_ty, tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        conv2dk3_dw_stride2_relu_ui8_ui8 = external_func("conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[tensorLayer2In_ty,tensorLayer2In_ty,tensorLayer2In_ty, weightsLayer2_ty, tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        conv2dk3_dw_stride1_relu_ui8_ui8 = external_func("conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[tensorLayer2In_ty,tensorLayer2In_ty,tensorLayer2In_ty, weightsLayer2_ty, tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        conv2dk1_skip_ui8_i8_i8 = external_func("conv2dk1_skip_ui8_i8_i8",inputs=[tensorLayer3In_ty, weightsLayer3_ty, tensorLayer3Out_ty, tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        conv2dk1_ui8_i8 = external_func("conv2dk1_ui8_i8",inputs=[tensorLayer3In_ty, weightsLayer3_ty, tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])

        ShimTile = tile(tileColIndex, 0)
        MemTile = tile(tileColIndex, 1)
        ComputeTile2 = tile(tileColIndex, 2)
        ComputeTile3 = tile(tileColIndex, 3)

        # AIE-array data movement with object fifos
        
        # Input
        act_in = object_fifo("act_in", ShimTile, ComputeTile2, 2, tensorLayerIn_ty)
        
        # wts
        wts_OF_L3L1 = object_fifo("wts_OF_L3L1", ShimTile, [ComputeTile2, ComputeTile3], 1, weightsAllLayers_ty)
        
        # Output
        act_out = object_fifo("act_out", ComputeTile3, [ShimTile], 1, tensorLayerOut_ty)
                
        # Set up compute tiles
        rtpComputeTile2 = Buffer(ComputeTile2, [16], T.i32(), "rtp2")
        rtpComputeTile3 = Buffer(ComputeTile2, [16], T.i32(), "rtp3")
        
        # Compute tile 6
        bn6_objectArchiveName = "combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a" % (bn6_depthWiseStride, "skip" if (bn6_withSkip) else "")
        bn6_tensorLayer1Out_ty = MemRefType.get((tensorL6_2InW, 1, tensorL6_2InC),uint8_ty)
        bn6_tensorLayer2Out_ty = MemRefType.get((tensorL6_3InW, 1, tensorL6_3InC),uint8_ty)
        bn6_tensorLayer3Out_ty = MemRefType.get((tensorL6_3InW, 1, tensorL6_3OutC),int8_ty)

        # between compute tiles
        act_bn6_bn7 = object_fifo("act_bn6_bn7", ComputeTile2, ComputeTile3, 2, bn6_tensorLayer3Out_ty)

        bottleneckACore("bn6", ComputeTile2, act_in, wts_OF_L3L1, act_bn6_bn7, rtpComputeTile2, bn6_objectArchiveName, conv2dk1_relu_i8_ui8, conv2dk3_dw_stride1_relu_ui8_ui8, conv2dk3_dw_stride2_relu_ui8_ui8, conv2dk1_ui8_i8, conv2dk1_skip_ui8_i8_i8, bn6_tensorLayer1Out_ty, bn6_tensorLayer2Out_ty, tensorInW, tensorInH, tensorInC, bn6_depthWiseStride, bn6_depthWiseChannels, tensorOutC, bn6_withSkip)
#        bottleneckACore("bn7", ComputeTile3, act_bn6_bn7, wts_OF_L3L1, act_out, rtpComputeTile3, bn6_objectArchiveName, conv2dk1_relu_i8_ui8, conv2dk3_dw_stride1_relu_ui8_ui8, conv2dk3_dw_stride2_relu_ui8_ui8, conv2dk1_ui8_i8, conv2dk1_skip_ui8_i8_i8, bn6_tensorLayer1Out_ty, bn6_tensorLayer2Out_ty, tensorInW, tensorInH, tensorInC, bn6_depthWiseStride, bn6_depthWiseChannels, tensorOutC, bn6_withSkip)

with mlir_mod_ctx() as ctx:
    mobilenetV3_bn_6_7(tileColIndex = 0,tensorInW = 28, tensorInH = 28, tensorInC = 40, bn6_depthWiseStride = 2, bn6_depthWiseChannels = 240, bn6_withSkip = False, bn6_tensorOutC = 80, bn6_scaleFactor1 = 8, bn6_scaleFactor2 = 9, bn6_scaleFactor3 = 11, enableTrace = False, trace_size = 16384, traceSizeInInt32s = 4096)
    
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)

