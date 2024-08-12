#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

from aie2_bottleneckA import bottleneckACore
# from aie2_bottleneckA_TEST import bottleneckACoreTEST
from aie2_bottleneckFusedA import bottleneckAFused

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext import *
from aie.extras.dialects.ext.memref import view as memref_view
import json
def read_scale_factors(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Read the existing scale factors
file_path = 'scale_factors.json'
scale_factors = read_scale_factors(file_path)

def mobilenetV3_bn_0_1_2_3_4_5_6_7_8_9(tileColIndex = 0,tensorInW = 56, tensorInH = 56, tensorInC = 16, 
                                bn0_scaleFactor2 = 9, bn0_scaleFactor3 = 8,  bn0_scaleFactorAdd = 2,
                       bn1_depthWiseStride = 2, bn1_depthWiseChannels = 64, bn1_withSkip = False, bn1_tensorOutC = 24, bn1_scaleFactor1 = 8, bn1_scaleFactor2 = 8, bn1_scaleFactor3 = 11,  bn1_scaleFactorAdd = 0,
                       bn2_depthWiseStride = 1, bn2_depthWiseChannels = 72, bn2_withSkip = True, bn2_tensorOutC = 24, bn2_scaleFactor1 = 8, bn2_scaleFactor2 = 8, bn2_scaleFactor3 = 11,  bn2_scaleFactorAdd = 0,
                       bn3_depthWiseStride = 2, bn3_depthWiseChannels = 72, bn3_withSkip = False, bn3_tensorOutC = 40, bn3_scaleFactor1 = 8, bn3_scaleFactor2 = 8, bn3_scaleFactor3 = 11,  bn3_scaleFactorAdd = 0,
                       bn4_depthWiseStride = 1, bn4_depthWiseChannels = 120, bn4_withSkip = True, bn4_tensorOutC = 40, bn4_scaleFactor1 = 8, bn4_scaleFactor2 = 8, bn4_scaleFactor3 = 11,  bn4_scaleFactorAdd = 0,
                       bn5_depthWiseStride = 1, bn5_depthWiseChannels = 120, bn5_withSkip = True, bn5_tensorOutC = 80, bn5_scaleFactor1 = 8, bn5_scaleFactor2 = 8, bn5_scaleFactor3 = 11,  bn5_scaleFactorAdd = 0,
                       bn6_depthWiseStride = 2, bn6_depthWiseChannels = 240, bn6_withSkip = False, bn6_tensorOutC = 80, bn6_scaleFactor1 = 8, bn6_scaleFactor2 = 8, bn6_scaleFactor3 = 11,  bn6_scaleFactorAdd = 0,
                       bn7_depthWiseStride = 1, bn7_depthWiseChannels = 200, bn7_withSkip = True, bn7_tensorOutC = 80, bn7_scaleFactor1 = 9, bn7_scaleFactor2 = 8, bn7_scaleFactor3 = 11, bn7_scaleFactorAdd = 0,
                       bn8_depthWiseStride = 1, bn8_depthWiseChannels = 184, bn8_withSkip = True, bn8_tensorOutC = 80, bn8_scaleFactor1 = 9, bn8_scaleFactor2 = 8, bn8_scaleFactor3 = 11, bn8_scaleFactorAdd = 0,
                       bn9_depthWiseStride = 1, bn9_depthWiseChannels = 184, bn9_withSkip = True, bn9_tensorOutC = 80, bn9_scaleFactor1 = 9, bn9_scaleFactor2 = 8, bn9_scaleFactor3 = 11, bn9_scaleFactorAdd = 0,
                       enableTrace = False, trace_size = 16384, traceSizeInInt32s = 4096):

    
# bn0

    tensorL0_2InC = tensorInC 
    tensorL0_2InW = tensorInW
    tensorL0_2InH = tensorInH
    tensorL0_2OutC=tensorL0_2InC

    tensorL0_3InC = tensorL0_2OutC
    tensorL0_3InW = tensorL0_2InW
    tensorL0_3InH = tensorL0_2InH
    tensorL0_3OutC = tensorL0_3InC


# bn1
    tensorL1_1InC = tensorL0_3OutC
    tensorL1_1InW = tensorL0_3InW
    tensorL1_1InH = tensorL0_3InH
    tensorL1_1OutC=bn1_depthWiseChannels

    tensorL1_2InC = tensorL1_1OutC
    tensorL1_2InW = tensorL1_1InW
    tensorL1_2InH = tensorL1_1InH
    tensorL1_2OutC=tensorL1_2InC

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
    
    # tensorOutW = tensorL8_3InW
    # tensorOutH = tensorL8_3InH
    # tensorOutC = tensorL8_3OutC

    @device(AIEDevice.npu1_3col)
    def device_body():
        
        # define types
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int16_ty = IntegerType.get_signless(16)
        int32_ty = IntegerType.get_signless(32)

        tensorLayerIn_ty = MemRefType.get((tensorInW, 1, tensorInC), uint8_ty)
        tensorLayerOut_ty = MemRefType.get((tensorOutW, 1, tensorOutC), int8_ty)

        # setup all the weights here
        bn0_weights_size= 3 * 3 * tensorL0_3InC * 1 + 1 * 1 * tensorL0_3InC * tensorL0_3OutC
        bn0_weightsAllLayers_ty = MemRefType.get((bn0_weights_size,), int8_ty)
        bn1_weights_size=1 * 1 * tensorL1_1InC * tensorL1_2InC + 3 * 3 * tensorL1_3InC * 1 + 1 * 1 * tensorL1_3InC * tensorL1_3OutC
        bn1_weightsAllLayers_ty = MemRefType.get((bn1_weights_size,), int8_ty)
        bn0_1_weights_size=bn0_weights_size+bn1_weights_size
        bn0_1_weightsAllLayers_ty = MemRefType.get((bn0_1_weights_size,), int8_ty)

        bn2_weights_size=1 * 1 * tensorL2_1InC * tensorL2_2InC + 3 * 3 * tensorL2_3InC * 1 + 1 * 1 * tensorL2_3InC * tensorL2_3OutC
        bn2_weightsAllLayers_ty = MemRefType.get((bn2_weights_size,), int8_ty)
        bn3_weights_size=1 * 1 * tensorL3_1InC * tensorL3_2InC + 3 * 3 * tensorL3_3InC * 1 + 1 * 1 * tensorL3_3InC * tensorL3_3OutC
        bn3_weightsAllLayers_ty = MemRefType.get((bn3_weights_size,), int8_ty)
        bn4_weights_size=1 * 1 * tensorL4_1InC * tensorL4_2InC + 3 * 3 * tensorL4_3InC * 1 + 1 * 1 * tensorL4_3InC * tensorL4_3OutC
        bn4_weightsAllLayers_ty = MemRefType.get((bn4_weights_size,), int8_ty)
        bn5_weights_size=1 * 1 * tensorL5_1InC * tensorL5_2InC + 3 * 3 * tensorL5_3InC * 1 + 1 * 1 * tensorL5_3InC * tensorL5_3OutC
        bn5_weightsAllLayers_ty = MemRefType.get((bn5_weights_size,), int8_ty)
        bn6_weights_size=1 * 1 * tensorL6_1InC * tensorL6_2InC + 3 * 3 * tensorL6_3InC * 1 + 1 * 1 * tensorL6_3InC * tensorL6_3OutC
        bn6_weightsAllLayers_ty = MemRefType.get((bn6_weights_size,), int8_ty)
        bn7_weights_size=1 * 1 * tensorL7_1InC * tensorL7_2InC + 3 * 3 * tensorL7_3InC * 1 + 1 * 1 * tensorL7_3InC * tensorL7_3OutC
        bn7_weightsAllLayers_ty = MemRefType.get((bn7_weights_size,), int8_ty)
        bn8_weights_size=1 * 1 * tensorL8_1InC * tensorL8_2InC + 3 * 3 * tensorL8_3InC * 1 + 1 * 1 * tensorL8_3InC * tensorL8_3OutC
        bn8_weightsAllLayers_ty = MemRefType.get((bn8_weights_size,), int8_ty)

        bn9_weights_size=1 * 1 * tensorL9_1InC * tensorL9_2InC + 3 * 3 * tensorL9_3InC * 1 + 1 * 1 * tensorL9_3InC * tensorL9_3OutC
        bn9_weightsAllLayers_ty = MemRefType.get((bn9_weights_size,), int8_ty)
 
        memtile_01_wts=bn0_weights_size+bn1_weights_size+bn2_weights_size+bn3_weights_size+bn4_weights_size+bn5_weights_size
        memtile_01_wts_ty = MemRefType.get((memtile_01_wts,), int8_ty)
        
        memtile_11_wts=bn6_weights_size+bn7_weights_size+bn8_weights_size+bn9_weights_size
        memtile_11_wts_ty = MemRefType.get((memtile_11_wts,), int8_ty)

        total_weights=memtile_01_wts+memtile_11_wts
        total_weights_ty = MemRefType.get((total_weights,), int8_ty)


        ShimTile00 = tile(tileColIndex, 0)
        ShimTile10 = tile(tileColIndex+1, 0)
        
        MemTile01 = tile(tileColIndex, 1)
        MemTile11 = tile(tileColIndex+1, 1)


        ComputeTile03 = tile(tileColIndex, 3) #bn0+bn1
        ComputeTile04 = tile(tileColIndex, 4) #bn2
        ComputeTile05 = tile(tileColIndex, 5) #bn3
        ComputeTile15 = tile(tileColIndex+1, 5) #bn4
        ComputeTile14 = tile(tileColIndex+1, 4) #bn5
        ComputeTile12 = tile(tileColIndex+1, 2) #bn6
        ComputeTile13 = tile(tileColIndex+1, 3) #bn7
        ComputeTile22 = tile(tileColIndex+2, 2) #bn8
        ComputeTile23 = tile(tileColIndex+2, 3) #bn9

                
        # Set up compute tiles
        rtpComputeTile03 = Buffer(ComputeTile03, [16], T.i32(), "rtp03") #bn0+bn1
        rtpComputeTile04 = Buffer(ComputeTile04, [16], T.i32(), "rtp04") #bn2
        rtpComputeTile05 = Buffer(ComputeTile05, [16], T.i32(), "rtp05") #bn3
        rtpComputeTile15 = Buffer(ComputeTile15, [16], T.i32(), "rtp15") #bn4
        rtpComputeTile14 = Buffer(ComputeTile14, [16], T.i32(), "rtp14") #bn5
        rtpComputeTile12 = Buffer(ComputeTile12, [16], T.i32(), "rtp12") #bn6
        rtpComputeTile13 = Buffer(ComputeTile13, [16], T.i32(), "rtp13") #bn7
        rtpComputeTile22 = Buffer(ComputeTile22, [16], T.i32(), "rtp22") #bn8
        rtpComputeTile23 = Buffer(ComputeTile23, [16], T.i32(), "rtp23") #bn9
        # AIE-array data movement with object fifos
        
        # Input
        act_in = object_fifo("act_in", ShimTile00, ComputeTile03, [3, 3], tensorLayerIn_ty)

        # wts
        wts_OF_01_L3L2 = object_fifo("wts_OF_01_L3L2", ShimTile00, MemTile01, 1, memtile_01_wts_ty)
        bn0_1_wts_OF_L3L1 = object_fifo("bn0_1_wts_OF_L2L1", MemTile01, ComputeTile03, [1,1], bn0_1_weightsAllLayers_ty)
        bn2_wts_OF_L3L1 = object_fifo("bn2_wts_OF_L2L1", MemTile01, ComputeTile04, [1,1], bn2_weightsAllLayers_ty)
        bn3_wts_OF_L3L1 = object_fifo("bn3_wts_OF_L2L1", MemTile01, ComputeTile05, [1,1], bn3_weightsAllLayers_ty)
        bn4_wts_OF_L3L1 = object_fifo("bn4_wts_OF_L2L1", MemTile01, ComputeTile15, [1,1], bn4_weightsAllLayers_ty)
        bn5_wts_OF_L3L1 = object_fifo("bn5_wts_OF_L2L1", MemTile01, ComputeTile14, [1,1], bn5_weightsAllLayers_ty)
        object_fifo_link(wts_OF_01_L3L2, [bn0_1_wts_OF_L3L1,bn2_wts_OF_L3L1,bn3_wts_OF_L3L1,bn4_wts_OF_L3L1,bn5_wts_OF_L3L1],[],[0,
                                                                                                                                 bn0_1_weights_size,
                                                                                                                                 bn0_1_weights_size+bn2_weights_size,
                                                                                                                                 bn0_1_weights_size+bn2_weights_size+bn3_weights_size,
                                                                                                                                 bn0_1_weights_size+bn2_weights_size+bn3_weights_size+bn4_weights_size
                                                                                                                                 ])

        #  # wts
        wts_OF_11_L3L2 = object_fifo("wts_OF_11_L3L2", ShimTile10, MemTile11, 1, memtile_11_wts_ty)
        bn6_wts_OF_L3L1 = object_fifo("bn6_wts_OF_L2L1", MemTile11, ComputeTile12, [1,1], bn6_weightsAllLayers_ty)
        bn7_wts_OF_L3L1 = object_fifo("bn7_wts_OF_L2L1", MemTile11, ComputeTile13, [1,1], bn7_weightsAllLayers_ty)
        bn8_wts_OF_L3L1 = object_fifo("bn8_wts_OF_L2L1", MemTile11, ComputeTile22, [1,1], bn8_weightsAllLayers_ty)
        bn9_wts_OF_L3L1 = object_fifo("bn9_wts_OF_L2L1", MemTile11, ComputeTile23, [1,1], bn9_weightsAllLayers_ty)
        object_fifo_link(wts_OF_11_L3L2, [bn6_wts_OF_L3L1,bn7_wts_OF_L3L1,bn8_wts_OF_L3L1,bn9_wts_OF_L3L1],[],[0,bn6_weights_size,bn6_weights_size+bn7_weights_size,bn6_weights_size+bn7_weights_size+bn8_weights_size])


        # # # ******************************************************************bn0+bn1******************************************************************
        bn0_tensorLayer2In_ty = MemRefType.get((tensorL0_2InW, 1, tensorL0_2InC), uint8_ty)
        bn0_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL0_3InC * 1,), int8_ty)
        bn0_tensorLayer2Out_ty = MemRefType.get((tensorL0_3InW, 1, tensorL0_3InC), uint8_ty)

        bn0_tensorLayer3In_ty = bn0_tensorLayer2Out_ty
        bn0_weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL0_3InC * tensorL0_3OutC,), int8_ty)
        bn0_tensorLayer3Out_ty = MemRefType.get((tensorL0_3InW, 1, tensorL0_3OutC),int8_ty)

         # temporary types for tensor to enable intial test
        bn1_tensorLayer1In_ty = MemRefType.get((tensorL1_1InW, 1, tensorL1_1InC), int8_ty)
        bn1_weightsLayer1_ty = MemRefType.get((1 * 1 * tensorL1_1InC * tensorL1_2InC,), int8_ty)
        bn1_tensorLayer1Out_ty = MemRefType.get((tensorL1_2InW, 1, tensorL1_2InC), uint8_ty)
    
        bn1_tensorLayer2In_ty = bn1_tensorLayer1Out_ty
        bn1_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL1_3InC * 1,), int8_ty)
        bn1_tensorLayer2Out_ty = MemRefType.get((tensorL1_3InW, 1, tensorL1_3InC), uint8_ty)

        bn1_tensorLayer3In_ty = bn1_tensorLayer2Out_ty
        bn1_weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL1_3InC * tensorL1_3OutC,), int8_ty)
        bn1_tensorLayer3Out_ty = MemRefType.get((tensorL1_3InW, 1, tensorL1_3OutC),int8_ty)

        # AIE Core Function declarations
        bn0_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func("bn0_conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[bn0_tensorLayer2In_ty,bn0_tensorLayer2In_ty,bn0_tensorLayer2In_ty, bn0_weightsLayer2_ty, bn0_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn0_conv2dk1_skip_ui8_ui8_i8 = external_func("bn0_conv2dk1_skip_ui8_ui8_i8",inputs=[bn0_tensorLayer3In_ty, bn0_weightsLayer3_ty, bn0_tensorLayer3Out_ty, bn0_tensorLayer2In_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])

        bn1_conv2dk1_relu_i8_ui8 = external_func("bn1_conv2dk1_relu_i8_ui8",inputs=[bn1_tensorLayer1In_ty, bn1_weightsLayer1_ty, bn1_tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn1_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func("bn1_conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[bn1_tensorLayer2In_ty,bn1_tensorLayer2In_ty,bn1_tensorLayer2In_ty, bn1_weightsLayer2_ty, bn1_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn1_conv2dk1_ui8_i8 = external_func("bn1_conv2dk1_ui8_i8",inputs=[bn1_tensorLayer3In_ty, bn1_weightsLayer3_ty, bn1_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])

        # Compute tile 
        bn01_objectArchiveName = "fused_bn0_bn1.a"
        bn0_tensorLayer0_2Out_ty = MemRefType.get((tensorL0_2InW, 1, tensorL0_2OutC),uint8_ty)
        bn0_tensorLayer0_3Out_ty = MemRefType.get((tensorL0_3InW, 1, tensorL0_3OutC),int8_ty)
        bn1_tensorLayer1_1Out_ty = MemRefType.get((tensorL1_1InW, 1, tensorL1_1OutC),uint8_ty)
        bn1_tensorLayer1_2Out_ty = MemRefType.get((tensorL1_3InW, 1, tensorL1_2OutC),uint8_ty)
        bn1_tensorLayer1_3Out_ty = MemRefType.get((tensorL1_3InW, 1, tensorL1_3OutC),int8_ty)

        # between compute tiles
        act_bn01_bn2 = object_fifo("act_bn01_bn2", ComputeTile03, ComputeTile04, [3, 2], bn1_tensorLayer1_3Out_ty)
        # act_out = object_fifo("act_out", ComputeTile03, ShimTile10, 1, bn1_tensorLayer1_3Out_ty)
        bottleneckAFused("bn01", ComputeTile03, act_in, bn0_1_wts_OF_L3L1, act_bn01_bn2, rtpComputeTile03, bn01_objectArchiveName,
                         bn0_conv2dk3_dw_stride1_relu_ui8_ui8, bn0_conv2dk1_skip_ui8_ui8_i8, bn1_conv2dk1_relu_i8_ui8, bn1_conv2dk3_dw_stride2_relu_ui8_ui8, bn1_conv2dk1_ui8_i8,
                         bn0_tensorLayer0_2Out_ty, bn0_tensorLayer0_3Out_ty,bn1_tensorLayer1_1Out_ty,bn1_tensorLayer1_2Out_ty, tensorL0_2InW, tensorL0_2InH, tensorL0_2InC,  bn1_depthWiseStride, bn1_depthWiseChannels, tensorL1_3OutC)

        # # # # ******************************************************************bn2******************************************************************
        #  # temporary types for tensor to enable intial test
        bn2_tensorLayer1In_ty = MemRefType.get((tensorL2_1InW, 1, tensorL2_1InC), int8_ty)
        bn2_weightsLayer1_ty = MemRefType.get((1 * 1 * tensorL2_1InC * tensorL2_2InC,), int8_ty)
        bn2_tensorLayer1Out_ty = MemRefType.get((tensorL2_2InW, 1, tensorL2_2InC), uint8_ty)
    
        bn2_tensorLayer2In_ty = bn2_tensorLayer1Out_ty
        bn2_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL2_3InC * 1,), int8_ty)
        bn2_tensorLayer2Out_ty = MemRefType.get((tensorL2_3InW, 1, tensorL2_3InC), uint8_ty)

        bn2_tensorLayer3In_ty = bn2_tensorLayer2Out_ty
        bn2_weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL2_3InC * tensorL2_3OutC,), int8_ty)
        bn2_tensorLayer3Out_ty = MemRefType.get((tensorL2_3InW, 1, tensorL2_3OutC),int8_ty)
        
        # AIE Core Function declarations
        bn2_conv2dk1_relu_i8_ui8 = external_func("bn2_conv2dk1_relu_i8_ui8",inputs=[bn2_tensorLayer1In_ty, bn2_weightsLayer1_ty, bn2_tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn2_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func("bn2_conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[bn2_tensorLayer2In_ty,bn2_tensorLayer2In_ty,bn2_tensorLayer2In_ty, bn2_weightsLayer2_ty, bn2_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn2_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func("bn2_conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[bn2_tensorLayer2In_ty,bn2_tensorLayer2In_ty,bn2_tensorLayer2In_ty, bn2_weightsLayer2_ty, bn2_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn2_conv2dk1_skip_ui8_i8_i8 = external_func("bn2_conv2dk1_skip_ui8_i8_i8",inputs=[bn2_tensorLayer3In_ty, bn2_weightsLayer3_ty, bn2_tensorLayer3Out_ty, bn2_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        
        bn2_conv2dk1_ui8_i8 = external_func("bn2_conv2dk1_ui8_i8",inputs=[bn2_tensorLayer3In_ty, bn2_weightsLayer3_ty, bn2_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])

        # Compute tile 
        bn2_objectArchiveName = "bn2_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a" % (bn2_depthWiseStride, "skip" if (bn2_withSkip) else "")
        bn2_tensorLayer1Out_ty = MemRefType.get((tensorL2_2InW, 1, tensorL2_2InC),uint8_ty)
        bn2_tensorLayer2Out_ty = MemRefType.get((tensorL2_3InW, 1, tensorL2_3InC),uint8_ty)
        bn2_tensorLayer3Out_ty = MemRefType.get((tensorL2_3InW, 1, tensorL2_3OutC),int8_ty)        

       

        # between compute tiles
        act_bn2_bn3 = object_fifo("act_bn2_bn3", ComputeTile04, ComputeTile05, [3, 2], bn2_tensorLayer3Out_ty)
        # act_out = object_fifo("act_out", ComputeTile04, [ShimTile10], 1, bn2_tensorLayer3Out_ty)

        bottleneckACore("bn2", ComputeTile04, act_bn01_bn2, bn2_wts_OF_L3L1, act_bn2_bn3, rtpComputeTile04, bn2_objectArchiveName,
                        bn2_conv2dk1_relu_i8_ui8, bn2_conv2dk3_dw_stride1_relu_ui8_ui8, bn2_conv2dk3_dw_stride2_relu_ui8_ui8, bn2_conv2dk1_ui8_i8, bn2_conv2dk1_skip_ui8_i8_i8,
                        bn2_tensorLayer1Out_ty, bn2_tensorLayer2Out_ty, tensorL2_1InW, tensorL2_1InH, tensorL2_1InC,  bn2_depthWiseStride, bn2_depthWiseChannels, tensorL2_3OutC, bn2_withSkip,bn2_scaleFactor1, bn2_scaleFactor2, bn2_scaleFactor3,  bn2_scaleFactorAdd)


        # # # # # ******************************************************************bn3******************************************************************
        # #  # temporary types for tensor to enable intial test
        bn3_tensorLayer1In_ty = MemRefType.get((tensorL3_1InW, 1, tensorL3_1InC), int8_ty)
        bn3_weightsLayer1_ty = MemRefType.get((1 * 1 * tensorL3_1InC * tensorL3_2InC,), int8_ty)
        bn3_tensorLayer1Out_ty = MemRefType.get((tensorL3_2InW, 1, tensorL3_2InC), uint8_ty)
        
        bn3_tensorLayer2In_ty = bn3_tensorLayer1Out_ty
        bn3_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL3_3InC * 1,), int8_ty)
        bn3_tensorLayer2Out_ty = MemRefType.get((tensorL3_3InW, 1, tensorL3_3InC), uint8_ty)

        bn3_tensorLayer3In_ty = bn3_tensorLayer2Out_ty
        bn3_weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL3_3InC * tensorL3_3OutC,), int8_ty)
        bn3_tensorLayer3Out_ty = MemRefType.get((tensorL3_3InW, 1, tensorL3_3OutC),int8_ty)
        
        # AIE Core Function declarations
        bn3_conv2dk1_relu_i8_ui8 = external_func("bn3_conv2dk1_relu_i8_ui8",inputs=[bn3_tensorLayer1In_ty, bn3_weightsLayer1_ty, bn3_tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn3_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func("bn3_conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[bn3_tensorLayer2In_ty,bn3_tensorLayer2In_ty,bn3_tensorLayer2In_ty, bn3_weightsLayer2_ty, bn3_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn3_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func("bn3_conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[bn3_tensorLayer2In_ty,bn3_tensorLayer2In_ty,bn3_tensorLayer2In_ty, bn3_weightsLayer2_ty, bn3_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn3_conv2dk1_skip_ui8_i8_i8 = external_func("bn3_conv2dk1_skip_ui8_i8_i8",inputs=[bn3_tensorLayer3In_ty, bn3_weightsLayer3_ty, bn3_tensorLayer3Out_ty, bn3_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn3_conv2dk1_ui8_i8 = external_func("bn3_conv2dk1_ui8_i8",inputs=[bn3_tensorLayer3In_ty, bn3_weightsLayer3_ty, bn3_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])

        # Compute tile 
        bn3_objectArchiveName = "bn3_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a" % (bn3_depthWiseStride, "skip" if (bn3_withSkip) else "")
        bn3_tensorLayer1Out_ty = MemRefType.get((tensorL3_2InW, 1, tensorL3_2InC),uint8_ty)
        bn3_tensorLayer2Out_ty = MemRefType.get((tensorL3_3InW, 1, tensorL3_3InC),uint8_ty)
        bn3_tensorLayer3Out_ty = MemRefType.get((tensorL3_3InW, 1, tensorL3_3OutC),int8_ty)        
       

        # # # between compute tiles
        act_bn3_bn4 = object_fifo("act_bn3_bn4", ComputeTile05, ComputeTile15, [3, 2], bn3_tensorLayer3Out_ty)
        # act_out = object_fifo("act_out", ComputeTile05, [ShimTile10], 1, bn3_tensorLayer3Out_ty)
        bottleneckACore("bn3", ComputeTile05, act_bn2_bn3, bn3_wts_OF_L3L1, act_bn3_bn4, rtpComputeTile05, bn3_objectArchiveName,
                         bn3_conv2dk1_relu_i8_ui8, bn3_conv2dk3_dw_stride1_relu_ui8_ui8, bn3_conv2dk3_dw_stride2_relu_ui8_ui8, bn3_conv2dk1_ui8_i8, bn3_conv2dk1_skip_ui8_i8_i8,
                           bn3_tensorLayer1Out_ty, bn3_tensorLayer2Out_ty, tensorL3_1InW, tensorL3_1InH, tensorL3_1InC, bn3_depthWiseStride, bn3_depthWiseChannels, tensorL3_3OutC, bn3_withSkip,bn3_scaleFactor1, bn3_scaleFactor2, bn3_scaleFactor3,  bn3_scaleFactorAdd)

        # # # # # ******************************************************************bn4******************************************************************

         # temporary types for tensor to enable intial test
        bn4_tensorLayer1In_ty = MemRefType.get((tensorL4_1InW, 1, tensorL4_1InC), int8_ty)
        bn4_weightsLayer1_ty = MemRefType.get((1 * 1 * tensorL4_1InC * tensorL4_2InC,), int8_ty)
        bn4_tensorLayer1Out_ty = MemRefType.get((tensorL4_2InW, 1, tensorL4_2InC), uint8_ty)
    
        bn4_tensorLayer2In_ty = bn4_tensorLayer1Out_ty
        bn4_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL4_3InC * 1,), int8_ty)
        bn4_tensorLayer2Out_ty = MemRefType.get((tensorL4_3InW, 1, tensorL4_3InC), uint8_ty)

        bn4_tensorLayer3In_ty = bn4_tensorLayer2Out_ty
        bn4_weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL4_3InC * tensorL4_3OutC,), int8_ty)
        bn4_tensorLayer3Out_ty = MemRefType.get((tensorL4_3InW, 1, tensorL4_3OutC),int8_ty)
        
        # AIE Core Function declarations
        bn4_conv2dk1_relu_i8_ui8 = external_func("bn4_conv2dk1_relu_i8_ui8",inputs=[bn4_tensorLayer1In_ty, bn4_weightsLayer1_ty, bn4_tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn4_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func("bn4_conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[bn4_tensorLayer2In_ty,bn4_tensorLayer2In_ty,bn4_tensorLayer2In_ty, bn4_weightsLayer2_ty, bn4_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn4_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func("bn4_conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[bn4_tensorLayer2In_ty,bn4_tensorLayer2In_ty,bn4_tensorLayer2In_ty, bn4_weightsLayer2_ty, bn4_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn4_conv2dk1_skip_ui8_i8_i8 = external_func("bn4_conv2dk1_skip_ui8_i8_i8",inputs=[bn4_tensorLayer3In_ty, bn4_weightsLayer3_ty, bn4_tensorLayer3Out_ty, bn4_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn4_conv2dk1_ui8_i8 = external_func("bn4_conv2dk1_ui8_i8",inputs=[bn4_tensorLayer3In_ty, bn4_weightsLayer3_ty, bn4_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])

        # Compute tile 6
        bn4_objectArchiveName = "bn4_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a" % (bn4_depthWiseStride, "skip" if (bn4_withSkip) else "")
        bn4_tensorLayer1Out_ty = MemRefType.get((tensorL4_2InW, 1, tensorL4_2InC),uint8_ty)
        bn4_tensorLayer2Out_ty = MemRefType.get((tensorL4_3InW, 1, tensorL4_3InC),uint8_ty)
        bn4_tensorLayer3Out_ty = MemRefType.get((tensorL4_3InW, 1, tensorL4_3OutC),int8_ty)        

       

        # # between compute tiles
        act_bn4_bn5 = object_fifo("act_bn4_bn5", ComputeTile15, ComputeTile14, [3, 2], bn4_tensorLayer3Out_ty)
        # act_out = object_fifo("act_out", ComputeTile15, [ShimTile10], 1, bn4_tensorLayer3Out_ty)
        bottleneckACore("bn4", ComputeTile15, act_bn3_bn4, bn4_wts_OF_L3L1, act_bn4_bn5, rtpComputeTile15, bn4_objectArchiveName,
                         bn4_conv2dk1_relu_i8_ui8, bn4_conv2dk3_dw_stride1_relu_ui8_ui8, bn4_conv2dk3_dw_stride2_relu_ui8_ui8, bn4_conv2dk1_ui8_i8, bn4_conv2dk1_skip_ui8_i8_i8,
                           bn4_tensorLayer1Out_ty, bn4_tensorLayer2Out_ty, tensorL4_1InW, tensorL4_1InH, tensorL4_1InC,  bn4_depthWiseStride, bn4_depthWiseChannels, tensorL4_3OutC, bn4_withSkip)

        # # # # # # ******************************************************************bn5******************************************************************

        # temporary types for tensor to enable intial test
        bn5_tensorLayer1In_ty = MemRefType.get((tensorL5_1InW, 1, tensorL5_1InC), int8_ty)
        bn5_weightsLayer1_ty = MemRefType.get((1 * 1 * tensorL5_1InC * tensorL5_2InC,), int8_ty)
        bn5_tensorLayer1Out_ty = MemRefType.get((tensorL5_2InW, 1, tensorL5_2InC), uint8_ty)
    
        bn5_tensorLayer2In_ty = bn5_tensorLayer1Out_ty
        bn5_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL5_3InC * 1,), int8_ty)
        bn5_tensorLayer2Out_ty = MemRefType.get((tensorL5_3InW, 1, tensorL5_3InC), uint8_ty)

        bn5_tensorLayer3In_ty = bn5_tensorLayer2Out_ty
        bn5_weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL5_3InC * tensorL5_3OutC,), int8_ty)
        bn5_tensorLayer3Out_ty = MemRefType.get((tensorL5_3InW, 1, tensorL5_3OutC),int8_ty)
        
        # AIE Core Function declarations
        bn5_conv2dk1_relu_i8_ui8 = external_func("bn5_conv2dk1_relu_i8_ui8",inputs=[bn5_tensorLayer1In_ty, bn5_weightsLayer1_ty, bn5_tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn5_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func("bn5_conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[bn5_tensorLayer2In_ty,bn5_tensorLayer2In_ty,bn5_tensorLayer2In_ty, bn5_weightsLayer2_ty, bn5_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn5_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func("bn5_conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[bn5_tensorLayer2In_ty,bn5_tensorLayer2In_ty,bn5_tensorLayer2In_ty, bn5_weightsLayer2_ty, bn5_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn5_conv2dk1_skip_ui8_i8_i8 = external_func("bn5_conv2dk1_skip_ui8_i8_i8",inputs=[bn5_tensorLayer3In_ty, bn5_weightsLayer3_ty, bn5_tensorLayer3Out_ty, bn5_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn5_conv2dk1_ui8_i8 = external_func("bn5_conv2dk1_ui8_i8",inputs=[bn5_tensorLayer3In_ty, bn5_weightsLayer3_ty, bn5_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])

        # Compute tile 6
        bn5_objectArchiveName = "bn5_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a" % (bn5_depthWiseStride, "skip" if (bn5_withSkip) else "")
        bn5_tensorLayer1Out_ty = MemRefType.get((tensorL5_2InW, 1, tensorL5_2InC),uint8_ty)
        bn5_tensorLayer2Out_ty = MemRefType.get((tensorL5_3InW, 1, tensorL5_3InC),uint8_ty)
        bn5_tensorLayer3Out_ty = MemRefType.get((tensorL5_3InW, 1, tensorL5_3OutC),int8_ty)        

       

        # between compute tiles
        act_bn5_bn6 = object_fifo("act_bn5_bn6", ComputeTile14, ComputeTile12, 2, bn5_tensorLayer3Out_ty)
        # act_out = object_fifo("act_out", ComputeTile14, [ShimTile10], 1, bn5_tensorLayer3Out_ty)
        bottleneckACore("bn5", ComputeTile14, act_bn4_bn5, bn5_wts_OF_L3L1, act_bn5_bn6, rtpComputeTile14, bn5_objectArchiveName,
                         bn5_conv2dk1_relu_i8_ui8, bn5_conv2dk3_dw_stride1_relu_ui8_ui8, bn5_conv2dk3_dw_stride2_relu_ui8_ui8, bn5_conv2dk1_ui8_i8, bn5_conv2dk1_skip_ui8_i8_i8,
                           bn5_tensorLayer1Out_ty, bn5_tensorLayer2Out_ty, tensorL5_1InW, tensorL5_1InH, tensorL5_1InC,  bn5_depthWiseStride, bn5_depthWiseChannels, tensorL5_3OutC, bn5_withSkip)


        # # temporary types for tensor to enable intial test
        bn6_tensorLayer1In_ty = MemRefType.get((tensorL6_1InW, 1, tensorL6_1InC), int8_ty)
        bn6_weightsLayer1_ty = MemRefType.get((1 * 1 * tensorL6_1InC * tensorL6_2InC,), int8_ty)
        bn6_tensorLayer2In_ty = MemRefType.get((tensorL6_2InW, 1, tensorL6_2InC), uint8_ty)
        bn6_tensorLayer1Out_ty = bn6_tensorLayer2In_ty
        bn6_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL6_3InC * 1,), int8_ty)
        bn6_tensorLayer3In_ty = MemRefType.get((tensorL6_3InW, 1, tensorL6_3InC), uint8_ty)
        bn6_tensorLayer2Out_ty = bn6_tensorLayer3In_ty
        bn6_weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL6_3InC * tensorL6_3OutC,), int8_ty)
        bn6_tensorLayer3Out_ty = MemRefType.get((tensorL6_3InW, 1, tensorL6_3OutC),int8_ty)
        
        # AIE Core Function declarations
        bn6_conv2dk1_relu_i8_ui8 = external_func("bn6_conv2dk1_relu_i8_ui8",inputs=[bn6_tensorLayer1In_ty, bn6_weightsLayer1_ty, bn6_tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn6_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func("bn6_conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[bn6_tensorLayer2In_ty,bn6_tensorLayer2In_ty,bn6_tensorLayer2In_ty, bn6_weightsLayer2_ty, bn6_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn6_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func("bn6_conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[bn6_tensorLayer2In_ty,bn6_tensorLayer2In_ty,bn6_tensorLayer2In_ty, bn6_weightsLayer2_ty, bn6_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn6_conv2dk1_skip_ui8_i8_i8 = external_func("bn6_conv2dk1_skip_ui8_i8_i8",inputs=[bn6_tensorLayer3In_ty, bn6_weightsLayer3_ty, bn6_tensorLayer3Out_ty, bn6_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn6_conv2dk1_ui8_i8 = external_func("bn6_conv2dk1_ui8_i8",inputs=[bn6_tensorLayer3In_ty, bn6_weightsLayer3_ty, bn6_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])

        
        # # Compute tile 6
        bn6_objectArchiveName = "bn6_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a" % (bn6_depthWiseStride, "skip" if (bn6_withSkip) else "")
        bn6_tensorLayer1Out_ty = MemRefType.get((tensorL6_2InW, 1, tensorL6_2InC),uint8_ty)
        bn6_tensorLayer2Out_ty = MemRefType.get((tensorL6_3InW, 1, tensorL6_3InC),uint8_ty)
        bn6_tensorLayer3Out_ty = MemRefType.get((tensorL6_3InW, 1, tensorL6_3OutC),int8_ty)

        # between compute tiles
        act_bn6_bn7 = object_fifo("act_bn6_bn7", ComputeTile12, ComputeTile13, 2, bn6_tensorLayer3Out_ty)

        # act_out = object_fifo("act_out", ComputeTile12, [ShimTile10], 1, bn6_tensorLayer3Out_ty)
        bottleneckACore("bn6", ComputeTile12, act_bn5_bn6, bn6_wts_OF_L3L1, act_bn6_bn7, rtpComputeTile12, bn6_objectArchiveName,
                         bn6_conv2dk1_relu_i8_ui8, bn6_conv2dk3_dw_stride1_relu_ui8_ui8, bn6_conv2dk3_dw_stride2_relu_ui8_ui8, bn6_conv2dk1_ui8_i8, bn6_conv2dk1_skip_ui8_i8_i8,
                           bn6_tensorLayer1Out_ty, bn6_tensorLayer2Out_ty, tensorL6_1InW, tensorL6_1InH, tensorL6_1InC,  bn6_depthWiseStride, bn6_depthWiseChannels, tensorL6_3OutC, bn6_withSkip)

        # ##### ******************************************************************************************************************************
        bn7_tensorLayer1In_ty = MemRefType.get((tensorL7_1InW, 1, tensorL7_1InC), int8_ty)
        bn7_weightsLayer1_ty = MemRefType.get((1 * 1 * tensorL7_1InC * tensorL7_2InC,), int8_ty)
        bn7_tensorLayer2In_ty = MemRefType.get((tensorL7_2InW, 1, tensorL7_2InC), uint8_ty)
        bn7_tensorLayer1Out_ty = bn7_tensorLayer2In_ty
        bn7_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL7_3InC * 1,), int8_ty)
        bn7_tensorLayer3In_ty = MemRefType.get((tensorL7_3InW, 1, tensorL7_3InC), uint8_ty)
        bn7_tensorLayer2Out_ty = bn7_tensorLayer3In_ty
        bn7_weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL7_3InC * tensorL7_3OutC,), int8_ty)
        bn7_tensorLayer3Out_ty = MemRefType.get((tensorL7_3InW, 1, tensorL7_3OutC),int8_ty)
        
        

        # AIE Core Function declarations
        bn7_conv2dk1_relu_i8_ui8 = external_func("bn7_conv2dk1_relu_i8_ui8",inputs=[bn7_tensorLayer1In_ty, bn7_weightsLayer1_ty, bn7_tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn7_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func("bn7_conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[bn7_tensorLayer2In_ty,bn7_tensorLayer2In_ty,bn7_tensorLayer2In_ty, bn7_weightsLayer2_ty, bn7_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn7_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func("bn7_conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[bn7_tensorLayer2In_ty,bn7_tensorLayer2In_ty,bn7_tensorLayer2In_ty, bn7_weightsLayer2_ty, bn7_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn7_conv2dk1_skip_ui8_i8_i8 = external_func("bn7_conv2dk1_skip_ui8_i8_i8",inputs=[bn7_tensorLayer3In_ty, bn7_weightsLayer3_ty, bn7_tensorLayer3Out_ty, bn7_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn7_conv2dk1_ui8_i8 = external_func("bn7_conv2dk1_ui8_i8",inputs=[bn7_tensorLayer3In_ty, bn7_weightsLayer3_ty, bn7_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])

        bn7_objectArchiveName = "bn7_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a" % (bn7_depthWiseStride, "skip" if (bn7_withSkip) else "")

        # between compute tiles
        act_bn7_bn8 = object_fifo("act_bn7_bn8", ComputeTile13, ComputeTile22, 2, bn7_tensorLayer3Out_ty)
        # act_out = object_fifo("act_out", ComputeTile13, [ShimTile10], 1, bn7_tensorLayer3Out_ty)
        bottleneckACore("bn7", ComputeTile13, act_bn6_bn7, bn7_wts_OF_L3L1, act_bn7_bn8, rtpComputeTile13, bn7_objectArchiveName, 
                        bn7_conv2dk1_relu_i8_ui8, bn7_conv2dk3_dw_stride1_relu_ui8_ui8, bn7_conv2dk3_dw_stride2_relu_ui8_ui8, bn7_conv2dk1_ui8_i8, bn7_conv2dk1_skip_ui8_i8_i8, 
                        bn7_tensorLayer1Out_ty, bn7_tensorLayer2Out_ty, tensorL7_1InW, tensorL7_1InH, tensorL7_1InC, bn7_depthWiseStride, bn7_depthWiseChannels, tensorL7_3OutC, bn7_withSkip)


        # ##### ******************************************************************************************************************************        
        bn8_tensorLayer1In_ty = MemRefType.get((tensorL8_1InW, 1, tensorL8_1InC), int8_ty)
        bn8_weightsLayer1_ty = MemRefType.get((1 * 1 * tensorL8_1InC * tensorL8_2InC,), int8_ty)
        bn8_tensorLayer2In_ty = MemRefType.get((tensorL8_2InW, 1, tensorL8_2InC), uint8_ty)
        bn8_tensorLayer1Out_ty = bn8_tensorLayer2In_ty
        bn8_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL8_3InC * 1,), int8_ty)
        bn8_tensorLayer3In_ty = MemRefType.get((tensorL8_3InW, 1, tensorL8_3InC), uint8_ty)
        bn8_tensorLayer2Out_ty = bn8_tensorLayer3In_ty
        bn8_weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL8_3InC * tensorL8_3OutC,), int8_ty)
        bn8_tensorLayer3Out_ty = MemRefType.get((tensorL8_3InW, 1, tensorL8_3OutC),int8_ty)
        
       
        # AIE Core Function declarations
        bn8_conv2dk1_relu_i8_ui8 = external_func("bn8_conv2dk1_relu_i8_ui8",inputs=[bn8_tensorLayer1In_ty, bn8_weightsLayer1_ty, bn8_tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn8_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func("bn8_conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[bn8_tensorLayer2In_ty,bn8_tensorLayer2In_ty,bn8_tensorLayer2In_ty, bn8_weightsLayer2_ty, bn8_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn8_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func("bn8_conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[bn8_tensorLayer2In_ty,bn8_tensorLayer2In_ty,bn8_tensorLayer2In_ty, bn8_weightsLayer2_ty, bn8_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn8_conv2dk1_skip_ui8_i8_i8 = external_func("bn8_conv2dk1_skip_ui8_i8_i8",inputs=[bn8_tensorLayer3In_ty, bn8_weightsLayer3_ty, bn8_tensorLayer3Out_ty, bn8_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn8_conv2dk1_ui8_i8 = external_func("bn8_conv2dk1_ui8_i8",inputs=[bn8_tensorLayer3In_ty, bn8_weightsLayer3_ty, bn8_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])

        bn8_objectArchiveName = "bn8_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a" % (bn8_depthWiseStride, "skip" if (bn8_withSkip) else "")

         # Output
        act_bn8_bn9 = object_fifo("act_bn8_bn9", ComputeTile22, ComputeTile23, 2, bn8_tensorLayer3Out_ty)
        # act_out = object_fifo("act_out", ComputeTile22, [ShimTile10], 1, bn8_tensorLayer3Out_ty)

        bottleneckACore("bn8", ComputeTile22, act_bn7_bn8, bn8_wts_OF_L3L1, act_bn8_bn9, rtpComputeTile22, bn8_objectArchiveName, 
                        bn8_conv2dk1_relu_i8_ui8, bn8_conv2dk3_dw_stride1_relu_ui8_ui8, bn8_conv2dk3_dw_stride2_relu_ui8_ui8, bn8_conv2dk1_ui8_i8, bn8_conv2dk1_skip_ui8_i8_i8, 
                        bn8_tensorLayer1Out_ty, bn8_tensorLayer2Out_ty, tensorL8_1InW, tensorL8_1InH, tensorL8_1InC, bn8_depthWiseStride, bn8_depthWiseChannels, tensorL8_3OutC, bn8_withSkip)

        # ##### ******************************************************************************************************************************        
        bn9_tensorLayer1In_ty = MemRefType.get((tensorL9_1InW, 1, tensorL9_1InC), int8_ty)
        bn9_weightsLayer1_ty = MemRefType.get((1 * 1 * tensorL9_1InC * tensorL9_2InC,), int8_ty)
        bn9_tensorLayer2In_ty = MemRefType.get((tensorL9_2InW, 1, tensorL9_2InC), uint8_ty)
        bn9_tensorLayer1Out_ty = bn9_tensorLayer2In_ty
        bn9_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL9_3InC * 1,), int8_ty)
        bn9_tensorLayer3In_ty = MemRefType.get((tensorL9_3InW, 1, tensorL9_3InC), uint8_ty)
        bn9_tensorLayer2Out_ty = bn9_tensorLayer3In_ty
        bn9_weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL9_3InC * tensorL9_3OutC,), int8_ty)
        bn9_tensorLayer3Out_ty = MemRefType.get((tensorL9_3InW, 1, tensorL9_3OutC),int8_ty)
        
        # Output
        act_out = object_fifo("act_out", ComputeTile23, ShimTile10, 2, tensorLayerOut_ty)

        # AIE Core Function declarations
        bn9_conv2dk1_relu_i8_ui8 = external_func("bn9_conv2dk1_relu_i8_ui8",inputs=[bn9_tensorLayer1In_ty, bn9_weightsLayer1_ty, bn9_tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn9_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func("bn9_conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[bn9_tensorLayer2In_ty,bn9_tensorLayer2In_ty,bn9_tensorLayer2In_ty, bn9_weightsLayer2_ty, bn9_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn9_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func("bn9_conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[bn9_tensorLayer2In_ty,bn9_tensorLayer2In_ty,bn9_tensorLayer2In_ty, bn9_weightsLayer2_ty, bn9_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn9_conv2dk1_skip_ui8_i8_i8 = external_func("bn9_conv2dk1_skip_ui8_i8_i8",inputs=[bn9_tensorLayer3In_ty, bn9_weightsLayer3_ty, bn9_tensorLayer3Out_ty, bn9_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn9_conv2dk1_ui8_i8 = external_func("bn9_conv2dk1_ui8_i8",inputs=[bn9_tensorLayer3In_ty, bn9_weightsLayer3_ty, bn9_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])

        bn9_objectArchiveName = "bn9_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a" % (bn9_depthWiseStride, "skip" if (bn9_withSkip) else "")

        bottleneckACore("bn9", ComputeTile23, act_bn8_bn9, bn9_wts_OF_L3L1, act_out, rtpComputeTile23, bn9_objectArchiveName, 
                        bn9_conv2dk1_relu_i8_ui8, bn9_conv2dk3_dw_stride1_relu_ui8_ui8, bn9_conv2dk3_dw_stride2_relu_ui8_ui8, bn9_conv2dk1_ui8_i8, bn9_conv2dk1_skip_ui8_i8_i8, 
                        bn9_tensorLayer1Out_ty, bn9_tensorLayer2Out_ty, tensorL9_1InW, tensorL9_1InH, tensorL9_1InC, bn9_depthWiseStride, bn9_depthWiseChannels, tensorL9_3OutC, bn9_withSkip)

         # instruction stream generation
        activationsInSize32b = (tensorInW * tensorInH * tensorInC) // 4
        activationsOutSize32b = (tensorOutW * tensorOutH * tensorOutC) // 4
        activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
        weightsInL3_ty = MemRefType.get((total_weights//4,), int32_ty)
        activationsOutL3_ty = MemRefType.get((activationsOutSize32b,), int32_ty)

        memtile_01_wts32b=(memtile_01_wts)//4
        memtile_11_wts32b=(memtile_11_wts)//4

        @FuncOp.from_py_func(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
        def sequence(inputFromL3, weightsFromL3, outputToL3):
            
            # # bn0+1
            NpuWriteRTPOp("rtp03", col=tileColIndex, row=3, index=0, value=bn0_scaleFactor2)
            NpuWriteRTPOp("rtp03", col=tileColIndex, row=3, index=1, value=bn0_scaleFactor3)
            NpuWriteRTPOp("rtp03", col=tileColIndex, row=3, index=2, value=bn0_scaleFactorAdd)
            NpuWriteRTPOp("rtp03", col=tileColIndex, row=3, index=3, value=bn1_scaleFactor1)
            NpuWriteRTPOp("rtp03", col=tileColIndex, row=3, index=4, value=bn1_scaleFactor2)
            NpuWriteRTPOp("rtp03", col=tileColIndex, row=3, index=5, value=bn1_scaleFactor3)

            # bn2
            NpuWriteRTPOp("rtp04", col=tileColIndex, row=4, index=0, value=bn2_scaleFactor1)
            NpuWriteRTPOp("rtp04", col=tileColIndex, row=4, index=1, value=bn2_scaleFactor2)
            NpuWriteRTPOp("rtp04", col=tileColIndex, row=4, index=2, value=bn2_scaleFactor3)
            NpuWriteRTPOp("rtp04", col=tileColIndex, row=4, index=3, value=bn2_scaleFactorAdd)

            # # bn3
            NpuWriteRTPOp("rtp05", col=tileColIndex, row=5, index=0, value=bn3_scaleFactor1)
            NpuWriteRTPOp("rtp05", col=tileColIndex, row=5, index=1, value=bn3_scaleFactor2)
            NpuWriteRTPOp("rtp05", col=tileColIndex, row=5, index=2, value=bn3_scaleFactor3)
            NpuWriteRTPOp("rtp05", col=tileColIndex, row=5, index=3, value=bn3_scaleFactorAdd)

            # bn4
            NpuWriteRTPOp("rtp15", col=tileColIndex+1, row=5, index=0, value=bn4_scaleFactor1)
            NpuWriteRTPOp("rtp15", col=tileColIndex+1, row=5, index=1, value=bn4_scaleFactor2)
            NpuWriteRTPOp("rtp15", col=tileColIndex+1, row=5, index=2, value=bn4_scaleFactor3)
            NpuWriteRTPOp("rtp15", col=tileColIndex+1, row=5, index=3, value=bn4_scaleFactorAdd)

            # # bn5
            NpuWriteRTPOp("rtp14", col=tileColIndex+1, row=4, index=0, value=bn5_scaleFactor1)
            NpuWriteRTPOp("rtp14", col=tileColIndex+1, row=4, index=1, value=bn5_scaleFactor2)
            NpuWriteRTPOp("rtp14", col=tileColIndex+1, row=4, index=2, value=bn5_scaleFactor3)
            NpuWriteRTPOp("rtp14", col=tileColIndex+1, row=4, index=3, value=bn5_scaleFactorAdd)

            NpuWriteRTPOp("rtp12", col=tileColIndex+1, row=2, index=0, value=bn6_scaleFactor1)
            NpuWriteRTPOp("rtp12", col=tileColIndex+1, row=2, index=1, value=bn6_scaleFactor2)
            NpuWriteRTPOp("rtp12", col=tileColIndex+1, row=2, index=2, value=bn6_scaleFactor3)
            NpuWriteRTPOp("rtp12", col=tileColIndex+1, row=2, index=3, value=bn6_scaleFactorAdd)


            NpuWriteRTPOp("rtp13", col=tileColIndex+1, row=3, index=0, value=bn7_scaleFactor1)
            NpuWriteRTPOp("rtp13", col=tileColIndex+1, row=3, index=1, value=bn7_scaleFactor2)
            NpuWriteRTPOp("rtp13", col=tileColIndex+1, row=3, index=2, value=bn7_scaleFactor3)
            NpuWriteRTPOp("rtp13", col=tileColIndex+1, row=3, index=3, value=bn7_scaleFactorAdd)

            NpuWriteRTPOp("rtp22", col=tileColIndex+2, row=2, index=0, value=bn8_scaleFactor1)
            NpuWriteRTPOp("rtp22", col=tileColIndex+2, row=2, index=1, value=bn8_scaleFactor2)
            NpuWriteRTPOp("rtp22", col=tileColIndex+2, row=2, index=2, value=bn8_scaleFactor3)
            NpuWriteRTPOp("rtp22", col=tileColIndex+2, row=2, index=3, value=bn8_scaleFactorAdd)

            NpuWriteRTPOp("rtp23", col=tileColIndex+2, row=3, index=0, value=bn9_scaleFactor1)
            NpuWriteRTPOp("rtp23", col=tileColIndex+2, row=3, index=1, value=bn9_scaleFactor2)
            NpuWriteRTPOp("rtp23", col=tileColIndex+2, row=3, index=2, value=bn9_scaleFactor3)
            NpuWriteRTPOp("rtp23", col=tileColIndex+2, row=3, index=3, value=bn9_scaleFactorAdd)
            
            npu_dma_memcpy_nd(
                metadata="act_in",
                bd_id=0,
                mem=inputFromL3,
                sizes=[1, 1, 1, activationsInSize32b],
            )
            npu_dma_memcpy_nd(
                metadata="act_out",
                bd_id=2,
                mem=outputToL3,
                sizes=[1, 1, 1, activationsOutSize32b],
            )
            npu_dma_memcpy_nd(
                metadata="wts_OF_01_L3L2",
                bd_id=1,
                mem=weightsFromL3,
                sizes=[1, 1, 1, memtile_01_wts32b],
            )
            npu_dma_memcpy_nd(
                metadata="wts_OF_11_L3L2",
                bd_id=1,
                mem=weightsFromL3,
                offsets=[0, 0, 0, memtile_01_wts32b],
                sizes=[1, 1, 1, memtile_11_wts32b],
            )
            npu_sync(column=1, row=0, direction=0, channel=0)



with mlir_mod_ctx() as ctx:

    mobilenetV3_bn_0_1_2_3_4_5_6_7_8_9(tileColIndex = 0, tensorInW = 112, tensorInH = 112, tensorInC = 16, 
                                                                        bn0_scaleFactor2 = scale_factors["BN0"]["conv3x3"], bn0_scaleFactor3 = scale_factors["BN0"]["conv1x1_2"],  bn0_scaleFactorAdd = scale_factors["BN0"]["skip_add"],
                    bn1_depthWiseStride = 2, bn1_depthWiseChannels = 64,  bn1_withSkip = False, bn1_tensorOutC = 24, bn1_scaleFactor1 = scale_factors["BN1"]["conv1x1_1"], bn1_scaleFactor2 = scale_factors["BN1"]["conv3x3"], bn1_scaleFactor3 = scale_factors["BN1"]["conv1x1_2"], bn1_scaleFactorAdd = scale_factors["BN1"]["skip_add"],
                    bn2_depthWiseStride = 1, bn2_depthWiseChannels = 72,  bn2_withSkip = True,  bn2_tensorOutC = 24, bn2_scaleFactor1 = scale_factors["BN2"]["conv1x1_1"], bn2_scaleFactor2 =scale_factors["BN2"]["conv3x3"], bn2_scaleFactor3 = scale_factors["BN2"]["conv1x1_2"],  bn2_scaleFactorAdd = scale_factors["BN2"]["skip_add"],
                    bn3_depthWiseStride = 2, bn3_depthWiseChannels = 72,  bn3_withSkip = False, bn3_tensorOutC = 40, bn3_scaleFactor1 = scale_factors["BN3"]["conv1x1_1"], bn3_scaleFactor2 =scale_factors["BN3"]["conv3x3"], bn3_scaleFactor3 = scale_factors["BN3"]["conv1x1_2"],  bn3_scaleFactorAdd = scale_factors["BN3"]["skip_add"],
                    bn4_depthWiseStride = 1, bn4_depthWiseChannels = 120, bn4_withSkip = True,  bn4_tensorOutC = 40, bn4_scaleFactor1 = scale_factors["BN4"]["conv1x1_1"], bn4_scaleFactor2 =scale_factors["BN4"]["conv3x3"], bn4_scaleFactor3 = scale_factors["BN4"]["conv1x1_2"],  bn4_scaleFactorAdd = scale_factors["BN4"]["skip_add"],
                    bn5_depthWiseStride = 1, bn5_depthWiseChannels = 120, bn5_withSkip = False,  bn5_tensorOutC = 40, bn5_scaleFactor1 = scale_factors["BN5"]["conv1x1_1"], bn5_scaleFactor2 =scale_factors["BN5"]["conv3x3"], bn5_scaleFactor3 = scale_factors["BN5"]["conv1x1_2"],  bn5_scaleFactorAdd = scale_factors["BN5"]["skip_add"],
                    bn6_depthWiseStride = 2, bn6_depthWiseChannels = 240, bn6_withSkip = False, bn6_tensorOutC = 80, bn6_scaleFactor1 = scale_factors["BN6"]["conv1x1_1"], bn6_scaleFactor2 =scale_factors["BN6"]["conv3x3"], bn6_scaleFactor3 = scale_factors["BN6"]["conv1x1_2"],  bn6_scaleFactorAdd = scale_factors["BN6"]["skip_add"],
                    bn7_depthWiseStride = 1, bn7_depthWiseChannels = 200, bn7_withSkip = True, bn7_tensorOutC = 80, bn7_scaleFactor1 = scale_factors["BN7"]["conv1x1_1"], bn7_scaleFactor2 =scale_factors["BN7"]["conv3x3"], bn7_scaleFactor3 = scale_factors["BN7"]["conv1x1_2"],  bn7_scaleFactorAdd = scale_factors["BN7"]["skip_add"],
                    bn8_depthWiseStride = 1, bn8_depthWiseChannels = 184, bn8_withSkip = False, bn8_tensorOutC = 80, bn8_scaleFactor1 = scale_factors["BN8"]["conv1x1_1"], bn8_scaleFactor2 =scale_factors["BN8"]["conv3x3"], bn8_scaleFactor3 = scale_factors["BN8"]["conv1x1_2"],  bn8_scaleFactorAdd = scale_factors["BN8"]["skip_add"],
                    bn9_depthWiseStride = 1, bn9_depthWiseChannels = 184, bn9_withSkip = True, bn9_tensorOutC = 80, bn9_scaleFactor1 = scale_factors["BN9"]["conv1x1_1"], bn9_scaleFactor2 =scale_factors["BN9"]["conv3x3"], bn9_scaleFactor3 = scale_factors["BN9"]["conv1x1_2"],  bn9_scaleFactorAdd = scale_factors["BN9"]["skip_add"],
                    enableTrace = False, trace_size = 16384, traceSizeInInt32s = 4096)
    
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)