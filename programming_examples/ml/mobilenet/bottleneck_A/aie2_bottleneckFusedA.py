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
from aie.extras.context import mlir_mod_ctx
#from aie.dialects.memref import *
from aie.extras.dialects.ext import *
from aie.extras.dialects.ext.memref import view as memref_view

import aie.utils.trace as trace_utils

class bottleneckAFused:
    def __init__(self, _bottleneckName,_computeTile,_actIn, _weightsIn, _actOut, _rtpsIn, 
                  _objectArchive,  _f3x3dwStride1Relu,_f1x1Skip,_f1x1Relu, _f3x3dwStride2Relu, _f1x1, 
                  _tensorLayer0_2Out_ty,_tensorLayer0_3Out_ty,_tensorLayer1_1Out_ty,_tensorLayer1_2Out_ty,
                  _tensorInW = 112, _tensorInH = 112, _tensorInC = 16, _depthWiseStride = 2, _depthWiseChannels = 64, _tensorOutC = 24):
        
        self.bottleneckName = _bottleneckName
        self.computeTile = _computeTile

        self.actIn = _actIn
        self.weightsIn = _weightsIn
        self.actOut = _actOut
        self.rtpsIn = _rtpsIn
        
        self.objectArchive = _objectArchive
        self.f3x3dwReluS1 = _f3x3dwStride1Relu
        self.f1x1Skip = _f1x1Skip
        
        self.f1x1Relu = _f1x1Relu       
        self.f1x1 = _f1x1
        self.f3x3dwReluS2 = _f3x3dwStride2Relu
        


        self.tensorLayer0_2Out_ty=_tensorLayer0_2Out_ty
        self.tensorLayer0_3Out_ty=_tensorLayer0_3Out_ty
        self.tensorLayer1_1Out_ty=_tensorLayer1_1Out_ty
        self.tensorLayer1_2Out_ty=_tensorLayer1_2Out_ty
        self.tensorInW = _tensorInW
        self.tensorInH = _tensorInH
        self.tensorInC = _tensorInC
        self.depthWiseStride = _depthWiseStride
        self.depthWiseChannels = _depthWiseChannels
        self.tensorOutC = _tensorOutC
 
        
        self.tensorOutH = self.tensorInH // self.depthWiseStride
        self.tensorOutW = self.tensorInW // self.depthWiseStride

        self.tensorL0_2InC = self.tensorInC 
        self.tensorL0_2OutC = self.tensorL0_2InC

        self.tensorL0_3InC = self.tensorL0_2InC
        self.tensorL0_3OutC = self.tensorL0_3InC

        self.tensorL1_1InC = self.tensorL0_3OutC
        self.tensorL1_1OutC = self.depthWiseChannels

        self.tensorL1_2InC = self.tensorL1_1OutC
        self.tensorL1_2OutC = self.tensorL1_2InC

        self.tensorL1_3InC = self.tensorL1_2InC
        self.tensorL1_3OutC = self.tensorOutC

      
        
        # Intermediate
        self.of_act_bn0_2_3 = object_fifo(self.bottleneckName+"_"+"act_bn0_2_3", self.computeTile, self.computeTile, 1, self.tensorLayer0_2Out_ty)
        self.of_act_bn0_bn1 = object_fifo(self.bottleneckName+"_"+"act_bn0_bn1", self.computeTile, self.computeTile, 1, self.tensorLayer0_3Out_ty)
        self.of_act_bn1_1_2 = object_fifo(self.bottleneckName+"_"+"act_bn1_1_2", self.computeTile, self.computeTile, 3, self.tensorLayer1_1Out_ty)
        self.of_act_bn1_2_3 = object_fifo(self.bottleneckName+"_"+"act_bn1_2_3", self.computeTile, self.computeTile, 1, self.tensorLayer1_2Out_ty)
        
        
        # Compute tile
        @core(self.computeTile, self.objectArchive)
        def core_body():

            for _ in for_(1): #for _ in for_(sys.maxsize):
                
                # acquire weights and rtps NOTE: needs to become once so outside for loop
                weightsAllLayers = self.weightsIn.acquire(ObjectFifoPort.Consume, 1)
            
                weightsLayer0_2 = memref_view(weightsAllLayers.output, [3 * 3 * self.tensorL0_2OutC * 1], shift=0)
                weightsLayer0_3 = memref_view(weightsAllLayers.output, [1 * 1 * self.tensorL0_3OutC * self.tensorL0_3InC], shift=(3 * 3 * self.tensorL0_2OutC * 1))
                weightsLayer1_1 = memref_view(weightsAllLayers.output, [1 * 1 * self.tensorL1_1OutC * self.tensorL1_1InC], shift=(3 * 3 * self.tensorL0_2OutC * 1 + 1 * 1 * self.tensorL0_3OutC * self.tensorL0_3InC))
                weightsLayer1_2 = memref_view(weightsAllLayers.output, [3 * 3 * self.tensorL1_2OutC * 1], shift=(3 * 3 * self.tensorL0_2OutC * 1 + 1 * 1 * self.tensorL0_3OutC * self.tensorL0_3InC + 1 * 1 * self.tensorL1_1OutC * self.tensorL1_1InC))
                weightsLayer1_3 = memref_view(weightsAllLayers.output, [1 * 1 * self.tensorL1_3OutC * self.tensorL1_3InC], shift=(3 * 3 * self.tensorL0_2OutC * 1 + 1 * 1 * self.tensorL0_3OutC * self.tensorL0_3InC + 1 * 1 * self.tensorL1_1OutC * self.tensorL1_1InC + 3 * 3 * self.tensorL1_2OutC * 1))
                scaleLayer0_2 = memref.load(self.rtpsIn, [0]) # bn0 scaleFactor2
                scaleLayer0_3 = memref.load(self.rtpsIn, [1]) # bn0 scaleFactor3
                skipScaleLayer0_3 = memref.load(self.rtpsIn, [2]) # bn0 scaleFactorAdd
                scaleLayer1_1 = memref.load(self.rtpsIn, [3]) # bn1 scaleFactor1
                scaleLayer1_2 = memref.load(self.rtpsIn, [4]) # bn1 scaleFactor2
                scaleLayer1_3 = memref.load(self.rtpsIn, [5]) # bn1 scaleFactor3

                # pre-amble 0: row 0 in layer 0_2 3x3 dw; row 0 in layer 0_3 1x1 conv; row 0 on layer 1_1 1x1 conv
                actInLayer0_2Rows = self.actIn.acquire(ObjectFifoPort.Consume, 2)
                actOutLayer0_2Row = self.of_act_bn0_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(self.f3x3dwReluS1, [actInLayer0_2Rows[0], actInLayer0_2Rows[0], actInLayer0_2Rows[1], weightsLayer0_2, actOutLayer0_2Row, self.tensorInW, 1, self.tensorL0_2OutC, 3, 3, 0, scaleLayer0_2, 0])
                self.of_act_bn0_2_3.release(ObjectFifoPort.Produce, 1)

                actInLayer0_3Row = self.of_act_bn0_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer0_3Row = self.of_act_bn0_bn1.acquire(ObjectFifoPort.Produce, 1)
                call(self.f1x1Skip, [actInLayer0_3Row, weightsLayer0_3, actOutLayer0_3Row, actInLayer0_2Rows[0], self.tensorInW, self.tensorL0_3InC, self.tensorL0_3OutC, scaleLayer0_3, skipScaleLayer0_3])
                self.of_act_bn0_2_3.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn0_bn1.release(ObjectFifoPort.Produce, 1)

                actInLayer1_1Row = self.of_act_bn0_bn1.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer1_1Row = self.of_act_bn1_1_2.acquire(ObjectFifoPort.Produce, 1)
                call(self.f1x1Relu, [actInLayer1_1Row, weightsLayer1_1, actOutLayer1_1Row, self.tensorInW, self.tensorL1_1InC, self.tensorL1_1OutC, scaleLayer1_1])
                self.of_act_bn0_bn1.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn1_1_2.release(ObjectFifoPort.Produce, 1)

                # pre-amble 1: row 1 in layer 0_2 3x3 dw; row 1 in layer 0_3 1x1 conv; row 1 on layer 1_1 1x1 conv; row 0 on layer 1_2 3x3 dw
                actInLayer0_2Rows = self.actIn.acquire(ObjectFifoPort.Consume, 3)
                actOutLayer0_2Row = self.of_act_bn0_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(self.f3x3dwReluS1, [actInLayer0_2Rows[0], actInLayer0_2Rows[1], actInLayer0_2Rows[2], weightsLayer0_2, actOutLayer0_2Row, self.tensorInW, 1, self.tensorL0_2OutC, 3, 3, 1, scaleLayer0_2, 0]) 
                self.of_act_bn0_2_3.release(ObjectFifoPort.Produce, 1)
                
                actInLayer0_3Row = self.of_act_bn0_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer0_3Row = self.of_act_bn0_bn1.acquire(ObjectFifoPort.Produce, 1)
                call(self.f1x1Skip, [actInLayer0_3Row, weightsLayer0_3, actOutLayer0_3Row, actInLayer0_2Rows[1], self.tensorInW, self.tensorL0_3InC, self.tensorL0_3OutC, scaleLayer0_3, skipScaleLayer0_3])
                self.actIn.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn0_2_3.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn0_bn1.release(ObjectFifoPort.Produce, 1)

                actInLayer1_1Row = self.of_act_bn0_bn1.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer1_1Row = self.of_act_bn1_1_2.acquire(ObjectFifoPort.Produce, 1)
                call(self.f1x1Relu, [actInLayer1_1Row, weightsLayer1_1, actOutLayer1_1Row, self.tensorInW, self.tensorL1_1InC, self.tensorL1_1OutC, scaleLayer1_1])
                self.of_act_bn0_bn1.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn1_1_2.release(ObjectFifoPort.Produce, 1)

                actInLayer1_2Rows = self.of_act_bn1_1_2.acquire(ObjectFifoPort.Consume, 2)
                actOutLayer1_2Row = self.of_act_bn1_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(self.f3x3dwReluS2, [actInLayer1_2Rows[0], actInLayer1_2Rows[0], actInLayer1_2Rows[1], weightsLayer1_2, actOutLayer1_2Row, self.tensorInW, 1, self.tensorL1_2OutC, 3, 3, 0, scaleLayer1_2, 0]) 
                self.of_act_bn1_1_2.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn1_2_3.release(ObjectFifoPort.Produce, 1)

                actInLayer1_3Row = self.of_act_bn1_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer1_3Row = self.actOut.acquire(ObjectFifoPort.Produce, 1)
                call(self.f1x1, [actInLayer1_3Row, weightsLayer1_3, actOutLayer1_3Row, self.tensorOutW, self.tensorL1_3InC, self.tensorL1_3OutC, scaleLayer1_3])
                self.of_act_bn1_2_3.release(ObjectFifoPort.Consume, 1)
                self.actOut.release(ObjectFifoPort.Produce, 1)
                
                # middle: layer 3 1x1 conv and layer 2 3x3 dw and layer 1 1x1 conv
                
                for _ in for_(self.tensorOutH  - 2):
                    for _ in for_(2):
                        actInLayer0_2Rows = self.actIn.acquire(ObjectFifoPort.Consume, 3)
                        actOutLayer0_2Row = self.of_act_bn0_2_3.acquire(ObjectFifoPort.Produce, 1)
                        call(self.f3x3dwReluS1, [actInLayer0_2Rows[0], actInLayer0_2Rows[1], actInLayer0_2Rows[2], weightsLayer0_2, actOutLayer0_2Row, self.tensorInW, 1, self.tensorL0_2OutC, 3, 3, 1, scaleLayer0_2, 0]) 
                        self.of_act_bn0_2_3.release(ObjectFifoPort.Produce, 1)
                
                        actInLayer0_3Row = self.of_act_bn0_2_3.acquire(ObjectFifoPort.Consume, 1)
                        actOutLayer0_3Row = self.of_act_bn0_bn1.acquire(ObjectFifoPort.Produce, 1)
                        call(self.f1x1Skip, [actInLayer0_3Row, weightsLayer0_3, actOutLayer0_3Row, actInLayer0_2Rows[1], self.tensorInW, self.tensorL0_3InC, self.tensorL0_3OutC, scaleLayer0_3, skipScaleLayer0_3])
                        self.actIn.release(ObjectFifoPort.Consume, 1)
                        self.of_act_bn0_2_3.release(ObjectFifoPort.Consume, 1)
                        self.of_act_bn0_bn1.release(ObjectFifoPort.Produce, 1)

                        actInLayer1_1Row = self.of_act_bn0_bn1.acquire(ObjectFifoPort.Consume, 1)
                        actOutLayer1_1Row = self.of_act_bn1_1_2.acquire(ObjectFifoPort.Produce, 1)
                        call(self.f1x1Relu, [actInLayer1_1Row, weightsLayer1_1, actOutLayer1_1Row, self.tensorInW, self.tensorL1_1InC, self.tensorL1_1OutC, scaleLayer1_1])
                        self.of_act_bn0_bn1.release(ObjectFifoPort.Consume, 1)
                        self.of_act_bn1_1_2.release(ObjectFifoPort.Produce, 1)

                        yield_([])

                    actInLayer1_2Rows = self.of_act_bn1_1_2.acquire(ObjectFifoPort.Consume, 3)
                    actOutLayer1_2Row = self.of_act_bn1_2_3.acquire(ObjectFifoPort.Produce, 1)
                    call(self.f3x3dwReluS2, [actInLayer1_2Rows[0], actInLayer1_2Rows[1], actInLayer1_2Rows[2], weightsLayer1_2, actOutLayer1_2Row, self.tensorInW, 1, self.tensorL1_2OutC, 3, 3, 1, scaleLayer1_2, 0]) 
                    self.of_act_bn1_1_2.release(ObjectFifoPort.Consume, 2)
                    self.of_act_bn1_2_3.release(ObjectFifoPort.Produce, 1)

                    actInLayer1_3Row = self.of_act_bn1_2_3.acquire(ObjectFifoPort.Consume, 1)
                    actOutLayer1_3Row = self.actOut.acquire(ObjectFifoPort.Produce, 1)
                    call(self.f1x1, [actInLayer1_3Row, weightsLayer1_3, actOutLayer1_3Row, self.tensorOutW, self.tensorL1_3InC, self.tensorL1_3OutC, scaleLayer1_3])
                    self.of_act_bn1_2_3.release(ObjectFifoPort.Consume, 1)
                    self.actOut.release(ObjectFifoPort.Produce, 1)
                    
                    yield_([])
                
                # last part

                actInLayer0_2Rows = self.actIn.acquire(ObjectFifoPort.Consume, 3)
                actOutLayer0_2Row = self.of_act_bn0_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(self.f3x3dwReluS1, [actInLayer0_2Rows[0], actInLayer0_2Rows[1], actInLayer0_2Rows[2], weightsLayer0_2, actOutLayer0_2Row, self.tensorInW, 1, self.tensorL0_2OutC, 3, 3, 1, scaleLayer0_2, 0]) 
                self.of_act_bn0_2_3.release(ObjectFifoPort.Produce, 1)
                
                actInLayer0_3Row = self.of_act_bn0_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer0_3Row = self.of_act_bn0_bn1.acquire(ObjectFifoPort.Produce, 1)
                call(self.f1x1Skip, [actInLayer0_3Row, weightsLayer0_3, actOutLayer0_3Row, actInLayer0_2Rows[1], self.tensorInW, self.tensorL0_3InC, self.tensorL0_3OutC, scaleLayer0_3, skipScaleLayer0_3])
                self.actIn.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn0_2_3.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn0_bn1.release(ObjectFifoPort.Produce, 1)

                actInLayer1_1Row = self.of_act_bn0_bn1.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer1_1Row = self.of_act_bn1_1_2.acquire(ObjectFifoPort.Produce, 1)
                call(self.f1x1Relu, [actInLayer1_1Row, weightsLayer1_1, actOutLayer1_1Row, self.tensorInW, self.tensorL1_1InC, self.tensorL1_1OutC, scaleLayer1_1])
                self.of_act_bn0_bn1.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn1_1_2.release(ObjectFifoPort.Produce, 1)
                
                actInLayer0_2Rows = self.actIn.acquire(ObjectFifoPort.Consume, 2)
                actOutLayer0_2Row = self.of_act_bn0_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(self.f3x3dwReluS1, [actInLayer0_2Rows[0], actInLayer0_2Rows[1], actInLayer0_2Rows[1], weightsLayer0_2, actOutLayer0_2Row, self.tensorInW, 1, self.tensorL0_2OutC, 3, 3, 2, scaleLayer0_2, 0]) 
                self.of_act_bn0_2_3.release(ObjectFifoPort.Produce, 1)

                actInLayer0_3Row = self.of_act_bn0_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer0_3Row = self.of_act_bn0_bn1.acquire(ObjectFifoPort.Produce, 1)
                call(self.f1x1Skip, [actInLayer0_3Row, weightsLayer0_3, actOutLayer0_3Row, actInLayer0_2Rows[1], self.tensorInW, self.tensorL0_3InC, self.tensorL0_3OutC, scaleLayer0_3, skipScaleLayer0_3])
                self.actIn.release(ObjectFifoPort.Consume, 2)
                self.of_act_bn0_2_3.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn0_bn1.release(ObjectFifoPort.Produce, 1)

                actInLayer1_1Row = self.of_act_bn0_bn1.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer1_1Row = self.of_act_bn1_1_2.acquire(ObjectFifoPort.Produce, 1)
                call(self.f1x1Relu, [actInLayer1_1Row, weightsLayer1_1, actOutLayer1_1Row, self.tensorInW, self.tensorL1_1InC, self.tensorL1_1OutC, scaleLayer1_1])
                self.of_act_bn0_bn1.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn1_1_2.release(ObjectFifoPort.Produce, 1)

                actInLayer1_2Rows = self.of_act_bn1_1_2.acquire(ObjectFifoPort.Consume, 3)
                actOutLayer1_2Row = self.of_act_bn1_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(self.f3x3dwReluS2, [actInLayer1_2Rows[0], actInLayer1_2Rows[1], actInLayer1_2Rows[2], weightsLayer1_2, actOutLayer1_2Row, self.tensorInW, 1, self.tensorL1_2OutC, 3, 3, 1, scaleLayer1_2, 0]) 
                self.of_act_bn1_1_2.release(ObjectFifoPort.Consume, 3)
                self.of_act_bn1_2_3.release(ObjectFifoPort.Produce, 1)

                actInLayer1_3Row = self.of_act_bn1_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer1_3Row = self.actOut.acquire(ObjectFifoPort.Produce, 1)
                call(self.f1x1, [actInLayer1_3Row, weightsLayer1_3, actOutLayer1_3Row, self.tensorOutW, self.tensorL1_3InC, self.tensorL1_3OutC, scaleLayer1_3])
                self.of_act_bn1_2_3.release(ObjectFifoPort.Consume, 1)
                self.actOut.release(ObjectFifoPort.Produce, 1)
                
                self.weightsIn.release(ObjectFifoPort.Consume, 1)
                yield_([])
