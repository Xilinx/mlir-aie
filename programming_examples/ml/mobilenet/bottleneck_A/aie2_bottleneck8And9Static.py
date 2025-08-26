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

# from aie.dialects.memref import *
from aie.extras.dialects.ext import *
from aie.extras.dialects.ext.memref import view as memref_view
from aie.extras.dialects.ext import memref

import aie.utils.trace as trace_utils


class bottleneckAFused_8and9:
    def __init__(
        self,
        _bottleneckName,
        _computeTile,
        _L1Tile,
        _actIn,
        _weightsIn,
        _actOut,
        _rtpsIn,
        _objectArchive,
        _f1x1Relu8,
        _f3x3dwStrideRelu8,
        _f1x1Skip8,
        _f1x1Relu9,
        _f3x3dwStrideRelu9,
        _f1x1Skip9,
        _tensorLayer8_1Out_ty,
        _tensorLayer8_2Out_ty,
        _tensorLayer8_3Out_ty,
        _tensorLayer9_1Out_ty,
        _tensorLayer9_2Out_ty,
        _tensorInW=14,
        _tensorInH=14,
        _tensorInC=80,
        _bn8_depthWiseStride=1,
        bn8_depthWiseChannels=184,
        _bn9_depthWiseStride=1,
        _bn9_depthWiseChannels=184,
        _tensorOutC=80,
        _scaleLayer8_1=8,
        _scaleLayer8_2=8,
        _scaleLayer8_3=8,
        _skipScaleLayer8_3=0,
        _scaleLayer9_1=8,
        _scaleLayer9_2=8,
        _scaleLayer9_3=8,
        _skipScaleLayer9_3=0,
    ):

        self.bottleneckName = _bottleneckName
        self.computeTile = _computeTile
        self.L1Tile = _L1Tile

        self.actIn = _actIn
        # self.weightsIn = _weightsIn
        weightsAllLayers = _weightsIn
        self.actOut = _actOut
        self.rtpsIn = _rtpsIn

        self.objectArchiveName = _objectArchive
        self.f1x1Relu8 = _f1x1Relu8
        self.f3x3dwStrideRelu8 = _f3x3dwStrideRelu8
        self.f1x1Skip8 = _f1x1Skip8
        self.f1x1Relu9 = _f1x1Relu9
        self.f3x3dwStrideRelu9 = _f3x3dwStrideRelu9
        self.f1x1Skip9 = _f1x1Skip9

        self.tensorLayer8_1Out_ty = _tensorLayer8_1Out_ty
        self.tensorLayer8_2Out_ty = _tensorLayer8_2Out_ty
        self.tensorLayer8_3Out_ty = _tensorLayer8_3Out_ty
        self.tensorLayer9_1Out_ty = _tensorLayer9_1Out_ty
        self.tensorLayer9_2Out_ty = _tensorLayer9_2Out_ty

        self.tensorInW = _tensorInW
        self.tensorInH = _tensorInH
        self.tensorInC = _tensorInC
        self.bn8_depthWiseStride = _bn8_depthWiseStride
        self.bn8_depthWiseChannels = bn8_depthWiseChannels
        self.bn9_depthWiseStride = _bn9_depthWiseStride
        self.bn9_depthWiseChannels = _bn9_depthWiseChannels
        self.tensorOutC = _tensorOutC

        self.tensorOutW = (
            self.tensorInW // self.bn8_depthWiseStride
        ) // self.bn9_depthWiseStride
        self.tensorOutH = (
            self.tensorInH // self.bn8_depthWiseStride
        ) // self.bn9_depthWiseStride

        self.tensorL8_1InC = _tensorInC
        self.tensorL8_1OutC = bn8_depthWiseChannels

        self.tensorL8_2InC = self.tensorL8_1OutC
        self.tensorL8_2OutC = self.tensorL8_2InC

        self.tensorL8_3InC = self.tensorL8_2InC
        self.tensorL8_3OutC = self.tensorInC

        self.tensorL9_1InC = self.tensorL8_3OutC
        self.tensorL9_1OutC = self.bn9_depthWiseChannels

        self.tensorL9_2InC = self.tensorL9_1OutC
        self.tensorL9_2OutC = self.tensorL9_2InC

        self.tensorL9_3InC = self.tensorL9_2InC
        self.tensorL9_3OutC = self.tensorOutC

        # Intermediate
        self.of_act_bn8_1_2 = object_fifo(
            self.bottleneckName + "_" + "act_bn8_1_2",
            self.computeTile,
            self.computeTile,
            3,
            self.tensorLayer8_1Out_ty,
            disable_synchronization=True,
        )
        self.of_act_bn8_1_2.allocate(self.L1Tile)  # TODO
        self.of_act_bn8_2_3 = object_fifo(
            self.bottleneckName + "_" + "act_bn8_2_3",
            self.computeTile,
            self.computeTile,
            1,
            self.tensorLayer8_2Out_ty,
            disable_synchronization=True,
        )
        self.of_act_bn8_2_3.allocate(self.L1Tile)  # TODO
        self.of_act_bn8_bn9 = object_fifo(
            self.bottleneckName + "_" + "act_bn8_bn9",
            self.computeTile,
            self.computeTile,
            2,
            self.tensorLayer8_3Out_ty,
            disable_synchronization=True,
        )
        self.of_act_bn8_bn9.allocate(self.L1Tile)  # TODO
        self.of_act_bn9_1_2 = object_fifo(
            self.bottleneckName + "_" + "act_bn9_1_2",
            self.computeTile,
            self.computeTile,
            3,
            self.tensorLayer9_1Out_ty,
            disable_synchronization=True,
        )
        self.of_act_bn9_1_2.allocate(self.L1Tile)  # TODO
        self.of_act_bn9_2_3 = object_fifo(
            self.bottleneckName + "_" + "act_bn9_2_3",
            self.computeTile,
            self.computeTile,
            1,
            self.tensorLayer9_2Out_ty,
            disable_synchronization=True,
        )
        self.of_act_bn9_2_3.allocate(self.L1Tile)  # TODO

        # Compute tile
        @core(self.computeTile, self.objectArchiveName, False)
        def core_body():

            for _ in for_(1):  # for _ in for_(sys.maxsize):

                # acquire weights and rtps NOTE: needs to become once so outside for loop
                # weightsAllLayers = self.weightsIn.acquire(ObjectFifoPort.Consume, 1)
                weightsLayer8_1 = memref_view(
                    weightsAllLayers,
                    [self.tensorL8_1InC * self.tensorL8_1OutC],
                    shift=0,
                )
                weightsLayer8_2 = memref_view(
                    weightsAllLayers,
                    [3 * 3 * self.tensorL8_2OutC * 1],
                    shift=self.tensorL8_1InC * self.tensorL8_1OutC,
                )
                weightsLayer8_3 = memref_view(
                    weightsAllLayers,
                    [1 * 1 * self.tensorL8_3OutC * self.tensorL8_3InC],
                    shift=(
                        self.tensorL8_1InC * self.tensorL8_1OutC
                        + 3 * 3 * self.tensorL8_2OutC * 1
                    ),
                )
                weightsLayer9_1 = memref_view(
                    weightsAllLayers,
                    [1 * 1 * self.tensorL9_1OutC * self.tensorL9_1InC],
                    shift=(
                        self.tensorL8_1InC * self.tensorL8_1OutC
                        + 3 * 3 * self.tensorL8_2OutC * 1
                        + 1 * 1 * self.tensorL8_3OutC * self.tensorL8_3InC
                    ),
                )
                weightsLayer9_2 = memref_view(
                    weightsAllLayers,
                    [3 * 3 * self.tensorL9_2OutC * 1],
                    shift=(
                        self.tensorL8_1InC * self.tensorL8_1OutC
                        + 3 * 3 * self.tensorL8_2OutC * 1
                        + 1 * 1 * self.tensorL8_3OutC * self.tensorL8_3InC
                        + 1 * 1 * self.tensorL9_1OutC * self.tensorL9_1InC
                    ),
                )
                weightsLayer9_3 = memref_view(
                    weightsAllLayers,
                    [1 * 1 * self.tensorL9_3OutC * self.tensorL9_3InC],
                    shift=(
                        self.tensorL8_1InC * self.tensorL8_1OutC
                        + 3 * 3 * self.tensorL8_2OutC * 1
                        + 1 * 1 * self.tensorL8_3OutC * self.tensorL8_3InC
                        + 1 * 1 * self.tensorL9_1OutC * self.tensorL9_1InC
                        + 3 * 3 * self.tensorL9_2OutC * 1
                    ),
                )

                scaleLayer8_1 = _scaleLayer8_1  # bn8 scaleFactor1
                scaleLayer8_2 = _scaleLayer8_2  # bn8 scaleFactor2
                scaleLayer8_3 = _scaleLayer8_3  # bn8 scaleFactor3
                skipScaleLayer8_3 = _skipScaleLayer8_3  # bn8 scaleFactorAdd
                scaleLayer9_1 = _scaleLayer9_1
                scaleLayer9_2 = _scaleLayer9_2
                scaleLayer9_3 = _scaleLayer9_3  # bn9 scaleFactor3
                skipScaleLayer9_3 = _skipScaleLayer9_3  # bn9 scaleFactorAdd

                # scaleLayer8_1 = memref.load(self.rtpsIn, [0]) # bn8 scaleFactor1
                # scaleLayer8_2 = memref.load(self.rtpsIn, [1]) # bn8 scaleFactor2
                # scaleLayer8_3 = memref.load(self.rtpsIn, [2]) # bn8 scaleFactor3
                # skipScaleLayer8_3 = memref.load(self.rtpsIn, [3]) # bn8 scaleFactorAdd
                # scaleLayer9_1 = memref.load(self.rtpsIn, [4])
                # scaleLayer9_2 = memref.load(self.rtpsIn, [5])
                # scaleLayer9_3 = memref.load(self.rtpsIn, [6]) # bn9 scaleFactor3
                # skipScaleLayer9_3 = memref.load(self.rtpsIn, [7]) # bn9 scaleFactorAdd

                # pre-amble 0: rows 0, 1 in layer 0_1 1x1 conv; row 0 in layer 0_2 3x3 dw; row 0 in layer 0_3 1x1 conv; row 0 on layer 1_1 1x1 conv
                actInLayer8_1Rows = self.actIn.acquire(ObjectFifoPort.Consume, 2)
                actOutLayer8_1Rows = self.of_act_bn8_1_2.acquire(
                    ObjectFifoPort.Produce, 2
                )
                call(
                    self.f1x1Relu8,
                    [
                        actInLayer8_1Rows[0],
                        weightsLayer8_1,
                        actOutLayer8_1Rows[0],
                        self.tensorInW,
                        self.tensorL8_1InC,
                        self.tensorL8_1OutC,
                        scaleLayer8_1,
                    ],
                )
                call(
                    self.f1x1Relu8,
                    [
                        actInLayer8_1Rows[1],
                        weightsLayer8_1,
                        actOutLayer8_1Rows[1],
                        self.tensorInW,
                        self.tensorL8_1InC,
                        self.tensorL8_1OutC,
                        scaleLayer8_1,
                    ],
                )
                self.of_act_bn8_1_2.release(ObjectFifoPort.Produce, 2)

                actInLayer8_2Rows = self.of_act_bn8_1_2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                actOutLayer8_2Row = self.of_act_bn8_2_3.acquire(
                    ObjectFifoPort.Produce, 1
                )
                call(
                    self.f3x3dwStrideRelu8,
                    [
                        actInLayer8_2Rows[0],
                        actInLayer8_2Rows[0],
                        actInLayer8_2Rows[1],
                        weightsLayer8_2,
                        actOutLayer8_2Row,
                        self.tensorInW,
                        1,
                        self.tensorL8_2OutC,
                        3,
                        3,
                        0,
                        scaleLayer8_2,
                        0,
                    ],
                )
                self.of_act_bn8_2_3.release(ObjectFifoPort.Produce, 1)

                actInLayer8_3Row = self.of_act_bn8_2_3.acquire(
                    ObjectFifoPort.Consume, 1
                )
                actOutLayer8_3Row = self.of_act_bn8_bn9.acquire(
                    ObjectFifoPort.Produce, 1
                )
                call(
                    self.f1x1Skip8,
                    [
                        actInLayer8_3Row,
                        weightsLayer8_3,
                        actOutLayer8_3Row,
                        actInLayer8_1Rows[0],
                        self.tensorInW,
                        self.tensorL8_3InC,
                        self.tensorL8_3OutC,
                        scaleLayer8_3,
                        skipScaleLayer8_3,
                    ],
                )
                self.actIn.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn8_2_3.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn8_bn9.release(ObjectFifoPort.Produce, 1)

                actInLayer9_1Row = self.of_act_bn8_bn9.acquire(
                    ObjectFifoPort.Consume, 1
                )
                actOutLayer9_1Row = self.of_act_bn9_1_2.acquire(
                    ObjectFifoPort.Produce, 1
                )
                call(
                    self.f1x1Relu9,
                    [
                        actInLayer9_1Row,
                        weightsLayer9_1,
                        actOutLayer9_1Row,
                        self.tensorInW,
                        self.tensorL9_1InC,
                        self.tensorL9_1OutC,
                        scaleLayer9_1,
                    ],
                )
                self.of_act_bn9_1_2.release(ObjectFifoPort.Produce, 1)

                # pre-amble 1: rows 2 in layer 0_1 1x1 conv; row 1 in layer 0_2 3x3 dw; row 1 in layer 0_3 1x1 conv; row 1 on layer 1_1 1x1 conv; row 0 on layer 1_2 3x3 dw

                actInLayer8_1Rows = self.actIn.acquire(ObjectFifoPort.Consume, 2)
                actOutLayer8_1Row = self.of_act_bn8_1_2.acquire(
                    ObjectFifoPort.Produce, 1
                )
                call(
                    self.f1x1Relu8,
                    [
                        actInLayer8_1Rows[1],
                        weightsLayer8_1,
                        actOutLayer8_1Row,
                        self.tensorInW,
                        self.tensorL8_1InC,
                        self.tensorL8_1OutC,
                        scaleLayer8_1,
                    ],
                )
                self.of_act_bn8_1_2.release(ObjectFifoPort.Produce, 1)

                actInLayer8_2Rows = self.of_act_bn8_1_2.acquire(
                    ObjectFifoPort.Consume, 3
                )
                actOutLayer8_2Row = self.of_act_bn8_2_3.acquire(
                    ObjectFifoPort.Produce, 1
                )
                call(
                    self.f3x3dwStrideRelu8,
                    [
                        actInLayer8_2Rows[0],
                        actInLayer8_2Rows[1],
                        actInLayer8_2Rows[2],
                        weightsLayer8_2,
                        actOutLayer8_2Row,
                        self.tensorInW,
                        1,
                        self.tensorL8_2OutC,
                        3,
                        3,
                        1,
                        scaleLayer8_2,
                        0,
                    ],
                )
                self.of_act_bn8_1_2.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn8_2_3.release(ObjectFifoPort.Produce, 1)

                actInLayer8_3Row = self.of_act_bn8_2_3.acquire(
                    ObjectFifoPort.Consume, 1
                )
                actOutLayer8_3Row = self.of_act_bn8_bn9.acquire(
                    ObjectFifoPort.Produce, 1
                )
                call(
                    self.f1x1Skip8,
                    [
                        actInLayer8_3Row,
                        weightsLayer8_3,
                        actOutLayer8_3Row,
                        actInLayer8_1Rows[0],
                        self.tensorInW,
                        self.tensorL8_3InC,
                        self.tensorL8_3OutC,
                        scaleLayer8_3,
                        skipScaleLayer8_3,
                    ],
                )
                self.actIn.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn8_2_3.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn8_bn9.release(ObjectFifoPort.Produce, 1)

                actInLayer9_1Rows = self.of_act_bn8_bn9.acquire(
                    ObjectFifoPort.Consume, 2
                )
                actOutLayer9_1Row = self.of_act_bn9_1_2.acquire(
                    ObjectFifoPort.Produce, 1
                )
                call(
                    self.f1x1Relu9,
                    [
                        actInLayer9_1Rows[1],
                        weightsLayer9_1,
                        actOutLayer9_1Row,
                        self.tensorInW,
                        self.tensorL9_1InC,
                        self.tensorL9_1OutC,
                        scaleLayer9_1,
                    ],
                )
                self.of_act_bn9_1_2.release(ObjectFifoPort.Produce, 1)

                actInLayer9_2Rows = self.of_act_bn9_1_2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                actOutLayer9_2Row = self.of_act_bn9_2_3.acquire(
                    ObjectFifoPort.Produce, 1
                )
                call(
                    self.f3x3dwStrideRelu9,
                    [
                        actInLayer9_2Rows[0],
                        actInLayer9_2Rows[0],
                        actInLayer9_2Rows[1],
                        weightsLayer9_2,
                        actOutLayer9_2Row,
                        self.tensorInW,
                        1,
                        self.tensorL9_2OutC,
                        3,
                        3,
                        0,
                        scaleLayer9_2,
                        0,
                    ],
                )
                self.of_act_bn9_2_3.release(ObjectFifoPort.Produce, 1)

                actInLayer9_3Row = self.of_act_bn9_2_3.acquire(
                    ObjectFifoPort.Consume, 1
                )
                actOutLayer9_3Row = self.actOut.acquire(ObjectFifoPort.Produce, 1)
                call(
                    self.f1x1Skip9,
                    [
                        actInLayer9_3Row,
                        weightsLayer9_3,
                        actOutLayer9_3Row,
                        actInLayer9_1Rows[0],
                        self.tensorOutW,
                        self.tensorL9_3InC,
                        self.tensorL9_3OutC,
                        scaleLayer9_3,
                        skipScaleLayer9_3,
                    ],
                )
                self.of_act_bn9_2_3.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn8_bn9.release(ObjectFifoPort.Consume, 1)
                self.actOut.release(ObjectFifoPort.Produce, 1)

                # middle: layer 3 1x1 conv and layer 2 3x3 dw and layer 1 1x1 conv
                for _ in for_(self.tensorOutH - 3):

                    actInLayer8_1Rows = self.actIn.acquire(ObjectFifoPort.Consume, 2)
                    actOutLayer8_1Row = self.of_act_bn8_1_2.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    call(
                        self.f1x1Relu8,
                        [
                            actInLayer8_1Rows[1],
                            weightsLayer8_1,
                            actOutLayer8_1Row,
                            self.tensorInW,
                            self.tensorL8_1InC,
                            self.tensorL8_1OutC,
                            scaleLayer8_1,
                        ],
                    )
                    self.of_act_bn8_1_2.release(ObjectFifoPort.Produce, 1)

                    actInLayer8_2Rows = self.of_act_bn8_1_2.acquire(
                        ObjectFifoPort.Consume, 3
                    )
                    actOutLayer8_2Row = self.of_act_bn8_2_3.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    call(
                        self.f3x3dwStrideRelu8,
                        [
                            actInLayer8_2Rows[0],
                            actInLayer8_2Rows[1],
                            actInLayer8_2Rows[2],
                            weightsLayer8_2,
                            actOutLayer8_2Row,
                            self.tensorInW,
                            1,
                            self.tensorL8_2OutC,
                            3,
                            3,
                            1,
                            scaleLayer8_2,
                            0,
                        ],
                    )
                    self.of_act_bn8_1_2.release(ObjectFifoPort.Consume, 1)
                    self.of_act_bn8_2_3.release(ObjectFifoPort.Produce, 1)

                    actInLayer8_3Row = self.of_act_bn8_2_3.acquire(
                        ObjectFifoPort.Consume, 1
                    )
                    actOutLayer8_3Row = self.of_act_bn8_bn9.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    call(
                        self.f1x1Skip8,
                        [
                            actInLayer8_3Row,
                            weightsLayer8_3,
                            actOutLayer8_3Row,
                            actInLayer8_1Rows[0],
                            self.tensorInW,
                            self.tensorL8_3InC,
                            self.tensorL8_3OutC,
                            scaleLayer8_3,
                            skipScaleLayer8_3,
                        ],
                    )
                    self.actIn.release(ObjectFifoPort.Consume, 1)
                    self.of_act_bn8_2_3.release(ObjectFifoPort.Consume, 1)
                    self.of_act_bn8_bn9.release(ObjectFifoPort.Produce, 1)

                    actInLayer9_1Rows = self.of_act_bn8_bn9.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    actOutLayer9_1Row = self.of_act_bn9_1_2.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    call(
                        self.f1x1Relu9,
                        [
                            actInLayer9_1Rows[1],
                            weightsLayer9_1,
                            actOutLayer9_1Row,
                            self.tensorInW,
                            self.tensorL9_1InC,
                            self.tensorL9_1OutC,
                            scaleLayer9_1,
                        ],
                    )
                    self.of_act_bn9_1_2.release(ObjectFifoPort.Produce, 1)

                    actInLayer9_2Rows = self.of_act_bn9_1_2.acquire(
                        ObjectFifoPort.Consume, 3
                    )
                    actOutLayer9_2Row = self.of_act_bn9_2_3.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    call(
                        self.f3x3dwStrideRelu9,
                        [
                            actInLayer9_2Rows[0],
                            actInLayer9_2Rows[1],
                            actInLayer9_2Rows[2],
                            weightsLayer9_2,
                            actOutLayer9_2Row,
                            self.tensorInW,
                            1,
                            self.tensorL9_2OutC,
                            3,
                            3,
                            1,
                            scaleLayer9_2,
                            0,
                        ],
                    )
                    self.of_act_bn9_1_2.release(ObjectFifoPort.Consume, 1)
                    self.of_act_bn9_2_3.release(ObjectFifoPort.Produce, 1)

                    actInLayer9_3Row = self.of_act_bn9_2_3.acquire(
                        ObjectFifoPort.Consume, 1
                    )
                    actOutLayer9_3Row = self.actOut.acquire(ObjectFifoPort.Produce, 1)
                    call(
                        self.f1x1Skip9,
                        [
                            actInLayer9_3Row,
                            weightsLayer9_3,
                            actOutLayer9_3Row,
                            actInLayer9_1Rows[0],
                            self.tensorOutW,
                            self.tensorL9_3InC,
                            self.tensorL9_3OutC,
                            scaleLayer9_3,
                            skipScaleLayer9_3,
                        ],
                    )
                    self.of_act_bn9_2_3.release(ObjectFifoPort.Consume, 1)
                    self.of_act_bn8_bn9.release(ObjectFifoPort.Consume, 1)
                    self.actOut.release(ObjectFifoPort.Produce, 1)

                    yield_([])

                # last part
                # post-amble 0

                actInLayer8_2Rows = self.of_act_bn8_1_2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                actOutLayer8_2Row = self.of_act_bn8_2_3.acquire(
                    ObjectFifoPort.Produce, 1
                )
                call(
                    self.f3x3dwStrideRelu8,
                    [
                        actInLayer8_2Rows[0],
                        actInLayer8_2Rows[1],
                        actInLayer8_2Rows[1],
                        weightsLayer8_2,
                        actOutLayer8_2Row,
                        self.tensorInW,
                        1,
                        self.tensorL8_2OutC,
                        3,
                        3,
                        2,
                        scaleLayer8_2,
                        0,
                    ],
                )
                self.of_act_bn8_1_2.release(ObjectFifoPort.Consume, 2)
                self.of_act_bn8_2_3.release(ObjectFifoPort.Produce, 1)

                actInLayer8_1Row = self.actIn.acquire(ObjectFifoPort.Consume, 1)
                actInLayer8_3Row = self.of_act_bn8_2_3.acquire(
                    ObjectFifoPort.Consume, 1
                )
                actOutLayer8_3Row = self.of_act_bn8_bn9.acquire(
                    ObjectFifoPort.Produce, 1
                )
                call(
                    self.f1x1Skip8,
                    [
                        actInLayer8_3Row,
                        weightsLayer8_3,
                        actOutLayer8_3Row,
                        actInLayer8_1Row,
                        self.tensorInW,
                        self.tensorL8_3InC,
                        self.tensorL8_3OutC,
                        scaleLayer8_3,
                        skipScaleLayer8_3,
                    ],
                )
                self.actIn.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn8_2_3.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn8_bn9.release(ObjectFifoPort.Produce, 1)

                actInLayer9_1Rows = self.of_act_bn8_bn9.acquire(
                    ObjectFifoPort.Consume, 2
                )
                actOutLayer9_1Row = self.of_act_bn9_1_2.acquire(
                    ObjectFifoPort.Produce, 1
                )
                call(
                    self.f1x1Relu9,
                    [
                        actInLayer9_1Rows[1],
                        weightsLayer9_1,
                        actOutLayer9_1Row,
                        self.tensorInW,
                        self.tensorL9_1InC,
                        self.tensorL9_1OutC,
                        scaleLayer9_1,
                    ],
                )
                self.of_act_bn9_1_2.release(ObjectFifoPort.Produce, 1)

                actInLayer9_2Rows = self.of_act_bn9_1_2.acquire(
                    ObjectFifoPort.Consume, 3
                )
                actOutLayer9_2Row = self.of_act_bn9_2_3.acquire(
                    ObjectFifoPort.Produce, 1
                )
                call(
                    self.f3x3dwStrideRelu9,
                    [
                        actInLayer9_2Rows[0],
                        actInLayer9_2Rows[1],
                        actInLayer9_2Rows[2],
                        weightsLayer9_2,
                        actOutLayer9_2Row,
                        self.tensorInW,
                        1,
                        self.tensorL9_2OutC,
                        3,
                        3,
                        1,
                        scaleLayer9_2,
                        0,
                    ],
                )
                self.of_act_bn9_1_2.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn9_2_3.release(ObjectFifoPort.Produce, 1)

                actInLayer9_3Row = self.of_act_bn9_2_3.acquire(
                    ObjectFifoPort.Consume, 1
                )
                actOutLayer9_3Row = self.actOut.acquire(ObjectFifoPort.Produce, 1)
                call(
                    self.f1x1Skip9,
                    [
                        actInLayer9_3Row,
                        weightsLayer9_3,
                        actOutLayer9_3Row,
                        actInLayer9_1Rows[0],
                        self.tensorOutW,
                        self.tensorL9_3InC,
                        self.tensorL9_3OutC,
                        scaleLayer9_3,
                        skipScaleLayer9_3,
                    ],
                )
                self.of_act_bn9_2_3.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn8_bn9.release(ObjectFifoPort.Consume, 1)
                self.actOut.release(ObjectFifoPort.Produce, 1)

                # post-amble 1

                actInLayer9_2Rows = self.of_act_bn9_1_2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                actOutLayer9_2Row = self.of_act_bn9_2_3.acquire(
                    ObjectFifoPort.Produce, 1
                )
                call(
                    self.f3x3dwStrideRelu9,
                    [
                        actInLayer9_2Rows[0],
                        actInLayer9_2Rows[1],
                        actInLayer9_2Rows[1],
                        weightsLayer9_2,
                        actOutLayer9_2Row,
                        self.tensorInW,
                        1,
                        self.tensorL9_2OutC,
                        3,
                        3,
                        2,
                        scaleLayer9_2,
                        0,
                    ],
                )
                self.of_act_bn9_1_2.release(ObjectFifoPort.Consume, 2)
                self.of_act_bn9_2_3.release(ObjectFifoPort.Produce, 1)

                actInLayer9_1Row = self.of_act_bn8_bn9.acquire(
                    ObjectFifoPort.Consume, 1
                )
                actInLayer9_3Row = self.of_act_bn9_2_3.acquire(
                    ObjectFifoPort.Consume, 1
                )
                actOutLayer9_3Row = self.actOut.acquire(ObjectFifoPort.Produce, 1)
                call(
                    self.f1x1Skip9,
                    [
                        actInLayer9_3Row,
                        weightsLayer9_3,
                        actOutLayer9_3Row,
                        actInLayer9_1Row,
                        self.tensorOutW,
                        self.tensorL9_3InC,
                        self.tensorL9_3OutC,
                        scaleLayer9_3,
                        skipScaleLayer9_3,
                    ],
                )
                self.of_act_bn9_2_3.release(ObjectFifoPort.Consume, 1)
                self.of_act_bn8_bn9.release(ObjectFifoPort.Consume, 1)
                self.actOut.release(ObjectFifoPort.Produce, 1)

                # self.weightsIn.release(ObjectFifoPort.Consume, 1)
                yield_([])


def mobilenetV3Bottleneck8And9(
    tileRowIndex=2,
    tileColIndex=2,
    tensorInW=56,
    tensorInH=56,
    tensorInC=24,
    tensorOutC=40,
    bn8_depthWiseStride=1,
    bn8_depthWiseChannels=72,
    bn9_depthWiseStride=1,
    bn9_depthWiseChannels=72,
    scaleFactor8_1=8,
    scaleFactor8_2=8,
    scaleFactor8_3=9,
    scaleFactorAdd8=1,
    scaleFactor9_1=8,
    scaleFactor9_2=9,
    scaleFactor9_3=11,
    scaleFactorAdd9=1,
    enableTrace=False,
    trace_size=16384,
    traceSizeInInt32s=4096,
):

    tensorOutW = (tensorInW // bn8_depthWiseStride) // bn9_depthWiseStride
    tensorOutH = (tensorInH // bn8_depthWiseStride) // bn9_depthWiseStride

    tensorL8_1InC = tensorInC
    tensorL8_1OutC = bn8_depthWiseChannels

    tensorL8_2InC = tensorL8_1OutC
    tensorL8_2OutC = tensorL8_2InC

    tensorL8_3InC = tensorL8_2InC
    tensorL8_3OutC = tensorInC

    tensorL9_1InC = tensorL8_3OutC
    tensorL9_1OutC = bn9_depthWiseChannels

    tensorL9_2InC = tensorL9_1OutC
    tensorL9_2OutC = tensorL9_2InC

    tensorL9_3InC = tensorL9_2InC
    tensorL9_3OutC = tensorOutC

    @device(AIEDevice.npu2)
    def device_body():

        # define types
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int16_ty = IntegerType.get_signless(16)
        int32_ty = IntegerType.get_signless(32)

        tensorLayer8_1In_ty = MemRefType.get((tensorInW, 1, tensorL8_1InC), int8_ty)
        weightsLayer8_1_ty = MemRefType.get((tensorL8_1OutC * tensorL8_1InC,), int8_ty)
        tensorLayer8_1Out_ty = MemRefType.get((tensorInW, 1, tensorL8_1OutC), uint8_ty)

        tensorLayer8_2In_ty = MemRefType.get((tensorInW, 1, tensorL8_2InC), uint8_ty)
        weightsLayer8_2_ty = MemRefType.get((3 * 3 * tensorL8_2OutC * 1,), int8_ty)
        tensorLayer8_2Out_ty = MemRefType.get((tensorInW, 1, tensorL8_2OutC), uint8_ty)

        tensorLayer8_3In_ty = MemRefType.get((tensorInW, 1, tensorL8_3InC), uint8_ty)
        weightsLayer8_3_ty = MemRefType.get(
            (1 * 1 * tensorL8_3OutC * tensorL8_3InC,), int8_ty
        )
        tensorLayer8_3Out_ty = MemRefType.get((tensorInW, 1, tensorL8_3OutC), int8_ty)

        tensorLayer9_1In_ty = MemRefType.get((tensorInW, 1, tensorL9_1InC), int8_ty)
        weightsLayer9_1_ty = MemRefType.get(
            (1 * 1 * tensorL9_1OutC * tensorL9_1InC,), int8_ty
        )
        tensorLayer9_1Out_ty = MemRefType.get((tensorInW, 1, tensorL9_1OutC), uint8_ty)

        tensorLayer9_2In_ty = MemRefType.get((tensorInW, 1, tensorL9_2InC), uint8_ty)
        weightsLayer9_2_ty = MemRefType.get((3 * 3 * tensorL9_2OutC * 1,), int8_ty)
        tensorLayer9_2Out_ty = MemRefType.get((tensorOutW, 1, tensorL9_2OutC), uint8_ty)
        tensorLayer9_3In_ty = MemRefType.get((tensorOutW, 1, tensorL9_3InC), uint8_ty)
        weightsLayer9_3_ty = MemRefType.get(
            (1 * 1 * tensorL9_3OutC * tensorL9_3InC,), int8_ty
        )
        tensorLayer9_3Out_ty = MemRefType.get((tensorOutW, 1, tensorL9_3OutC), int8_ty)

        weightsAllLayers = (
            tensorL8_1OutC * tensorL8_1InC
            + 3 * 3 * tensorL8_2OutC * 1
            + 1 * 1 * tensorL8_3OutC * tensorL8_3InC
            + 1 * 1 * tensorL9_1OutC * tensorL9_1InC
            + 3 * 3 * tensorL9_2OutC * 1
            + 1 * 1 * tensorL9_3OutC * tensorL9_3InC
        )

        weightsAllLayers_ty = MemRefType.get(
            (
                tensorL8_1OutC * tensorL8_1InC
                + 3 * 3 * tensorL8_2OutC * 1
                + 1 * 1 * tensorL8_3OutC * tensorL8_3InC
                + 1 * 1 * tensorL9_1OutC * tensorL9_1InC
                + 3 * 3 * tensorL9_2OutC * 1
                + 1 * 1 * tensorL9_3OutC * tensorL9_3InC,
            ),
            int8_ty,
        )

        # AIE Core Function declarations
        bn8_conv2dk1_relu_i8_ui8 = external_func(
            "bn8_conv2dk1_relu_i8_ui8",
            inputs=[
                tensorLayer8_1In_ty,
                weightsLayer8_1_ty,
                tensorLayer8_1Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn8_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func(
            "bn8_conv2dk3_dw_stride1_relu_ui8_ui8",
            inputs=[
                tensorLayer8_2In_ty,
                tensorLayer8_2In_ty,
                tensorLayer8_2In_ty,
                weightsLayer8_2_ty,
                tensorLayer8_2Out_ty,
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
                tensorLayer8_3In_ty,
                weightsLayer8_3_ty,
                tensorLayer8_3Out_ty,
                tensorLayer8_1In_ty,
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
                tensorLayer9_1In_ty,
                weightsLayer9_1_ty,
                tensorLayer9_1Out_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn9_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func(
            "bn9_conv2dk3_dw_stride1_relu_ui8_ui8",
            inputs=[
                tensorLayer9_2In_ty,
                tensorLayer9_2In_ty,
                tensorLayer9_2In_ty,
                weightsLayer9_2_ty,
                tensorLayer9_2Out_ty,
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
                tensorLayer9_3In_ty,
                weightsLayer9_3_ty,
                tensorLayer9_3Out_ty,
                tensorLayer9_1In_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        # Tile declarations
        ShimTile = tile(tileColIndex, 0)
        # MemTile = tile(tileColIndex, 1)
        ComputeTile = tile(tileColIndex, tileRowIndex)
        ComputeTileForL1 = tile(tileColIndex - 1, tileRowIndex)

        # AIE-array data movement with object fifos

        # Input
        act_in_tmp = object_fifo(
            "act_in_tmp", ShimTile, ComputeTileForL1, 3, tensorLayer8_1In_ty
        )
        act_in = object_fifo(
            "act_in",
            ComputeTileForL1,
            ComputeTile,
            3,
            tensorLayer8_1In_ty,
            via_DMA=False,
        )
        # act_in.set_via_shared_mem(ObjectFifoPort.Produce)
        act_in.allocate(ComputeTileForL1)
        # object_fifo_link(act_in_tmp, act_in)

        # wts
        # wts_OF_L3L2 = object_fifo(
        #     "wts_OF_L3L2", ShimTile, ComputeTile, 1, weightsAllLayers_ty
        # )

        file_path = "weights/"
        wts_ary = np.fromfile(file_path + "bn8_9_chain.txt", sep=",", dtype=np.int8)

        wts_OF_L3L2 = buffer(
            ComputeTile,
            np.ndarray[(weightsAllLayers,), np.dtype[np.int8]],
            "wts_OF_L3L2",
            initial_value=wts_ary,
        )

        # Output
        act_out = object_fifo(
            "act_out", ComputeTile, ComputeTileForL1, 2, tensorLayer9_3Out_ty
        )
        # act_out.set_via_shared_mem(ObjectFifoPort.Consume)
        act_out.allocate(ComputeTileForL1)
        act_out_tmp = object_fifo(
            "act_out_tmp", ComputeTileForL1, [ShimTile], 2, tensorLayer9_3Out_ty
        )

        # Set up compute tiles
        rtpComputeTile = buffer(
            ComputeTile, np.ndarray[(16,), np.dtype[np.int32]], "rtp"
        )

        @core(ComputeTileForL1)
        def core_body():
            for _ in for_(tensorInH):
                elem_in = act_in_tmp.acquire(ObjectFifoPort.Consume, 1)
                elem_out = act_in.acquire(ObjectFifoPort.Produce, 1)
                for i in for_(tensorInW):
                    for j in for_(tensorInC):
                        v0 = memref.load(elem_in, [i, 1, j])
                        memref.store(v0, elem_out, [i, 1, j])
                        yield_([])

                    yield_([])

                act_in.release(ObjectFifoPort.Produce, 1)
                act_in_tmp.release(ObjectFifoPort.Consume, 1)

                yield_([])

        bottleneckAFused_8and9(
            "bn8_bn9",
            ComputeTile,
            ComputeTileForL1,
            act_in,
            wts_OF_L3L2,
            act_out,
            rtpComputeTile,
            "combined_bn_8_9.a",
            bn8_conv2dk1_relu_i8_ui8,
            bn8_conv2dk3_dw_stride1_relu_ui8_ui8,
            bn8_conv2dk1_skip_ui8_i8_i8,
            bn9_conv2dk1_relu_i8_ui8,
            bn9_conv2dk3_dw_stride1_relu_ui8_ui8,
            bn9_conv2dk1_skip_ui8_i8_i8,
            tensorLayer8_1Out_ty,
            tensorLayer8_2Out_ty,
            tensorLayer8_3Out_ty,
            tensorLayer9_1Out_ty,
            tensorLayer9_2Out_ty,
            tensorInW,
            tensorInH,
            tensorInC,
            bn8_depthWiseStride,
            bn8_depthWiseChannels,
            bn9_depthWiseStride,
            bn9_depthWiseChannels,
            tensorOutC,
            scaleFactor8_1,
            scaleFactor8_2,
            scaleFactor8_3,
            scaleFactorAdd8,
            scaleFactor9_1,
            scaleFactor9_2,
            scaleFactor9_3,
            scaleFactorAdd9,
        )

        # instruction stream generation
        activationsInSize32b = (tensorInW * tensorInH * tensorInC) // 4
        activationsOutSize32b = (tensorOutW * tensorOutH * tensorOutC) // 4
        # totalWeightsSize32b = (tensorL8_1InC* tensorL8_1OutC+3*3*tensorL8_2OutC*1 + 1*1*tensorL8_3InC*tensorL8_3OutC + 1*1*tensorL9_1InC*tensorL9_1OutC + 3*3*tensorL9_2OutC + 1*1*tensorL9_3InC*tensorL9_3OutC) // 4
        totalWeightsSize32b = (
            tensorL8_1InC * tensorL8_1OutC
            + 3 * 3 * tensorL8_2OutC * 1
            + 1 * 1 * tensorL8_3InC * tensorL8_3OutC
            + tensorL9_1InC * tensorL9_1OutC
            + 3 * 3 * tensorL9_2OutC
            + tensorL9_3InC * tensorL9_3OutC
        ) // 4
        activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
        weightsInL3_ty = MemRefType.get((totalWeightsSize32b,), int32_ty)
        activationsOutL3_ty = MemRefType.get((activationsOutSize32b,), int32_ty)

        @runtime_sequence(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
        def sequence(inputFromL3, weightsFromL3, outputToL3):
            NpuWriteRTPOp("rtp", index=0, value=scaleFactor8_1)
            NpuWriteRTPOp("rtp", index=1, value=scaleFactor8_2)
            NpuWriteRTPOp("rtp", index=2, value=scaleFactor8_3)
            NpuWriteRTPOp("rtp", index=3, value=scaleFactorAdd8)
            NpuWriteRTPOp("rtp", index=4, value=scaleFactor9_1)
            NpuWriteRTPOp("rtp", index=5, value=scaleFactor9_2)
            NpuWriteRTPOp("rtp", index=6, value=scaleFactor9_3)
            NpuWriteRTPOp("rtp", index=7, value=scaleFactorAdd9)

            npu_dma_memcpy_nd(
                metadata="act_in_tmp",
                bd_id=0,
                mem=inputFromL3,
                sizes=[1, 1, 1, activationsInSize32b],
            )
            npu_dma_memcpy_nd(
                metadata="act_out_tmp",
                bd_id=2,
                mem=outputToL3,
                sizes=[1, 1, 1, activationsOutSize32b],
            )
            # npu_dma_memcpy_nd(
            #     metadata="wts_OF_L3L2",
            #     bd_id=1,
            #     mem=weightsFromL3,
            #     sizes=[1, 1, 1, totalWeightsSize32b],
            # )
            npu_sync(column=0, row=0, direction=0, channel=0)
