#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext.scf import _for as range_
from aie.extras.util import np_ndarray_type_get_shape

# tracing definitions
trace_sz_in_bytes = 8192
trace_sz_in_i32s = trace_sz_in_bytes // 4
enableTrace = False

# Define bottleneck layer sizes
tensorInW = 32
tensorInH = 32
tensorInCInit = 64
tensorInCRest = 4 * tensorInCInit
n_cols = 3
repeat = 2

activationsIn = tensorInW * tensorInH * tensorInCInit
acitivationsOut = tensorInW * tensorInH * tensorInCRest

totalWeights_init = (
    tensorInCInit * tensorInCInit
    + 3 * 3 * tensorInCInit * tensorInCInit
    + 2 * tensorInCInit * tensorInCRest
)

totalWeights_rest = (
    tensorInCInit * tensorInCRest
    + 3 * 3 * tensorInCInit * tensorInCInit
    + tensorInCInit * tensorInCRest
)

totalWeights_complete = totalWeights_init + repeat * totalWeights_rest


def resnet_conv_x():

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_3col)
        def deviceBody():

            # define types
            tensorLayer1In_ty_init = np.ndarray[
                (tensorInW, 1, tensorInCInit), np.dtype[np.int8]
            ]
            tensorLayer1In_ty_rest = np.ndarray[
                (tensorInW, 1, tensorInCRest), np.dtype[np.uint8]
            ]
            weightsLayer1_ty_init = np.ndarray[
                (tensorInCInit * tensorInCInit,), np.dtype[np.int8]
            ]
            weightsLayer1_ty_rest = np.ndarray[
                (tensorInCRest * tensorInCInit,), np.dtype[np.int8]
            ]

            tensorLayer1Out_ty = np.ndarray[
                (tensorInW, 1, tensorInCInit), np.dtype[np.uint8]
            ]

            tensorLayer2In_ty = np.ndarray[
                (
                    tensorInW,
                    1,
                    tensorInCInit,
                ),
                np.dtype[np.uint8],
            ]
            weightsLayer2_ty = np.ndarray[
                (3 * 3 * tensorInCInit * tensorInCInit,), np.dtype[np.int8]
            ]
            tensorLayer2Out_ty = np.ndarray[
                (tensorInW, 1, tensorInCInit // 2), np.dtype[np.uint8]
            ]

            tensorLayer3In_ty = np.ndarray[
                (tensorInW, 1, tensorInCInit // 2), np.dtype[np.uint8]
            ]
            weightsLayer3_ty_init = np.ndarray[
                (2 * tensorInCInit * tensorInCRest,), np.dtype[np.int8]
            ]
            weightsLayer3_ty_rest = np.ndarray[
                (tensorInCRest // 4 * tensorInCRest,), np.dtype[np.int8]
            ]

            tensorLayer3Out_ty = np.ndarray[
                (tensorInW, 1, tensorInCRest), np.dtype[np.uint8]
            ]

            allWeights_ty_init = np.ndarray[
                (
                    tensorInCInit * tensorInCInit
                    + 3 * 3 * tensorInCInit * tensorInCInit
                    + tensorInCInit * tensorInCRest
                    + tensorInCInit * tensorInCRest,
                ),
                np.dtype[np.int8],
            ]

            allWeights_ty_rest = np.ndarray[
                (
                    tensorInCRest * tensorInCInit
                    + 3 * 3 * tensorInCInit * tensorInCInit
                    + tensorInCInit * tensorInCRest,
                ),
                np.dtype[np.int8],
            ]

            activationsInL3_ty = np.ndarray[(activationsIn,), np.dtype[np.int8]]
            activationsOutL3_ty = np.ndarray[(acitivationsOut,), np.dtype[np.int8]]

            weightsInL3_ty_complete = np.ndarray[
                (totalWeights_complete,), np.dtype[np.int8]
            ]

            # kernel definitions
            conv2dk1_i8 = external_func(
                "conv2dk1_i8",
                inputs=[
                    tensorLayer1In_ty_init,
                    weightsLayer1_ty_init,
                    tensorLayer1Out_ty,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                ],
            )
            conv2dk3 = external_func(
                "conv2dk3_ui8",
                inputs=[
                    tensorLayer2In_ty,
                    tensorLayer2In_ty,
                    tensorLayer2In_ty,
                    weightsLayer2_ty,
                    tensorLayer2Out_ty,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                ],
            )
            conv2dk1_skip_init_i8 = external_func(
                "conv2dk1_skip_init_i8",
                inputs=[
                    tensorLayer3In_ty,
                    tensorLayer3In_ty,
                    weightsLayer3_ty_init,
                    tensorLayer3Out_ty,
                    tensorLayer1In_ty_init,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                ],
            )
            conv2dk1_ui8 = external_func(
                "conv2dk1_ui8",
                inputs=[
                    tensorLayer3Out_ty,
                    weightsLayer1_ty_rest,
                    tensorLayer1Out_ty,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                ],
            )

            conv2dk1_skip_ui8 = external_func(
                "conv2dk1_skip_ui8",
                inputs=[
                    tensorLayer3In_ty,
                    tensorLayer3In_ty,
                    weightsLayer3_ty_rest,
                    tensorLayer3Out_ty,
                    tensorLayer3Out_ty,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                ],
            )

            ShimTile00 = tile(0, 0)
            MemTile01 = tile(0, 1)
            ComputeTile02 = tile(0, 2)
            ComputeTile03 = tile(0, 3)
            ComputeTile04 = tile(0, 4)
            ComputeTile05 = tile(0, 5)

            ShimTile10 = tile(1, 0)
            MemTile11 = tile(1, 1)
            ComputeTile12 = tile(1, 2)
            ComputeTile13 = tile(1, 3)
            ComputeTile14 = tile(1, 4)
            ComputeTile15 = tile(1, 5)

            ShimTile20 = tile(2, 0)
            MemTile21 = tile(2, 1)
            ComputeTile22 = tile(2, 2)
            ComputeTile23 = tile(2, 3)
            ComputeTile24 = tile(2, 4)
            ComputeTile25 = tile(2, 5)

            shims = [ShimTile00, ShimTile10, ShimTile20]
            mems = [MemTile01, MemTile11, MemTile21]
            wts_sizes = [allWeights_ty_init, allWeights_ty_rest, allWeights_ty_rest]
            layer1_wts_sizes = [
                weightsLayer1_ty_init,
                weightsLayer1_ty_rest,
                weightsLayer1_ty_rest,
            ]
            laye1_act_sizes = [
                tensorLayer1In_ty_init,
                tensorLayer1In_ty_rest,
                tensorLayer1In_ty_rest,
            ]
            layer3_wts_sizes = [
                weightsLayer3_ty_init,
                weightsLayer3_ty_rest,
                weightsLayer3_ty_rest,
            ]

            cores = [
                [ComputeTile02, ComputeTile03, ComputeTile04, ComputeTile05],
                [ComputeTile15, ComputeTile14, ComputeTile13, ComputeTile12],
                [ComputeTile22, ComputeTile23, ComputeTile24, ComputeTile25],
            ]

            if enableTrace:
                flow(ComputeTile04, WireBundle.Trace, 0, ShimTile00, WireBundle.DMA, 1)

            # runtime parameters

            rtpComputeTile02 = Buffer(ComputeTile02, [16], np.int32, "rtpComputeTile02")
            rtpComputeTile03 = Buffer(ComputeTile03, [16], np.int32, "rtpComputeTile03")
            rtpComputeTile04 = Buffer(ComputeTile05, [16], np.int32, "rtpComputeTile04")
            rtpComputeTile05 = Buffer(ComputeTile04, [16], np.int32, "rtpComputeTile05")

            rtpComputeTile12 = Buffer(ComputeTile12, [16], np.int32, "rtpComputeTile12")
            rtpComputeTile13 = Buffer(ComputeTile13, [16], np.int32, "rtpComputeTile13")
            rtpComputeTile14 = Buffer(ComputeTile14, [16], np.int32, "rtpComputeTile14")
            rtpComputeTile15 = Buffer(ComputeTile15, [16], np.int32, "rtpComputeTile15")

            rtpComputeTile22 = Buffer(ComputeTile22, [16], np.int32, "rtpComputeTile22")
            rtpComputeTile23 = Buffer(ComputeTile23, [16], np.int32, "rtpComputeTile23")
            rtpComputeTile24 = Buffer(ComputeTile24, [16], np.int32, "rtpComputeTile24")
            rtpComputeTile25 = Buffer(ComputeTile25, [16], np.int32, "rtpComputeTile25")

            rtp = [
                [
                    rtpComputeTile02,
                    rtpComputeTile03,
                    rtpComputeTile04,
                    rtpComputeTile05,
                ],
                [
                    rtpComputeTile15,
                    rtpComputeTile14,
                    rtpComputeTile13,
                    rtpComputeTile12,
                ],
                [
                    rtpComputeTile22,
                    rtpComputeTile23,
                    rtpComputeTile24,
                    rtpComputeTile25,
                ],
            ]

            # set up data movement with OFs
            conv1_kernels = ["conv2dk1_i8.o", "conv2dk1_ui8.o", "conv2dk1_ui8.o"]
            conv1_kernels_call = [conv2dk1_i8, conv2dk1_ui8, conv2dk1_ui8]

            conv3_kernels = [
                "conv2dk1_skip_init.o",
                "conv2dk1_skip.o",
                "conv2dk1_skip.o",
            ]
            conv3_kernels_call = [
                conv2dk1_skip_init_i8,
                conv2dk1_skip_ui8,
                conv2dk1_skip_ui8,
            ]

            act1_fifo_names = ["act1_00_02_01", "act1_04_15_01", "act1_13_22_21"]
            act1_fifos = {}

            # input tensor (with broadcast for skip connection)
            act1_fifo_names = ["act1_00_02_01", "act1_04_15_11", "act1_13_22_21"]
            act1_fifos = {}

            skip_fifo_names = ["skip_0", "skip_1", "skip_2"]
            skip_fifos = {}

            act1_fifos[act1_fifo_names[0]] = object_fifo(
                act1_fifo_names[0],
                shims[0],
                [cores[0][0], mems[0]],
                [2, 2, 4],
                laye1_act_sizes[0],
            )
            skip_fifos[skip_fifo_names[0]] = object_fifo(
                skip_fifo_names[0], mems[0], cores[0][2], 2, laye1_act_sizes[0]
            )
            object_fifo_link(act1_fifo_names[0], skip_fifo_names[0])

            for i in range(1, repeat + 1):
                if i == 1:
                    act1_fifos[act1_fifo_names[i]] = object_fifo(
                        act1_fifo_names[i],
                        cores[i - 1][2],
                        [cores[i][0], mems[i - 1]],
                        [2, 2, 4],
                        laye1_act_sizes[i],
                    )
                    skip_fifos[skip_fifo_names[i]] = object_fifo(
                        skip_fifo_names[i],
                        mems[i - 1],
                        cores[i][2],
                        2,
                        laye1_act_sizes[i],
                    )
                    object_fifo_link(act1_fifo_names[i], skip_fifo_names[i])
                else:
                    act1_fifos[act1_fifo_names[i]] = object_fifo(
                        act1_fifo_names[i],
                        cores[i - 1][2],
                        [cores[i][0], mems[i]],
                        [2, 2, 4],
                        laye1_act_sizes[i],
                    )
                    skip_fifos[skip_fifo_names[i]] = object_fifo(
                        skip_fifo_names[i],
                        mems[i],
                        cores[i][2],
                        2,
                        laye1_act_sizes[i],
                    )
                    object_fifo_link(act1_fifo_names[i], skip_fifo_names[i])

            act2_fifo_names = ["act2_02_03_05", "act2_15_12_14", "act2_22_23_25"]
            act2_fifos = {}

            act3_fifo_names_1 = ["act3_03_04", "act3_14_13", "act3_23_24"]
            act3_fifo_1 = {}

            act3_fifo_names_2 = ["act3_05_04", "act3_12_13", "act3_25_24"]
            act3_fifo_2 = {}

            for i in range(n_cols):
                if i == 1:
                    # 1x1 -> 3x3
                    act2_fifos[act2_fifo_names[i]] = object_fifo(
                        act2_fifo_names[i],
                        cores[i][0],
                        [cores[i][3], cores[i][1]],
                        4,
                        tensorLayer1Out_ty,
                    )

                    # 3x3 -> 1x1
                    act3_fifo_1[act3_fifo_names_1[i]] = object_fifo(
                        act3_fifo_names_1[i],
                        cores[i][1],
                        cores[i][2],
                        2,
                        tensorLayer2Out_ty,
                    )
                    # 3x3 -> 1x1
                    act3_fifo_2[act3_fifo_names_2[i]] = object_fifo(
                        act3_fifo_names_2[i],
                        cores[i][3],
                        cores[i][2],
                        2,
                        tensorLayer2Out_ty,
                    )
                else:
                    # 1x1 -> 3x3
                    act2_fifos[act2_fifo_names[i]] = object_fifo(
                        act2_fifo_names[i],
                        cores[i][0],
                        [cores[i][1], cores[i][3]],
                        4,
                        tensorLayer1Out_ty,
                    )

                    # 3x3 -> 1x1
                    act3_fifo_1[act3_fifo_names_1[i]] = object_fifo(
                        act3_fifo_names_1[i],
                        cores[i][1],
                        cores[i][2],
                        2,
                        tensorLayer2Out_ty,
                    )
                    # 3x3 -> 1x1
                    act3_fifo_2[act3_fifo_names_2[i]] = object_fifo(
                        act3_fifo_names_2[i],
                        cores[i][3],
                        cores[i][2],
                        2,
                        tensorLayer2Out_ty,
                    )
            wts_fifo_names = ["wts_0_L3L2", "wts_1_L3L2", "wts_2_L3L2"]
            wts_fifos = {}
            wts_sub_fifo_names = [
                ["wts_buf_00", "wts_buf_01", "wts_buf_02"],
                ["wts_buf_10", "wts_buf_11", "wts_buf_12"],
                ["wts_buf_20", "wts_buf_21", "wts_buf_22"],
            ]
            wts_sub_fifos = {}

            for i in range(n_cols):

                wts_fifos[wts_fifo_names[i]] = object_fifo(
                    wts_fifo_names[i], shims[i], mems[i], 1, wts_sizes[i]
                )
                wts_sub_fifos[wts_sub_fifo_names[i][0]] = object_fifo(
                    wts_sub_fifo_names[i][0],
                    mems[i],
                    cores[i][0],
                    1,
                    layer1_wts_sizes[i],
                )
                if i == 1:
                    wts_sub_fifos[wts_sub_fifo_names[i][1]] = object_fifo(
                        wts_sub_fifo_names[i][1],
                        mems[i],
                        [cores[i][3], cores[i][1]],
                        1,
                        weightsLayer2_ty,
                    )

                else:
                    wts_sub_fifos[wts_sub_fifo_names[i][1]] = object_fifo(
                        wts_sub_fifo_names[i][1],
                        mems[i],
                        [cores[i][1], cores[i][3]],
                        1,
                        weightsLayer2_ty,
                    )
                wts_sub_fifos[wts_sub_fifo_names[i][2]] = object_fifo(
                    wts_sub_fifo_names[i][2],
                    mems[i],
                    cores[i][2],
                    1,
                    layer3_wts_sizes[i],
                )
                object_fifo_link(
                    wts_fifo_names[i],
                    [
                        wts_sub_fifo_names[i][0],
                        wts_sub_fifo_names[i][1],
                        wts_sub_fifo_names[i][2],
                    ],
                    [],
                    [
                        0,
                        np.prod(np_ndarray_type_get_shape(layer1_wts_sizes[i])),
                        np.prod(np_ndarray_type_get_shape(layer1_wts_sizes[i]))
                        + np.prod(np_ndarray_type_get_shape(weightsLayer2_ty)),
                    ],
                )
            # output tensor
            outOFL2L3 = object_fifo(
                "outOFL2L3", cores[2][2], shims[1], 2, tensorLayer3Out_ty
            )
            conv3_out_fifo = [
                act1_fifos[act1_fifo_names[1]],
                act1_fifos[act1_fifo_names[2]],
                outOFL2L3,
            ]
            conv3_out_fifo_names = ["act1_04_15_11", "act1_13_22_21", "outOFL2L3"]
            # # 1x1 conv2d
            for i in range(n_cols):

                @core(cores[i][0], conv1_kernels[i])
                def core_body():
                    for _ in range_(sys.maxsize):

                        # acquire weights once
                        element0Weights = wts_sub_fifos[
                            wts_sub_fifo_names[i][0]
                        ].acquire(ObjectFifoPort.Consume, 1)
                        scale = rtp[i][0][0]
                        for _ in range_(tensorInH):
                            element0ActivactionsIn = act1_fifos[
                                act1_fifo_names[i]
                            ].acquire(ObjectFifoPort.Consume, 1)
                            element0ActivactionsOut = act2_fifos[
                                act2_fifo_names[i]
                            ].acquire(ObjectFifoPort.Produce, 1)
                            if i == 0:
                                conv1_kernels_call[i](
                                    element0ActivactionsIn,
                                    element0Weights,
                                    element0ActivactionsOut,
                                    tensorInW,
                                    tensorInCInit,
                                    tensorInCInit,
                                    scale,
                                )
                            else:
                                conv1_kernels_call[i](
                                    element0ActivactionsIn,
                                    element0Weights,
                                    element0ActivactionsOut,
                                    tensorInW,
                                    tensorInCRest,
                                    tensorInCInit,
                                    scale,
                                )

                            objectfifo_release(
                                ObjectFifoPort.Consume, act1_fifo_names[i], 1
                            )

                            objectfifo_release(
                                ObjectFifoPort.Produce, act2_fifo_names[i], 1
                            )
                        objectfifo_release(
                            ObjectFifoPort.Consume, wts_sub_fifo_names[i][0], 1
                        )

            # 3x3 conv2d OFM 0-31
            for i in range(n_cols):

                @core(cores[i][1], "conv2dk3.o")
                def core_body():
                    scale = 1
                    for _ in range_(sys.maxsize):

                        # acquire weights and rtps once
                        element0Weights = wts_sub_fifos[
                            wts_sub_fifo_names[i][1]
                        ].acquire(ObjectFifoPort.Consume, 1)
                        # scale = memref.load(rtpComputeTile03, 0)

                        # pre-amble: top row
                        elementActivactionsIn = act2_fifos[act2_fifo_names[i]].acquire(
                            ObjectFifoPort.Consume, 2
                        )
                        element0ActivactionsOut = act3_fifo_1[
                            act3_fifo_names_1[i]
                        ].acquire(ObjectFifoPort.Produce, 1)
                        conv2dk3(
                            elementActivactionsIn[0],
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            element0Weights,
                            element0ActivactionsOut,
                            tensorInW,
                            tensorInCInit,
                            tensorInCInit // 2,
                            3,
                            3,
                            0,
                            scale,
                            0,
                        )
                        objectfifo_release(
                            ObjectFifoPort.Produce, act3_fifo_names_1[i], 1
                        )

                        # middle
                        for _ in range_(tensorInH - 2):
                            elementActivactionsIn = act2_fifos[
                                act2_fifo_names[i]
                            ].acquire(ObjectFifoPort.Consume, 3)
                            element0ActivactionsOut = act3_fifo_1[
                                act3_fifo_names_1[i]
                            ].acquire(ObjectFifoPort.Produce, 1)
                            conv2dk3(
                                elementActivactionsIn[0],
                                elementActivactionsIn[1],
                                elementActivactionsIn[2],
                                element0Weights,
                                element0ActivactionsOut,
                                tensorInW,
                                tensorInCInit,
                                tensorInCInit // 2,
                                3,
                                3,
                                1,
                                scale,
                                0,
                            )

                            objectfifo_release(
                                ObjectFifoPort.Consume, act2_fifo_names[i], 1
                            )
                            objectfifo_release(
                                ObjectFifoPort.Produce, act3_fifo_names_1[i], 1
                            )

                        # last part
                        elementActivactionsIn = act2_fifos[act2_fifo_names[i]].acquire(
                            ObjectFifoPort.Consume, 2
                        )
                        element0ActivactionsOut = act3_fifo_1[
                            act3_fifo_names_1[i]
                        ].acquire(ObjectFifoPort.Produce, 1)
                        conv2dk3(
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[1],
                            element0Weights,
                            element0ActivactionsOut,
                            tensorInW,
                            tensorInCInit,
                            tensorInCInit // 2,
                            3,
                            3,
                            2,
                            scale,
                            0,
                        )

                        objectfifo_release(
                            ObjectFifoPort.Consume, act2_fifo_names[i], 2
                        )
                        objectfifo_release(
                            ObjectFifoPort.Produce, act3_fifo_names_1[i], 1
                        )

                        objectfifo_release(
                            ObjectFifoPort.Consume, wts_sub_fifo_names[i][1], 1
                        )

            # 3x3 conv2d OFM 32-63

            for i in range(n_cols):

                @core(cores[i][3], "conv2dk3.o")
                def core_body():
                    scale = 1
                    for _ in range_(sys.maxsize):

                        # acquire weights and rtps once
                        element0Weights = wts_sub_fifos[
                            wts_sub_fifo_names[i][1]
                        ].acquire(ObjectFifoPort.Consume, 1)
                        # scale = memref.load(rtpComputeTile05, 0)

                        # pre-amble: top row
                        elementActivactionsIn = act2_fifos[act2_fifo_names[i]].acquire(
                            ObjectFifoPort.Consume, 2
                        )
                        element0ActivactionsOut = act3_fifo_2[
                            act3_fifo_names_2[i]
                        ].acquire(ObjectFifoPort.Produce, 1)
                        conv2dk3(
                            elementActivactionsIn[0],
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            element0Weights,
                            element0ActivactionsOut,
                            tensorInW,
                            tensorInCInit,
                            tensorInCInit // 2,
                            3,
                            3,
                            0,
                            scale,
                            tensorInCInit // 2,
                        )

                        objectfifo_release(
                            ObjectFifoPort.Produce, act3_fifo_names_2[i], 1
                        )

                        # middle
                        for _ in range_(tensorInH - 2):
                            elementActivactionsIn = act2_fifos[
                                act2_fifo_names[i]
                            ].acquire(ObjectFifoPort.Consume, 3)
                            element0ActivactionsOut = act3_fifo_2[
                                act3_fifo_names_2[i]
                            ].acquire(ObjectFifoPort.Produce, 1)
                            conv2dk3(
                                elementActivactionsIn[0],
                                elementActivactionsIn[1],
                                elementActivactionsIn[2],
                                element0Weights,
                                element0ActivactionsOut,
                                tensorInW,
                                tensorInCInit,
                                tensorInCInit // 2,
                                3,
                                3,
                                1,
                                scale,
                                tensorInCInit // 2,
                            )

                            objectfifo_release(
                                ObjectFifoPort.Consume, act2_fifo_names[i], 1
                            )
                            objectfifo_release(
                                ObjectFifoPort.Produce, act3_fifo_names_2[i], 1
                            )

                        # last part
                        elementActivactionsIn = act2_fifos[act2_fifo_names[i]].acquire(
                            ObjectFifoPort.Consume, 2
                        )
                        element0ActivactionsOut = act3_fifo_2[
                            act3_fifo_names_2[i]
                        ].acquire(ObjectFifoPort.Produce, 1)
                        conv2dk3(
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[1],
                            element0Weights,
                            element0ActivactionsOut,
                            tensorInW,
                            tensorInCInit,
                            tensorInCInit // 2,
                            3,
                            3,
                            2,
                            scale,
                            tensorInCInit // 2,
                        )
                        objectfifo_release(
                            ObjectFifoPort.Consume, act2_fifo_names[i], 2
                        )
                        objectfifo_release(
                            ObjectFifoPort.Produce, act3_fifo_names_2[i], 1
                        )
                        objectfifo_release(
                            ObjectFifoPort.Consume, wts_sub_fifo_names[i][1], 1
                        )

            # # 1x1 conv2d and add skip
            for i in range(n_cols):

                @core(cores[i][2], conv3_kernels[i])
                def core_body():
                    for _ in range_(sys.maxsize):

                        # acquire weights and rtps once
                        element0Weights = wts_sub_fifos[
                            wts_sub_fifo_names[i][2]
                        ].acquire(ObjectFifoPort.Consume, 1)
                        if i == 0:
                            scale = rtp[0][3][0]
                            skipScale = rtp[0][3][1]
                            skipConvScale = rtp[0][3][2]
                        else:
                            scale = rtp[i][2][0]
                            skipScale = rtp[i][2][1]

                        for _ in range_(tensorInH):
                            element0ActivactionsIn = act3_fifo_1[
                                act3_fifo_names_1[i]
                            ].acquire(ObjectFifoPort.Consume, 1)
                            element1ActivactionsIn = act3_fifo_2[
                                act3_fifo_names_2[i]
                            ].acquire(ObjectFifoPort.Consume, 1)

                            elementActivactionsOut = conv3_out_fifo[i].acquire(
                                ObjectFifoPort.Produce, 1
                            )
                            elementSkipsIn = skip_fifos[skip_fifo_names[i]].acquire(
                                ObjectFifoPort.Consume, 1
                            )
                            if i == 0:
                                conv3_kernels_call[0](
                                    element0ActivactionsIn,
                                    element1ActivactionsIn,
                                    element0Weights,
                                    elementActivactionsOut,
                                    elementSkipsIn,
                                    tensorInW,
                                    tensorInCInit,
                                    tensorInCRest,
                                    tensorInCInit,
                                    scale,
                                    skipScale,
                                    skipConvScale,
                                )
                            else:
                                conv3_kernels_call[i](
                                    element0ActivactionsIn,
                                    element1ActivactionsIn,
                                    element0Weights,
                                    elementActivactionsOut,
                                    elementSkipsIn,
                                    tensorInW,
                                    tensorInCInit,
                                    tensorInCRest,
                                    scale,
                                    skipScale,
                                )
                            objectfifo_release(
                                ObjectFifoPort.Consume, act3_fifo_names_1[i], 1
                            )
                            objectfifo_release(
                                ObjectFifoPort.Consume, act3_fifo_names_2[i], 1
                            )
                            objectfifo_release(
                                ObjectFifoPort.Produce, conv3_out_fifo_names[i], 1
                            )

                            objectfifo_release(
                                ObjectFifoPort.Consume, skip_fifo_names[i], 1
                            )
                        objectfifo_release(
                            ObjectFifoPort.Consume, wts_sub_fifo_names[i][2], 1
                        )

            # instruction stream generation
            @runtime_sequence(
                activationsInL3_ty, weightsInL3_ty_complete, activationsOutL3_ty
            )
            def sequence(inputFromL3, weightsFromL3, outputToL3):

                NpuWriteRTPOp("rtpComputeTile02", index=0, value=1)
                NpuWriteRTPOp("rtpComputeTile03", index=0, value=1)
                NpuWriteRTPOp("rtpComputeTile04", index=0, value=1)
                NpuWriteRTPOp("rtpComputeTile05", index=0, value=1)
                NpuWriteRTPOp("rtpComputeTile05", index=1, value=0)
                NpuWriteRTPOp("rtpComputeTile05", index=2, value=1)

                NpuWriteRTPOp("rtpComputeTile15", index=0, value=1)
                NpuWriteRTPOp("rtpComputeTile14", index=0, value=1)
                NpuWriteRTPOp("rtpComputeTile12", index=0, value=1)
                NpuWriteRTPOp("rtpComputeTile13", index=0, value=1)
                NpuWriteRTPOp("rtpComputeTile13", index=1, value=0)

                NpuWriteRTPOp("rtpComputeTile22", index=0, value=1)
                NpuWriteRTPOp("rtpComputeTile23", index=0, value=1)
                NpuWriteRTPOp("rtpComputeTile25", index=0, value=1)
                NpuWriteRTPOp("rtpComputeTile24", index=0, value=1)
                NpuWriteRTPOp("rtpComputeTile24", index=1, value=0)

                npu_dma_memcpy_nd(
                    metadata="act1_00_02_01",
                    bd_id=0,
                    mem=inputFromL3,
                    sizes=[1, 1, 1, activationsIn],
                )
                npu_dma_memcpy_nd(
                    metadata="wts_0_L3L2",
                    bd_id=1,
                    mem=weightsFromL3,
                    sizes=[1, 1, 1, totalWeights_init],
                )

                npu_dma_memcpy_nd(
                    metadata="wts_1_L3L2",
                    bd_id=1,
                    mem=weightsFromL3,
                    offsets=[0, 0, 0, totalWeights_init],
                    sizes=[1, 1, 1, totalWeights_rest],
                )

                npu_dma_memcpy_nd(
                    metadata="wts_2_L3L2",
                    bd_id=1,
                    mem=weightsFromL3,
                    offsets=[
                        0,
                        0,
                        0,
                        totalWeights_init + totalWeights_rest,
                    ],
                    sizes=[1, 1, 1, totalWeights_rest],
                )
                npu_dma_memcpy_nd(
                    metadata="outOFL2L3",
                    bd_id=2,
                    mem=outputToL3,
                    sizes=[1, 1, 1, acitivationsOut],
                )
                # outOFL2L3 will only complete after inputs complete, so we just wait on outOFL2L3 instead of all
                dma_wait("outOFL2L3")

    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


resnet_conv_x()
