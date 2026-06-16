#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
"""ResNet conv2_x layers — IRON + ``@iron.jit``, kernel-library backed."""

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.helpers.taplib import TensorAccessPattern


@iron.jit
def resnet_conv2_x(
    activations_in: In,
    weights_in: In,
    activations_out: Out,
    *,
    tensorInW: CompileTime[int] = 32,
    tensorInH: CompileTime[int] = 32,
    tensorInCInit: CompileTime[int] = 64,
    repeat: CompileTime[int] = 2,
):
    tensorInCRest = 4 * tensorInCInit
    n_cols = repeat + 1

    activationsIn = tensorInW * tensorInH * tensorInCInit
    acitivationsOut = tensorInW * tensorInH * tensorInCRest

    layer1_wts_init_size = tensorInCInit * tensorInCInit
    layer1_wts_rest_size = tensorInCRest * tensorInCInit
    layer2_wts_size = 3 * 3 * tensorInCInit * tensorInCInit
    layer3_wts_init_size = 2 * tensorInCInit * tensorInCRest
    layer3_wts_rest_size = (tensorInCRest // 4) * tensorInCRest

    totalWeights_init = layer1_wts_init_size + layer2_wts_size + layer3_wts_init_size
    totalWeights_rest = layer1_wts_rest_size + layer2_wts_size + layer3_wts_rest_size
    totalWeights_complete = totalWeights_init + repeat * totalWeights_rest

    # All FIFO element types are 1D — matches the kernel-library convention
    # (``conv*`` factories declare flat ``(W * C,)`` buffers).
    tensorLayer1In_ty_init = np.ndarray[(tensorInW * tensorInCInit,), np.dtype[np.int8]]
    tensorLayer1In_ty_rest = np.ndarray[
        (tensorInW * tensorInCRest,), np.dtype[np.uint8]
    ]
    weightsLayer1_ty_init = np.ndarray[(layer1_wts_init_size,), np.dtype[np.int8]]
    weightsLayer1_ty_rest = np.ndarray[(layer1_wts_rest_size,), np.dtype[np.int8]]

    tensorLayer1Out_ty = np.ndarray[(tensorInW * tensorInCInit,), np.dtype[np.uint8]]
    weightsLayer2_ty = np.ndarray[(layer2_wts_size,), np.dtype[np.int8]]
    tensorLayer2Out_ty = np.ndarray[
        (tensorInW * (tensorInCInit // 2),), np.dtype[np.uint8]
    ]

    weightsLayer3_ty_init = np.ndarray[(layer3_wts_init_size,), np.dtype[np.int8]]
    weightsLayer3_ty_rest = np.ndarray[(layer3_wts_rest_size,), np.dtype[np.int8]]
    tensorLayer3Out_ty = np.ndarray[(tensorInW * tensorInCRest,), np.dtype[np.uint8]]

    allWeights_ty_init = np.ndarray[
        (
            layer1_wts_init_size
            + layer2_wts_size
            + tensorInCInit * tensorInCRest
            + tensorInCInit * tensorInCRest,
        ),
        np.dtype[np.int8],
    ]
    allWeights_ty_rest = np.ndarray[
        (layer1_wts_rest_size + layer2_wts_size + tensorInCInit * tensorInCRest,),
        np.dtype[np.int8],
    ]

    activationsInL3_ty = np.ndarray[(activationsIn,), np.dtype[np.int8]]
    activationsOutL3_ty = np.ndarray[(acitivationsOut,), np.dtype[np.int8]]
    weightsInL3_ty_complete = np.ndarray[(totalWeights_complete,), np.dtype[np.int8]]

    wts_sizes = [allWeights_ty_init, allWeights_ty_rest, allWeights_ty_rest]
    layer1_wts_sizes = [
        weightsLayer1_ty_init,
        weightsLayer1_ty_rest,
        weightsLayer1_ty_rest,
    ]
    layer1_wts_byte_counts = [
        layer1_wts_init_size,
        layer1_wts_rest_size,
        layer1_wts_rest_size,
    ]
    layer1_act_sizes = [
        tensorLayer1In_ty_init,
        tensorLayer1In_ty_rest,
        tensorLayer1In_ty_rest,
    ]
    layer3_wts_sizes = [
        weightsLayer3_ty_init,
        weightsLayer3_ty_rest,
        weightsLayer3_ty_rest,
    ]

    # ------------------------------------------------------------------
    # Kernel-library calls.  The factories embed source paths +
    # -D{INT8,UINT8}_ACT flags so the JIT compiles them on demand.
    # ------------------------------------------------------------------
    # First-column 1x1: int8 in / uint8 out (with ReLU) — conv2dk1.cc + INT8_ACT
    conv2dk1_init = kernels.conv2dk1(
        input_width=tensorInW,
        input_channels=tensorInCInit,
        output_channels=tensorInCInit,
        act_dtype=np.int8,
    )
    # 3x3 conv: weights buffer holds full 64 output channels (shared by two
    # workers that each compute 32 via channel_offset); per-call output is 32.
    conv2dk3_kernel = kernels.conv2dk3(
        input_width=tensorInW,
        input_channels=tensorInCInit,
        output_channels=tensorInCInit // 2,
        weight_output_channels=tensorInCInit,
        act_dtype=np.uint8,
    )
    # First-column 1x1 + skip-projection (weights buffer holds both main
    # 1x1 and the skip 1x1 projection inline).
    conv2dk1_skip_init_kernel = kernels.conv2dk1_skip_init(
        input_width=tensorInW,
        input_channels=tensorInCInit,
        output_channels=tensorInCRest,
        act_dtype=np.int8,
        skip_input_channels=tensorInCInit,
    )
    # Repeated-column 1x1: uint8 in/out
    conv2dk1_rest = kernels.conv2dk1(
        input_width=tensorInW,
        input_channels=tensorInCRest,
        output_channels=tensorInCInit,
        act_dtype=np.uint8,
    )
    # Repeated-column 1x1 + skip add
    conv2dk1_skip_rest = kernels.conv2dk1_skip(
        input_width=tensorInW,
        input_channels=tensorInCInit,
        output_channels=tensorInCRest,
        act_dtype=np.uint8,
    )
    conv1_kernels_call = [conv2dk1_init, conv2dk1_rest, conv2dk1_rest]
    conv3_kernels_call = [
        conv2dk1_skip_init_kernel,
        conv2dk1_skip_rest,
        conv2dk1_skip_rest,
    ]

    # Cores - we move in a snake-like pattern, that depends on
    # shared memory between neighbors, so we'll explicitly place all cores
    cores = [
        [Tile(0, 2), Tile(0, 3), Tile(0, 4), Tile(0, 5)],
        [Tile(1, 5), Tile(1, 4), Tile(1, 3), Tile(1, 2)],
        [Tile(2, 2), Tile(2, 3), Tile(2, 4), Tile(2, 5)],
    ]

    # input tensor (with broadcast for skip connection)
    act1_fifo_names = ["act1_00_02_01", "act1_04_15_11", "act1_13_22_21"]
    act1_fifos = []
    skip_fifos = []

    act1_fifos.append(ObjectFifo(layer1_act_sizes[0], name=act1_fifo_names[0]))
    skip_fifos.append(
        act1_fifos[0].cons(4).forward(tile=Tile(0, 1), depth=2, name="skip_0")
    )

    for i in range(1, repeat + 1):
        act1_fifos.append(ObjectFifo(layer1_act_sizes[i], name=act1_fifo_names[i]))
        if i == 1:
            skip_tile = Tile(0, 1)
        else:
            skip_tile = Tile(i, 1)
        skip_fifos.append(
            act1_fifos[-1].cons(4).forward(tile=skip_tile, depth=2, name=f"skip_{i}")
        )

    act2_fifo_names = ["act2_02_03_05", "act2_15_12_14", "act2_22_23_25"]
    act2_fifos = []

    act3_fifo_names_1 = ["act3_03_04", "act3_14_13", "act3_23_24"]
    act3_fifos_1 = []

    act3_fifo_names_2 = ["act3_05_04", "act3_12_13", "act3_25_24"]
    act3_fifos_2 = []

    for i in range(n_cols):
        # 1x1 -> 3x3
        act2_fifos.append(
            ObjectFifo(tensorLayer1Out_ty, depth=4, name=act2_fifo_names[i])
        )
        # 3x3 -> 1x1
        act3_fifos_1.append(ObjectFifo(tensorLayer2Out_ty, name=act3_fifo_names_1[i]))
        act3_fifos_2.append(ObjectFifo(tensorLayer2Out_ty, name=act3_fifo_names_2[i]))

    wts_fifos = []
    wts_sub_fifos = [[], [], []]

    for i in range(n_cols):
        wts_fifos.append(ObjectFifo(wts_sizes[i], name=f"wts_{i}_L3L2", depth=1))
        wts_offsets = [
            0,
            layer1_wts_byte_counts[i],
            layer1_wts_byte_counts[i] + layer2_wts_size,
        ]
        wts_sub_fifos[i] = (
            wts_fifos[i]
            .cons()
            .split(
                wts_offsets,
                depths=[1, 1, 1],
                obj_types=[layer1_wts_sizes[i], weightsLayer2_ty, layer3_wts_sizes[i]],
                names=[f"wts_buf_{i}{j}" for j in range(3)],
                tile=Tile(i, 1),
            )
        )

    # output tensor
    outOFL2L3 = ObjectFifo(tensorLayer3Out_ty, name="outOFL2L3")
    conv3_out_fifos = [
        act1_fifos[1],
        act1_fifos[2],
        outOFL2L3,
    ]

    # Task for a worker to perform.  ``idx`` and the scale literals are
    # metaprogramming values (Python-side constants that compile to scalars).
    def conv1_fn(of_wts, of_act1, of_act2, conv1_kernel, scale, idx):
        element0Weights = of_wts.acquire(1)
        for _ in range_(tensorInH):
            element0ActivactionsIn = of_act1.acquire(1)
            element0ActivactionsOut = of_act2.acquire(1)
            if idx == 0:
                conv1_kernel(
                    element0ActivactionsIn,
                    element0Weights,
                    element0ActivactionsOut,
                    tensorInW,
                    tensorInCInit,
                    tensorInCInit,
                    scale,
                )
            else:
                conv1_kernel(
                    element0ActivactionsIn,
                    element0Weights,
                    element0ActivactionsOut,
                    tensorInW,
                    tensorInCRest,
                    tensorInCInit,
                    scale,
                )
            of_act1.release(1)
            of_act2.release(1)
        of_wts.release(1)

    # 3x3 conv2d
    def conv2_fn(of_wts, of_act2, of_act3, conv2dk3_k, last_arg_zero=False):
        last_arg = 0 if last_arg_zero else tensorInCInit // 2
        scale = 1

        element0Weights = of_wts.acquire(1)

        # pre-amble: top row
        elementActivactionsIn = of_act2.acquire(2)
        element0ActivactionsOut = of_act3.acquire(1)
        conv2dk3_k(
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
            last_arg,
        )
        of_act3.release(1)

        # middle
        for _ in range_(tensorInH - 2):
            elementActivactionsIn = of_act2.acquire(3)
            element0ActivactionsOut = of_act3.acquire(1)
            conv2dk3_k(
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
                last_arg,
            )
            of_act2.release(1)
            of_act3.release(1)

        # last part
        elementActivactionsIn = of_act2.acquire(2)
        element0ActivactionsOut = of_act3.acquire(1)
        conv2dk3_k(
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
            last_arg,
        )
        of_act2.release(2)
        of_act3.release(1)
        of_wts.release(1)

    # 1x1 conv2d and add skip
    def conv1_skip_fn(
        of_wts,
        of_act3_1,
        of_act3_2,
        of_conv3,
        of_skip,
        conv3_kernel,
        scale,
        skipScale,
        skipConvScale,
        idx,
    ):
        element0Weights = of_wts.acquire(1)
        for _ in range_(tensorInH):
            element0ActivactionsIn = of_act3_1.acquire(1)
            element1ActivactionsIn = of_act3_2.acquire(1)
            elementActivactionsOut = of_conv3.acquire(1)
            elementSkipsIn = of_skip.acquire(1)
            if idx == 0:
                conv3_kernel(
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
                conv3_kernel(
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
            of_act3_1.release(1)
            of_act3_2.release(1)
            of_conv3.release(1)
            of_skip.release(1)
        of_wts.release(1)

    # ------------------------------------------------------------------
    # Workers — scales are inlined Python literals (was RTP-fed before).
    # ------------------------------------------------------------------
    workers = []
    for i in range(n_cols):
        workers.append(
            Worker(
                conv1_fn,
                [
                    wts_sub_fifos[i][0].cons(),
                    act1_fifos[i].cons(),
                    act2_fifos[i].prod(),
                    conv1_kernels_call[i],
                    1,  # scale
                    i,
                ],
                tile=cores[i][0],
            )
        )
        workers.append(
            Worker(
                conv2_fn,
                [
                    wts_sub_fifos[i][1].cons(),
                    act2_fifos[i].cons(),
                    act3_fifos_1[i].prod(),
                    conv2dk3_kernel,
                    False,
                ],
                tile=cores[i][1],
            )
        )
        skip_args = [1, 0, 1] if i == 0 else [1, 0, 0]
        workers.append(
            Worker(
                conv1_skip_fn,
                [
                    wts_sub_fifos[i][2].cons(),
                    act3_fifos_1[i].cons(),
                    act3_fifos_2[i].cons(),
                    conv3_out_fifos[i].prod(),
                    skip_fifos[i].cons(),
                    conv3_kernels_call[i],
                    *skip_args,
                    i,
                ],
                tile=cores[i][2],
                stack_size=0xA00,
            )
        )
        workers.append(
            Worker(
                conv2_fn,
                [
                    wts_sub_fifos[i][1].cons(),
                    act2_fifos[i].cons(),
                    act3_fifos_2[i].prod(),
                    conv2dk3_kernel,
                    True,
                ],
                tile=cores[i][3],
            )
        )

    # Runtime: stream activations + weights in, drain output.
    rt = Runtime()

    def sequence(inputFromL3, weightsFromL3, outputToL3):
        act1_fifos[0].prod().fill(inputFromL3, tile=Tile(0, 0))

        tap = TensorAccessPattern(
            (totalWeights_complete,),
            offset=0,
            sizes=[1, 1, 1, totalWeights_init],
            strides=[0, 0, 0, 1],
        )
        wts_fifos[0].prod().fill(weightsFromL3, tap, tile=Tile(0, 0))

        tap = TensorAccessPattern(
            (totalWeights_complete,),
            offset=totalWeights_init,
            sizes=[1, 1, 1, totalWeights_rest],
            strides=[0, 0, 0, 1],
        )
        wts_fifos[1].prod().fill(weightsFromL3, tap, tile=Tile(1, 0))

        tap = TensorAccessPattern(
            (totalWeights_complete,),
            offset=totalWeights_init + totalWeights_rest,
            sizes=[1, 1, 1, totalWeights_rest],
            strides=[0, 0, 0, 1],
        )
        wts_fifos[2].prod().fill(weightsFromL3, tap, tile=Tile(2, 0))
        outOFL2L3.cons().drain(outputToL3, tile=Tile(1, 0), wait=True)

    rt.sequence(
        sequence,
        [activationsInL3_ty, weightsInL3_ty_complete, activationsOutL3_ty],
    )

    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()
