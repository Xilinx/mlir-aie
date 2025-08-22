#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc.
import numpy as np
import sys

from aie.iron import GlobalBuffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col3, NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.util import np_ndarray_type_get_shape
from aie.helpers.taplib import TensorAccessPattern

if len(sys.argv) > 1:
    if sys.argv[1] == "npu":
        dev = NPU1Col3()
    elif sys.argv[1] == "npu2":
        dev = NPU2()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

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

# define types
tensorLayer1In_ty_init = np.ndarray[(tensorInW, 1, tensorInCInit), np.dtype[np.int8]]
tensorLayer1In_ty_rest = np.ndarray[(tensorInW, 1, tensorInCRest), np.dtype[np.uint8]]
weightsLayer1_ty_init = np.ndarray[(tensorInCInit * tensorInCInit,), np.dtype[np.int8]]
weightsLayer1_ty_rest = np.ndarray[(tensorInCRest * tensorInCInit,), np.dtype[np.int8]]

tensorLayer1Out_ty = np.ndarray[(tensorInW, 1, tensorInCInit), np.dtype[np.uint8]]
tensorLayer2In_ty = np.ndarray[(tensorInW, 1, tensorInCInit), np.dtype[np.uint8]]
weightsLayer2_ty = np.ndarray[
    (3 * 3 * tensorInCInit * tensorInCInit,), np.dtype[np.int8]
]
tensorLayer2Out_ty = np.ndarray[(tensorInW, 1, tensorInCInit // 2), np.dtype[np.uint8]]

tensorLayer3In_ty = np.ndarray[(tensorInW, 1, tensorInCInit // 2), np.dtype[np.uint8]]
weightsLayer3_ty_init = np.ndarray[
    (2 * tensorInCInit * tensorInCRest,), np.dtype[np.int8]
]
weightsLayer3_ty_rest = np.ndarray[
    (tensorInCRest // 4 * tensorInCRest,), np.dtype[np.int8]
]

tensorLayer3Out_ty = np.ndarray[(tensorInW, 1, tensorInCRest), np.dtype[np.uint8]]

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

weightsInL3_ty_complete = np.ndarray[(totalWeights_complete,), np.dtype[np.int8]]

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

# kernel definitions
conv2dk1_i8 = Kernel(
    "conv2dk1_i8",
    "conv2dk1_i8.o",
    [
        tensorLayer1In_ty_init,
        weightsLayer1_ty_init,
        tensorLayer1Out_ty,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
    ],
)
conv2dk3 = Kernel(
    "conv2dk3_ui8",
    "conv2dk3.o",
    [
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
conv2dk1_skip_init_i8 = Kernel(
    "conv2dk1_skip_init_i8",
    "conv2dk1_skip_init.o",
    [
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
conv2dk1_ui8 = Kernel(
    "conv2dk1_ui8",
    "conv2dk1_ui8.o",
    [
        tensorLayer3Out_ty,
        weightsLayer1_ty_rest,
        tensorLayer1Out_ty,
        np.int32,
        np.int32,
        np.int32,
        np.int32,
    ],
)

conv2dk1_skip_ui8 = Kernel(
    "conv2dk1_skip_ui8",
    "conv2dk1_skip.o",
    [
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
conv1_kernels_call = [conv2dk1_i8, conv2dk1_ui8, conv2dk1_ui8]
conv3_kernels_call = [
    conv2dk1_skip_init_i8,
    conv2dk1_skip_ui8,
    conv2dk1_skip_ui8,
]

# runtime parameters
rtp = []
for i in range(3):
    rtp.append([])
    for j in range(2, 6):
        rtp[i].append(
            GlobalBuffer(
                np.ndarray[(16,), np.dtype[np.int32]],
                name=f"rtpComputeTile{i}{j}",
                use_write_rtp=True,
            )
        )

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

act1_fifos.append(ObjectFifo(laye1_act_sizes[0], name=act1_fifo_names[0]))
skip_fifos.append(
    act1_fifos[0].cons(4).forward(placement=Tile(0, 1), depth=2, name="skip_0")
)

for i in range(1, repeat + 1):
    act1_fifos.append(ObjectFifo(laye1_act_sizes[i], name=act1_fifo_names[i]))
    if i == 1:
        placement = Tile(0, 1)
    else:
        placement = Tile(i, 1)
    skip_fifos.append(
        act1_fifos[-1].cons(4).forward(placement=placement, depth=2, name=f"skip_{i}")
    )

act2_fifo_names = ["act2_02_03_05", "act2_15_12_14", "act2_22_23_25"]
act2_fifos = []

act3_fifo_names_1 = ["act3_03_04", "act3_14_13", "act3_23_24"]
act3_fifos_1 = []

act3_fifo_names_2 = ["act3_05_04", "act3_12_13", "act3_25_24"]
act3_fifos_2 = []

for i in range(n_cols):
    # 1x1 -> 3x3
    act2_fifos.append(ObjectFifo(tensorLayer1Out_ty, depth=4, name=act2_fifo_names[i]))
    # 3x3 -> 1x1
    act3_fifos_1.append(ObjectFifo(tensorLayer2Out_ty, name=act3_fifo_names_1[i]))
    act3_fifos_2.append(ObjectFifo(tensorLayer2Out_ty, name=act3_fifo_names_2[i]))

wts_fifos = []
wts_sub_fifos = [[], [], []]

for i in range(n_cols):
    wts_fifos.append(ObjectFifo(wts_sizes[i], name=f"wts_{i}_L3L2", depth=1))
    wts_offsets = [
        0,
        np.prod(np_ndarray_type_get_shape(layer1_wts_sizes[i])),
        np.prod(np_ndarray_type_get_shape(layer1_wts_sizes[i]))
        + np.prod(np_ndarray_type_get_shape(weightsLayer2_ty)),
    ]
    wts_sub_fifos[i] = (
        wts_fifos[i]
        .cons()
        .split(
            wts_offsets,
            depths=[1, 1, 1],
            obj_types=[layer1_wts_sizes[i], weightsLayer2_ty, layer3_wts_sizes[i]],
            names=[f"wts_buf_{i}{j}" for j in range(3)],
            placement=Tile(i, 1),
        )
    )

# output tensor
outOFL2L3 = ObjectFifo(tensorLayer3Out_ty, name="outOFL2L3")
conv3_out_fifos = [
    act1_fifos[1],
    act1_fifos[2],
    outOFL2L3,
]


# Task for a worker to perform. Note that idx is a metaprogramming value,
# e.g., it is a Python variable but will compile to a scalar constant.
def conv1_fn(of_wts, of_act1, of_act2, conv1_kernel, rtp, idx):
    # acquire weights once
    element0Weights = of_wts.acquire(1)
    scale = rtp[0]
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
# Task for a worker to perform. Note that last_arg_zero is a metaprogramming value,
# e.g., it is a Python variable that will not be present in compiled code.
def conv2_fn(of_wts, of_act2, of_act3, conv2dk3_kernel, last_arg_zero=False):
    last_arg = 0
    if not last_arg_zero:
        last_arg = tensorInCInit // 2
    scale = 1

    # acquire weights and rtps once
    element0Weights = of_wts.acquire(1)
    # scale = memref.load(rtpComputeTile03, 0)

    # pre-amble: top row
    elementActivactionsIn = of_act2.acquire(2)
    element0ActivactionsOut = of_act3.acquire(1)
    conv2dk3_kernel(
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
        conv2dk3_kernel(
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
    conv2dk3_kernel(
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


# # 1x1 conv2d and add skip
# Task for a worker to perform. Note that idx is a metaprogramming value,
# e.g., it is a Python variable that will not be present in compiled code.
def conv1_skip_fn(
    of_wts, of_act3_1, of_act3_2, of_conv3, of_skip, conv3_kernel, my_rtp, idx
):
    # acquire weights and rtps once
    element0Weights = of_wts.acquire(1)
    if idx == 0:
        scale = my_rtp[0]
        skipScale = my_rtp[1]
        skipConvScale = my_rtp[2]
    else:
        scale = my_rtp[0]
        skipScale = my_rtp[1]

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


# Create workers and place each one on a particular compute core
workers = []
for i in range(n_cols):
    placement = cores[i][0]
    w = Worker(
        conv1_fn,
        [
            wts_sub_fifos[i][0].cons(),
            act1_fifos[i].cons(),
            act2_fifos[i].prod(),
            conv1_kernels_call[i],
            rtp[placement.col][placement.row - 2],
            i,
        ],
        placement=placement,
    )
    workers.append(w)
    w = Worker(
        conv2_fn,
        [
            wts_sub_fifos[i][1].cons(),
            act2_fifos[i].cons(),
            act3_fifos_1[i].prod(),
            conv2dk3,
            False,
        ],
        placement=cores[i][1],
    )
    workers.append(w)
    placement = cores[i][2]
    if i == 0:
        skip_rtp = rtp[0][3]
    else:
        skip_rtp = rtp[placement.col][placement.row - 2]
    w = Worker(
        conv1_skip_fn,
        [
            wts_sub_fifos[i][2].cons(),
            act3_fifos_1[i].cons(),
            act3_fifos_2[i].cons(),
            conv3_out_fifos[i].prod(),
            skip_fifos[i].cons(),
            conv3_kernels_call[i],
            skip_rtp,
            i,
        ],
        placement=placement,
        stack_size=0xA00,
    )
    workers.append(w)
    w = Worker(
        conv2_fn,
        [
            wts_sub_fifos[i][1].cons(),
            act2_fifos[i].cons(),
            act3_fifos_2[i].prod(),
            conv2dk3,
            True,
        ],
        placement=cores[i][3],
    )
    workers.append(w)

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(activationsInL3_ty, weightsInL3_ty_complete, activationsOutL3_ty) as (
    inputFromL3,
    weightsFromL3,
    outputToL3,
):

    # Set runtime parameters
    def set_rtps(rtp):

        rtp[0][0][0] = 1
        rtp[0][1][0] = 1
        rtp[0][2][0] = 1
        rtp[0][3][0] = 1
        rtp[0][3][1] = 0
        rtp[0][3][2] = 1

        rtp[1][3][0] = 1
        rtp[1][2][0] = 1
        rtp[1][0][0] = 1
        rtp[1][1][0] = 1
        rtp[1][1][1] = 0

        rtp[2][0][0] = 1
        rtp[2][1][0] = 1
        rtp[2][3][0] = 1
        rtp[2][2][0] = 1
        rtp[2][2][1] = 0

    rt.inline_ops(set_rtps, [rtp])

    # Start workers
    rt.start(*workers)

    # Fill/drain input/output object FIFOs
    rt.fill(act1_fifos[0].prod(), inputFromL3, placement=Tile(0, 0))

    tap = TensorAccessPattern(
        (totalWeights_complete,),
        offset=0,
        sizes=[1, 1, 1, totalWeights_init],
        strides=[0, 0, 0, 1],
    )
    rt.fill(wts_fifos[0].prod(), weightsFromL3, tap, placement=Tile(0, 0))

    tap = TensorAccessPattern(
        (totalWeights_complete,),
        offset=totalWeights_init,
        sizes=[1, 1, 1, totalWeights_rest],
        strides=[0, 0, 0, 1],
    )
    rt.fill(wts_fifos[1].prod(), weightsFromL3, tap, placement=Tile(1, 0))

    tap = TensorAccessPattern(
        (totalWeights_complete,),
        offset=totalWeights_init + totalWeights_rest,
        sizes=[1, 1, 1, totalWeights_rest],
        strides=[0, 0, 0, 1],
    )
    rt.fill(wts_fifos[2].prod(), weightsFromL3, tap, placement=Tile(2, 0))
    rt.drain(outOFL2L3.cons(), outputToL3, placement=Tile(1, 0), wait=True)

# Place components (assign them resources on the device) and generate an MLIR module
module = Program(dev, rt).resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
