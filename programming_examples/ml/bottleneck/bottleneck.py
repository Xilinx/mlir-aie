#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc.
import numpy as np
import sys

from aie.iron import (
    GlobalBuffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    WorkerRuntimeBarrier,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import AnyMemTile, NPU1Col1, NPU2, Tile
from aie.iron.controlflow import range_

if len(sys.argv) > 1:
    if sys.argv[1] == "npu":
        dev = NPU1Col1()
    elif sys.argv[1] == "npu2":
        dev = NPU2()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

# Define bottleneck layer sizes

tensorInW = 32
tensorInH = 32
tensorInC = 256

tensorL1InC = tensorInC
tensorL1OutC = tensorL1InC // 4

tensorL2InC = tensorL1OutC
tensorL2OutC = tensorL2InC

tensorL3InC = tensorL2OutC
tensorL3OutC = tensorL3InC * 4

activationsIn = tensorInW * tensorInH * tensorInC
acitivationsOut = activationsIn
weightsL1_sz = tensorL1InC * tensorL1OutC
weightsL2_sz = 3 * 3 * tensorL2InC * tensorL2OutC
weightsL3_sz = tensorL3InC * tensorL3OutC
totalWeights = weightsL1_sz + weightsL2_sz + weightsL3_sz


def bottleneck4AIEs():
    # define types
    activationsInL3_ty = np.ndarray[(activationsIn,), np.dtype[np.int8]]
    weightsInL3_ty = np.ndarray[(totalWeights,), np.dtype[np.uint8]]
    weightsAll_ty = np.ndarray[(totalWeights,), np.dtype[np.int8]]

    tensorLayer1In_ty = np.ndarray[(tensorInW, 1, tensorL1InC), np.dtype[np.int8]]
    weightsLayer1_ty = np.ndarray[(weightsL1_sz,), np.dtype[np.int8]]
    tensorLayer1Out_ty = np.ndarray[(tensorInW, 1, tensorL1OutC), np.dtype[np.uint8]]

    tensorLayer2In_ty = np.ndarray[(tensorInW, 1, tensorL2InC), np.dtype[np.uint8]]
    weightsLayer2_ty = np.ndarray[(weightsL2_sz,), np.dtype[np.int8]]
    tensorLayer2Out_ty = np.ndarray[
        (tensorInW, 1, tensorL2OutC // 2), np.dtype[np.uint8]
    ]

    tensorLayer3In_ty = np.ndarray[(tensorInW, 1, tensorL3InC // 2), np.dtype[np.uint8]]
    weightsLayer3_ty = np.ndarray[(weightsL3_sz,), np.dtype[np.int8]]
    tensorLayer3Out_ty = np.ndarray[(tensorInW, 1, tensorL3OutC), np.dtype[np.uint8]]

    # kernel definitions
    conv2dk1 = Kernel(
        "conv2dk1_i8",
        "conv2dk1.o",
        [
            tensorLayer1In_ty,
            weightsLayer1_ty,
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
    conv2dk1_skip = Kernel(
        "conv2dk1_skip_i8",
        "conv2dk1_skip.o",
        [
            tensorLayer3In_ty,
            tensorLayer3In_ty,
            weightsLayer3_ty,
            tensorLayer3Out_ty,
            tensorLayer1In_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )

    # Buffers used to hold runtime parameters
    rtp2 = GlobalBuffer(
        np.ndarray[(16,), np.dtype[np.int32]],
        name="rtp2",
        use_write_rtp=True,
    )
    rtp4 = GlobalBuffer(
        np.ndarray[(16,), np.dtype[np.int32]],
        name="rtp4",
        use_write_rtp=True,
    )

    rtp_barrier = WorkerRuntimeBarrier()

    # AIE-array data movement with object fifos
    of_inOF_act_L3L2 = ObjectFifo(tensorLayer1In_ty, name="inOF_act_L3L2")
    of_skip_buf = of_inOF_act_L3L2.cons(4).forward(
        depth=2, placement=AnyMemTile, name="skip_buf"
    )

    # weights
    inOF_wts_0_L3L2 = ObjectFifo(weightsAll_ty, depth=1, name="inOF_wts_0_L3L2")
    of_offsets = [0, weightsL1_sz, weightsL1_sz + weightsL2_sz]
    of_wts_buf_00, wts_buf_01, wts_buf_02 = inOF_wts_0_L3L2.cons().split(
        of_offsets,
        obj_types=[weightsLayer1_ty, weightsLayer2_ty, weightsLayer3_ty],
        names=[f"wts_buf_0{i}" for i in range(3)],
    )

    # activation tensor
    of_act_2_3_5 = ObjectFifo(
        tensorLayer1Out_ty,
        name="act_2_3_5",
    )  # 1x1 -> 3x3

    # 3x3 -> 1x1
    act_3_4 = ObjectFifo(tensorLayer2Out_ty, name="act_3_4")
    # 3x3 -> 1x1
    act_5_4 = ObjectFifo(tensorLayer2Out_ty, name="act_5_4")

    # output tensor
    outOFL2L3 = ObjectFifo(tensorLayer3Out_ty, name="outOFL2L3")

    workers = []

    # 1x1 conv2d
    def worker_conv2dk1_fn(
        of_wts, of_act_in, of_act_out, conv2dk1_kernel, rtp_buff, barrier
    ):
        # acquire weights amd rtps once
        barrier.wait_for_value(1)
        scale = rtp_buff[0]
        element0Weights = of_wts.acquire(1)
        for _ in range_(tensorInH):
            element0ActivactionsIn = of_act_in.acquire(1)
            element0ActivactionsOut = of_act_out.acquire(1)
            conv2dk1_kernel(
                element0ActivactionsIn,
                element0Weights,
                element0ActivactionsOut,
                tensorInW,
                tensorL1InC,
                tensorL1OutC,
                scale,
            )
            of_act_in.release(1)
            of_act_out.release(1)
        of_wts.release(1)

    worker = Worker(
        worker_conv2dk1_fn,
        fn_args=[
            of_wts_buf_00.cons(),
            of_inOF_act_L3L2.cons(),
            of_act_2_3_5.prod(),
            conv2dk1,
            rtp2,
            rtp_barrier,
        ],
    )
    workers.append(worker)

    # 3x3 conv2d OFM 0-31
    def worker_conv2dk3_fn(of_wts, of_act_in, of_act_out, conv2dk3_fn, conv_last_arg):
        scale = 11

        # acquire weights and rtps once
        element0Weights = of_wts.acquire(1)
        # scale = memref.load(rtpComputeTile3, 0)

        # pre-amble: top row
        elementActivactionsIn = of_act_in.acquire(2)
        element0ActivactionsOut = of_act_out.acquire(1)
        conv2dk3_fn(
            elementActivactionsIn[0],
            elementActivactionsIn[0],
            elementActivactionsIn[1],
            element0Weights,
            element0ActivactionsOut,
            tensorInW,
            tensorL2InC,
            tensorL2OutC,
            3,
            3,
            0,
            scale,
            conv_last_arg,
        )
        of_act_out.release(1)

        # middle
        for _ in range_(tensorInH - 2):
            elementActivactionsIn = of_act_in.acquire(3)
            element0ActivactionsOut = of_act_out.acquire(1)
            conv2dk3_fn(
                elementActivactionsIn[0],
                elementActivactionsIn[1],
                elementActivactionsIn[2],
                element0Weights,
                element0ActivactionsOut,
                tensorInW,
                tensorL2InC,
                tensorL2OutC,
                3,
                3,
                1,
                scale,
                conv_last_arg,
            )
            of_act_in.release(1)
            of_act_out.release(1)

        # last part
        elementActivactionsIn = of_act_in.acquire(2)
        element0ActivactionsOut = of_act_out.acquire(1)
        conv2dk3_fn(
            elementActivactionsIn[0],
            elementActivactionsIn[1],
            elementActivactionsIn[1],
            element0Weights,
            element0ActivactionsOut,
            tensorInW,
            tensorL2InC,
            tensorL2OutC,
            3,
            3,
            2,
            scale,
            conv_last_arg,
        )

        of_act_in.release(2)
        of_act_out.release(1)
        of_wts.release(1)

    worker = Worker(
        worker_conv2dk3_fn,
        fn_args=[wts_buf_01.cons(), of_act_2_3_5.cons(4), act_3_4.prod(), conv2dk3, 0],
        placement=Tile(0, 3),
    )
    workers.append(worker)
    worker = Worker(
        worker_conv2dk3_fn,
        fn_args=[
            wts_buf_01.cons(),
            of_act_2_3_5.cons(4),
            act_5_4.prod(),
            conv2dk3,
            tensorL2OutC // 2,
        ],
        placement=Tile(0, 5),
    )
    workers.append(worker)

    # # 1x1 conv2d and add skip
    def worker_conv2dk1_skip_fn(
        of_wts,
        of_act_in0,
        of_act_in1,
        of_skip,
        of_out,
        conv2dk1_skip_fn,
        rtp_buff,
        barrier,
    ):
        # acquire weights and rtps once
        barrier.wait_for_value(1)
        scale = rtp_buff[0]
        skipScale = rtp_buff[1]
        element0Weights = of_wts.acquire(1)

        for _ in range_(tensorInH):
            element0ActivactionsIn = of_act_in0.acquire(1)
            element1ActivactionsIn = of_act_in1.acquire(1)
            elementSkipsIn = of_skip.acquire(1)
            elementActivactionsOut = of_out.acquire(1)

            conv2dk1_skip_fn(
                element0ActivactionsIn,
                element1ActivactionsIn,
                element0Weights,
                elementActivactionsOut,
                elementSkipsIn,
                tensorInW,
                tensorL3InC,
                tensorL3OutC,
                scale,
                skipScale,
            )
            of_out.release(1)
            of_act_in0.release(1)
            of_act_in1.release(1)
            of_skip.release(1)
        of_wts.release(1)

    worker = Worker(
        worker_conv2dk1_skip_fn,
        fn_args=[
            wts_buf_02.cons(),
            act_3_4.cons(),
            act_5_4.cons(),
            of_skip_buf.cons(),
            outOFL2L3.prod(),
            conv2dk1_skip,
            rtp4,
            rtp_barrier,
        ],
        placement=Tile(0, 4),
        stack_size=0xA00,
    )
    workers.append(worker)

    # Runtime operations to move data to/from the device and set the runtime parameters
    rt = Runtime()
    with rt.sequence(activationsInL3_ty, weightsInL3_ty, activationsInL3_ty) as (
        inputFromL3,
        weightsFromL3,
        outputToL3,
    ):
        # write RTP parameters
        def runtime_ops(p2, p4):
            p2[0] = 1  # scale
            # scale: conv1x1 with the same scale as the input so we match the scaling factor of output after conv1x1 and the initial input
            p4[0] = 1
            p4[1] = 0  # skip_scale

        rt.inline_ops(runtime_ops, [rtp2, rtp4])

        rt.set_barrier(rtp_barrier, 1)

        # TODO: the order of operations here is a little misleading,
        # as workers are started immediately
        rt.start(*workers)

        rt.fill(of_inOF_act_L3L2.prod(), inputFromL3)
        rt.fill(inOF_wts_0_L3L2.prod(), weightsFromL3)
        rt.drain(outOFL2L3.cons(), outputToL3, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


module = bottleneck4AIEs()
print(module)
