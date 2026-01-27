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
from aie.iron.controlflow import range_
import aie.utils.trace as trace_utils


def conv2dk1(
    dev, width: int, height: int, in_channels: int, out_channels: int, trace_size: int
):
    with mlir_mod_ctx() as ctx:

        actIn = width * in_channels  # 32*64 = 2048
        bufIn = actIn * 2  # double buffer

        weights = in_channels * out_channels

        actOut = width * out_channels  # 32*64 = 2048
        bufOut = actOut * 2  # double buffer

        tensorInSize = width * height * in_channels
        tensorOutSize = width * height * out_channels

        N_in_bytes = tensorOutSize  # Number of bytes of output data (1 byte/elem)

        @device(dev)
        def device_body():

            actIn_ty = np.ndarray[(actIn,), np.dtype[np.int8]]
            bufIn_ty = np.ndarray[(bufIn,), np.dtype[np.int8]]

            weights_ty = np.ndarray[(weights,), np.dtype[np.int8]]

            out_ty = np.ndarray[(actOut,), np.dtype[np.int8]]
            bufOut_ty = np.ndarray[(bufOut,), np.dtype[np.int8]]
            tensorIn_ty = np.ndarray[(tensorInSize,), np.dtype[np.int8]]
            tensorOut_ty = np.ndarray[(tensorOutSize,), np.dtype[np.int8]]

            # AIE Core Function declarations
            conv2dk1_i8 = external_func(
                "conv2dk1_i8",
                inputs=[
                    actIn_ty,
                    weights_ty,
                    out_ty,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                ],
            )

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)

            lock2 = lock(ComputeTile2, init=0)

            # AIE-array data movement with object fifos
            # Input
            of_inOF_act_L3L2 = object_fifo(
                "inOF_act_L3L2", ShimTile, MemTile, 2, bufIn_ty
            )
            of_act_L2_02 = object_fifo("act_L2_02", MemTile, ComputeTile2, 2, actIn_ty)
            object_fifo_link(of_inOF_act_L3L2, of_act_L2_02)

            # wts
            of_inOF_wts_0_L3L2 = object_fifo(
                "inOF_wts_0_L3L2", ShimTile, [ComputeTile2], 1, weights_ty
            )

            # Output
            of_out_02_L2 = object_fifo("out_02_L2", ComputeTile2, [MemTile], 2, out_ty)
            of_outOFL2L3 = object_fifo("outOFL2L3", MemTile, [ShimTile], 2, bufOut_ty)
            object_fifo_link(of_out_02_L2, of_outOFL2L3)

            # Set up a packet-switched flow from core to shim for tracing information
            tiles_to_trace = [ComputeTile2]
            if trace_size > 0:
                trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

            # Set up compute tiles

            rtp2 = buffer(
                ComputeTile2,
                np.ndarray[(16,), np.dtype[np.int32]],
                "rtp2",
                use_write_rtp=True,
            )

            # Compute tile 2
            @core(ComputeTile2, "conv2dk1_i8.o", stack_size=0x600)
            def core_body():
                y_dim = height
                x_dim = width
                ci = in_channels
                co = out_channels

                for _ in range_(0xFFFFFFFF):
                    use_lock(lock2, LockAction.Acquire, value=1)
                    scale = rtp2[0]

                    elemWts = of_inOF_wts_0_L3L2.acquire(ObjectFifoPort.Consume, 1)

                    for _ in range_(y_dim):
                        elemIn = of_act_L2_02.acquire(ObjectFifoPort.Consume, 1)
                        elemOut0 = of_out_02_L2.acquire(ObjectFifoPort.Produce, 1)

                        conv2dk1_i8(elemIn, elemWts, elemOut0, x_dim, ci, co, scale)
                        of_act_L2_02.release(ObjectFifoPort.Consume, 1)
                        of_out_02_L2.release(ObjectFifoPort.Produce, 1)
                    of_inOF_wts_0_L3L2.release(ObjectFifoPort.Consume, 1)

            # To/from AIE-array data movement
            @runtime_sequence(tensorIn_ty, weights_ty, tensorOut_ty)
            def sequence(I, W, O):

                if trace_size > 0:
                    trace_utils.configure_packet_tracing_aie2(
                        tiles_to_trace, ShimTile, trace_size, N_in_bytes, ddr_id=2
                    )

                rtp2[0] = 10

                set_lock_value(lock2, 1)

                in_act_task = shim_dma_single_bd_task(
                    of_inOF_act_L3L2,
                    I,
                    sizes=[1, 1, 1, tensorInSize],
                    issue_token=True,
                )
                in_wts_task = shim_dma_single_bd_task(
                    of_inOF_wts_0_L3L2,
                    W,
                    sizes=[1, 1, 1, weights],
                    issue_token=True,
                )
                out_task = shim_dma_single_bd_task(
                    of_outOFL2L3,
                    O,
                    sizes=[1, 1, 1, tensorOutSize],
                    issue_token=True,
                )

                dma_start_task(in_act_task, in_wts_task, out_task)
                dma_await_task(in_act_task, in_wts_task, out_task)

                trace_utils.gen_trace_done_aie2(ShimTile)

    #    print(ctx.module.operation.verify())
    print(ctx.module)


if __name__ == "__main__":
    try:
        device_name = str(sys.argv[1])
        if device_name == "npu":
            dev = AIEDevice.npu1_1col
        elif device_name == "npu2":
            dev = AIEDevice.npu2_1col
        else:
            raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
        width = int(sys.argv[2])
        if width % 8 != 0 or width < 8:
            print("Width size must be a multiple of 8 and greater than or equal to 8")
            raise ValueError
        height = int(sys.argv[3])
        if height % 8 != 0 or height < 8:
            print("Height size must be a multiple of 8 and greater than or equal to 8")
            raise ValueError
        in_channels = int(sys.argv[4])
        if in_channels % 8 != 0 or in_channels < 8:
            print(
                "Input channels size must be a multiple of 8 and greater than or equal to 8"
            )
            raise ValueError
        out_channels = int(sys.argv[5])
        if out_channels % 8 != 0 or out_channels < 8:
            print(
                "Output channel size must be a multiple of 8 and greater than or equal to 8"
            )
            raise ValueError
        trace_size = 0 if (len(sys.argv) != 7) else int(sys.argv[6])
    except ValueError:
        print("Argument has inappropriate value")
    conv2dk1(dev, width, height, in_channels, out_channels, trace_size)
