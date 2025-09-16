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
from aie.helpers.dialects.ext.scf import _for as range_
import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent


def conv2dk14(
    dev,
    width: int,
    height: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    trace_size: int,
):
    with mlir_mod_ctx() as ctx:

        # Kernel processes 16 tiles and 16 output channels at a time
        sub_out_channels = 16
        sub_tiles = 16

        actIn = kernel_size * kernel_size * in_channels * sub_tiles
        weights = kernel_size * kernel_size * in_channels * sub_out_channels
        actOut = sub_tiles * sub_out_channels

        out_channels_group = out_channels // sub_out_channels  # 72
        width_out = width // kernel_size
        height_out = height // kernel_size

        # we reload inputs 72 times (out_channels // sub_out_channels)
        tensorInSize = width * height * in_channels * out_channels_group
        tensorWeightsSize = weights * out_channels_group
        tensorOutSize = width_out * height_out * sub_out_channels * out_channels_group

        N_in_bytes = tensorOutSize  # Number of bytes of output data (1 byte/elem)

        @device(dev)
        def device_body():

            actIn_ty = np.ndarray[(actIn,), np.dtype[np.uint8]]
            weights_ty = np.ndarray[(weights,), np.dtype[np.int8]]
            out_ty = np.ndarray[(actOut,), np.dtype[np.int8]]

            tensorIn_ty = np.ndarray[(tensorInSize,), np.dtype[np.uint8]]
            tensorWeights_ty = np.ndarray[(tensorWeightsSize,), np.dtype[np.int8]]
            tensorOut_ty = np.ndarray[(tensorOutSize,), np.dtype[np.int8]]

            # AIE Core Function declarations
            conv2dk14_i8 = external_func(
                "conv2dk14_i8",
                inputs=[
                    actIn_ty,
                    weights_ty,
                    out_ty,
                    np.int32,
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
                "inOF_act_L3L2",
                ShimTile,
                MemTile,
                2,
                np.ndarray[
                    (kernel_size, width * in_channels), np.dtype[np.uint8]
                ],  # (14, 3584)
                dimensionsToStream=None,
                dimensionsFromStreamPerConsumer=[
                    [
                        (kernel_size, kernel_size * in_channels),  # (14, 56)
                        (64, kernel_size * kernel_size * in_channels),  # (64, 784)
                        (kernel_size * in_channels, 1),  # (56, 1)
                    ],
                ],
            )
            of_act_L2_02 = object_fifo(
                "act_L2_02",
                MemTile,
                ComputeTile2,
                2,
                np.ndarray[(actIn,), np.dtype[np.uint8]],
                dimensionsToStream=[
                    (2, kernel_size * kernel_size * in_channels * 8),  # (2, 6272)
                    (kernel_size * kernel_size // 2, 2 * in_channels),  # (98, 8)
                    (8, kernel_size * kernel_size * in_channels),  # (8, 784)
                    (2 * in_channels, 1),  # (8, 1)
                ],
            )
            object_fifo_link(of_inOF_act_L3L2, of_act_L2_02)

            # wts
            of_inOF_wts_0_L3L2 = object_fifo(
                "inOF_wts_0_L3L2", ShimTile, [ComputeTile2], 2, weights_ty
            )

            # Output
            of_out_02_L2 = object_fifo(
                "out_02_L2",
                ComputeTile2,
                [MemTile],
                2,
                np.ndarray[(actOut,), np.dtype[np.int8]],
                # dimensionsFromStreamPerConsumer=[
                #     [
                #         (2, 8),
                #         (16, 16),
                #         (8, 1),
                #     ],
                # ],
            )
            of_outOFL2L3 = object_fifo(
                "outOFL2L3",
                MemTile,
                [ShimTile],
                2,
                np.ndarray[
                    (sub_out_channels, width_out * height_out), np.dtype[np.int8]
                ],
                # dimensionsToStream=[(16, 16), (256, 256), (16, 1)],
                dimensionsToStream=[(256, 256), (16, 8), (2, 128), (8, 1)],
            )
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
            @core(ComputeTile2, "conv2dk14.o", stack_size=0xC00)
            def core_body():
                y_dim = height // kernel_size
                x_blocks = 4
                x_dim = width // x_blocks  # num pixels for 1/4 of a row
                ci = in_channels
                co = sub_out_channels

                for _ in range_(0xFFFFFFFF):
                    use_lock(lock2, LockAction.Acquire, value=1)
                    scale = rtp2[0]

                    elemWts = of_inOF_wts_0_L3L2.acquire(ObjectFifoPort.Consume, 1)

                    for _ in range_(y_dim):
                        for _ in range_(x_blocks):
                            elemIn = of_act_L2_02.acquire(ObjectFifoPort.Consume, 1)
                            elemOut0 = of_out_02_L2.acquire(ObjectFifoPort.Produce, 1)
                            conv2dk14_i8(
                                elemIn,
                                elemWts,
                                elemOut0,
                                x_dim,  # input_width
                                ci,  # input_channels
                                co,  # output_channels
                                kernel_size,  # kernel_width
                                scale,
                            )
                            of_act_L2_02.release(ObjectFifoPort.Consume, 1)
                            of_out_02_L2.release(ObjectFifoPort.Produce, 1)

                    of_inOF_wts_0_L3L2.release(ObjectFifoPort.Consume, 1)

            # To/from AIE-array data movement
            # @runtime_sequence(tensorIn_ty, weights_ty, tensorOut_ty)
            @runtime_sequence(tensorIn_ty, tensorWeights_ty, tensorOut_ty)
            def sequence(I, W, O):

                if trace_size > 0:
                    trace_utils.configure_packet_tracing_aie2(
                        tiles_to_trace=tiles_to_trace,
                        shim=ShimTile,
                        trace_size=trace_size,
                        trace_offset=N_in_bytes,
                        ddr_id=2,
                        coretile_events=[
                            CoreEvent.INSTR_EVENT_0,
                            CoreEvent.INSTR_EVENT_1,
                            CoreEvent.INSTR_VECTOR,
                            PortEvent(CoreEvent.PORT_RUNNING_0, 1, True),  # master(1)
                            PortEvent(CoreEvent.PORT_RUNNING_1, 2, True),  # master(2)
                            PortEvent(CoreEvent.PORT_RUNNING_2, 1, False),  # slave(1)
                            CoreEvent.MEMORY_STALL,
                            CoreEvent.LOCK_STALL,
                        ],
                    )

                rtp2[0] = 14

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
                    sizes=[1, 1, 1, tensorWeightsSize],
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
        if in_channels != 4:
            print("Input channels size must be equal to 4")
            raise ValueError
        out_channels = int(sys.argv[5])
        if out_channels != 1152:
            print("Output channel size must be equal to 1152")
            raise ValueError
        kernel_size = int(sys.argv[6])
        if kernel_size != 14:
            print("Kernel size must be 14 right now.")
            raise ValueError
        trace_size = 0 if (len(sys.argv) != 8) else int(sys.argv[7])
    except ValueError:
        print("Argument has inappropriate value")

    conv2dk14(dev, width, height, in_channels, out_channels, kernel_size, trace_size)
