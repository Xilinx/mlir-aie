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


def conv2dk14(
    dev, width: int, height: int, in_channels: int, out_channels: int, kernel_size: int, trace_size: int
):
    with mlir_mod_ctx() as ctx:

        # Kernel transfer sizes
        actIn = 16 * kernel_size * kernel_size * in_channels # 16 tiles
        # bufIn = actIn * 4 # 64 tiles (1 tile row)
        bufIn = width * kernel_size * in_channels # 64 tiles (1 tile row)

        weights = 16 * kernel_size * kernel_size * in_channels
        bufWeights = weights * (out_channels/ 16)

        actOut = 16 * 16
        # bufOut = actOut * 2  # double buffer
        bufOut = (64*64*out_channels)

        # we reload inputs 72 times (not big enough to keep in memtile)
        tensorInSize = width * height * in_channels * (out_channels / 16)
        # tensorOutSize = (width/kernel_size) * (height/kernel_size) * out_channels
        tensorOutSize = bufOut

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
                ]
            )

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)

            lock2 = lock(ComputeTile2, init=0)

            # AIE-array data movement with object fifos
            # Input
            of_inOF_act_L3L2 = object_fifo(
                "inOF_act_L3L2", ShimTile, MemTile, 2, 
                np.ndarray[(kernel_size, width*in_channels), np.dtype[np.int8]],
                dimensionsToStream = None,
                dimensionsFromStreamPerConsumer =
                [
                    (kernel_size,kernel_size*in_channels),
                    (64, kernel_size*kernel_size*in_channels),
                    (kernel_size*in_channels,1),
                ], 
            )
            of_act_L2_02 = object_fifo(
                "act_L2_02", MemTile, ComputeTile2, 2,
                np.ndarray[(kernel_size, (width/4)*in_channels), np.dtype[np.int8]],
                # [
                #     (4,16*kernel_size*in_channels),
                #     (16,kernel_size*in_channels),
                #     (kernel_size,width*in_channels),
                #     (kernel_size*in_channels,1),
                # ], 
                dimensionsToStream =
                [
                    (2,kernel_size*kernel_size*in_channels*8),
                    # (kernel_size*kernel_size/2,2*in_channels),
                    (98,2*in_channels),
                    (8, kernel_size*kernel_size*in_channels),
                    (2*in_channels, 1),
                ],
            )
            object_fifo_link(of_inOF_act_L3L2, of_act_L2_02)

            # wts
            of_inOF_wts_0_L3L2 = object_fifo(
                "inOF_wts_0_L3L2", ShimTile, [ComputeTile2], 2, weights_ty
            )

            # Output
            of_out_02_L2 = object_fifo("out_02_L2", ComputeTile2, [MemTile], 2,
                np.ndarray[(16, 16), np.dtype[np.int8]],
                dimensionsFromStreamPerConsumer =
                [
                    (2,128),
                    (8,1),
                    (2,64), 
                    (8,8),                                                
                ],
            )
            of_outOFL2L3 = object_fifo("outOFL2L3", MemTile, [ShimTile], 2, 
                np.ndarray[(1152, 4096), np.dtype[np.int8]],
                dimensionsToStream =
                [
                    (16,16),
                    (64,256),
                    (16,1)
                ],
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
            @core(ComputeTile2, "conv2dk14_i8.o", stack_size=0x600)
            def core_body():
                # y_dim = height // kernel_size
                y_dim = 64
                # x_dim = width
                x_blocks = 4
                # x_dim = width * in_channels // x_blocks
                x_dim = 14*16*4
                ci = in_channels
                co = out_channels
                # co16 = out_channels // 16
                co16 = 72

                for _ in range_(0xFFFFFFFF):
                    use_lock(lock2, LockAction.Acquire, value=1)
                    scale = rtp2[0]

                    for _ in range_(co16):
                        elemWts = of_inOF_wts_0_L3L2.acquire(ObjectFifoPort.Consume, 1)

                        for _ in range_(y_dim):
                            for _ in range_(x_blocks):
                                elemIn = of_act_L2_02.acquire(ObjectFifoPort.Consume, 1)
                                elemOut0 = of_out_02_L2.acquire(ObjectFifoPort.Produce, 1)
                                conv2dk14_i8(elemIn, elemWts, elemOut0, x_dim, ci, co, kernel_size, scale)
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
        kernel_size = int(sys.argv[6])
        if kernel_size != 14:
            print(
                "Kernel size must be 14 right now."
            )
            raise ValueError
        trace_size = 0 if (len(sys.argv) != 8) else int(sys.argv[7])
    except ValueError:
        print("Argument has inappropriate value")

    out_channels = 1152 # TODO why do we need this here?
    kernel_size = 14  # TODO why do we need this here?
    trace_size = 16384  # TODO why do we need this here?

    conv2dk14(dev, width, height, in_channels, out_channels, kernel_size, trace_size)
