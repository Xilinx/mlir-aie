# packet_switch/aie_add_placed.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
from curses import meta
from operator import le, ne
from re import I
from struct import pack
from matplotlib import use
import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

import aie.utils.trace as trace_utils


from aie.dialects import memref, vector


# In this example, it calls low-level python->mlir translation api.
# See example such as https://github.com/Xilinx/mlir-aie/blob/dfe7d1d532ba8c4ac88da72ee77bc2591bda8943/test/npu-xrt/dma_task_large_linear/aie2.py
def packet_switch_kernel(dev, in_out_size):
    in_out_ty = np.dtype[np.int8]  # input data type

    @device(dev)
    def device_body():
        # define types
        vector_ty = np.ndarray[(in_out_size,), in_out_ty]  # Size of input vector

        memref.global_(
            "objFifo_in0", T.memref(in_out_size, T.i8()), sym_visibility="public"
        )
        memref.global_(
            "objFifo_out0", T.memref(in_out_size, T.i8()), sym_visibility="public"
        )
        memref.global_(
            "objFifo_in1", T.memref(in_out_size, T.i8()), sym_visibility="public"
        )
        memref.global_(
            "objFifo_out1", T.memref(in_out_size, T.i8()), sym_visibility="public"
        )

        # Declare the kernel functions that will be linked to at compile stage
        add_func = external_func("add", [vector_ty, vector_ty])
        mult_func = external_func("mul", [vector_ty, vector_ty])

        ShimTile_0_0 = tile(0, 0)
        MemTile_0_1 = tile(0, 1)
        CT_0_2 = tile(0, 2)
        CT_0_3 = tile(0, 3)

        # core_0_2
        objFifo_core02_cons_buff_0 = buffer(
            tile=CT_0_2, datatype=vector_ty, name="objFifo_core02_cons_buff_0"
        )
        objFifo_core02_buff_0 = buffer(
            tile=CT_0_2, datatype=vector_ty, name="objFifo_core02_buff_0"
        )

        # Instead of using aie.objectfifo, it use aie.buffer().
        # aie.buffer() is the underhood implementation of aie.objectfifo, which can be verified by running
        # "aie-opt -aie-objectFifo-stateful-transform aie_add.mlir"
        # TODO: use objectfifo once it also support packet_flow
        objFifo_core02_cons_prod_lock = lock(
            tile=CT_0_2, lock_id=0, init=1, sym_name="objFifo_core02_cons_prod_lock"
        )
        objFifo_core02_cons_cons_lock = lock(
            tile=CT_0_2, lock_id=1, init=0, sym_name="objFifo_core02_cons_cons_lock"
        )
        objFifo_core02_prod_lock = lock(
            tile=CT_0_2, lock_id=2, init=1, sym_name="objFifo_core02_prod_lock"
        )
        objFifo_core02_cons_lock = lock(
            tile=CT_0_2, lock_id=3, init=0, sym_name="objFifo_core02_cons_lock"
        )

        # core_0_3
        objFifo_core03_cons_buff_0 = buffer(
            tile=CT_0_3, datatype=vector_ty, name="objFifo_core03_cons_buff_0"
        )
        objFifo_core03_buff_0 = buffer(
            tile=CT_0_3, datatype=vector_ty, name="objFifo_core03_buff_0"
        )

        objFifo_core03_cons_prod_lock = aie.lock(
            CT_0_3, lock_id=0, init=1, sym_name="objFifo_core03_cons_prod_lock"
        )
        objFifo_core03_cons_cons_lock = aie.lock(
            CT_0_3, lock_id=1, init=0, sym_name="objFifo_core03_cons_cons_lock"
        )
        objFifo_core03_prod_lock = aie.lock(
            CT_0_3, lock_id=2, init=1, sym_name="objFifo_core03_prod_lock"
        )
        objFifo_core03_cons_lock = aie.lock(
            CT_0_3, lock_id=3, init=0, sym_name="objFifo_core03_cons_lock"
        )

        # TODO: probably another issue to change the packet id. Because when doing trace, it use packet_id 0 an so forth?
        # Configure the packet flow path
        packetflow(
            pkt_id=0,
            source=ShimTile_0_0,
            source_port=WireBundle.DMA,
            source_channel=0,
            dest=MemTile_0_1,
            dest_port=WireBundle.DMA,
            dest_channel=0,
            keep_pkt_header=True,  # By keeping the pkt_header, the 4Byte(packet header) will not be dropped when Memtile receives
        )
        packetflow(
            pkt_id=1,
            source=ShimTile_0_0,
            source_port=WireBundle.DMA,
            source_channel=0,
            dest=MemTile_0_1,
            dest_port=WireBundle.DMA,
            dest_channel=0,
            keep_pkt_header=True,
        )
        packetflow(
            pkt_id=2,
            source=MemTile_0_1,
            source_port=WireBundle.DMA,
            source_channel=2,
            dest=ShimTile_0_0,
            dest_port=WireBundle.DMA,
            dest_channel=0,
        )
        packetflow(
            pkt_id=0,
            source=MemTile_0_1,
            source_port=WireBundle.DMA,
            source_channel=0,
            dest=CT_0_2,
            dest_port=WireBundle.DMA,
            dest_channel=0,
        )
        packetflow(
            pkt_id=4,
            source=CT_0_2,
            source_port=WireBundle.DMA,
            source_channel=0,
            dest=MemTile_0_1,
            dest_port=WireBundle.DMA,
            dest_channel=2,
        )
        packetflow(
            pkt_id=1,
            source=MemTile_0_1,
            source_port=WireBundle.DMA,
            source_channel=0,
            dest=CT_0_3,
            dest_port=WireBundle.DMA,
            dest_channel=0,
        )
        packetflow(
            pkt_id=6,
            source=CT_0_3,
            source_port=WireBundle.DMA,
            source_channel=0,
            dest=MemTile_0_1,
            dest_port=WireBundle.DMA,
            dest_channel=2,
        )

        @core(CT_0_2, "add_mul.o")
        def core_body():
            for _ in range_(sys.maxsize):
                use_lock(
                    objFifo_core02_cons_cons_lock,
                    LockAction.AcquireGreaterEqual,
                    value=1,
                )
                use_lock(
                    objFifo_core02_prod_lock, LockAction.AcquireGreaterEqual, value=1
                )
                add_func(objFifo_core02_cons_buff_0, objFifo_core02_buff_0)
                use_lock(objFifo_core02_cons_prod_lock, LockAction.Release, value=1)
                use_lock(objFifo_core02_cons_lock, LockAction.Release, value=1)

        @mem(CT_0_2)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(
                    objFifo_core02_cons_prod_lock,
                    LockAction.AcquireGreaterEqual,
                    value=1,
                )
                dma_bd(objFifo_core02_cons_buff_0)
                use_lock(objFifo_core02_cons_cons_lock, LockAction.Release, value=1)
                next_bd(block[1])
            with block[2]:
                s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
            with block[3]:
                use_lock(
                    objFifo_core02_cons_lock, LockAction.AcquireGreaterEqual, value=1
                )
                dma_bd(objFifo_core02_buff_0, packet=(0, 4))
                use_lock(objFifo_core02_prod_lock, LockAction.Release, value=1)
                next_bd(block[3])
            with block[4]:
                EndOp()

        @core(CT_0_3, "add_mul.o")
        def core_body():
            for _ in range_(start=sys.maxsize):  # Infinity loop.
                use_lock(
                    objFifo_core03_cons_cons_lock,
                    LockAction.AcquireGreaterEqual,
                    value=1,
                )
                use_lock(
                    objFifo_core03_prod_lock, LockAction.AcquireGreaterEqual, value=1
                )
                mult_func(objFifo_core03_cons_buff_0, objFifo_core03_buff_0)
                use_lock(objFifo_core03_cons_prod_lock, LockAction.Release, value=1)
                use_lock(objFifo_core03_cons_lock, LockAction.Release, value=1)

        # DMA logic for CT 0_3
        @mem(CT_0_3)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(
                    objFifo_core03_cons_prod_lock,
                    LockAction.AcquireGreaterEqual,
                    value=1,
                )
                dma_bd(
                    objFifo_core03_cons_buff_0,
                )
                use_lock(objFifo_core03_cons_cons_lock, LockAction.Release, value=1)
                next_bd(block[1])  # goto block[1]
            with block[2]:
                s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
            with block[3]:
                use_lock(
                    objFifo_core03_cons_lock, LockAction.AcquireGreaterEqual, value=1
                )
                dma_bd(objFifo_core03_buff_0, packet=(0, 6))
                use_lock(objFifo_core03_prod_lock, LockAction.Release, value=1)
                next_bd(block[3])
            with block[4]:
                EndOp()

        shim_dma_allocation("objFifo_in0", DMAChannelDir.MM2S, 0, 0)

        @runtime_sequence(
            np.ndarray[(in_out_size,), in_out_ty], np.ndarray[(in_out_size,), in_out_ty]
        )
        def sequence(A, B):
            in_task = shim_dma_single_bd_task(
                "objFifo_in0",
                A,
                offset=0,
                sizes=[1, 1, 1, in_out_size],
                strides=[0, 0, 0, 1],
                packet=(0, 0),
            )
            out_task = shim_dma_single_bd_task(
                "objFifo_out0",
                B,
                offset=0,
                sizes=[1, 1, 1, in_out_size],
                strides=[0, 0, 0, 1],
                issue_token=True,
            )
            dma_start_task(in_task, out_task)
            dma_await_task(out_task)

        shim_dma_allocation("objFifo_out0", DMAChannelDir.S2MM, 0, 0)
        shim_dma_allocation("objFifo_out1", DMAChannelDir.S2MM, 0, 0)

        # The extract 4 byte is for packet header.
        vector_with_packet_ty = np.ndarray[(in_out_size + 4,), in_out_ty]

        objFifo_in0_cons_buff_0 = buffer(
            MemTile_0_1, datatype=vector_with_packet_ty, name="objFifo_in0_cons_buff_0"
        )
        objFifo_out0_buff_0 = buffer(
            tile=MemTile_0_1, datatype=vector_ty, name="objFifo_out0_buff_0"
        )

        objFifo_in0_cons_prod_lock = lock(
            MemTile_0_1, lock_id=0, init=1, sym_name="objFifo_in0_cons_prod_lock"
        )
        objFifo_in0_cons_cons_lock = lock(
            MemTile_0_1, lock_id=1, init=0, sym_name="objFifo_in0_cons_cons_lock"
        )
        objFifo_out0_prod_lock = lock(
            MemTile_0_1, lock_id=2, init=1, sym_name="objFifo_out0_prod_lock"
        )
        objFifo_out0_cons_lock = lock(
            MemTile_0_1, lock_id=3, init=0, sym_name="objFifo_out0_cons_lock"
        )

        # MemTile dma logic
        @memtile_dma(MemTile_0_1)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(
                    objFifo_in0_cons_prod_lock, LockAction.AcquireGreaterEqual, value=1
                )
                dma_bd(
                    objFifo_in0_cons_buff_0
                )  # First 4 bye is the packet header becasue in packet_flow(pkt_id=0) configured "keep_pkt_header"
                use_lock(objFifo_in0_cons_cons_lock, LockAction.Release, value=1)
                next_bd(block[1])  # goto block[1]
            with block[2]:
                s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
            with block[3]:
                use_lock(
                    objFifo_in0_cons_cons_lock, LockAction.AcquireGreaterEqual, value=1
                )
                # Send the message to corresponding CT. This works because the packet header from Shimtile to Memtile
                # This works because the packet header from Shimtile to Memtile is saved in the buffer and the
                # packet_flow from Memtile to ComputeTile uses the same packet_id (0 or 1)
                dma_bd(objFifo_in0_cons_buff_0)
                use_lock(objFifo_in0_cons_prod_lock, LockAction.Release, value=1)
                next_bd(block[3])
            with block[4]:
                s2 = dma_start(DMAChannelDir.MM2S, 2, dest=block[5], chain=block[6])
            with block[5]:
                use_lock(
                    objFifo_out0_cons_lock, LockAction.AcquireGreaterEqual, value=1
                )
                dma_bd(objFifo_out0_buff_0, packet=(0, 2))
                use_lock(objFifo_out0_prod_lock, LockAction.Release, value=1)
                next_bd(block[5])
            with block[6]:
                s3 = dma_start(DMAChannelDir.S2MM, 2, dest=block[7], chain=block[8])
            with block[7]:
                use_lock(
                    objFifo_out0_prod_lock, LockAction.AcquireGreaterEqual, value=1
                )
                dma_bd(objFifo_out0_buff_0)
                use_lock(objFifo_out0_cons_lock, LockAction.Release, value=1)
                next_bd(block[7])
            with block[8]:
                EndOp()


if len(sys.argv) < 3:
    raise ValueError("[ERROR] Need at least 2 arguments (dev)")


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = AIEDevice.npu1_1col
elif opts.device == "npu2":
    dev = AIEDevice.npu2
elif opts.device == "npu2_1":
    dev = AIEDevice.npu2_1col
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

with mlir_mod_ctx() as ctx:
    packet_switch_kernel(dev, in_out_size=256)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
