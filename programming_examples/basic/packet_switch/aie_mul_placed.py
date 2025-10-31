# packet_switch/aie_mul_placed.py -*- Python -*-
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


def packet_switch_kernel(dev, in_out_size):
    in_out_ty = np.dtype[np.int8]

    @device(dev)
    def device_body():
        # Define datatypes
        # Size of input vector
        vector_ty = np.ndarray[(in_out_size,), in_out_ty]
        # Size of input vector + 4 bytes for the packet header (used in memtile_0_1 DMA logic)
        vector_with_packet_ty = np.ndarray[(in_out_size + 4,), in_out_ty]

        add_func = external_func("add", [vector_ty, vector_ty])
        mult_func = external_func("mul", [vector_ty, vector_ty])

        ShimTile_0_0 = tile(0, 0)
        MemTile_0_1 = tile(0, 1)
        CT_0_2 = tile(0, 2)
        CT_0_3 = tile(0, 3)

        # Data allocations and synchronization with aie.buffer() and aie.lock()
        # (TODO: use objectfifo once it also supports packet_flow)
        # aie.buffer() and aie.lock() is the underhood implementation of aie.objectfifo, which can be
        # verified by running "aie-opt -aie-objectFifo-stateful-transform aie_add.mlir"
        # core_0_2
        core02_buff_in = buffer(tile=CT_0_2, datatype=vector_ty, name="core02_buff_in")
        core02_prod_lock_in = lock(
            tile=CT_0_2, lock_id=0, init=1, sym_name="core02_prod_lock_in"
        )
        core02_cons_lock_in = lock(
            tile=CT_0_2, lock_id=1, init=0, sym_name="core02_cons_lock_in"
        )
        core02_buff_out = buffer(
            tile=CT_0_2, datatype=vector_ty, name="core02_buff_out"
        )
        core02_prod_lock_out = lock(
            tile=CT_0_2, lock_id=2, init=1, sym_name="core02_prod_lock_out"
        )
        core02_cons_lock_out = lock(
            tile=CT_0_2, lock_id=3, init=0, sym_name="core02_cons_lock_out"
        )

        # core_0_3
        core03_buff_in = buffer(tile=CT_0_3, datatype=vector_ty, name="core03_buff_in")
        core03_prod_lock_in = aie.lock(
            CT_0_3, lock_id=0, init=1, sym_name="core03_prod_lock_in"
        )
        core03_cons_lock_in = aie.lock(
            CT_0_3, lock_id=1, init=0, sym_name="core03_cons_lock_in"
        )
        core03_buff_out = buffer(
            tile=CT_0_3, datatype=vector_ty, name="core03_buff_out"
        )
        core03_prod_lock_out = aie.lock(
            CT_0_3, lock_id=2, init=1, sym_name="core03_prod_lock_out"
        )
        core03_cons_lock_out = aie.lock(
            CT_0_3, lock_id=3, init=0, sym_name="core03_cons_lock_out"
        )

        # memtile_0_1
        mem01_buff_in = buffer(
            MemTile_0_1, datatype=vector_with_packet_ty, name="mem01_buff_in"
        )
        mem01_prod_lock_in = lock(
            MemTile_0_1, lock_id=0, init=1, sym_name="mem01_prod_lock_in"
        )
        mem01_cons_lock_in = lock(
            MemTile_0_1, lock_id=1, init=0, sym_name="mem01_cons_lock_in"
        )
        mem01_buff_out = buffer(
            tile=MemTile_0_1, datatype=vector_ty, name="mem01_buff_out"
        )
        mem01_prod_lock_out = lock(
            MemTile_0_1, lock_id=2, init=1, sym_name="mem01_prod_lock_out"
        )
        mem01_cons_lock_out = lock(
            MemTile_0_1, lock_id=3, init=0, sym_name="mem01_cons_lock_out"
        )

        # Setup packet flows
        # TODO: change the packet ids, because trace uses packet_id 0 and so forth
        packetflow(
            pkt_id=0,
            source=ShimTile_0_0,
            source_port=WireBundle.DMA,
            source_channel=0,
            dests={"dest": MemTile_0_1, "port": WireBundle.DMA, "channel": 0},
            keep_pkt_header=True,
        )
        packetflow(
            pkt_id=1,
            source=ShimTile_0_0,
            source_port=WireBundle.DMA,
            source_channel=0,
            dests={"dest": MemTile_0_1, "port": WireBundle.DMA, "channel": 0},
            keep_pkt_header=True,
        )
        packetflow(
            pkt_id=2,
            source=MemTile_0_1,
            source_port=WireBundle.DMA,
            source_channel=2,
            dests={"dest": ShimTile_0_0, "port": WireBundle.DMA, "channel": 0},
        )
        packetflow(
            pkt_id=0,
            source=MemTile_0_1,
            source_port=WireBundle.DMA,
            source_channel=0,
            dests={"dest": CT_0_2, "port": WireBundle.DMA, "channel": 0},
        )
        packetflow(
            pkt_id=4,
            source=CT_0_2,
            source_port=WireBundle.DMA,
            source_channel=0,
            dests={"dest": MemTile_0_1, "port": WireBundle.DMA, "channel": 2},
        )
        packetflow(
            pkt_id=1,
            source=MemTile_0_1,
            source_port=WireBundle.DMA,
            source_channel=0,
            dests={"dest": CT_0_3, "port": WireBundle.DMA, "channel": 0},
        )
        packetflow(
            pkt_id=6,
            source=CT_0_3,
            source_port=WireBundle.DMA,
            source_channel=0,
            dests={"dest": MemTile_0_1, "port": WireBundle.DMA, "channel": 2},
        )

        # core_0_2 compute
        @core(CT_0_2, "add_mul.o")
        def core_body():
            for _ in range_(sys.maxsize):
                # Acquire locks to read core02_buff_in and write core02_buff_out
                use_lock(core02_cons_lock_in, LockAction.AcquireGreaterEqual, value=1)
                use_lock(core02_prod_lock_out, LockAction.AcquireGreaterEqual, value=1)
                add_func(core02_buff_in, core02_buff_out)
                # Release locks to write core02_buff_in and read core02_buff_out
                use_lock(core02_prod_lock_in, LockAction.Release, value=1)
                use_lock(core02_cons_lock_out, LockAction.Release, value=1)

        # core_0_2 DMA logic
        @mem(CT_0_2)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
            with block[1]:
                # Write data from stream to core02_buff_in
                use_lock(core02_prod_lock_in, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(core02_buff_in)
                use_lock(core02_cons_lock_in, LockAction.Release, value=1)
                next_bd(block[1])
            with block[2]:
                s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
            with block[3]:
                # Write data from core02_buff_out + packet header to stream
                # Data is then routed to memtile_0_1
                use_lock(core02_cons_lock_out, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(core02_buff_out, packet=(0, 4))
                use_lock(core02_prod_lock_out, LockAction.Release, value=1)
                next_bd(block[3])
            with block[4]:
                EndOp()

        # core_0_3 compute
        @core(CT_0_3, "add_mul.o")
        def core_body():
            for _ in range_(sys.maxsize):
                # Acquire locks to read core03_buff_in and write core03_buff_out
                use_lock(core03_cons_lock_in, LockAction.AcquireGreaterEqual, value=1)
                use_lock(core03_prod_lock_out, LockAction.AcquireGreaterEqual, value=1)
                mult_func(core03_buff_in, core03_buff_out)
                # Release locks to write core03_buff_in and read core03_buff_out
                use_lock(core03_prod_lock_in, LockAction.Release, value=1)
                use_lock(core03_cons_lock_out, LockAction.Release, value=1)

        # core_0_3 DMA logic
        @mem(CT_0_3)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
            with block[1]:
                # Write data from stream to core03_buff_in
                use_lock(core03_prod_lock_in, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(core03_buff_in)
                use_lock(core03_cons_lock_in, LockAction.Release, value=1)
                next_bd(block[1])
            with block[2]:
                s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
            with block[3]:
                # Write data from core03_buff_out + packet header to stream
                # Data is then routed to memtile_0_1
                use_lock(core03_cons_lock_out, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(core03_buff_out, packet=(0, 6))
                use_lock(core03_prod_lock_out, LockAction.Release, value=1)
                next_bd(block[3])
            with block[4]:
                EndOp()

        # memtile_0_1 DMA logic
        @memtile_dma(MemTile_0_1)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
            with block[1]:
                # Write data from stream to mem01_buff_in
                use_lock(mem01_prod_lock_in, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(
                    mem01_buff_in
                )  # First 4 byte is the packet header becasue in packet_flow(pkt_id=0) configured "keep_pkt_header"
                use_lock(mem01_cons_lock_in, LockAction.Release, value=1)
                next_bd(block[1])
            with block[2]:
                s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
            with block[3]:
                # Write data from mem01_buff_in to stream (no additional packet header!)
                # Sends the message to corresponding ComputeTile core. This works because the packet header from
                # Shimtile to Memtile is saved in the buffer and the packet_flow from Memtile to ComputeTile uses
                # the same packet_id (0 or 1).
                use_lock(mem01_cons_lock_in, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(mem01_buff_in)
                use_lock(mem01_prod_lock_in, LockAction.Release, value=1)
                next_bd(block[3])
            with block[4]:
                s2 = dma_start(DMAChannelDir.S2MM, 2, dest=block[5], chain=block[6])
            with block[5]:
                # Write data from stream to mem01_buff_out
                use_lock(mem01_prod_lock_out, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(mem01_buff_out)
                use_lock(mem01_cons_lock_out, LockAction.Release, value=1)
                next_bd(block[5])
            with block[6]:
                s3 = dma_start(DMAChannelDir.MM2S, 2, dest=block[7], chain=block[8])
            with block[7]:
                # Write data from mem01_buff_out + packet header to stream
                # Data is then routed to shimtile_0_0
                use_lock(mem01_cons_lock_out, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(mem01_buff_out, packet=(0, 2))
                use_lock(mem01_prod_lock_out, LockAction.Release, value=1)
                next_bd(block[7])
            with block[8]:
                EndOp()

        # Data movement to / from NPU
        @runtime_sequence(
            np.ndarray[(in_out_size,), in_out_ty], np.ndarray[(in_out_size,), in_out_ty]
        )
        def sequence(A, B):
            # Write data from host buffer A + packet header to stream
            in_task = dma_configure_task(ShimTile_0_0, DMAChannelDir.MM2S, 0)
            with bds(in_task) as bd:
                with bd[0]:
                    shim_dma_bd(
                        A,
                        offset=0,
                        sizes=[1, 1, 1, in_out_size],
                        strides=[0, 0, 0, 1],
                        packet=(0, 1),
                    )
                    EndOp()
            # Write data from stream to host buffer B
            out_task = dma_configure_task(
                ShimTile_0_0, DMAChannelDir.S2MM, 0, issue_token=True
            )
            with bds(out_task) as bd:
                with bd[0]:
                    shim_dma_bd(
                        B,
                        offset=0,
                        sizes=[1, 1, 1, in_out_size],
                        strides=[0, 0, 0, 1],
                    )
                    EndOp()
            dma_start_task(in_task, out_task)
            dma_await_task(out_task)


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
