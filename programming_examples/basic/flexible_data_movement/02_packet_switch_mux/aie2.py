# 02_packet_switch_mux/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
# Prototype 2: Packet-Switched Channel Multiplexing
#
# Demonstrates multiplexing 4 logical data streams over 1 physical shimDMA
# channel using packet IDs. MemTile routes packets by ID to 4 different cores.
#
# Data flow:
#   DDR -[1 shimDMA MM2S ch0, pkt_id=0..3]-> MemTile -[packet route]-> 4 cores
#   4 cores -[pkt_id=4..7]-> MemTile -[pkt_id=8]-> shimDMA S2MM ch0 -> DDR
#
# Each core receives unique data (256 bytes), adds 1, and sends result back.
# All 4 logical input streams share 1 physical shimDMA MM2S channel.
#
# ShimDMA channels used: 1 MM2S + 1 S2MM = 2 of 4 total
# Key win: 4 cores fed through 1 physical channel via packet multiplexing

import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_


def packet_switch_mux(dev, n_cores=2, chunk_size=256):
    """
    Multiplex n_cores data streams over 1 shimDMA channel using packet IDs.
    """
    total_size = n_cores * chunk_size
    dtype = np.dtype[np.uint8]

    @device(dev)
    def device_body():
        chunk_ty = np.ndarray[(chunk_size,), dtype]
        # MemTile buffer type: chunk + 4 bytes for packet header
        chunk_with_pkt_ty = np.ndarray[(chunk_size + 4,), dtype]
        full_ty = np.ndarray[(total_size,), dtype]

        add_func = external_func(
            "passThroughLine",
            [chunk_ty, chunk_ty, np.int32],
            link_with="passThrough.cc.o",
        )

        ShimTile = tile(0, 0)
        MemTile_0_1 = tile(0, 1)
        cores = [tile(0, 2 + i) for i in range(n_cores)]

        # --- Buffers and locks for each core ---
        core_bufs_in = []
        core_locks_prod_in = []
        core_locks_cons_in = []
        core_bufs_out = []
        core_locks_prod_out = []
        core_locks_cons_out = []

        for i in range(n_cores):
            buf_in = buffer(cores[i], chunk_ty, name=f"core{i}_buf_in")
            prod_in = lock(cores[i], lock_id=0, init=1, sym_name=f"core{i}_prod_in")
            cons_in = lock(cores[i], lock_id=1, init=0, sym_name=f"core{i}_cons_in")
            buf_out = buffer(cores[i], chunk_ty, name=f"core{i}_buf_out")
            prod_out = lock(cores[i], lock_id=2, init=1, sym_name=f"core{i}_prod_out")
            cons_out = lock(cores[i], lock_id=3, init=0, sym_name=f"core{i}_cons_out")
            core_bufs_in.append(buf_in)
            core_locks_prod_in.append(prod_in)
            core_locks_cons_in.append(cons_in)
            core_bufs_out.append(buf_out)
            core_locks_prod_out.append(prod_out)
            core_locks_cons_out.append(cons_out)

        # --- MemTile buffers: 1 input (with packet header), 1 output ---
        mem_buf_in = buffer(MemTile_0_1, chunk_with_pkt_ty, name="mem_buf_in")
        mem_lock_prod_in = lock(MemTile_0_1, lock_id=0, init=1, sym_name="mem_prod_in")
        mem_lock_cons_in = lock(MemTile_0_1, lock_id=1, init=0, sym_name="mem_cons_in")
        mem_buf_out = buffer(MemTile_0_1, chunk_ty, name="mem_buf_out")
        mem_lock_prod_out = lock(
            MemTile_0_1, lock_id=2, init=1, sym_name="mem_prod_out"
        )
        mem_lock_cons_out = lock(
            MemTile_0_1, lock_id=3, init=0, sym_name="mem_cons_out"
        )

        # --- Packet flows ---
        # NOTE: Avoid packet ID 0 (reserved for trace subsystem)
        # Input IDs: 1..n_cores, Return IDs: n_cores+1..2*n_cores, Output: 2*n_cores+1

        # ShimTile -> MemTile (all packets on same shim DMA channel 0)
        for i in range(n_cores):
            packetflow(
                pkt_id=i + 1,
                source=ShimTile,
                source_port=WireBundle.DMA,
                source_channel=0,
                dests={"dest": MemTile_0_1, "port": WireBundle.DMA, "channel": 0},
                keep_pkt_header=True,
            )

        # MemTile -> each core (packet ID demux)
        for i in range(n_cores):
            packetflow(
                pkt_id=i + 1,
                source=MemTile_0_1,
                source_port=WireBundle.DMA,
                source_channel=0,
                dests={"dest": cores[i], "port": WireBundle.DMA, "channel": 0},
            )

        # Each core -> MemTile (results back, unique packet IDs)
        for i in range(n_cores):
            packetflow(
                pkt_id=n_cores + 1 + i,
                source=cores[i],
                source_port=WireBundle.DMA,
                source_channel=0,
                dests={
                    "dest": MemTile_0_1,
                    "port": WireBundle.DMA,
                    "channel": 2,
                },
            )

        # MemTile -> ShimTile (results to DDR)
        packetflow(
            pkt_id=2 * n_cores + 1,
            source=MemTile_0_1,
            source_port=WireBundle.DMA,
            source_channel=2,
            dests={"dest": ShimTile, "port": WireBundle.DMA, "channel": 0},
        )

        # --- Core logic ---
        for i in range(n_cores):

            def make_core_fn(idx):
                @core(cores[idx])
                def core_body():
                    for _ in range_(sys.maxsize):
                        use_lock(
                            core_locks_cons_in[idx],
                            LockAction.AcquireGreaterEqual,
                            value=1,
                        )
                        use_lock(
                            core_locks_prod_out[idx],
                            LockAction.AcquireGreaterEqual,
                            value=1,
                        )
                        add_func(
                            core_bufs_in[idx],
                            core_bufs_out[idx],
                            chunk_size,
                        )
                        use_lock(
                            core_locks_prod_in[idx],
                            LockAction.Release,
                            value=1,
                        )
                        use_lock(
                            core_locks_cons_out[idx],
                            LockAction.Release,
                            value=1,
                        )

            make_core_fn(i)

        # --- Core DMA logic ---
        for i in range(n_cores):

            def make_mem_fn(idx):
                @mem(cores[idx])
                def m(block):
                    # S2MM ch0: stream -> core input buffer
                    s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
                    with block[1]:
                        use_lock(
                            core_locks_prod_in[idx],
                            LockAction.AcquireGreaterEqual,
                            value=1,
                        )
                        dma_bd(core_bufs_in[idx])
                        use_lock(
                            core_locks_cons_in[idx],
                            LockAction.Release,
                            value=1,
                        )
                        next_bd(block[1])
                    with block[2]:
                        # MM2S ch0: core output buffer -> stream (with packet header)
                        s1 = dma_start(
                            DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4]
                        )
                    with block[3]:
                        use_lock(
                            core_locks_cons_out[idx],
                            LockAction.AcquireGreaterEqual,
                            value=1,
                        )
                        dma_bd(
                            core_bufs_out[idx],
                            packet=(0, n_cores + 1 + idx),
                        )
                        use_lock(
                            core_locks_prod_out[idx],
                            LockAction.Release,
                            value=1,
                        )
                        next_bd(block[3])
                    with block[4]:
                        EndOp()

            make_mem_fn(i)

        # --- MemTile DMA logic ---
        @memtile_dma(MemTile_0_1)
        def m(block):
            # S2MM ch0: receive from shim (with packet header preserved)
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(
                    mem_lock_prod_in,
                    LockAction.AcquireGreaterEqual,
                    value=1,
                )
                dma_bd(mem_buf_in)
                use_lock(mem_lock_cons_in, LockAction.Release, value=1)
                next_bd(block[1])
            with block[2]:
                # MM2S ch0: forward to cores (packet header in buffer does routing)
                s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
            with block[3]:
                use_lock(
                    mem_lock_cons_in,
                    LockAction.AcquireGreaterEqual,
                    value=1,
                )
                dma_bd(mem_buf_in)
                use_lock(mem_lock_prod_in, LockAction.Release, value=1)
                next_bd(block[3])
            with block[4]:
                # S2MM ch2: receive results from cores
                s2 = dma_start(DMAChannelDir.S2MM, 2, dest=block[5], chain=block[6])
            with block[5]:
                use_lock(
                    mem_lock_prod_out,
                    LockAction.AcquireGreaterEqual,
                    value=1,
                )
                dma_bd(mem_buf_out)
                use_lock(mem_lock_cons_out, LockAction.Release, value=1)
                next_bd(block[5])
            with block[6]:
                # MM2S ch2: forward results to shim (with packet header)
                s3 = dma_start(DMAChannelDir.MM2S, 2, dest=block[7], chain=block[8])
            with block[7]:
                use_lock(
                    mem_lock_cons_out,
                    LockAction.AcquireGreaterEqual,
                    value=1,
                )
                dma_bd(mem_buf_out, packet=(0, 2 * n_cores + 1))
                use_lock(mem_lock_prod_out, LockAction.Release, value=1)
                next_bd(block[7])
            with block[8]:
                EndOp()

        # --- Runtime sequence ---
        @runtime_sequence(full_ty, full_ty)
        def sequence(A, B):
            # Send each core's data as a separate packet
            for i in range(n_cores):
                in_task = dma_configure_task(ShimTile, DMAChannelDir.MM2S, 0)
                with bds(in_task) as bd:
                    with bd[0]:
                        shim_dma_bd(
                            A,
                            offset=i * chunk_size,
                            sizes=[1, 1, 1, chunk_size],
                            strides=[0, 0, 0, 1],
                            packet=(0, i + 1),
                        )
                        EndOp()

                out_task = dma_configure_task(
                    ShimTile,
                    DMAChannelDir.S2MM,
                    0,
                    issue_token=True,
                )
                with bds(out_task) as bd:
                    with bd[0]:
                        shim_dma_bd(
                            B,
                            offset=i * chunk_size,
                            sizes=[1, 1, 1, chunk_size],
                            strides=[0, 0, 0, 1],
                        )
                        EndOp()
                dma_start_task(in_task, out_task)
                dma_await_task(out_task)
                dma_free_task(in_task)
                dma_free_task(out_task)


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=False, default="npu", dest="device")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = AIEDevice.npu1_1col
elif opts.device == "npu2":
    dev = AIEDevice.npu2
else:
    raise ValueError(f"[ERROR] Unknown device: {opts.device}")

with mlir_mod_ctx() as ctx:
    packet_switch_mux(dev)
    res = ctx.module.operation.verify()
    if res is True:
        print(ctx.module)
    else:
        print(res)
