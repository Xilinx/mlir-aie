# packet_switch/packet_switch.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Packet-switched two-kernel demo — iron explicit-routing primitives.

A single shim → memtile → compute → memtile → shim pipeline with two
compute cores (one running ``add``, one running ``mul``).  Which core
handles a given input is selected by the input shim DMA's packet ID;
the ``--op`` flag picks which packet ID to use:

  * ``--op add`` → packet (0, 0) → routes to core_0_2 (add)
  * ``--op mul`` → packet (0, 1) → routes to core_0_3 (mul)

The whole topology is expressed in iron-level primitives:
:class:`PacketFlow` for the routes (with explicit ``pkt_id`` + the
``keep_pkt_header=True`` knob ObjectFifo doesn't expose),
:class:`TileDma` for each tile's DMA program (compute tile + memtile),
:class:`Worker` for the compute body, plus iron :class:`Buffer` and
:class:`Lock` for shared state.  The runtime sequence still needs the
dialect-level ``shim_dma_bd(packet=...)`` primitive to stamp the input
packet ID, so use ``rt.inline_ops`` as the escape hatch there.

Two invocation modes:

  * compile-only: ``... --op add --xclbin-path=PATH --insts-path=PATH``
  * emit-MLIR:    ``... --op add --emit-mlir``
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
    DmaChannel,
    ExternalFunction,
    Lock,
    PacketFlow,
    Program,
    Release,
    Runtime,
    TileDma,
    Worker,
)
from aie.iron.device import NPU1Col1, NPU2, NPU2Col1, Tile
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir, WireBundle
from aie.dialects.aie import EndOp
from aie.dialects.aiex import (
    bds,
    dma_await_task,
    dma_configure_task,
    dma_start_task,
    shim_dma_bd,
)
from aie.utils.compile import compile_mlir_module


_OP_PACKET_ID = {"add": 0, "mul": 1}


def _device_for(dev_str: str):
    if dev_str == "npu":
        return NPU1Col1()
    if dev_str == "npu2":
        return NPU2()
    if dev_str == "npu2_1":
        return NPU2Col1()
    raise ValueError(f"[ERROR] Device name {dev_str!r} is unknown")


def _build_program(dev, in_out_size: int, input_packet_id: int):
    in_out_ty = np.dtype[np.int8]
    vector_ty = np.ndarray[(in_out_size,), in_out_ty]
    # +4 bytes for the kept packet header at the memtile.
    vector_with_packet_ty = np.ndarray[(in_out_size + 4,), in_out_ty]

    # Pin tile coordinates to match the original placed design exactly.
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    memtile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    ct_0_2 = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)
    ct_0_3 = Tile(col=0, row=3, tile_type=AIETileType.CoreTile)

    # ----- Compute tile 0,2 (add) -----
    c02_buf_in = Buffer(type=vector_ty, name="core02_buff_in")
    c02_buf_out = Buffer(type=vector_ty, name="core02_buff_out")
    c02_prod_lock_in = Lock(tile=ct_0_2, lock_id=0, init=1, name="core02_prod_lock_in")
    c02_cons_lock_in = Lock(tile=ct_0_2, lock_id=1, init=0, name="core02_cons_lock_in")
    c02_prod_lock_out = Lock(tile=ct_0_2, lock_id=2, init=1, name="core02_prod_lock_out")
    c02_cons_lock_out = Lock(tile=ct_0_2, lock_id=3, init=0, name="core02_cons_lock_out")

    # ----- Compute tile 0,3 (mul) -----
    c03_buf_in = Buffer(type=vector_ty, name="core03_buff_in")
    c03_buf_out = Buffer(type=vector_ty, name="core03_buff_out")
    c03_prod_lock_in = Lock(tile=ct_0_3, lock_id=0, init=1, name="core03_prod_lock_in")
    c03_cons_lock_in = Lock(tile=ct_0_3, lock_id=1, init=0, name="core03_cons_lock_in")
    c03_prod_lock_out = Lock(tile=ct_0_3, lock_id=2, init=1, name="core03_prod_lock_out")
    c03_cons_lock_out = Lock(tile=ct_0_3, lock_id=3, init=0, name="core03_cons_lock_out")

    # ----- Memtile 0,1 -----
    mem_buf_in = Buffer(type=vector_with_packet_ty, name="mem01_buff_in", tile=memtile)
    mem_buf_out = Buffer(type=vector_ty, name="mem01_buff_out", tile=memtile)
    mem_prod_lock_in = Lock(tile=memtile, lock_id=0, init=1, name="mem01_prod_lock_in")
    mem_cons_lock_in = Lock(tile=memtile, lock_id=1, init=0, name="mem01_cons_lock_in")
    mem_prod_lock_out = Lock(tile=memtile, lock_id=2, init=1, name="mem01_prod_lock_out")
    mem_cons_lock_out = Lock(tile=memtile, lock_id=3, init=0, name="mem01_cons_lock_out")

    # ----- External kernels (compiled from add_mul.cc) -----
    _kernel_src = str(Path(__file__).parent / "add_mul.cc")
    add_func = ExternalFunction(
        "add",
        object_file_name="add_mul.o",
        source_file=_kernel_src,
        arg_types=[vector_ty, vector_ty],
    )
    mul_func = ExternalFunction(
        "mul",
        object_file_name="add_mul.o",
        source_file=_kernel_src,
        arg_types=[vector_ty, vector_ty],
    )

    # ----- Worker bodies -----
    def c02_body(buf_in, buf_out, prod_in, cons_in, prod_out, cons_out, add):
        cons_in.acquire()
        prod_out.acquire()
        add(buf_in, buf_out)
        prod_in.release()
        cons_out.release()

    c02_worker = Worker(
        c02_body,
        fn_args=[
            c02_buf_in, c02_buf_out,
            c02_prod_lock_in, c02_cons_lock_in,
            c02_prod_lock_out, c02_cons_lock_out,
            add_func,
        ],
        tile=ct_0_2,
    )

    def c03_body(buf_in, buf_out, prod_in, cons_in, prod_out, cons_out, mul):
        cons_in.acquire()
        prod_out.acquire()
        mul(buf_in, buf_out)
        prod_in.release()
        cons_out.release()

    c03_worker = Worker(
        c03_body,
        fn_args=[
            c03_buf_in, c03_buf_out,
            c03_prod_lock_in, c03_cons_lock_in,
            c03_prod_lock_out, c03_cons_lock_out,
            mul_func,
        ],
        tile=ct_0_3,
    )

    # ----- Per-tile DMA programs -----
    c02_dma = TileDma(
        tile=ct_0_2,
        channels=[
            DmaChannel(direction=DMAChannelDir.S2MM, channel=0, bds=[
                Bd(buffer=c02_buf_in,
                   acquires=[Acquire(c02_prod_lock_in)],
                   releases=[Release(c02_cons_lock_in)]),
            ]),
            DmaChannel(direction=DMAChannelDir.MM2S, channel=0, bds=[
                Bd(buffer=c02_buf_out,
                   acquires=[Acquire(c02_cons_lock_out)],
                   releases=[Release(c02_prod_lock_out)],
                   packet=(0, 4)),
            ]),
        ],
    )

    c03_dma = TileDma(
        tile=ct_0_3,
        channels=[
            DmaChannel(direction=DMAChannelDir.S2MM, channel=0, bds=[
                Bd(buffer=c03_buf_in,
                   acquires=[Acquire(c03_prod_lock_in)],
                   releases=[Release(c03_cons_lock_in)]),
            ]),
            DmaChannel(direction=DMAChannelDir.MM2S, channel=0, bds=[
                Bd(buffer=c03_buf_out,
                   acquires=[Acquire(c03_cons_lock_out)],
                   releases=[Release(c03_prod_lock_out)],
                   packet=(0, 6)),
            ]),
        ],
    )

    # Memtile: 4 channels.  The input S2MM/MM2S pair forwards the header-
    # carrying buffer (memtile re-emits without restamping; the original
    # pkt_id from the shim flows through).  The output S2MM/MM2S pair
    # forwards results from compute tiles to the shim with pkt (0, 2).
    mem_dma = TileDma(
        tile=memtile,
        channels=[
            DmaChannel(direction=DMAChannelDir.S2MM, channel=0, bds=[
                Bd(buffer=mem_buf_in,
                   acquires=[Acquire(mem_prod_lock_in)],
                   releases=[Release(mem_cons_lock_in)]),
            ]),
            DmaChannel(direction=DMAChannelDir.MM2S, channel=0, bds=[
                Bd(buffer=mem_buf_in,
                   acquires=[Acquire(mem_cons_lock_in)],
                   releases=[Release(mem_prod_lock_in)]),
            ]),
            DmaChannel(direction=DMAChannelDir.S2MM, channel=2, bds=[
                Bd(buffer=mem_buf_out,
                   acquires=[Acquire(mem_prod_lock_out)],
                   releases=[Release(mem_cons_lock_out)]),
            ]),
            DmaChannel(direction=DMAChannelDir.MM2S, channel=2, bds=[
                Bd(buffer=mem_buf_out,
                   acquires=[Acquire(mem_cons_lock_out)],
                   releases=[Release(mem_prod_lock_out)],
                   packet=(0, 2)),
            ]),
        ],
    )

    # ----- Packet routes -----
    # Two ingress flows from shim → memtile with both pkt_ids 0 and 1;
    # keep_pkt_header=True so the memtile re-emit preserves the routing ID.
    flow_shim_to_mem_pkt0 = PacketFlow(
        pkt_id=0, src=shim, dst=memtile,
        src_port=WireBundle.DMA, src_channel=0,
        dst_port=WireBundle.DMA, dst_channel=0,
        keep_pkt_header=True,
    )
    flow_shim_to_mem_pkt1 = PacketFlow(
        pkt_id=1, src=shim, dst=memtile,
        src_port=WireBundle.DMA, src_channel=0,
        dst_port=WireBundle.DMA, dst_channel=0,
        keep_pkt_header=True,
    )
    # Egress: memtile → shim (pkt 2).
    flow_mem_to_shim = PacketFlow(
        pkt_id=2, src=memtile, dst=shim,
        src_port=WireBundle.DMA, src_channel=2,
        dst_port=WireBundle.DMA, dst_channel=0,
    )
    # memtile → core_0_2 (pkt 0) ; core_0_2 → memtile (pkt 4).
    flow_mem_to_c02 = PacketFlow(
        pkt_id=0, src=memtile, dst=ct_0_2,
        src_port=WireBundle.DMA, src_channel=0,
        dst_port=WireBundle.DMA, dst_channel=0,
    )
    flow_c02_to_mem = PacketFlow(
        pkt_id=4, src=ct_0_2, dst=memtile,
        src_port=WireBundle.DMA, src_channel=0,
        dst_port=WireBundle.DMA, dst_channel=2,
    )
    # memtile → core_0_3 (pkt 1) ; core_0_3 → memtile (pkt 6).
    flow_mem_to_c03 = PacketFlow(
        pkt_id=1, src=memtile, dst=ct_0_3,
        src_port=WireBundle.DMA, src_channel=0,
        dst_port=WireBundle.DMA, dst_channel=0,
    )
    flow_c03_to_mem = PacketFlow(
        pkt_id=6, src=ct_0_3, dst=memtile,
        src_port=WireBundle.DMA, src_channel=0,
        dst_port=WireBundle.DMA, dst_channel=2,
    )

    # ----- Runtime sequence -----
    # The shim DMA stamps each input task with input_packet_id (chosen by
    # --op).  This per-task packet stamping needs the dialect-level
    # shim_dma_bd(packet=...) primitive, so use rt.inline_ops as the
    # escape hatch.
    def emit_seq(A_data, B_data):
        in_task = dma_configure_task(shim.op, DMAChannelDir.MM2S, 0)
        with bds(in_task) as bd:
            with bd[0]:
                shim_dma_bd(
                    A_data.op,
                    offset=0,
                    sizes=[1, 1, 1, in_out_size],
                    strides=[0, 0, 0, 1],
                    packet=(0, input_packet_id),
                )
                EndOp()
        out_task = dma_configure_task(
            shim.op, DMAChannelDir.S2MM, 0, issue_token=True
        )
        with bds(out_task) as bd:
            with bd[0]:
                shim_dma_bd(
                    B_data.op,
                    offset=0,
                    sizes=[1, 1, 1, in_out_size],
                    strides=[0, 0, 0, 1],
                )
                EndOp()
        dma_start_task(in_task, out_task)
        dma_await_task(out_task)

    rt = Runtime()
    for f in (
        flow_shim_to_mem_pkt0, flow_shim_to_mem_pkt1,
        flow_mem_to_shim,
        flow_mem_to_c02, flow_c02_to_mem,
        flow_mem_to_c03, flow_c03_to_mem,
    ):
        rt.add_flow(f)
    for lk in (
        c02_prod_lock_in, c02_cons_lock_in,
        c02_prod_lock_out, c02_cons_lock_out,
        c03_prod_lock_in, c03_cons_lock_in,
        c03_prod_lock_out, c03_cons_lock_out,
        mem_prod_lock_in, mem_cons_lock_in,
        mem_prod_lock_out, mem_cons_lock_out,
    ):
        rt.add_lock(lk)
    for td in (c02_dma, c03_dma, mem_dma):
        rt.add_tile_dma(td)

    with rt.sequence(vector_ty, vector_ty) as (A, B):
        rt.start(c02_worker, c03_worker)
        rt.inline_ops(emit_seq, [A, B])

    return Program(dev, rt)


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Packet Switch (two-kernel demo)")
    p.add_argument(
        "-d", "--dev", type=str, choices=["npu", "npu2", "npu2_1"], default="npu"
    )
    p.add_argument(
        "--op",
        choices=list(_OP_PACKET_ID.keys()),
        required=True,
        help="which compute path to route input through (add → pkt 0, mul → pkt 1)",
    )
    p.add_argument("-n", "--length", type=int, default=256, help="vector length")
    p.add_argument(
        "--emit-mlir",
        action="store_true",
        help="print the resolved MLIR module to stdout (legacy aiecc-on-a-file path)",
    )
    p.add_argument("--xclbin-path", type=str, default=None)
    p.add_argument("--insts-path", type=str, default=None)
    return p


def main():
    opts = _make_argparser().parse_args()
    program = _build_program(
        dev=_device_for(opts.dev),
        in_out_size=opts.length,
        input_packet_id=_OP_PACKET_ID[opts.op],
    )
    module = program.resolve_program()
    if opts.emit_mlir:
        print(module)
        return
    if opts.xclbin_path:
        if not opts.insts_path:
            sys.exit("--xclbin-path requires --insts-path (must be set together)")
        compile_mlir_module(
            mlir_module=str(module),
            xclbin_path=opts.xclbin_path,
            insts_path=opts.insts_path,
            work_dir=str(Path(opts.xclbin_path).resolve().parent),
        )
        return
    print(module)


if __name__ == "__main__":
    main()
