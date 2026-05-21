# packet_switch/packet_switch.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Packet-switched two-kernel demo — low-level placed IRON.

A single shim → memtile → compute → memtile → shim pipeline with two
compute cores (one running ``add``, one running ``mul``).  Which core
handles a given input is selected by the input shim DMA's packet ID;
the ``--op`` flag picks which packet ID to use:

  * ``--op add`` → packet (0, 0) → routes to core_0_2 (add_func)
  * ``--op mul`` → packet (0, 1) → routes to core_0_3 (mul_func)

The body intentionally stays low-level (``packetflow``, ``@mem`` /
``@memtile_dma``, ``buffer`` + ``lock``) because ObjectFifo doesn't yet
support packet_flow (TODO in the original).  This file just wraps it
with a small ``main()`` that supports both:

  * compile-only: ``... --op add --xclbin-path=PATH --insts-path=PATH``
  * emit-MLIR:    ``... --op add --emit-mlir``

Both compile modes look for ``add_mul.o`` in the work_dir (the parent
of ``--xclbin-path``); the Makefile builds it once with peano.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_
from aie.utils.compile import compile_mlir_module


def _build_module(dev, in_out_size: int, input_packet_id: int):
    in_out_ty = np.dtype[np.int8]

    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            vector_ty = np.ndarray[(in_out_size,), in_out_ty]
            vector_with_packet_ty = np.ndarray[(in_out_size + 4,), in_out_ty]

            add_func = external_func("add", [vector_ty, vector_ty], link_with="add_mul.o")
            mult_func = external_func("mul", [vector_ty, vector_ty], link_with="add_mul.o")

            ShimTile_0_0 = tile(0, 0)
            MemTile_0_1 = tile(0, 1)
            CT_0_2 = tile(0, 2)
            CT_0_3 = tile(0, 3)

            # ObjectFifo doesn't yet support packet_flow, so use raw buffer +
            # lock per the original design.
            core02_buff_in = buffer(tile=CT_0_2, datatype=vector_ty, name="core02_buff_in")
            core02_prod_lock_in = lock(tile=CT_0_2, lock_id=0, init=1, sym_name="core02_prod_lock_in")
            core02_cons_lock_in = lock(tile=CT_0_2, lock_id=1, init=0, sym_name="core02_cons_lock_in")
            core02_buff_out = buffer(tile=CT_0_2, datatype=vector_ty, name="core02_buff_out")
            core02_prod_lock_out = lock(tile=CT_0_2, lock_id=2, init=1, sym_name="core02_prod_lock_out")
            core02_cons_lock_out = lock(tile=CT_0_2, lock_id=3, init=0, sym_name="core02_cons_lock_out")

            core03_buff_in = buffer(tile=CT_0_3, datatype=vector_ty, name="core03_buff_in")
            core03_prod_lock_in = lock(CT_0_3, lock_id=0, init=1, sym_name="core03_prod_lock_in")
            core03_cons_lock_in = lock(CT_0_3, lock_id=1, init=0, sym_name="core03_cons_lock_in")
            core03_buff_out = buffer(tile=CT_0_3, datatype=vector_ty, name="core03_buff_out")
            core03_prod_lock_out = lock(CT_0_3, lock_id=2, init=1, sym_name="core03_prod_lock_out")
            core03_cons_lock_out = lock(CT_0_3, lock_id=3, init=0, sym_name="core03_cons_lock_out")

            mem01_buff_in = buffer(MemTile_0_1, datatype=vector_with_packet_ty, name="mem01_buff_in")
            mem01_prod_lock_in = lock(MemTile_0_1, lock_id=0, init=1, sym_name="mem01_prod_lock_in")
            mem01_cons_lock_in = lock(MemTile_0_1, lock_id=1, init=0, sym_name="mem01_cons_lock_in")
            mem01_buff_out = buffer(tile=MemTile_0_1, datatype=vector_ty, name="mem01_buff_out")
            mem01_prod_lock_out = lock(MemTile_0_1, lock_id=2, init=1, sym_name="mem01_prod_lock_out")
            mem01_cons_lock_out = lock(MemTile_0_1, lock_id=3, init=0, sym_name="mem01_cons_lock_out")

            # Two ingress flows from shim → memtile (pkt 0 and pkt 1, both
            # keep_pkt_header=True so the memtile can re-route by pkt ID).
            packetflow(pkt_id=0, source=ShimTile_0_0, source_port=WireBundle.DMA, source_channel=0,
                       dests={"dest": MemTile_0_1, "port": WireBundle.DMA, "channel": 0},
                       keep_pkt_header=True)
            packetflow(pkt_id=1, source=ShimTile_0_0, source_port=WireBundle.DMA, source_channel=0,
                       dests={"dest": MemTile_0_1, "port": WireBundle.DMA, "channel": 0},
                       keep_pkt_header=True)
            # Egress from memtile → shim (pkt 2).
            packetflow(pkt_id=2, source=MemTile_0_1, source_port=WireBundle.DMA, source_channel=2,
                       dests={"dest": ShimTile_0_0, "port": WireBundle.DMA, "channel": 0})
            # memtile → core_0_2 (pkt 0) and core_0_2 → memtile (pkt 4).
            packetflow(pkt_id=0, source=MemTile_0_1, source_port=WireBundle.DMA, source_channel=0,
                       dests={"dest": CT_0_2, "port": WireBundle.DMA, "channel": 0})
            packetflow(pkt_id=4, source=CT_0_2, source_port=WireBundle.DMA, source_channel=0,
                       dests={"dest": MemTile_0_1, "port": WireBundle.DMA, "channel": 2})
            # memtile → core_0_3 (pkt 1) and core_0_3 → memtile (pkt 6).
            packetflow(pkt_id=1, source=MemTile_0_1, source_port=WireBundle.DMA, source_channel=0,
                       dests={"dest": CT_0_3, "port": WireBundle.DMA, "channel": 0})
            packetflow(pkt_id=6, source=CT_0_3, source_port=WireBundle.DMA, source_channel=0,
                       dests={"dest": MemTile_0_1, "port": WireBundle.DMA, "channel": 2})

            @core(CT_0_2)
            def core_body():
                for _ in range_(sys.maxsize):
                    use_lock(core02_cons_lock_in, LockAction.AcquireGreaterEqual, value=1)
                    use_lock(core02_prod_lock_out, LockAction.AcquireGreaterEqual, value=1)
                    add_func(core02_buff_in, core02_buff_out)
                    use_lock(core02_prod_lock_in, LockAction.Release, value=1)
                    use_lock(core02_cons_lock_out, LockAction.Release, value=1)

            @mem(CT_0_2)
            def m(block):
                s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
                with block[1]:
                    use_lock(core02_prod_lock_in, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(core02_buff_in)
                    use_lock(core02_cons_lock_in, LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
                with block[3]:
                    use_lock(core02_cons_lock_out, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(core02_buff_out, packet=(0, 4))
                    use_lock(core02_prod_lock_out, LockAction.Release, value=1)
                    next_bd(block[3])
                with block[4]:
                    EndOp()

            @core(CT_0_3)
            def core_body():
                for _ in range_(sys.maxsize):
                    use_lock(core03_cons_lock_in, LockAction.AcquireGreaterEqual, value=1)
                    use_lock(core03_prod_lock_out, LockAction.AcquireGreaterEqual, value=1)
                    mult_func(core03_buff_in, core03_buff_out)
                    use_lock(core03_prod_lock_in, LockAction.Release, value=1)
                    use_lock(core03_cons_lock_out, LockAction.Release, value=1)

            @mem(CT_0_3)
            def m(block):
                s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
                with block[1]:
                    use_lock(core03_prod_lock_in, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(core03_buff_in)
                    use_lock(core03_cons_lock_in, LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
                with block[3]:
                    use_lock(core03_cons_lock_out, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(core03_buff_out, packet=(0, 6))
                    use_lock(core03_prod_lock_out, LockAction.Release, value=1)
                    next_bd(block[3])
                with block[4]:
                    EndOp()

            @memtile_dma(MemTile_0_1)
            def m(block):
                s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
                with block[1]:
                    use_lock(mem01_prod_lock_in, LockAction.AcquireGreaterEqual, value=1)
                    # First 4 bytes are the packet header (keep_pkt_header=True).
                    dma_bd(mem01_buff_in)
                    use_lock(mem01_cons_lock_in, LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
                with block[3]:
                    # Re-emit the saved packet header from mem01_buff_in; the
                    # downstream packetflow uses the same pkt_id (0 or 1) so
                    # the correct compute core picks it up.
                    use_lock(mem01_cons_lock_in, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mem01_buff_in)
                    use_lock(mem01_prod_lock_in, LockAction.Release, value=1)
                    next_bd(block[3])
                with block[4]:
                    s2 = dma_start(DMAChannelDir.S2MM, 2, dest=block[5], chain=block[6])
                with block[5]:
                    use_lock(mem01_prod_lock_out, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mem01_buff_out)
                    use_lock(mem01_cons_lock_out, LockAction.Release, value=1)
                    next_bd(block[5])
                with block[6]:
                    s3 = dma_start(DMAChannelDir.MM2S, 2, dest=block[7], chain=block[8])
                with block[7]:
                    use_lock(mem01_cons_lock_out, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mem01_buff_out, packet=(0, 2))
                    use_lock(mem01_prod_lock_out, LockAction.Release, value=1)
                    next_bd(block[7])
                with block[8]:
                    EndOp()

            @runtime_sequence(
                np.ndarray[(in_out_size,), in_out_ty], np.ndarray[(in_out_size,), in_out_ty]
            )
            def sequence(A, B):
                in_task = dma_configure_task(ShimTile_0_0, DMAChannelDir.MM2S, 0)
                with bds(in_task) as bd:
                    with bd[0]:
                        shim_dma_bd(
                            A,
                            offset=0,
                            sizes=[1, 1, 1, in_out_size],
                            strides=[0, 0, 0, 1],
                            packet=(0, input_packet_id),
                        )
                        EndOp()
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

        res = ctx.module.operation.verify()
        if res is not True:
            raise RuntimeError(f"MLIR verify failed: {res}")
        return ctx.module


_OP_PACKET_ID = {"add": 0, "mul": 1}


def _device_for(dev_str: str):
    if dev_str == "npu":
        return AIEDevice.npu1_1col
    if dev_str == "npu2":
        return AIEDevice.npu2
    if dev_str == "npu2_1":
        return AIEDevice.npu2_1col
    raise ValueError(f"[ERROR] Device name {dev_str!r} is unknown")


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
    module = _build_module(
        dev=_device_for(opts.dev),
        in_out_size=opts.length,
        input_packet_id=_OP_PACKET_ID[opts.op],
    )
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
