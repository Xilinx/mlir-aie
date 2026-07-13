# manual_switchbox/manual_switchbox.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Manual stream routing — pin every switchbox hop by hand.

Almost every design should route with ``ObjectFifo`` or ``Flow`` and let
``--aie-create-pathfinder-flows`` choose the switchbox connections.  This
example is the escape hatch: it pins the *exact* physical path a
shim -> compute -> shim passthrough takes, hop by hop, using the raw
``aie.switchbox`` / ``aie.connect`` / ``aie.shim_mux`` dialect ops.

The point is to show that manual routing needs **no new IRON API**.  A
plain user-side class with ``tiles()`` + ``resolve()`` (the ``Resolvable``
protocol) is resolved at device scope when handed to a ``Worker`` via
``fn_args`` -- so it can emit any device-level dialect op, here the
switchbox configuration, right alongside the ObjectFifo-free explicit
DMA program.

Data path (single column, NPU2):

    DDR --shim DMA--> shim_mux --> shim SB --> memtile SB --> compute SB
        --> compute DMA (S2MM) --> [core copies buffer] --> compute DMA (MM2S)
        --> compute SB --> memtile SB --> shim SB --> shim_mux --> shim DMA --> DDR

Every SB/mux connection below is exactly what the pathfinder would have
emitted for the equivalent two ``aie.flow`` s; we just write them out.
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
    CompileTime,
    DmaChannel,
    In,
    Lock,
    Out,
    Program,
    Release,
    Runtime,
    Worker,
    TileDma,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir, WireBundle
from aie.dialects.aie import (
    EndOp,
    connect,
    shim_mux,
    switchbox,
)
from aie.dialects.aiex import (
    npu_address_patch,
    npu_push_queue,
    npu_sync,
    npu_writebd,
)

N = 64  # elements per transfer (int32)


class ManualSwitchbox:
    """A user-side Resolvable that emits one ``aie.switchbox`` region.

    No IRON binding is required: ``Program.resolve_program`` calls
    ``resolve()`` on any ``Resolvable`` handed to a Worker via ``fn_args``,
    at device scope, after resolving the tiles reported by ``tiles()``.
    """

    def __init__(self, tile, conns):
        # conns: list of (src_bundle, src_channel, dst_bundle, dst_channel)
        self._tile = tile
        self._conns = conns

    def tiles(self):
        return [self._tile]

    def resolve(self, loc=None, ip=None):
        @switchbox(self._tile.op)
        def _body():
            for sb, sc, db, dc in self._conns:
                connect(sb, sc, db, dc)
            # aie.switchbox has SingleBlockImplicitTerminator<EndOp>, which
            # region_op does not add for us.
            EndOp()


class ManualShimMux:
    """User-side Resolvable that emits the shim tile's ``aie.shim_mux``.

    The shim_mux translates between the shim DMA ports and the stream
    switch; a shim endpoint needs it in addition to the shim switchbox.
    """

    def __init__(self, tile, conns):
        self._tile = tile
        self._conns = conns

    def tiles(self):
        return [self._tile]

    def resolve(self, loc=None, ip=None):
        @shim_mux(self._tile.op)
        def _body():
            for sb, sc, db, dc in self._conns:
                connect(sb, sc, db, dc)
            EndOp()


@iron.jit
def manual_switchbox(a_in: In, c_out: Out, *, col: CompileTime[int] = 0):
    vec_ty = np.ndarray[(N,), np.dtype[np.int32]]

    shim = Tile(col=col, row=0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(col=col, row=1, tile_type=AIETileType.MemTile)
    comp = Tile(col=col, row=2, tile_type=AIETileType.CoreTile)

    # --- manual routing: every hop the pathfinder would have chosen -------
    # shim_mux: DDR-side DMA <-> stream switch.
    mux = ManualShimMux(
        shim,
        [
            (WireBundle.DMA, 0, WireBundle.North, 3),  # into array
            (WireBundle.North, 2, WireBundle.DMA, 0),  # out of array
        ],
    )
    sb_shim = ManualSwitchbox(
        shim,
        [
            (WireBundle.South, 3, WireBundle.North, 1),  # up toward memtile
            (WireBundle.North, 0, WireBundle.South, 2),  # down toward mux
        ],
    )
    sb_mem = ManualSwitchbox(
        mem,
        [
            (WireBundle.South, 1, WireBundle.North, 1),  # pass up to compute
            (WireBundle.North, 0, WireBundle.South, 0),  # pass down to shim
        ],
    )
    sb_comp = ManualSwitchbox(
        comp,
        [
            (WireBundle.South, 1, WireBundle.DMA, 0),  # stream in -> compute S2MM
            (WireBundle.DMA, 0, WireBundle.South, 0),  # compute MM2S -> stream out
        ],
    )

    # --- explicit data movement (no ObjectFifo, so nothing auto-routes) ---
    in_buf = Buffer(tile=comp, type=vec_ty, name="in_buf")
    out_buf = Buffer(tile=comp, type=vec_ty, name="out_buf")
    in_prod = Lock(tile=comp, lock_id=0, init=1, name="in_prod")
    in_cons = Lock(tile=comp, lock_id=1, init=0, name="in_cons")
    out_prod = Lock(tile=comp, lock_id=2, init=1, name="out_prod")
    out_cons = Lock(tile=comp, lock_id=3, init=0, name="out_cons")

    comp_dma = TileDma(
        tile=comp,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.S2MM,
                channel=0,
                bds=[
                    Bd(
                        buffer=in_buf,
                        acquires=[Acquire(in_prod, value=1)],
                        releases=[Release(in_cons, value=1)],
                        next="self",
                    )
                ],
            ),
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=0,
                bds=[
                    Bd(
                        buffer=out_buf,
                        acquires=[Acquire(out_cons, value=1)],
                        releases=[Release(out_prod, value=1)],
                        next="self",
                    )
                ],
            ),
        ],
    )

    def core_body(i_cons, i_prod, o_prod, o_cons, ib, ob, *_routing):
        # *_routing swallows the ManualSwitchbox / ManualShimMux objects: they
        # are in fn_args only so Program resolves them at device scope; the core
        # function itself ignores them.
        for _ in range_(1):
            i_cons.acquire(1)
            o_prod.acquire(1)
            for k in range_(N):
                ob[k] = ib[k]
            i_prod.release(1)
            o_cons.release(1)

    worker = Worker(
        core_body,
        # The routing Resolvables ride along in fn_args: they emit at device
        # scope during resolve, with no IRON API dedicated to them.
        [
            in_cons,
            in_prod,
            out_prod,
            out_cons,
            in_buf,
            out_buf,
            mux,
            sb_shim,
            sb_mem,
            sb_comp,
        ],
        tile=comp,
        while_true=False,
    )

    rt = Runtime()
    for lk in (in_prod, in_cons, out_prod, out_cons):
        rt.add_lock(lk)
    rt.add_tile_dma(comp_dma)

    def host_bd_writes(a, c):
        # shim S2MM ch0: DDR (a) -> array
        npu_writebd(
            bd_id=0,
            buffer_length=N,
            buffer_offset=0,
            column=col,
            row=0,
            enable_packet=0,
            out_of_order_id=0,
            packet_id=0,
            packet_type=0,
            d0_size=0,
            d0_stride=0,
            d0_zero_before=0,
            d0_zero_after=0,
            d1_size=0,
            d1_stride=0,
            d1_zero_before=0,
            d1_zero_after=0,
            d2_size=0,
            d2_stride=0,
            d2_zero_before=0,
            d2_zero_after=0,
            iteration_current=0,
            iteration_size=0,
            iteration_stride=0,
            lock_acq_enable=0,
            lock_acq_id=0,
            lock_acq_val=0,
            lock_rel_id=0,
            lock_rel_val=0,
            next_bd=0,
            use_next_bd=0,
            valid_bd=1,
        )
        npu_address_patch(addr=0x1D004, arg_idx=0, arg_plus=0)
        # shim MM2S ch0: array -> DDR (c)
        npu_writebd(
            bd_id=1,
            buffer_length=N,
            buffer_offset=0,
            column=col,
            row=0,
            enable_packet=0,
            out_of_order_id=0,
            packet_id=0,
            packet_type=0,
            d0_size=0,
            d0_stride=0,
            d0_zero_before=0,
            d0_zero_after=0,
            d1_size=0,
            d1_stride=0,
            d1_zero_before=0,
            d1_zero_after=0,
            d2_size=0,
            d2_stride=0,
            d2_zero_before=0,
            d2_zero_after=0,
            iteration_current=0,
            iteration_size=0,
            iteration_stride=0,
            lock_acq_enable=0,
            lock_acq_id=0,
            lock_acq_val=0,
            lock_rel_id=0,
            lock_rel_val=0,
            next_bd=0,
            use_next_bd=0,
            valid_bd=1,
        )
        npu_address_patch(addr=0x1D024, arg_idx=1, arg_plus=0)
        npu_push_queue(
            column=col,
            row=0,
            direction=0,
            channel=0,
            bd_id=0,
            issue_token=False,
            repeat_count=0,
        )
        npu_push_queue(
            column=col,
            row=0,
            direction=1,
            channel=0,
            bd_id=1,
            issue_token=True,
            repeat_count=0,
        )
        npu_sync(column=col, row=0, direction=1, channel=0, column_num=1, row_num=1)

    with rt.sequence(vec_ty, vec_ty) as (a, c):
        rt.start(worker)
        rt.inline_ops(host_bd_writes, [a, c])

    return Program(iron.get_current_device(), rt).resolve_program()


def _compile_kwargs(opts):
    return dict(col=opts.col)


def _run_and_verify(opts):
    a = iron.arange(1, N + 1, dtype=np.int32, device="npu")
    c = iron.zeros_like(a)
    manual_switchbox(a, c, **_compile_kwargs(opts))
    assert_pass(c.numpy(), a.numpy(), fail_msg="manual-switchbox passthrough mismatch")


def main():
    p = argparse.ArgumentParser(prog="AIE Manual Switchbox")
    add_compile_args(p, dev_choices=("npu2",), default_dev="npu2", with_emit_mlir=True)
    p.add_argument("-c", "--col", type=int, default=0)
    opts = p.parse_args()
    run_design_cli(
        manual_switchbox,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
