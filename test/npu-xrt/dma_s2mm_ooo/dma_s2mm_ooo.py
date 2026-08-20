# dma_s2mm_ooo/dma_s2mm_ooo.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Out-of-order S2MM receiver demo -- deterministic many-to-one merge.

N compute-core senders -- the full bottom compute row, one per column
(row=2, col=0..N-1) -- each self-generate a distinct slice of arange and emit
it as one packet per out-of-order S2MM channel, stamped with a first-class
out-of-order id via the dataflow BD (`Bd(out_of_order_id=...)`) -- no runtime
writebd. The receiver tile sits at the CENTER of the row (col N//2) so senders
funnel in from both sides (see the packet-id note below). Each channel merges the
N packets into its own buffer on the receiver tile in OUT-OF-ORDER mode, placing
each packet into the receive BD whose pinned bd_id equals the packet header's
out-of-order id -- regardless of arrival order.

Three axes are selectable so one file covers the matrix:
  * --recv-tile core|mem : the merge (receiver) tile type.
  * --channels 1|2       : out-of-order channels on that tile; 2 share one tile
                           (the per-tile 'one out-of-order channel' limit is
                           gone) with disjoint pinned bd_ids (memtile odd
                           channels require bd_id >= 24). Each sender fans its
                           slice out to every channel.
  * -n / --sources 2..8  : merge width. 1-channel and memtile-2-channel configs
                           reach the full-width n=8; a core receiver with 2
                           channels tops out at n=7 -- the 16-BD core-tile budget
                           is c*(n+1) (n receive BDs + 1 egress BD per channel),
                           so 2 channels at n=8 need 18 > 16 BDs.

Each sender uses a DISTINCT route packet id (all still routed to the one OoO S2MM
channel; placement is by the separate out_of_order_id). Sharing one pkt_id across
N senders over-subscribes a compute tile's switchbox arbiter when the receiver is
a core; distinct ids route cleanly, so core and memtile receivers use one code
path. Distinct ids do have a switchbox cost, though: a stream-switch slave port
holds at most 4 packet rules, so distinct-pkt_id flows funneling toward the
receiver pile rules onto the shared ports nearest it. CENTERING the receiver at
col N//2 splits that funnel -- the west half routes east, the east half routes
west -- so each direction carries only ~N/2 senders' rules instead of all N.
That halves the peak, keeping every port under 4 rules through the full-width
n=8; a col-0 receiver (one-sided funnel) would instead cap the 2-channel merge
at n=6 by overflowing a port.

Placement follows the header out-of-order id, pinned to a NON-sequential bd_id
(not the slot position), so matching the expected permutation proves bd_id-
directed placement. Sender s stamps the pinned id of slot (s + shift) % N, so
each merged buffer is the send order rotated by `shift`; the verifier runs every
non-identity rotation (an in-order S2MM could match at most one by luck).

Completion is on-chip and needs no writebd: each receive BD releases a counting
lock the egress MM2S acquires N of, so the per-channel drain self-gates; the host
only drains each merged buffer to the output via a high-level shim DMA task.
(A shim-tile receiver is out of scope: it would scatter straight to DDR with no
on-chip consumer to gate completion and no routable token -- see the CDO lit test
dma_out_of_order_s2mm_shim.mlir for shim OoO S2MM lowering coverage.)

Invocation (standard basic/ 3-mode CLI):
  * emit-MLIR:    python dma_s2mm_ooo.py --recv-tile core --channels 2 -n 4 --emit-mlir
  * run + verify: python dma_s2mm_ooo.py --recv-tile mem --channels 2 -n 8
"""

import argparse
import sys

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.dialects.aiex import dma_await_task, dma_start_task, shim_dma_single_bd_task
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
    CompileTime,
    DmaChannel,
    Flow,
    Lock,
    Out,
    PacketFlow,
    Program,
    Release,
    Runtime,
    TileDma,
    Worker,
)
from aie.iron.device import Tile
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

OOO_ID_MASK = 0x3F  # 6-bit out-of-order id field


def _perm(n, shift):
    # Sender s stamps the out-of-order id of slot (s + shift) % n: a
    # deterministic rotation of the merge by `shift`.
    return [(s + shift) % n for s in range(n)]


def _slot_ids(recv_is_core, k, n):
    # Pinned, NON-sequential receive-BD ids for channel k. Ids are disjoint
    # across channels and fit the receiver's BD budget (core 16 / memtile 48),
    # leaving room for the per-channel egress MM2S BD.
    if recv_is_core:
        # 16 BDs shared by c channels + their egress BDs. Channel 0 -> odd ids
        # (1,3,5,...), channel 1 -> even ids >=2 (2,4,6,...): disjoint, strided
        # (bd_id != slot position), leaving free ids for the egress MM2S BD(s).
        return [(k + 1) + 2 * j for j in range(n)]
    # Memtile: 48 BDs; parity bases keep channels disjoint and honor the
    # hardware rule that an odd channel needs bd_id >= 24.
    base = 24 if (k % 2 == 1) else 0
    return [base + 3 + 2 * j for j in range(n)]


@iron.jit
def dma_s2mm_ooo(
    c_out: Out,
    *,
    n: CompileTime[int] = 3,
    tile_words: CompileTime[int] = 16,
    shift: CompileTime[int] = 1,
    channels: CompileTime[int] = 1,
    recv_is_core: CompileTime[int] = 0,
):
    tw = tile_words
    c = channels
    index = _perm(n, shift)

    egress = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    # Center the receiver so packets funnel in from both sides (see docstring).
    rc = n // 2
    if recv_is_core:
        receiver = Tile(col=rc, row=3, tile_type=AIETileType.CoreTile)
    else:
        receiver = Tile(col=rc, row=1, tile_type=AIETileType.MemTile)
    senders = [Tile(col=s, row=2, tile_type=AIETileType.CoreTile) for s in range(n)]

    # Per-channel merge buffer + release-only completion counter on the receiver.
    bufs = [
        Buffer(
            type=np.ndarray[(n * tw,), np.dtype[np.int32]],
            name=f"buf{k}",
            tile=receiver,
        )
        for k in range(c)
    ]
    cons = [Lock(receiver, init=0, name=f"ooo_cons{k}") for k in range(c)]

    recv_channels = []
    for k in range(c):
        ids = _slot_ids(recv_is_core, k, n)
        recv_bds = [
            Bd(
                buffer=bufs[k],
                offset=j * tw,
                length=tw,
                bd_id=ids[j],
                packet=(0, 0),  # packet-enabled; placement is by out_of_order_id
                releases=[Release(cons[k], value=1)],
            )
            for j in range(n)
        ]
        egress_bd = Bd(
            buffer=bufs[k],
            offset=0,
            length=n * tw,
            acquires=[Acquire(cons[k], value=n)],
            releases=[Release(cons[k], value=0)],
            next=0,
        )
        recv_channels += [
            DmaChannel(
                direction=DMAChannelDir.S2MM,
                channel=k,
                bds=recv_bds,
                out_of_order=True,
                repeat_count=n,
            ),
            DmaChannel(direction=DMAChannelDir.MM2S, channel=k, bds=[egress_bd]),
        ]
    recv_dma = TileDma(tile=receiver, channels=recv_channels)

    # Senders: each core owns a pre-initialized slice and fans it out to every
    # channel as one packet, stamped with the target slot's pinned bd_id and a
    # distinct route pkt_id. A trivial worker releases `filled` once to launch the
    # (chained) sends.
    sender_dmas, workers, sender_locks = [], [], []

    def pkt_id(s, k):
        return k * n + s  # distinct per (sender, channel), routed to channel k

    for s in range(n):
        pat = np.arange(s * tw, (s + 1) * tw, dtype=np.int32)
        sbuf = Buffer(initial_value=pat, name=f"sbuf{s}", tile=senders[s])
        filled = Lock(senders[s], init=0, name=f"filled{s}")
        done = Lock(senders[s], init=0, name=f"done{s}")
        sender_locks += [filled, done]

        def make_body(fl):
            def body(fl_):
                fl_.release(1)

            return body

        workers.append(
            Worker(make_body(filled), [filled], tile=senders[s], while_true=False)
        )

        # c chained send BDs (one per channel); the first gates on `filled` (and
        # releases `done` for the in-order lock invariant); the chain loops back
        # to it, so exactly one pass (c packets) is sent.
        send_bds = [
            Bd(
                buffer=sbuf,
                length=tw,
                packet=(0, pkt_id(s, k)),
                out_of_order_id=_slot_ids(recv_is_core, k, n)[index[s]] & OOO_ID_MASK,
                acquires=[Acquire(filled, value=1)] if k == 0 else [],
                releases=[Release(done, value=1)] if k == 0 else [],
                next=(k + 1) % c,
            )
            for k in range(c)
        ]
        sender_dmas.append(
            TileDma(
                tile=senders[s],
                channels=[
                    DmaChannel(direction=DMAChannelDir.MM2S, channel=0, bds=send_bds)
                ],
            )
        )

    # Flows: each sender fans out to every channel (distinct pkt id, dst channel
    # k); each channel's egress MM2S drains to its own shim S2MM channel.
    flows = []
    for s in range(n):
        for k in range(c):
            flows.append(
                PacketFlow(
                    pkt_id=pkt_id(s, k),
                    src=senders[s],
                    dst=receiver,
                    dst_channel=k,
                    keep_pkt_header=True,
                )
            )
    for k in range(c):
        flows.append(
            Flow(
                src=receiver,
                dst=egress,
                src_channel=k,
                dst_channel=k,
                shim_symbol=f"egress{k}",
            )
        )

    def sequence(c_h):
        # The senders and merge run autonomously (config-armed dataflow + the
        # launch workers); the host only drains each channel's merged buffer to
        # its slice of the output with a high-level shim DMA task (no writebd).
        # Each egress self-gates on-chip on its ooo_cons counter.
        for k in range(c):
            task = shim_dma_single_bd_task(
                f"egress{k}",
                c_h.op,
                offset=k * n * tw,
                sizes=[1, 1, 1, n * tw],
                issue_token=True,
            )
            dma_start_task(task)
            dma_await_task(task)

    rt = Runtime(sequence, [np.ndarray[(c * n * tw,), np.dtype[np.int32]]])
    for f in flows:
        rt.add_flow(f)
    for lk in cons:
        rt.add_lock(lk)
    for lk in sender_locks:
        rt.add_lock(lk)
    rt.add_tile_dma(recv_dma)
    for sd in sender_dmas:
        rt.add_tile_dma(sd)
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _compile_kwargs(opts):
    return dict(
        n=opts.sources,
        tile_words=opts.tile_words,
        shift=1,
        channels=opts.channels,
        recv_is_core=1 if opts.recv_tile == "core" else 0,
    )


def _run_and_verify(opts):
    n, tw, c = opts.sources, opts.tile_words, opts.channels
    a_np = np.arange(n * tw, dtype=np.int32).reshape(n, tw)
    recv_is_core = 1 if opts.recv_tile == "core" else 0

    # Every non-identity rotation is a distinct deterministic permutation that
    # fully overwrites each channel's buffer, so matching all of them proves
    # placement follows the pinned out-of-order id. Each of the `channels`
    # channels merges the same senders, so the output is that permutation tiled.
    for shift in range(1, n):
        out = iron.zeros((c * n * tw,), dtype=np.int32, device="npu")
        dma_s2mm_ooo(
            out, n=n, tile_words=tw, shift=shift, channels=c, recv_is_core=recv_is_core
        )
        one = np.empty_like(a_np)
        for s in range(n):
            one[(s + shift) % n] = a_np[s]
        expected = np.tile(one.reshape(-1), c)
        assert_pass(
            expected,
            out.numpy(),
            fail_msg=(
                f"out-of-order placement wrong for shift={shift} "
                f"recv={opts.recv_tile} channels={c}"
            ),
            print_pass=False,
        )
    print("PASS!")


def main():
    p = argparse.ArgumentParser(prog="AIE out-of-order S2MM merge")
    add_compile_args(p, dev_choices=("npu2",), default_dev="npu2", with_emit_mlir=True)
    p.add_argument(
        "-n", "--sources", type=int, default=3, help="number of senders to merge (2..8)"
    )
    p.add_argument(
        "--tile-words", type=int, default=16, help="int32 words per packet (>=1)"
    )
    p.add_argument(
        "--channels",
        type=int,
        default=1,
        choices=(1, 2),
        help="out-of-order channels on the receiver tile",
    )
    p.add_argument(
        "--recv-tile",
        choices=("core", "mem"),
        default="mem",
        help="receiver (merge) tile type",
    )
    opts = p.parse_args()
    if not (2 <= opts.sources <= 8):
        sys.exit("--sources must be between 2 and 8")
    if opts.recv_tile == "core" and opts.channels == 2 and opts.sources > 7:
        sys.exit(
            "core receiver with 2 channels supports at most 7 senders: "
            "c*(n+1) = 2*(8+1) = 18 > the 16-BD core tile budget"
        )
    if opts.tile_words < 1:
        sys.exit("--tile-words must be >= 1")
    run_design_cli(
        dma_s2mm_ooo,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()
