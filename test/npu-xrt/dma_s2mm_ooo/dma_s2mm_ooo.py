# dma_s2mm_ooo/dma_s2mm_ooo.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Out-of-order S2MM receiver demo -- deterministic many-to-one merge.

N compute-core senders -- the full bottom compute row, one per column
(row=2, col=0..N-1) -- each self-generate a distinct m*tw slice of arange and
emit it as m packets into one out-of-order S2MM channel, each packet carrying a
distinct tw-word sub-slice via a send-side BD iteration (size=m, stride=tw)
run m times by the sender's repeat count, all m reusing that source's single
out-of-order id, stamped via the dataflow BD (`Bd(out_of_order_id=...)`) -- no
runtime writebd.
The receiver tile sits at the CENTER of the row (col N//2) so senders funnel in
from both sides (see the packet-id note below). Each channel merges the N*m
packets into its own buffer on the receiver tile in OUT-OF-ORDER mode: a
receive-side BD iteration (size=m, stride=tw) advances the write offset so each
source's m packets land in m consecutive sub-buffers, for N*m sub-buffers total,
placed by the packet header's out-of-order id -- regardless of arrival order.

Selectable options so one file covers the matrix:
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
  * --packets m (default 1) : sub-buffers per source. Each source sends m
                           packets -- a distinct tw sub-slice each, via a
                           send-side BD iteration (size=m, stride=tw) run m
                           times by the sender's repeat count -- landing in m
                           consecutive sub-buffers.
                           n*m must be <= 63 -- the egress lock acquires n*m and
                           63 is the 6-bit lock-value max (n=8 1-channel tops out
                           at m=4); m is orthogonal to the routing wall above, so
                           the n ceilings hold.
  * --nonuniform          : give slot j iteration size j+1 -- a different packet
                           count per slot in one merge (total = 1+2+...+n). The
                           receiver derives the total from the per-BD iteration
                           sizes. Overrides --packets.

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
lock the egress MM2S acquires the total packet count of, so the drain
self-gates; the host only drains each merged buffer to the output via a
high-level shim DMA task.
(A shim-tile receiver is out of scope: it would scatter straight to DDR with no
on-chip consumer to gate completion and no routable token -- see the CDO lit test
dma_out_of_order_s2mm_shim.mlir for shim OoO S2MM lowering coverage.)

Invocation (standard basic/ 3-mode CLI):
  * emit-MLIR:    python dma_s2mm_ooo.py --recv-tile core --channels 2 -n 4 --emit-mlir
  * run + verify: python dma_s2mm_ooo.py --recv-tile mem --channels 2 -n 8
  * run + verify: python dma_s2mm_ooo.py --recv-tile mem --channels 1 -n 4 --packets 4
  * run + verify: python dma_s2mm_ooo.py --recv-tile mem --channels 1 -n 4 --nonuniform
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
    BdIteration,
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
# An AIE lock value is 6-bit (max 63). The egress completion lock acquires n*m,
# so the merge width is bounded by n*m <= MAX_LOCK_VALUE.
MAX_LOCK_VALUE = 0x3F


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
    packets: CompileTime[int] = 1,
    nonuniform: CompileTime[int] = 0,
):
    tw = tile_words
    c = channels
    m = packets
    # Per-slot packet counts: uniform m, or 1,2,...,n when nonuniform (different
    # m per slot in one merge). off[j] is slot j's start (in tw sub-buffers);
    # M is the total packet count.
    ms = [j + 1 for j in range(n)] if nonuniform else [m] * n
    off = [sum(ms[:j]) for j in range(n)]
    M = sum(ms)
    # Self-guard the reusable API (the CLI re-checks these for friendlier exits):
    # a hang or silent-wrong lowering otherwise, not a diagnostic.
    if M > MAX_LOCK_VALUE:
        raise ValueError(
            f"total packets {M} exceeds the completion-lock ceiling "
            f"{MAX_LOCK_VALUE} (an AIE lock value is 6-bit)"
        )
    if recv_is_core and c == 2 and n > 7:
        raise ValueError(
            f"core receiver with 2 channels supports at most 7 senders (got n={n}): "
            "c*(n+1) receive+egress BDs must fit the 16-BD core-tile budget"
        )
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
            type=np.ndarray[(M * tw,), np.dtype[np.int32]],
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
                offset=off[j] * tw,
                length=tw,
                bd_id=ids[j],
                packet=(0, 0),  # packet-enabled; placement is by out_of_order_id
                iteration=BdIteration(size=ms[j], stride=tw),
                releases=[Release(cons[k], value=1)],
            )
            for j in range(n)
        ]
        egress_bd = Bd(
            buffer=bufs[k],
            offset=0,
            length=M * tw,
            acquires=[Acquire(cons[k], value=M)],
            releases=[Release(cons[k], value=0)],
            next=0,
        )
        recv_channels += [
            DmaChannel(
                direction=DMAChannelDir.S2MM,
                channel=k,
                bds=recv_bds,
                out_of_order=True,
                # No repeat_count: an out-of-order channel derives its packet
                # count from the receive BDs (the sum of their iteration sizes).
            ),
            DmaChannel(direction=DMAChannelDir.MM2S, channel=k, bds=[egress_bd]),
        ]
    recv_dma = TileDma(tile=receiver, channels=recv_channels)

    # Senders: each core owns a pre-initialized slice and fans it out to every
    # channel as `cnt` packets -- each a distinct tw sub-slice via a send-side BD
    # iteration (size=cnt, stride=tw) -- stamped with the target slot's pinned
    # bd_id and a distinct route pkt_id. `cnt` is the target slot's iteration
    # size (equal for all slots unless nonuniform). The uniform slice is source
    # s's own arange chunk; nonuniform sends the target slot's chunk. A trivial
    # worker releases `filled` by `cnt` to launch the (chained) sends.
    sender_dmas, workers, sender_locks = [], [], []

    def pkt_id(s, k):
        return k * n + s  # distinct per (sender, channel), routed to channel k

    for s in range(n):
        t = index[s]  # target slot
        cnt = ms[t]  # packets this source sends = slot t's iteration size
        lo = off[t] if nonuniform else s * m  # start of this source's chunk
        pat = np.arange(lo * tw, (lo + cnt) * tw, dtype=np.int32)
        sbuf = Buffer(initial_value=pat, name=f"sbuf{s}", tile=senders[s])
        filled = Lock(senders[s], init=0, name=f"filled{s}")
        done = Lock(senders[s], init=0, name=f"done{s}")
        sender_locks += [filled, done]

        def make_body(reps):
            def body(fl_):
                fl_.release(reps)

            return body

        workers.append(
            Worker(make_body(cnt), [filled], tile=senders[s], while_true=False)
        )

        # c chained send BDs (one per channel); the first gates on `filled` (and
        # releases `done` for the in-order lock invariant). repeat_count is
        # 0-based, so cnt-1 runs the chain exactly cnt times; each BD's iteration
        # advances its read offset by tw per pass, so cnt distinct tw sub-slices
        # are sent per channel.
        send_bds = [
            Bd(
                buffer=sbuf,
                length=tw,
                iteration=BdIteration(size=cnt, stride=tw),
                packet=(0, pkt_id(s, k)),
                out_of_order_id=_slot_ids(recv_is_core, k, n)[t] & OOO_ID_MASK,
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
                    DmaChannel(
                        direction=DMAChannelDir.MM2S,
                        channel=0,
                        bds=send_bds,
                        repeat_count=cnt - 1,
                    )
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
                offset=k * M * tw,
                sizes=[1, 1, 1, M * tw],
                issue_token=True,
            )
            dma_start_task(task)
            dma_await_task(task)

    rt = Runtime(sequence, [np.ndarray[(c * M * tw,), np.dtype[np.int32]]])
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
        packets=opts.packets,
        nonuniform=1 if opts.nonuniform else 0,
    )


def _run_and_verify(opts):
    n, tw, c, m = opts.sources, opts.tile_words, opts.channels, opts.packets
    # Per-slot packet counts, prefix offsets, and total -- must match the design.
    ms = [j + 1 for j in range(n)] if opts.nonuniform else [m] * n
    off = [sum(ms[:j]) for j in range(n)]
    M = sum(ms)
    recv_is_core = 1 if opts.recv_tile == "core" else 0

    # Every non-identity rotation is a distinct deterministic permutation that
    # fully overwrites each channel's buffer, so matching all of them proves
    # placement follows the pinned out-of-order id. Each channel merges the same
    # senders, so the output is one channel's buffer tiled.
    for shift in range(1, n):
        out = iron.zeros((c * M * tw,), dtype=np.int32, device="npu")
        dma_s2mm_ooo(
            out,
            n=n,
            tile_words=tw,
            shift=shift,
            channels=c,
            recv_is_core=recv_is_core,
            packets=m,
            nonuniform=1 if opts.nonuniform else 0,
        )
        one = np.empty(M * tw, dtype=np.int32)
        for j in range(n):
            s = (j - shift) % n  # source landing in slot j
            lo = off[j] if opts.nonuniform else s * m
            one[off[j] * tw : (off[j] + ms[j]) * tw] = np.arange(
                lo * tw, (lo + ms[j]) * tw, dtype=np.int32
            )
        expected = np.tile(one, c)
        assert_pass(
            expected,
            out.numpy(),
            fail_msg=(
                f"placement wrong: nonuniform={bool(opts.nonuniform)} m={m} "
                f"shift={shift} recv={opts.recv_tile} channels={c}"
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
    p.add_argument(
        "--packets", type=int, default=1, help="packets per source (m); m>=1"
    )
    p.add_argument(
        "--nonuniform",
        action="store_true",
        help="give slot j iteration size j+1 (different m per slot); "
        "overrides --packets",
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
    if opts.packets < 1:
        sys.exit("--packets must be >= 1")
    if opts.nonuniform and opts.packets != 1:
        sys.exit("--nonuniform sets the per-slot count itself; leave --packets at 1")
    # Total packets = sum of per-slot counts (1..n when nonuniform, else n*m).
    total = (
        opts.sources * (opts.sources + 1) // 2
        if opts.nonuniform
        else (opts.sources * opts.packets)
    )
    if total > MAX_LOCK_VALUE:
        sys.exit(
            f"total packets must be <= {MAX_LOCK_VALUE} "
            "(out-of-order completion-lock value ceiling)"
        )
    run_design_cli(
        dma_s2mm_ooo,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()
