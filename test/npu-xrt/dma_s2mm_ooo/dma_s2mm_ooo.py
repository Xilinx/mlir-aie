# dma_s2mm_ooo/dma_s2mm_ooo.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Out-of-order S2MM merge: many senders into one channel, placed by header id.

`N` compute cores each stream a slice of data into one S2MM channel that runs in
out-of-order mode. Each packet carries an *out-of-order id* in its header, and the
receiver writes the packet into the slot whose pinned `bd_id` matches that id. A
packet always lands in the same slot because its header id picks the destination,
not its arrival order. This is the many-to-one merge primitive.

The senders occupy the bottom compute row, one per column. The receiver sits at
the center of that row (column `n//2`) because a centered receiver splits the
incoming traffic to both sides. Centering is necessary because a stream-switch
port holds only four packet rules, and a one-sided funnel would overflow a port
past `n=6`.

Each sender routes a distinct packet id to the one merge channel, while the
receiver places by the separate out-of-order id. Distinct route ids are needed
because sharing one id across the senders over-subscribes a compute tile's
switchbox arbiter. Every sender generates its own data and stamps the id on a
dataflow BD (`Bd(out_of_order_id=...)`), so no runtime writebd is involved.

Completion happens on-chip, with no host round-trip and no completion token. Each
receive BD releases a counting lock, and the egress MM2S must acquire the total
packet count before it drains. The drain cannot start early because that acquire
blocks until every packet has landed. The host only moves each merged buffer to
the output with a high-level shim DMA task.

The test proves placement follows the header id. Sender `s` targets slot
`(s + shift) % n`, which rotates the merged buffer by `shift`, and the verifier
checks every non-identity rotation. An in-order channel could match at most one
rotation by luck. Because the receive slots use non-sequential `bd_id`s, a correct
result cannot be explained by slot position.

Options (one file covers the whole matrix):

  --recv-tile core|mem   Merge (receiver) tile type.
  --channels 1|2         Out-of-order channels on the receiver tile. Two channels
                         share one tile with disjoint pinned bd_ids, and each
                         sender fans a distinct sub-slice to each channel.
  -n / --sources 1..8    Merge width (n=1 is a degenerate one-way merge).
  --packets m            Sub-buffers per source. Each source sends m packets, one
                         distinct sub-slice each, via a send-side BD iteration.
  --nonuniform           Give slot j the count j+1, a different packet count per
                         slot in one merge. Overrides --packets.
  --repeat-count k       Run k+1 merge rounds reusing the one buffer. The rounds
                         are separated by a sender-side credit-token barrier
                         because a single out-of-order channel is a FIFO. Within a
                         round the senders still race.

See README.md for the resource limits and the bound formulas.

Invocation (standard 3-mode CLI):
  emit MLIR:    python dma_s2mm_ooo.py --recv-tile core --channels 2 -n 4 --emit-mlir
  run + verify: python dma_s2mm_ooo.py --recv-tile mem --channels 2 -n 8
  run + verify: python dma_s2mm_ooo.py --recv-tile mem --channels 1 -n 4 --packets 4
  run + verify: python dma_s2mm_ooo.py --recv-tile mem --channels 1 -n 4 --nonuniform
  run + verify: python dma_s2mm_ooo.py --recv-tile mem --channels 1 -n 2 --repeat-count 2
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
    PacketDest,
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

OOO_ID_MASK = 0x3F  # the out-of-order id header field is 6 bits

# The merge width is bounded by M <= MAX_LOCK_VALUE because an AIE lock value is
# 6 bits and the egress completion lock acquires the total packet count M.
MAX_LOCK_VALUE = 0x3F

# The all-rounds total M*(k+1) cannot exceed MAX_REPEAT_FIELD + 1 because
# out-of-order encodes M*(k+1)-1 into the 8-bit DMA start-queue repeat field.
MAX_REPEAT_FIELD = 0xFF

# A multi-round sender walks every round from one BD whose iteration size is
# max(ms)*rounds, which cannot exceed the BD iteration wrap of 64 (aie-rt
# IterWrapMax + 1).
MAX_BD_ITER = 64


def _perm(n, shift):
    # Sender s stamps the out-of-order id of slot (s + shift) % n, which rotates
    # the merge by `shift`.
    return [(s + shift) % n for s in range(n)]


def _slot_ids(recv_is_core, kc, n):
    # Pinned, non-sequential receive-BD ids for channel kc. The ids stay disjoint
    # across channels and leave room for each channel's egress MM2S BD.
    if recv_is_core:
        # A core tile has 16 BDs shared by all channels and their egress BDs.
        # Channel 0 takes odd ids and channel 1 takes even ids >= 2, which keeps
        # the channels disjoint and makes bd_id differ from slot position.
        return [(kc + 1) + 2 * j for j in range(n)]
    # A memtile has 48 BDs. The parity bases keep channels disjoint and honor the
    # hardware rule that an odd channel needs bd_id >= 24.
    base = 24 if (kc % 2 == 1) else 0
    return [base + 3 + 2 * j for j in range(n)]


def _chan_base(kc, r, lo, M, rounds):
    # First sub-buffer (in tw units) of one source's chunk for channel kc, round r.
    # Every (channel, round, source) chunk gets a globally distinct base because
    # channel kc is offset by a whole channel span (rounds*M) and round r by M.
    # That distinctness lets the verifier catch a channel swap or a cross-channel
    # misroute, not just a wrong within-channel slot. `lo` is the source's own
    # within-round start.
    return lo + r * M + kc * rounds * M


def _recv_bds(buf, ids, off, ms, tw, con):
    # The n out-of-order receive BDs for one channel. Each BD is pinned to a slot's
    # bd_id, packet-enabled because placement follows the header id, and
    # release-only on the completion counter `con`. Both the single-round and
    # multi-round receivers use this.
    return [
        Bd(
            buffer=buf,
            offset=off[j] * tw,
            length=tw,
            bd_id=ids[j],
            packet=(0, 0),
            iteration=BdIteration(size=ms[j], stride=tw),
            releases=[Release(con, value=1)],
        )
        for j in range(len(ids))
    ]


def _source_pat(lo, cnt, M, rounds, c, tw):
    # One source's send buffer. The per-channel, per-round arange chunks are laid
    # out channel-major, with channel kc's `rounds` chunks contiguous. Each
    # per-channel send BD then reads its own channel's data at offset kc*rounds*cnt.
    return np.concatenate(
        [
            np.arange(
                _chan_base(kc, r, lo, M, rounds) * tw,
                (_chan_base(kc, r, lo, M, rounds) + cnt) * tw,
                dtype=np.int32,
            )
            for kc in range(c)
            for r in range(rounds)
        ]
    )


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
    repeat_count: CompileTime[int] = 0,
):
    tw = tile_words
    c = channels
    m = packets

    # Per-slot packet counts. Uniform m by default, or 1,2,...,n under nonuniform
    # (a different m per slot in one merge). off[j] is slot j's start in tw
    # sub-buffers, and M is the total packet count.
    ms = [j + 1 for j in range(n)] if nonuniform else [m] * n
    off = [sum(ms[:j]) for j in range(n)]
    M = sum(ms)
    k = repeat_count  # extra merge rounds; rounds = k + 1
    rounds = k + 1

    # These guards protect the reusable API directly because an out-of-range config
    # would otherwise hang or lower silently wrong rather than raise. The CLI
    # repeats them with friendlier exit messages.
    if k < 0:
        raise ValueError("repeat_count must be >= 0")
    if M > MAX_LOCK_VALUE:
        raise ValueError(
            f"total packets {M} exceeds the completion-lock ceiling "
            f"{MAX_LOCK_VALUE} (an AIE lock value is 6-bit)"
        )
    if M * rounds - 1 > MAX_REPEAT_FIELD:
        raise ValueError(
            f"total packets over all rounds (M*(k+1) = {M * rounds}) exceed the "
            f"out-of-order repeat field, which encodes M*(k+1)-1 <= {MAX_REPEAT_FIELD}"
        )
    if k > 0 and c * max(ms) > MAX_LOCK_VALUE:
        raise ValueError(
            f"per-round send credit c*max(ms) = {c * max(ms)} exceeds the "
            f"6-bit lock ceiling {MAX_LOCK_VALUE} (the sender go credit is one lock)"
        )
    if rounds > MAX_LOCK_VALUE:
        raise ValueError(
            f"rounds (k+1) = {rounds} exceeds the 6-bit lock ceiling "
            f"{MAX_LOCK_VALUE} (each sender's token-credit lock is initialized to "
            "rounds)"
        )
    if max(ms) * rounds > MAX_BD_ITER:
        raise ValueError(
            f"send BD iteration size max(ms)*(k+1) = {max(ms) * rounds} "
            f"exceeds the BD iteration cap {MAX_BD_ITER} (the sender walks all "
            "rounds from one BD)"
        )
    if k > 0 and recv_is_core and c == 2 and n > 6:
        raise ValueError(
            f"core receiver, 2 channels, repeat_count>0 supports at most 6 senders "
            f"(got n={n}): 2n receive + 3 drain/token BDs must fit the 16-BD budget"
        )
    if recv_is_core and c == 2 and n > 7:
        raise ValueError(
            f"core receiver with 2 channels supports at most 7 senders (got n={n}): "
            "c*(n+1) receive+egress BDs must fit the 16-BD core-tile budget"
        )

    index = _perm(n, shift)

    # Tiles. Senders fill the bottom compute row; the receiver is centered on that
    # row (see the module docstring for why); the egress is the shim NOC tile.
    egress = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    rc = n // 2
    if recv_is_core:
        receiver = Tile(col=rc, row=3, tile_type=AIETileType.CoreTile)
    else:
        receiver = Tile(col=rc, row=1, tile_type=AIETileType.MemTile)
    senders = [Tile(col=s, row=2, tile_type=AIETileType.CoreTile) for s in range(n)]

    # One merge buffer and one release-only completion counter per channel, both on
    # the receiver.
    bufs = [
        Buffer(
            type=np.ndarray[(M * tw,), np.dtype[np.int32]],
            name=f"buf{kc}",
            tile=receiver,
        )
        for kc in range(c)
    ]
    cons = [Lock(receiver, init=0, name=f"ooo_cons{kc}") for kc in range(c)]

    sender_dmas, workers, sender_locks, recv_locks, flows = [], [], [], [], []

    def pkt_id(s, kc):
        # Distinct id per (sender, channel), all routed to channel kc.
        return kc * n + s

    def add_sender_flows():
        # One packet flow per (sender, channel). keep_pkt_header carries the
        # out-of-order id through to placement on the receiver.
        for s in range(n):
            for kc in range(c):
                flows.append(
                    PacketFlow(
                        pkt_id=pkt_id(s, kc),
                        src=senders[s],
                        dst=receiver,
                        dst_channel=kc,
                        keep_pkt_header=True,
                    )
                )

    if k == 0:
        # Single round (the default). Each sender owns a preloaded arange slice and
        # fans it to every channel as `cnt` packets, one distinct tw sub-slice each,
        # via a send-side BD iteration (size=cnt, stride=tw). Each packet carries
        # the target slot's pinned bd_id and a distinct route pkt_id. `cnt` is the
        # target slot's iteration size, equal for all slots unless nonuniform. A
        # trivial worker releases `filled` to launch the sends.

        # Receiver: n out-of-order receive BDs plus one draining egress BD per
        # channel.
        recv_channels = []
        for kc in range(c):
            ids = _slot_ids(recv_is_core, kc, n)
            recv_bds = _recv_bds(bufs[kc], ids, off, ms, tw, cons[kc])
            egress_bd = Bd(
                buffer=bufs[kc],
                offset=0,
                length=M * tw,
                acquires=[Acquire(cons[kc], value=M)],
                releases=[Release(cons[kc], value=0)],
                next=0,
            )
            recv_channels += [
                DmaChannel(
                    direction=DMAChannelDir.S2MM,
                    channel=kc,
                    bds=recv_bds,
                    out_of_order=True,
                ),
                DmaChannel(direction=DMAChannelDir.MM2S, channel=kc, bds=[egress_bd]),
            ]
        recv_dma = TileDma(tile=receiver, channels=recv_channels)

        # Senders.
        for s in range(n):
            t = index[s]  # target slot
            cnt = ms[t]  # packets this source sends, equal to slot t's iteration size
            lo = off[t] if nonuniform else s * m
            pat = _source_pat(lo, cnt, M, rounds, c, tw)  # rounds == 1 here
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

            # One chained send BD per channel. Each BD reads its own channel's
            # region of sbuf at offset kc*cnt because each channel carries distinct
            # data. The first BD gates on `filled` and releases `done` to keep the
            # lock balanced.
            send_bds = [
                Bd(
                    buffer=sbuf,
                    offset=kc * cnt * tw,
                    length=tw,
                    iteration=BdIteration(size=cnt, stride=tw),
                    packet=(0, pkt_id(s, kc)),
                    out_of_order_id=_slot_ids(recv_is_core, kc, n)[t] & OOO_ID_MASK,
                    acquires=[Acquire(filled, value=1)] if kc == 0 else [],
                    releases=[Release(done, value=1)] if kc == 0 else [],
                    next=(kc + 1) % c,
                )
                for kc in range(c)
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

        # Flows: sender packets in, then one egress drain per channel.
        add_sender_flows()
        for kc in range(c):
            flows.append(
                Flow(
                    src=receiver,
                    dst=egress,
                    src_channel=kc,
                    dst_channel=kc,
                    shim_symbol=f"egress{kc}",
                )
            )
    else:
        # Multi-round (repeat_count > 0). This runs k+1 true n-way out-of-order
        # merges through the one reused buffer. A receiver lock cannot group the
        # rounds because a single out-of-order channel is a FIFO. A shared counter
        # would miscount a fast source's next-round packet, and a per-slot lock
        # would head-of-line deadlock. The barrier is therefore on the sender side.
        # After draining round r the receiver broadcasts a one-word credit token,
        # and each sender waits on it before the next round. The n senders still
        # race within a round, which keeps each round a genuine out-of-order merge.
        # With c channels the token fires once all c have drained (a `both` join).
        # The token packet-shares channel 0's MM2S with that channel's drain,
        # chained after it under a distinct pkt_id. Sharing the MM2S means the merge
        # needs no extra channel and fits a core tile's two MM2S even at c == 2.

        # tok_pkt is the first pkt_id past the send ids (which are < n*c). Both
        # tok_pkt and the channel-0 egress id tok_pkt+1 must fit the 5-bit pkt_id
        # field (max 31).
        tok_pkt = n * c
        both = Lock(receiver, init=0, name="ooo_both")  # all c drains done -> token
        tokbuf = Buffer(
            initial_value=np.zeros(1, np.int32), name="tokbuf", tile=receiver
        )
        recv_locks = [both]

        # Receiver: the out-of-order receive BDs plus the per-channel drain, with
        # the token BD chained after channel 0's drain.
        recv_channels = []
        for kc in range(c):
            ids = _slot_ids(recv_is_core, kc, n)
            recv_bds = _recv_bds(bufs[kc], ids, off, ms, tw, cons[kc])
            # Channel 0's drain packet-routes to the shim because its MM2S also
            # carries the token, chained after it under a second pkt_id. The other
            # channels drain circuit and loop back to themselves.
            drain_bds = [
                Bd(
                    buffer=bufs[kc],
                    offset=0,
                    length=M * tw,
                    packet=(0, tok_pkt + 1) if kc == 0 else None,
                    acquires=[Acquire(cons[kc], value=M)],
                    releases=[Release(both, value=1)],  # this channel drained a round
                    next=1 if kc == 0 else 0,
                )
            ]
            if kc == 0:
                drain_bds.append(
                    Bd(
                        buffer=tokbuf,
                        length=1,
                        packet=(0, tok_pkt),
                        acquires=[Acquire(both, value=c)],  # all c channels drained
                        releases=[Release(both, value=0)],  # dummy (lock invariant)
                        next=0,
                    )
                )
            recv_channels += [
                DmaChannel(
                    direction=DMAChannelDir.S2MM,
                    channel=kc,
                    bds=recv_bds,
                    out_of_order=True,
                    repeat_count=k,  # HW raw packet count = M*(k+1)-1
                ),
                DmaChannel(
                    direction=DMAChannelDir.MM2S,
                    channel=kc,
                    bds=drain_bds,
                    repeat_count=k,
                ),
            ]
        recv_dma = TileDma(tile=receiver, channels=recv_channels)

        # Senders.
        for s in range(n):
            t = index[s]
            cnt = ms[t]
            lo = off[t] if nonuniform else s * m  # this source's round-0 chunk start
            pat = _source_pat(lo, cnt, M, rounds, c, tw)
            sbuf = Buffer(initial_value=pat, name=f"sbuf{s}", tile=senders[s])
            # go is one round's send credit (c*cnt, cnt to each channel). The worker
            # launches round 0, and each credit token replenishes one round.
            go = Lock(senders[s], init=0, name=f"go{s}")
            tok_free = Lock(senders[s], init=rounds, name=f"tok_free{s}")
            tokrx = Buffer(
                type=np.ndarray[(1,), np.dtype[np.int32]],
                name=f"tokrx{s}",
                tile=senders[s],
            )
            sender_locks += [go, tok_free]

            def make_launch(reps):
                def body(g):
                    g.release(reps)  # launch round 0 with one round's credit

                return body

            workers.append(
                Worker(make_launch(c * cnt), [go], tile=senders[s], while_true=False)
            )

            # One send BD per channel (fan-out). Each BD reads its own channel's
            # region of sbuf at offset kc*rounds*cnt because each channel carries
            # distinct data. Each BD also gates on a `go` credit, which blocks the
            # next round until the token restores the whole round's credit.
            send_bds = [
                Bd(
                    buffer=sbuf,
                    offset=kc * rounds * cnt * tw,
                    length=tw,
                    iteration=BdIteration(size=cnt * rounds, stride=tw),
                    packet=(0, pkt_id(s, kc)),
                    out_of_order_id=_slot_ids(recv_is_core, kc, n)[t] & OOO_ID_MASK,
                    acquires=[Acquire(go, value=1)],  # a credit per packet
                    releases=[Release(go, value=0)],  # dummy; go is fed by the token
                    next=(kc + 1) % c,
                )
                for kc in range(c)
            ]
            tok_bd = Bd(
                buffer=tokrx,
                length=1,
                acquires=[Acquire(tok_free, value=1)],
                releases=[Release(go, value=c * cnt)],  # a token frees one round
                next=0,
            )
            sender_dmas.append(
                TileDma(
                    tile=senders[s],
                    channels=[
                        DmaChannel(
                            direction=DMAChannelDir.MM2S,
                            channel=0,
                            bds=send_bds,
                            repeat_count=cnt * rounds - 1,
                        ),
                        DmaChannel(
                            direction=DMAChannelDir.S2MM,
                            channel=0,
                            bds=[tok_bd],
                            repeat_count=k,
                        ),
                    ],
                )
            )

        # Flows: sender packets in, then the credit-token broadcast, then egress.
        add_sender_flows()
        # Credit-token broadcast from the receiver's MM2S ch0 to every sender's
        # S2MM ch0, one packet flow fanned out to all senders. It shares ch0 with
        # the egress drain below.
        flows.append(
            PacketFlow(
                pkt_id=tok_pkt,
                src=receiver,
                src_channel=0,
                dst=senders[0],
                dst_channel=0,
                extra_dsts=[PacketDest(senders[s], channel=0) for s in range(1, n)],
            )
        )
        # Channel 0's egress is packet-routed because its MM2S is shared with the
        # token. The other channels are circuit-routed. Each channel drains to its
        # own shim S2MM.
        flows.append(
            PacketFlow(
                pkt_id=tok_pkt + 1,
                src=receiver,
                src_channel=0,
                dst=egress,
                dst_channel=0,
                shim_symbol="egress0",
            )
        )
        for kc in range(1, c):
            flows.append(
                Flow(
                    src=receiver,
                    dst=egress,
                    src_channel=kc,
                    dst_channel=kc,
                    shim_symbol=f"egress{kc}",
                )
            )

    def sequence(c_h):
        # The senders and merge run autonomously. The host drains each channel's
        # merged buffer once per round into that round's output region (round-major
        # layout), and each drain self-gates on-chip on the channel's ooo_cons
        # count. The host must drain round-major (every channel of round r before
        # round r+1) because the multi-round barrier holds a round until every
        # channel has drained.
        for r in range(rounds):
            for kc in range(c):
                task = shim_dma_single_bd_task(
                    f"egress{kc}",
                    c_h.op,
                    offset=(r * c + kc) * M * tw,
                    sizes=[1, 1, 1, M * tw],
                    issue_token=True,
                )
                dma_start_task(task)
                dma_await_task(task)

    rt = Runtime(sequence, [np.ndarray[(rounds * c * M * tw,), np.dtype[np.int32]]])
    for f in flows:
        rt.add_flow(f)
    for lk in cons:
        rt.add_lock(lk)
    for lk in recv_locks:
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
        repeat_count=opts.repeat_count,
    )


def _run_and_verify(opts):
    n, tw, c, m = opts.sources, opts.tile_words, opts.channels, opts.packets
    k = opts.repeat_count
    rounds = k + 1
    # Per-slot packet counts, prefix offsets, and total. These must match the
    # design.
    ms = [j + 1 for j in range(n)] if opts.nonuniform else [m] * n
    off = [sum(ms[:j]) for j in range(n)]
    M = sum(ms)
    recv_is_core = 1 if opts.recv_tile == "core" else 0

    # Every non-identity rotation fully overwrites each channel's buffer, and
    # matching all of them proves placement follows the pinned out-of-order id.
    # Because n=1 has no non-identity rotation, it runs the single identity case.
    for shift in range(1, n) or [0]:
        out = iron.zeros((rounds * c * M * tw,), dtype=np.int32, device="npu")
        dma_s2mm_ooo(
            out,
            n=n,
            tile_words=tw,
            shift=shift,
            channels=c,
            recv_is_core=recv_is_core,
            packets=m,
            nonuniform=1 if opts.nonuniform else 0,
            repeat_count=k,
        )
        # In region (round r, channel kc), slot j holds source (j-shift)%n's chunk
        # for that channel and round. Because the chunks are distinct per channel
        # (see _chan_base), a channel swap or cross-channel misroute fails the
        # check, not just a wrong slot.
        expected = np.empty(rounds * c * M * tw, dtype=np.int32)
        for r in range(rounds):
            for kc in range(c):
                base = (r * c + kc) * M * tw
                for j in range(n):
                    s = (j - shift) % n
                    lo = off[j] if opts.nonuniform else s * m
                    cb = _chan_base(kc, r, lo, M, rounds)
                    expected[base + off[j] * tw : base + (off[j] + ms[j]) * tw] = (
                        np.arange(cb * tw, (cb + ms[j]) * tw, dtype=np.int32)
                    )
        assert_pass(
            expected,
            out.numpy(),
            fail_msg=(
                f"placement wrong: nonuniform={bool(opts.nonuniform)} m={m} "
                f"shift={shift} recv={opts.recv_tile} channels={c} "
                f"repeat_count={k}"
            ),
            print_pass=False,
        )
    print("PASS!")


def main():
    p = argparse.ArgumentParser(prog="AIE out-of-order S2MM merge")
    add_compile_args(p, dev_choices=("npu2",), default_dev="npu2", with_emit_mlir=True)
    p.add_argument(
        "-n", "--sources", type=int, default=3, help="number of senders to merge (1..8)"
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
    p.add_argument(
        "--repeat-count",
        type=int,
        default=0,
        help="extra merge rounds (k); the receiver runs k+1 rounds, default 0",
    )
    opts = p.parse_args()
    if not (1 <= opts.sources <= 8):
        sys.exit("--sources must be between 1 and 8")
    if (
        opts.repeat_count > 0
        and opts.recv_tile == "core"
        and opts.channels == 2
        and opts.sources > 6
    ):
        sys.exit(
            "core receiver, 2 channels, --repeat-count > 0 supports at most 6 "
            "senders: 2n receive + 3 drain/token BDs > the 16-BD core tile budget"
        )
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
    if opts.repeat_count < 0:
        sys.exit("--repeat-count must be >= 0")
    cnt_max = opts.sources if opts.nonuniform else opts.packets
    if opts.repeat_count > 0 and opts.channels * cnt_max > MAX_LOCK_VALUE:
        sys.exit(
            f"per-round send credit channels*max(ms) = "
            f"{opts.channels * cnt_max} must be <= {MAX_LOCK_VALUE} "
            "(the sender go credit is a single 6-bit lock)"
        )
    rounds = opts.repeat_count + 1
    if rounds > MAX_LOCK_VALUE:
        sys.exit(
            f"--repeat-count too large: rounds (k+1) = {rounds} must be <= "
            f"{MAX_LOCK_VALUE} (each sender's token-credit lock is initialized to "
            "rounds, a single 6-bit lock)"
        )
    if total * rounds - 1 > MAX_REPEAT_FIELD:
        sys.exit(
            f"total packets over all rounds ({total * rounds}) exceed the 8-bit "
            f"repeat field (the out-of-order channel encodes M*(k+1)-1 <= {MAX_REPEAT_FIELD})"
        )
    if opts.repeat_count > 0 and cnt_max * rounds > MAX_BD_ITER:
        sys.exit(
            f"send BD iteration size max(ms)*(k+1) = {cnt_max * rounds} "
            f"must be <= {MAX_BD_ITER} (the sender walks all rounds from one BD)"
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
