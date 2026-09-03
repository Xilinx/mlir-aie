# dma_s2mm_ooo/dma_s2mm_ooo.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# REQUIRES: ryzen_ai_npu2, peano
#
# RUN: %run_on_npu2% %python %s --recv-tile mem  --channels 1 -n 1
# RUN: %run_on_npu2% %python %s --recv-tile core --channels 1 -n 1
# RUN: %run_on_npu2% %python %s --recv-tile mem  --channels 1 -n 8
# RUN: %run_on_npu2% %python %s --recv-tile core --channels 1 -n 8
# RUN: %run_on_npu2% %python %s --recv-tile mem  --channels 2 -n 8
# RUN: %run_on_npu2% %python %s --recv-tile core --channels 2 -n 7
# RUN: %run_on_npu2% %python %s --recv-tile mem  --channels 1 -n 4 --packets 8
# RUN: %run_on_npu2% %python %s --recv-tile core --channels 1 -n 4 --packets 8
# RUN: %run_on_npu2% %python %s --recv-tile mem  --channels 2 -n 4 --packets 4
# RUN: %run_on_npu2% %python %s --recv-tile mem  --channels 1 -n 4 --nonuniform
# RUN: %run_on_npu2% %python %s --recv-tile core --channels 1 -n 4 --nonuniform
# RUN: %run_on_npu2% %python %s --recv-tile mem  --channels 1 -n 2 --repeat-count 8
# RUN: %run_on_npu2% %python %s --recv-tile core --channels 1 -n 4 --repeat-count 2
# RUN: %run_on_npu2% %python %s --recv-tile mem  --channels 2 -n 8 --repeat-count 2
# RUN: %run_on_npu2% %python %s --recv-tile core --channels 2 -n 6 --repeat-count 1
# RUN: %run_on_npu2% %python %s --recv-tile mem  --channels 1 -n 4 --packets 2 --repeat-count 2
# RUN: %run_on_npu2% %python %s --recv-tile mem  --channels 1 -n 4 --nonuniform --repeat-count 2
# RUN: %run_on_npu2% %python %s --recv-tile mem  --channels 1 -n 1 --repeat-count 8 --recv-backpressure
# RUN: %run_on_npu2% %python %s --recv-tile core --channels 1 -n 1 --repeat-count 8 --recv-backpressure
# RUN: %run_on_npu2% %python %s --recv-tile mem  --channels 1 -n 1 --packets 2 --repeat-count 8 --recv-backpressure
# RUN: %run_on_npu2% %python %s --recv-config runtime --recv-tile mem  --channels 1 -n 4
# RUN: %run_on_npu2% %python %s --recv-config runtime --recv-tile core --channels 1 -n 4
# RUN: %run_on_npu2% %python %s --recv-config runtime --recv-tile mem  --channels 1 -n 1
# RUN: %run_on_npu2% %python %s --recv-config runtime --recv-tile core --channels 1 -n 8
# RUN: %run_on_npu2% %python %s --recv-config runtime --recv-tile mem  --channels 1 -n 4 --packets 2
# RUN: %run_on_npu2% %python %s --recv-config runtime --recv-tile core --channels 1 -n 4 --packets 2
# RUN: %run_on_npu2% %python %s --recv-config runtime --recv-tile mem  --channels 1 -n 4 --nonuniform
# RUN: %run_on_npu2% %python %s --recv-config runtime --recv-tile mem  --channels 2 -n 8
# RUN: %run_on_npu2% %python %s --recv-config runtime --recv-tile core --channels 2 -n 4
# RUN: %run_on_npu2% %python %s --recv-config runtime --recv-tile mem  --channels 1 -n 4 --repeat-count 2
# RUN: %run_on_npu2% %python %s --recv-config runtime --recv-tile core --channels 1 -n 4 --repeat-count 1
# RUN: %run_on_npu2% %python %s --recv-config runtime --recv-tile core --channels 2 -n 6 --repeat-count 1
#
"""Out-of-order S2MM merge: N senders stream into one S2MM channel and each
packet lands in a fixed slot chosen by its header out-of-order id, not by
arrival order. See README.md for the design, options, limits, and bounds.
"""

import argparse
import sys

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir, LockAction
from aie.dialects.aie import EndOp, bds, dma_bd, next_bd, use_lock
from aie.dialects.aiex import (
    dma_await_task,
    dma_configure_task,
    dma_start_task,
    shim_dma_single_bd_task,
)
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

OOO_ID_MASK = 0x3F  # out-of-order id header field is 6-bit
MAX_LOCK_VALUE = 0x3F  # AIE lock value is 6-bit
MAX_REPEAT_FIELD = 0xFF  # DMA start-queue repeat field is 8-bit
MAX_BD_ITER = 64  # BD iteration wrap (aie-rt IterWrapMax + 1)


def _perm(n, shift):
    return [(s + shift) % n for s in range(n)]


def _slot_ids(recv_is_core, kc, n):
    # Non-sequential receive-BD ids, disjoint across channels (a memtile odd
    # channel needs bd_id >= 24).
    if recv_is_core:
        return [(kc + 1) + 2 * j for j in range(n)]
    base = 24 if (kc % 2 == 1) else 0
    return [base + 3 + 2 * j for j in range(n)]


def _chan_base(kc, r, lo, M, rounds):
    # Globally distinct base per (channel, round, source) so the verifier catches
    # a cross-channel misroute, not just a wrong slot.
    return lo + r * M + kc * rounds * M


def _recv_bds(buf, ids, off, ms, tw, con):
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
    recv_backpressure: CompileTime[int] = 0,
    recv_config: CompileTime[str] = "static",
):
    tw = tile_words
    c = channels
    m = packets
    recv_runtime = recv_config == "runtime"

    ms = [j + 1 for j in range(n)] if nonuniform else [m] * n
    off = [sum(ms[:j]) for j in range(n)]
    M = sum(ms)
    k = repeat_count  # extra merge rounds; rounds = k + 1
    rounds = k + 1

    # Guard the reusable API directly: an out-of-range config would otherwise hang
    # or lower silently wrong rather than raise. The CLI repeats these friendlier.
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
            f"(got n={n}): 2n receive + 3 drain/token BDs must fit the 16-BD core tile budget"
        )
    if recv_is_core and c == 2 and n > 7:
        raise ValueError(
            f"core receiver with 2 channels supports at most 7 senders (got n={n}): "
            "c*(n+1) receive+egress BDs must fit the 16-BD core tile budget"
        )
    if recv_backpressure:
        if n != 1:
            raise ValueError(
                "recv_backpressure supports only a single producer (n=1); use the "
                "default sender-side barrier for n>1"
            )
        if c != 1:
            raise ValueError("recv_backpressure supports only a single channel")
        if k == 0:
            raise ValueError(
                "recv_backpressure needs repeat_count>0 (it gates buffer reuse)"
            )
        if M * rounds > MAX_LOCK_VALUE:
            raise ValueError(
                f"recv_backpressure launch credit M*(k+1) = {M * rounds} exceeds the "
                f"6-bit lock ceiling {MAX_LOCK_VALUE} (the sender streams every round "
                "from one credit)"
            )

    if recv_runtime:
        if recv_backpressure:
            raise ValueError(
                "recv_config='runtime' is incompatible with recv_backpressure"
            )

    index = _perm(n, shift)

    # The receiver is centered at column n//2 to split the funnel (see README).
    egress = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    rc = n // 2
    if recv_is_core:
        receiver = Tile(col=rc, row=3, tile_type=AIETileType.CoreTile)
    else:
        receiver = Tile(col=rc, row=1, tile_type=AIETileType.MemTile)
    senders = [Tile(col=s, row=2, tile_type=AIETileType.CoreTile) for s in range(n)]

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
    runtime_recv = []

    def pkt_id(s, kc):
        return kc * n + s

    def add_sender_flows():
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
        # Single round (the default).
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
            if recv_runtime:
                # Runtime path: the ooo S2MM receive channel is emitted as a
                # dma_configure_task in sequence(); only the egress MM2S stays static.
                runtime_recv.append((kc, ids))
                recv_channels.append(
                    DmaChannel(
                        direction=DMAChannelDir.MM2S, channel=kc, bds=[egress_bd]
                    )
                )
            else:
                recv_channels += [
                    DmaChannel(
                        direction=DMAChannelDir.S2MM,
                        channel=kc,
                        bds=recv_bds,
                        out_of_order=True,
                    ),
                    DmaChannel(
                        direction=DMAChannelDir.MM2S, channel=kc, bds=[egress_bd]
                    ),
                ]
        recv_dma = TileDma(tile=receiver, channels=recv_channels)

        for s in range(n):
            t = index[s]  # target slot
            cnt = ms[t]
            lo = off[t] if nonuniform else s * m
            pat = _source_pat(lo, cnt, M, rounds, c, tw)
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
    elif recv_backpressure:
        # Single-producer (n=1) receiver-side backpressure: a free-slot credit
        # gates buffer reuse instead of the cross-tile token.
        ids = _slot_ids(recv_is_core, 0, n)
        prod = Lock(receiver, init=M, name="ooo_prod")  # free slots, init full
        recv_locks = [prod]
        recv_bd = Bd(
            buffer=bufs[0],
            offset=0,
            length=tw,
            bd_id=ids[0],
            packet=(0, 0),
            iteration=BdIteration(size=M, stride=tw),
            acquires=[Acquire(prod, value=1)],
            releases=[Release(cons[0], value=1)],
        )
        egress_bd = Bd(
            buffer=bufs[0],
            offset=0,
            length=M * tw,
            acquires=[Acquire(cons[0], value=M)],
            releases=[Release(prod, value=M)],  # return the round's slots
            next=0,
        )
        recv_dma = TileDma(
            tile=receiver,
            channels=[
                DmaChannel(
                    direction=DMAChannelDir.S2MM,
                    channel=0,
                    bds=[recv_bd],
                    out_of_order=True,
                    repeat_count=k,
                ),
                DmaChannel(
                    direction=DMAChannelDir.MM2S,
                    channel=0,
                    bds=[egress_bd],
                    repeat_count=k,
                ),
            ],
        )

        pat = _source_pat(0, M, M, rounds, c, tw)
        sbuf = Buffer(initial_value=pat, name="sbuf0", tile=senders[0])
        go = Lock(senders[0], init=0, name="go0")
        sender_locks += [go]

        def make_body(reps):
            def body(g):
                g.release(reps)

            return body

        workers.append(
            Worker(make_body(M * rounds), [go], tile=senders[0], while_true=False)
        )
        send_bd = Bd(
            buffer=sbuf,
            offset=0,
            length=tw,
            iteration=BdIteration(size=M * rounds, stride=tw),
            packet=(0, pkt_id(0, 0)),
            out_of_order_id=ids[0] & OOO_ID_MASK,
            acquires=[Acquire(go, value=1)],
            releases=[Release(go, value=0)],  # dummy; go is seeded once by the worker
            next=0,
        )
        sender_dmas.append(
            TileDma(
                tile=senders[0],
                channels=[
                    DmaChannel(
                        direction=DMAChannelDir.MM2S,
                        channel=0,
                        bds=[send_bd],
                        repeat_count=M * rounds - 1,
                    )
                ],
            )
        )

        add_sender_flows()
        flows.append(
            Flow(
                src=receiver,
                dst=egress,
                src_channel=0,
                dst_channel=0,
                shim_symbol="egress0",
            )
        )
    else:
        # Multi-round (repeat_count > 0): k+1 merges through the reused buffer with
        # a sender-side barrier. The receiver broadcasts a one-word credit token
        # once all channels drain a round; it packet-shares channel 0's drain MM2S.
        tok_pkt = n * c
        both = Lock(receiver, init=0, name="ooo_both")
        tokbuf = Buffer(
            initial_value=np.zeros(1, np.int32), name="tokbuf", tile=receiver
        )
        recv_locks = [both]

        recv_channels = []
        # On a core tile bd_ids are tile-wide, so the static drain/token allocator
        # (blind to the runtime receive ids pinned in sequence()) can reuse a
        # receive slot and deadlock (a scale-dependent BD-slot race). Pin the
        # drain/token off the receive ids. A memtile restricts ids per channel and
        # its allocator already avoids the collision, so leave it alone.
        pin_ids = recv_runtime and recv_is_core
        recv_id_union = {x for kc2 in range(c) for x in _slot_ids(recv_is_core, kc2, n)}
        free_ids = [i for i in range(16) if i not in recv_id_union]
        for kc in range(c):
            ids = _slot_ids(recv_is_core, kc, n)
            recv_bds = _recv_bds(bufs[kc], ids, off, ms, tw, cons[kc])
            drain_kw = dict(bd_id=free_ids[kc]) if pin_ids else {}
            drain_bds = [
                Bd(
                    buffer=bufs[kc],
                    offset=0,
                    length=M * tw,
                    packet=(0, tok_pkt + 1) if kc == 0 else None,
                    acquires=[Acquire(cons[kc], value=M)],
                    releases=[Release(both, value=1)],
                    next=1 if kc == 0 else 0,
                    **drain_kw,
                )
            ]
            if kc == 0:
                tok_kw = dict(bd_id=free_ids[c]) if pin_ids else {}
                drain_bds.append(
                    Bd(
                        buffer=tokbuf,
                        length=1,
                        packet=(0, tok_pkt),
                        acquires=[Acquire(both, value=c)],
                        releases=[Release(both, value=0)],  # dummy (lock invariant)
                        next=0,
                        **tok_kw,
                    )
                )
            drain_channel = DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=kc,
                bds=drain_bds,
                repeat_count=k,
            )
            if recv_runtime:
                # Runtime path: the ooo S2MM channel is armed in sequence() with
                # repeat_count = M*(k+1)-1; only the token-carrying drain stays static.
                runtime_recv.append((kc, ids))
                recv_channels.append(drain_channel)
            else:
                recv_channels += [
                    DmaChannel(
                        direction=DMAChannelDir.S2MM,
                        channel=kc,
                        bds=recv_bds,
                        out_of_order=True,
                        repeat_count=k,  # raw HW count M*(k+1)-1
                    ),
                    drain_channel,
                ]
        recv_dma = TileDma(tile=receiver, channels=recv_channels)

        for s in range(n):
            t = index[s]  # target slot
            cnt = ms[t]
            lo = off[t] if nonuniform else s * m
            pat = _source_pat(lo, cnt, M, rounds, c, tw)
            sbuf = Buffer(initial_value=pat, name=f"sbuf{s}", tile=senders[s])
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
                    g.release(reps)

                return body

            workers.append(
                Worker(make_launch(c * cnt), [go], tile=senders[s], while_true=False)
            )

            send_bds = [
                Bd(
                    buffer=sbuf,
                    offset=kc * rounds * cnt * tw,
                    length=tw,
                    iteration=BdIteration(size=cnt * rounds, stride=tw),
                    packet=(0, pkt_id(s, kc)),
                    out_of_order_id=_slot_ids(recv_is_core, kc, n)[t] & OOO_ID_MASK,
                    acquires=[Acquire(go, value=1)],
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

        add_sender_flows()
        # Credit-token broadcast: receiver MM2S ch0 fanned out to every sender's
        # S2MM ch0, sharing ch0 with the egress drain below.
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
        # Runtime receiver path: arm each ooo S2MM merge channel from the host
        # sequence with dma_configure_task {out_of_order}. The chain only configures
        # the BDs; hardware ignores Use_Next_BD and places each packet by header id.
        for kc, ids in runtime_recv:
            task = dma_configure_task(
                receiver.op,
                DMAChannelDir.S2MM,
                kc,
                repeat_count=M * rounds - 1,
                out_of_order=True,
            )
            with bds(task) as bd:
                for j in range(n):
                    with bd[j]:
                        # ms[j] packets per slot spread across ms[j] sub-buffers via
                        # BD iteration. The runtime-sequence path takes iteration from
                        # the outermost sizes/strides dim (repeat count), not the static
                        # BdIteration attr (rejected here); the tw-word contiguous
                        # transfer is the innermost dim. An ms[j]==1 slot stays linear.
                        nd = (
                            dict(sizes=[ms[j], 1, 1, tw], strides=[tw, 0, 0, 1])
                            if ms[j] > 1
                            else {}
                        )
                        dma_bd(
                            bufs[kc].op,
                            offset=off[j] * tw,
                            transfer_len=tw,
                            bd_id=ids[j],
                            packet=(0, 0),
                            **nd,
                        )
                        use_lock(cons[kc].op, LockAction.Release, value=1)
                        if j + 1 < n:
                            next_bd(bd[j + 1])
                        else:
                            EndOp()
            dma_start_task(task)

        # Drain each channel's merged buffer round-major; each drain self-gates
        # on-chip on the channel's ooo_cons count.
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
        recv_backpressure=1 if opts.recv_backpressure else 0,
        recv_config="runtime" if opts.recv_config == "runtime" else "static",
    )


def _run_and_verify(opts):
    n, tw, c, m = opts.sources, opts.tile_words, opts.channels, opts.packets
    k = opts.repeat_count
    rounds = k + 1
    ms = [j + 1 for j in range(n)] if opts.nonuniform else [m] * n
    off = [sum(ms[:j]) for j in range(n)]
    M = sum(ms)
    recv_is_core = 1 if opts.recv_tile == "core" else 0

    # Every non-identity rotation fully overwrites each channel's buffer, so
    # matching all of them proves placement follows the pinned out-of-order id.
    # n=1 has no non-identity rotation, so it runs the single identity case.
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
            recv_backpressure=1 if opts.recv_backpressure else 0,
            recv_config="runtime" if opts.recv_config == "runtime" else "static",
        )
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
        "--recv-config",
        choices=("static", "runtime"),
        default="static",
        help="how the out-of-order receive channel is configured: 'static' via "
        "the tile program, or 'runtime' via a dma_configure_task in the host "
        "sequence; supports the full merge matrix",
    )
    p.add_argument(
        "--packets", type=int, default=1, help="packets per source (m); m>=1"
    )
    p.add_argument(
        "--nonuniform",
        action="store_true",
        help="give slot j iteration size j+1 (a different m per slot); "
        "leave --packets at 1",
    )
    p.add_argument(
        "--repeat-count",
        type=int,
        default=0,
        help="extra merge rounds (k); the receiver runs k+1 rounds, default 0",
    )
    p.add_argument(
        "--recv-backpressure",
        action="store_true",
        help="single-producer reuse via a receiver-side credit instead of the "
        "sender-side barrier (requires -n 1, --channels 1, --repeat-count > 0)",
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
    if opts.recv_backpressure:
        if opts.sources != 1:
            sys.exit(
                "--recv-backpressure supports only a single producer (-n 1); use the "
                "default sender-side barrier for more sources"
            )
        if opts.channels != 1:
            sys.exit("--recv-backpressure supports only a single channel")
        if opts.repeat_count == 0:
            sys.exit(
                "--recv-backpressure needs --repeat-count > 0 (it gates buffer reuse)"
            )
        if total * rounds > MAX_LOCK_VALUE:
            sys.exit(
                f"--recv-backpressure launch credit M*(k+1) = {total * rounds} must "
                f"be <= {MAX_LOCK_VALUE} (the sender streams every round from one credit)"
            )
    if opts.recv_config == "runtime":
        if opts.recv_backpressure:
            sys.exit("--recv-config runtime is incompatible with --recv-backpressure")
    run_design_cli(
        dma_s2mm_ooo,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()
