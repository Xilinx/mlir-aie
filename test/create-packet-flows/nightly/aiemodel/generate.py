#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Design generators for the router's property test."""

import random
from collections import Counter, defaultdict

from .params import FAMILIES, PARAMS
from .fabric import CORE, CTRL, DIRECTIONAL, DMA, MM2S, NORTH, S2MM, SOUTH, Target
from .design import Design, agree_overlay_keep, design_signature, load_design
from .deadlock import Analysis, Stream, describe_stream
from .routing import (
    Construction,
    construct,
    pins_hops_fn,
    plan_routing,
    unroutable_arbiters,
)

# Design generators.


def bd_block(nbytes, pkt=None, acq=None, rel=None):
    ops = [] if acq is None else [("lock", 2, acq, 1)]
    ops.append(("bd", nbytes, pkt))
    return ops + ([] if rel is None else [("lock", 1, rel, 1)])


def add_program(d, tile, dr, ch, blocks, loops, repeat=0):
    p = dict(
        tile=tile,
        kind="start",
        dir=dr,
        ch=ch,
        seq=[list(b) for b in blocks] + ([] if loops else [[]]),
        loops=loops,
        loop_to=0,
        repeat=0 if loops else repeat,
        dyn_repeat=False,
        users=[],
    )
    d.programs.append(p)
    return p


def flow_endpoints(d):
    srcs = {s for s, _ in d.flows} | {s for f in d.packet_flows for s in f["srcs"]}
    dsts = {t for _, t in d.flows} | {t for f in d.packet_flows for t in f["dsts"]}
    return srcs, dsts


def ids_by_source(d):
    ids = defaultdict(list)
    for f in d.packet_flows:
        for s in f["srcs"]:
            ids[s].append(f["id"])
    return ids


def canonical(d):
    """`d` as the router reads it back: programs in document order."""
    out = load_design(d.emit())
    if out.unsupported:
        raise ValueError(f"generated unsupported ops {out.unsupported}")
    return out


def assign_ids(rng, d, masks=True):
    """Distinct ids among packet flows that share a source or a destination,
    directly or through others. Some flows whose sources are their own get a
    mask, claiming no id another flow uses."""
    flows = d.packet_flows
    parent = list(range(len(flows)))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for a, fa in enumerate(flows):
        for b in range(a):
            fb = flows[b]
            if set(fa["srcs"]) & set(fb["srcs"]) or set(fa["dsts"]) & set(fb["dsts"]):
                parent[find(a)] = find(b)
    groups = defaultdict(list)
    for a in range(len(flows)):
        groups[find(a)].append(a)
    for members in groups.values():
        for a, pid in zip(
            members, rng.sample(range(PARAMS["max_id"] + 1), len(members))
        ):
            flows[a]["id"] = pid
    if not masks:
        return
    used = Counter(s for f in flows for s in f["srcs"])
    ids = {f["id"] for f in flows}
    for f in flows:
        if rng.random() >= 0.15 or any(used[s] > 1 for s in f["srcs"]):
            continue
        free = [b for b in range(5) if not f["id"] >> b & 1]
        if not free:
            continue
        clear = sum(1 << b for b in rng.sample(free, rng.randint(1, min(2, len(free)))))
        mask = PARAMS["max_id"] & ~clear
        if any(x & mask == f["id"] for x in ids - {f["id"]}):
            continue
        f["mask"] = mask


def add_programs(rng, d, tiles):
    """DMA programs, locks and cores on `tiles` that make some receivers stall
    and some not, and chain some agents to others through locks. Returns the
    lock-bounded receivers as (endpoint, program, lock whose initial tokens
    bound it, None for a state lock)."""
    t = d.target
    srcs, dsts = flow_endpoints(d)
    ids_of = ids_by_source(d)
    bounded = []
    for tile in sorted(tiles):
        if t.kind(tile) == "shim":
            if t.aie1:
                add_shim_programs(rng, d, tile, srcs, dsts, ids_of)
            continue
        if t.aie1:
            bounded += add_state_lock_programs(rng, d, tile, srcs, dsts, ids_of)
            continue
        gated_in, gated_out = [], []
        for ch in range(t.masters[tile][DMA]):
            ep = (*tile, DMA, ch)
            if ep not in dsts and rng.random() >= 0.1:
                continue
            roll, n = rng.random(), rng.choice([32, 64, 128])
            if roll < 0.3:
                continue
            if roll < 0.55:
                add_program(d, tile, S2MM, ch, [bd_block(n)], True)
                continue
            prod, cons = d.lock(tile, rng.choice([1, 2])), d.lock(tile, 0)
            p = add_program(d, tile, S2MM, ch, [bd_block(n, None, prod, cons)], True)
            gated_in.append((prod, cons))
            bounded.append((ep, p, prod))
        for ch in range(t.slaves[tile][DMA]):
            ep = (*tile, DMA, ch)
            if ep not in srcs and rng.random() >= 0.1:
                continue
            ids = ids_of.get(ep, [])
            blocks = [
                bd_block(rng.choice([32, 64, 128]), rng.choice(ids * 3 + [None]))
                for _ in range(rng.randint(1, 2))
            ]
            roll = rng.random()
            if roll < 0.25:
                continue
            if roll < 0.45:
                add_program(d, tile, MM2S, ch, blocks, True)
            elif roll < 0.8:
                add_program(d, tile, MM2S, ch, blocks, False, rng.choice([0, 0, 1, 2]))
            else:
                gated_out.append((ch, blocks, rng.random() < 0.5))
        uses = []
        if t.kind(tile) == "core":
            for prod, cons in gated_in:
                uses += [(2, cons, 1), (1, prod, 1)]
            for ch, blocks, loops in gated_out:
                full, empty = d.lock(tile, 0), d.lock(tile, 1)
                blocks[0] = [("lock", 2, full, 1)] + blocks[0]
                blocks[-1] = blocks[-1] + [("lock", 1, empty, 1)]
                add_program(d, tile, MM2S, ch, blocks, loops)
                uses += [(2, empty, 1), (1, full, 1)]
            if uses:
                rng.shuffle(uses)
                d.cores[tile] = uses
        else:
            for (ch, blocks, loops), (prod, cons) in zip(gated_out, gated_in):
                blocks[0] = [("lock", 2, cons, 1)] + blocks[0]
                blocks[-1] = blocks[-1] + [("lock", 1, prod, 1)]
                add_program(d, tile, MM2S, ch, blocks, loops)
    return bounded


def add_state_lock_programs(rng, d, tile, srcs, dsts, ids_of):
    """add_programs on an AIE1 core tile, whose locks hold a state: 0 when
    a buffer is empty, 1 when full. The core empties what S2MM fills and
    fills what MM2S sends, or MM2S forwards what S2MM filled."""
    t = d.target
    bounded, gated_in, uses = [], [], []
    for ch in range(t.masters[tile][DMA]):
        ep = (*tile, DMA, ch)
        if ep not in dsts and rng.random() >= 0.1:
            continue
        roll, n = rng.random(), rng.choice([32, 64, 128])
        if roll < 0.3:
            continue
        if roll < 0.55:
            add_program(d, tile, S2MM, ch, [bd_block(n)], True)
            continue
        full = d.lock(tile, 0)
        block = [("lock", 0, full, 0), ("bd", n, None), ("lock", 1, full, 1)]
        p = add_program(d, tile, S2MM, ch, [block], True)
        gated_in.append(full)
        bounded.append((ep, p, None))
    for ch in range(t.slaves[tile][DMA]):
        ep = (*tile, DMA, ch)
        if ep not in srcs and rng.random() >= 0.1:
            continue
        ids = ids_of.get(ep, [])
        blocks = [
            bd_block(rng.choice([32, 64, 128]), rng.choice(ids * 3 + [None]))
            for _ in range(rng.randint(1, 2))
        ]
        roll = rng.random()
        if roll < 0.25:
            continue
        if roll < 0.45:
            add_program(d, tile, MM2S, ch, blocks, True)
            continue
        if roll < 0.8:
            add_program(d, tile, MM2S, ch, blocks, False)
            continue
        if gated_in and rng.random() < 0.4:
            full = gated_in.pop()
        else:
            full = d.lock(tile, 0)
            uses += [(0, full, 0), (1, full, 1)]
        blocks[0] = [("lock", 0, full, 1)] + blocks[0]
        blocks[-1] = blocks[-1] + [("lock", 1, full, 0)]
        add_program(d, tile, MM2S, ch, blocks, rng.random() < 0.5)
    for full in gated_in:
        uses += [(0, full, 1), (1, full, 0)]
    if uses:
        rng.shuffle(uses)
        d.cores[tile] = uses
    return bounded


def add_shim_programs(rng, d, tile, srcs, dsts, ids_of):
    """AIE1 shim DMA programs, in place of a runtime sequence."""
    for ep in d.target.endpoints(tile, False):
        if ep in dsts:
            n = rng.choice([64, 128, 256])
            add_program(d, tile, S2MM, ep[3], [bd_block(n)], rng.random() < 0.5)
    for ep in d.target.endpoints(tile, True):
        if ep not in srcs:
            continue
        ids = ids_of.get(ep, [])
        blocks = [
            bd_block(rng.choice([64, 128, 256]), rng.choice(ids * 3 + [None]))
            for _ in range(rng.randint(1, 2))
        ]
        add_program(d, tile, MM2S, ep[3], blocks, rng.random() < 0.3)


def add_host(rng, d):
    """A runtime sequence driving the shim DMA endpoints: memcpys (some with
    packet ids, some in loops), configured tasks, bd chains, and waits after
    them in varied orders. AIE1 devices have no runtime sequence."""
    t = d.target
    if t.aie1:
        return
    srcs, dsts = flow_endpoints(d)
    ids_of = ids_by_source(d)
    events, waits = [], []
    for ep in sorted(srcs):
        if t.kind(ep[:2]) != "shim":
            continue
        ids = ids_of.get(ep, [])
        if rng.random() < 0.3:
            k = len(d.programs)
            p = dict(
                tile=ep[:2],
                kind="task",
                alloc=None,
                dir=MM2S,
                ch=ep[3],
                seq=[
                    [("bd", rng.choice([64, 128, 256]), rng.choice(ids * 3 + [None]))]
                    for _ in range(rng.randint(1, 2))
                ],
                loops=False,
                repeat=rng.choice([0, 0, 1]),
                dyn_repeat=False,
                users=[],
                sequence=0,
            )
            d.programs.append(p)
            for _ in range(rng.choice([1, 1, 2])):
                in_loop = rng.random() < 0.15
                p["users"].append(in_loop)
                events.append(("start", k, in_loop))
            waits.append(("await", k))
            continue
        sym = f"in{ep[0]}_{ep[3]}"
        d.allocs[sym] = dict(
            tile=ep[:2], dir=MM2S, ch=ep[3], pkt=rng.choice(ids) if ids else None
        )
        if rng.random() < 0.1:
            events.append(("chain", None, sym, False))
        else:
            for _ in range(rng.choice([1, 1, 2])):
                pkt = rng.choice(ids * 2 + [None]) if ids else None
                events.append(
                    (
                        "memcpy",
                        sym,
                        pkt,
                        rng.choice([64, 128, 256]),
                        rng.random() < 0.15,
                        rng.choice([1, 1, 2, 4]),
                    )
                )
        waits.append(("wait", sym))
    for ep in sorted(dsts):
        if t.kind(ep[:2]) != "shim":
            continue
        sym = f"out{ep[0]}_{ep[3]}"
        d.allocs[sym] = dict(tile=ep[:2], dir=S2MM, ch=ep[3], pkt=None)
        events.append(("memcpy", sym, None, rng.choice([64, 128, 256]), False, 1))
        waits.append(("wait", sym))
    if not events:
        return
    rng.shuffle(events)
    rng.shuffle(waits)
    for w in waits:
        if rng.random() < 0.2:
            continue

        def issues(ev):
            if w[0] == "await":
                return ev[0] == "start" and ev[1] == w[1]
            return (
                ev[0] in ("memcpy", "chain")
                and (ev[1] if ev[0] == "memcpy" else ev[2]) == w[1]
            )

        last = max(i for i, ev in enumerate(events) if issues(ev))
        events.insert(rng.randint(last + 1, len(events)), w)
    d.sequences = [events]


def size_receivers(rng, d, bounded):
    """Give lock-bounded receivers one buffer: exactly what their senders
    send, a word less, that without packet headers kept, or less. Returns the
    modes."""
    an = Analysis(d)
    an.graph
    modes = []
    for ep, p, prod in bounded:
        streams = [s for s in an.streams if s.dst == ep]
        v = [an.volumes.send_volume(s) for s in streams]
        if not streams or None in v:
            continue
        v0 = sum(
            an.volumes.send_volume(Stream(s.src, s.dst, s.pid, mask=s.mask))
            for s in streams
        )
        if v0 < 8:
            continue
        mode = rng.choice(["exact", "short", "header", "overrun"])
        if mode == "header" and sum(v) == v0:
            mode = "exact"
        cap = {
            "exact": sum(v),
            "short": sum(v) - 4,
            "header": v0,
            "overrun": v0 // 2 // 4 * 4,
        }[mode]
        if prod is not None:
            c, r, lid, _ = d.locks[prod]
            d.locks[prod] = (c, r, lid, 1)
        p["seq"][0] = [
            op if op[0] != "bd" else ("bd", cap, op[2]) for op in p["seq"][0]
        ]
        modes.append(mode)
    return modes


def direct_ok(t, src, dst):
    """The router connects a flow within one tile directly, so a port it
    cannot connect there to the other is no flow."""
    return src[:2] != dst[:2] or t.legal(src[:2], src[2:], dst[2:])


def random_design(rng, dev):
    t = Target(dev)
    width = min(t.cols, rng.choice([1, 2, 2, 3]))
    c0 = rng.randrange(t.cols - width + 1)
    window = [(c, r) for c in range(c0, c0 + width) for r in range(t.rows)]
    send = [ep for tile in window for ep in t.endpoints(tile, True)]
    recv = [ep for tile in window for ep in t.endpoints(tile, False)]
    d = Design(dev)
    srcs = list(send)
    rng.shuffle(srcs)
    taken = {}
    circuit_srcs = set()
    for src in srcs[: rng.randint(1, 7)]:
        if rng.random() < 0.35:
            # A circuit flow from a shim DMA back into its own shim is lowered
            # to an illegal connect<South : 7, DMA : 0> (repros/).
            free = [
                x
                for x in recv
                if x not in taken
                and direct_ok(t, src, x)
                and not (src[1] == 0 and x[:2] == src[:2])
            ]
            for x in rng.sample(free, min(len(free), rng.choice([1, 1, 2]))):
                d.flows.append((src, x))
                taken[x] = "circuit"
                circuit_srcs.add(src)
            continue
        for _ in range(rng.choice([1, 1, 2, 3])):
            shared = [
                x for x, k in taken.items() if k == "packet" and direct_ok(t, src, x)
            ]
            dsts = set()
            for _ in range(rng.choice([1, 1, 2, 3])):
                free = [x for x in recv if x not in taken and direct_ok(t, src, x)]
                pool = shared if shared and rng.random() < 0.3 else free
                pool = [x for x in pool if x not in dsts]
                if pool:
                    dsts.add(rng.choice(pool))
            if not dsts:
                continue
            for x in dsts:
                taken[x] = "packet"
            roll = rng.random()
            keep = True if roll < 0.15 else False if roll < 0.25 else None
            roll = rng.random()
            priority = True if roll < 0.1 else False if roll < 0.2 else None
            d.add_packet_flow(0, [src], sorted(dsts), keep=keep, priority=priority)
    for f in d.packet_flows:
        if rng.random() < 0.15:
            extra = [
                s
                for s in send
                if s not in circuit_srcs
                and s not in f["srcs"]
                and all(direct_ok(t, s, x) for x in f["dsts"])
            ]
            if extra:
                f["srcs"].append(rng.choice(extra))
    assign_ids(rng, d)
    bounded = add_programs(rng, d, window)
    add_host(rng, d)
    if rng.random() < 0.3:
        add_task_tokens(d)
    agree_overlay_keep(d)
    d.sized = size_receivers(rng, d, bounded)
    return d


def add_task_tokens(d):
    """The routes aiecc's column control overlay adds on NPUs: the shim of
    each column in use sends the task-complete tokens the host's waits wait
    for from its TileControl port to South 0, as priority packets with id 15.
    Skipped when a flow of the design takes id 15."""
    tct = 15
    if not d.dev.startswith("npu") or any(
        tct & (f["mask"] or 31) == f["id"] & (f["mask"] or 31) for f in d.packet_flows
    ):
        return
    for c in sorted({c for c, _ in d.used_tiles()}):
        d.add_packet_flow(
            tct, [(c, 0, CTRL, 0)], [(c, 0, SOUTH, 0)], keep=True, priority=True
        )


def fix_ir(rng, d, con):
    """A copy of `d` with one circuit tree and one packet tree of the witness
    `con` turned into switchbox configuration the router has to keep."""
    out = d.copy()
    t = d.target
    changed = False
    circuit = sorted({s for s, _ in d.flows})
    if circuit and rng.random() < 0.7:
        src = rng.choice(circuit)
        tree = con.trees[src]
        for s, ms in tree["children"].items():
            for m in ms:
                out.boxes.setdefault(s[:2], []).append(("connect", s[2:], m[2:]))
        if t.kind(src[:2]) == "shim":
            out.muxes.setdefault(src[:2], []).append(
                ((DMA, src[3]), (NORTH, tree["root"][3]))
            )
        for m, dst in tree["ends"].items():
            if t.kind(dst[:2]) == "shim":
                out.muxes.setdefault(dst[:2], []).append(((NORTH, m[3]), (DMA, dst[3])))
        out.flows = [f for f in out.flows if f[0] != src]
        changed = True
    srcs = Counter(s for f in d.packet_flows for s in f["srcs"])
    dsts = Counter(x for f in d.packet_flows for x in f["dsts"])
    alone = [
        k
        for k, f in enumerate(d.packet_flows)
        if len(f["srcs"]) == 1
        and srcs[f["srcs"][0]] == 1
        and all(dsts[x] == 1 for x in f["dsts"])
        and not f["priority"]
        and f["mask"] is None
    ]
    if alone and (not changed or rng.random() < 0.5):
        f = out.packet_flows.pop(rng.choice(alone))
        src = f["srcs"][0]
        tree = con.trees[src]
        for s, ms in tree["children"].items():
            tile = s[:2]
            box = out.boxes.setdefault(tile, [])
            taken = {op[2] for op in box if op[0] == "amsel"}
            a = min(set(range(PARAMS["arbiters"])) - taken, default=None)
            if a is None:
                return None
            name = f"fx_{a}"
            box.append(("amsel", name, a, 0))
            for m in ms:
                keep = f["keep"] if m in tree["ends"] else None
                box.append(("masterset", m[2:], [name], keep, False))
            box.append(("rules", s[2:], [(PARAMS["max_id"], f["id"], name)], False))
        if t.kind(src[:2]) == "shim":
            out.muxes.setdefault(src[:2], []).append(
                ((DMA, src[3]), (NORTH, tree["root"][3]))
            )
        for m, dst in tree["ends"].items():
            if t.kind(dst[:2]) == "shim":
                out.muxes.setdefault(dst[:2], []).append(((NORTH, m[3]), (DMA, dst[3])))
        changed = True
    return out if changed else None


def devices_of(device):
    return FAMILIES.get(device, (device,))


def routable_case(seed, device):
    """A design the generator routed itself. None if it gave up."""
    devs = devices_of(device)
    rng = random.Random(seed)
    d = canonical(random_design(rng, devs[seed % len(devs)]))
    built = construct(d, seed)
    if built is None:
        return None
    shape = "random"
    if rng.random() < PARAMS["fixed_rate"]:
        fixed = fix_ir(rng, d, built[0])
        if fixed is not None:
            fd = canonical(fixed)
            fbuilt = construct(fd, seed)
            if fbuilt is not None and (fd.flows or fd.packet_flows):
                d, built, shape = fd, fbuilt, "fixed-ir"
    con, an, _ = built
    return dict(
        tier="routable",
        shape=shape,
        seed=seed,
        design=d,
        analysis=an,
        hops_on=None,
        truth=dict(routable=True, witness_hops=con.hops()),
    )


def hub_row(t):
    """The memtile row, or the first core row on AIE1, which has no memtiles."""
    return next(r for r in range(t.rows) if t.kind((0, r)) != "shim")


def gen_unroutable_arbiters(rng, dev, programmed):
    """More pairwise conflicting packet streams take an arbiter at memtile T
    than it has: streams into T's S2MM channels, and with circuit switched
    hops off, streams out of its MM2S channels. Unprogrammed, every pair
    conflicts; programmed, the model decides."""
    t = Target(dev)
    row = hub_row(t)
    tc, yc = rng.sample(range(t.cols), 2)
    d = Design(dev)
    recv, send = t.endpoints((tc, row), False), t.endpoints((tc, row), True)
    ins = rng.randint(max(1, 7 - len(send)), len(recv))
    outs = rng.randint(max(1, 7 - ins), len(send))
    others = [
        ep
        for tile in sorted(t.kinds)
        if tile not in ((tc, row), (yc, row))
        for ep in t.endpoints(tile, True)
    ]
    for dst, src in zip(rng.sample(recv, ins), rng.sample(others, ins)):
        d.add_packet_flow(0, [src], [dst])
    for a, b in zip(
        rng.sample(send, outs), rng.sample(t.endpoints((yc, row), False), outs)
    ):
        d.add_packet_flow(
            0,
            [a],
            [b],
            priority=True if rng.random() < 0.1 else None,
        )
    assign_ids(rng, d, masks=False)
    if programmed:
        for f in d.packet_flows:
            f["keep"] = rng.choice([None, True, False])
        agree_overlay_keep(d)
        srcs, dsts = flow_endpoints(d)
        bounded = add_programs(rng, d, {ep[:2] for ep in srcs | dsts})
        add_host(rng, d)
        size_receivers(rng, d, bounded)
    return canonical(d)


def gen_unroutable_ports(rng, dev):
    """More circuit flows from above a one-column memtile into its S2MM
    channels than its North inputs carry."""
    t = Target(dev)
    row = hub_row(t)
    d = Design(dev)
    k = rng.randint(5, 6)
    above = [ep for r in range(row + 1, t.rows) for ep in t.endpoints((0, r), True)]
    for ch, src in zip(rng.sample(range(6), k), rng.sample(above, k)):
        d.flows.append((src, (0, row, DMA, ch)))
    add_programs(rng, d, [(0, r) for r in range(t.rows)])
    return canonical(d)


def gen_unroutable_merge(rng, dev):
    """Seven streams must climb from row 2 to row 3 of one column, on six
    links. Five are circuit flows; of the packet streams, the two that could
    share a link conflict: one from the memtile and one from row 2, into two
    channels of an unprogrammed tile at the top. A third packet flow ties them
    into one group, sharing the memtile stream's destination and the other's
    source, so the pathfinder is free to merge them."""
    d = Design(dev)
    mem = rng.sample(range(6), 5)
    top = rng.sample([0, 1], 2)
    ids = rng.sample(range(32), 3)
    row2 = (0, 2, DMA, rng.randrange(2))
    upper = [
        (0, 3, DMA, 0),
        (0, 3, DMA, 1),
        (0, 3, CORE, 0),
        (0, 4, DMA, 0),
        (0, 4, DMA, 1),
    ]
    rng.shuffle(upper)
    srcs = [(0, 1, DMA, ch) for ch in mem[1:]] + [(0, 2, CORE, 0)]
    d.flows = list(zip(srcs, upper))
    for pid, src, dst in (
        (ids[0], (0, 1, DMA, mem[0]), (0, 5, DMA, top[0])),
        (ids[1], row2, (0, 5, DMA, top[0])),
        (ids[2], row2, (0, 5, DMA, top[1])),
    ):
        d.add_packet_flow(pid, [src], [dst])
    return canonical(d)


def reserve_arbiters(d, tile, ports, arbiters=range(1, 6)):
    """Mastersets on `ports` taking every msel of `arbiters` at `tile`."""
    box = d.boxes.setdefault(tile, [])
    for a, port in zip(arbiters, ports):
        names = [f"r{a}_{m}" for m in range(PARAMS["msels"])]
        box += [("amsel", n, a, m) for m, n in enumerate(names)]
        box.append(("masterset", port, names, None, False))


def gen_wormhole(rng, dev):
    """Two packet flows cross between cores (c,3) and (c,4), one going up and
    one down, where existing mastersets leave each switchbox one arbiter
    (arbiter_hold_cycle_wormhole.mlir, with its ends and ids varied)."""
    t = Target(dev)
    c = rng.randrange(t.cols)
    d = Design(dev)
    for r in (3, 4):
        reserve_arbiters(d, (c, r), [(NORTH, i) for i in range(5)])
    a, b = rng.sample(range(32), 2)
    d.add_packet_flow(
        a, [(c, 2, DMA, rng.randrange(2))], [(c, 4, DMA, rng.randrange(2))]
    )
    d.add_packet_flow(
        b, [(c, 5, DMA, rng.randrange(2))], [(c, 3, DMA, rng.randrange(2))]
    )
    return canonical(d)


def gen_chain(rng, dev):
    """Waits that can close through arbiters at several switchboxes:
    mastersets leave some switchboxes one arbiter, and a core takes buffers
    from two lock-bounded receivers in turn, so each can stall while the
    other's flow waits."""
    t = Target(dev)
    d = Design(dev)
    row = hub_row(t)
    x = (rng.randrange(t.cols), rng.randrange(row + 1, t.rows))
    near = [
        tile
        for tile in sorted(t.kinds)
        if tile != x and abs(tile[0] - x[0]) + abs(tile[1] - x[1]) <= 3
    ]
    for tile in rng.sample(near, min(len(near), rng.randint(1, 3))):
        ports = [p for p in t.master_ports(tile) if p[0] in DIRECTIONAL]
        reserve_arbiters(d, tile, rng.sample(ports, min(5, len(ports))))
    send = [
        ep
        for tile in sorted(t.kinds)
        if tile != x and tile not in d.boxes
        for ep in t.endpoints(tile, True)
    ]
    k = rng.randint(2, 4)
    srcs = rng.sample(send, k)
    ids = rng.sample(range(32), k)
    locks = []
    for ch in range(2):
        if t.aie1:
            full = d.lock(x, 0)
            block = [("lock", 0, full, 0), ("bd", 64, None), ("lock", 1, full, 1)]
            add_program(d, x, S2MM, ch, [block], True)
            locks.append(full)
            continue
        prod, cons = d.lock(x, 1), d.lock(x, 0)
        add_program(d, x, S2MM, ch, [bd_block(64, None, prod, cons)], True)
        locks.append((prod, cons))
    if t.aie1:
        d.cores[x] = [(0, locks[1], 1), (0, locks[0], 1)]
        d.cores[x] += [(1, locks[0], 0), (1, locks[1], 0)]
    else:
        d.cores[x] = [
            (2, locks[1][1], 1),
            (2, locks[0][1], 1),
            (1, locks[0][0], 1),
            (1, locks[1][0], 1),
        ]
    recv = [
        ep for tile in sorted(t.kinds) if tile != x for ep in t.endpoints(tile, False)
    ]
    for i, (src, pid) in enumerate(zip(srcs, ids)):
        dst = (
            (*x, DMA, i) if i < 2 else rng.choice([r for r in recv if r[:2] != src[:2]])
        )
        d.add_packet_flow(pid, [src], [dst])
    srcs, dsts = flow_endpoints(d)
    add_programs(rng, d, {ep[:2] for ep in srcs | dsts} - {x})
    add_host(rng, d)
    return canonical(d)


def unroutable_case(seed, device):
    """A design no routing fits, and what the router has to say."""
    devs = devices_of(device)
    rng = random.Random(f"unroutable-{seed}")
    wide = [x for x in devs if Target(x).cols > 1]
    shape = ("arbiters", "ports", "merge", "wormhole")[seed % 4]
    aie1 = Target(devs[0]).aie1
    if aie1:
        # The ports and merge shapes need a memtile's six channels, and the
        # wormhole shape a one-column device, where no detour breaks the cycle.
        shape = "arbiters"
    hops_on, expect = False, "Unable to find a legal routing"
    if shape == "arbiters":
        d = gen_unroutable_arbiters(
            rng, wide[seed // 4 % len(wide)], rng.random() < 0.5
        )
        an = Analysis(d)
        reason = unroutable_arbiters(d, an, pins_hops_fn(d, False))
        routable = False if reason else None
        expect = reason
    else:
        dev = devs[0]
        if shape == "ports":
            d, hops_on = gen_unroutable_ports(rng, dev), rng.random() < 0.5
        elif shape == "merge":
            d, hops_on = gen_unroutable_merge(rng, dev), rng.random() < 0.5
        else:
            d = gen_wormhole(rng, dev)
        an = Analysis(d)
        routable = False
        if shape == "wormhole":
            con = Construction(d, random.Random(seed))
            for f in d.packet_flows:
                con.route(f["srcs"][0], f["dsts"], True)
            expect = plan_routing(d, an, con.solution(), hops_on=False)["reason"]
    return dict(
        tier="unroutable",
        shape=shape,
        seed=seed,
        design=d,
        analysis=an,
        hops_on=hops_on,
        truth=dict(routable=routable, expect=expect),
    )


def unknown_case(seed, device):
    """Shapes the model does not decide. Whatever the router emits is still
    checked, hold cycles included."""
    devs = devices_of(device)
    rng = random.Random(f"unknown-{seed}")
    dev = devs[seed % len(devs)]
    wide = [x for x in devs if Target(x).cols > 1]
    shapes = (
        "arbiters-hops-on",
        "packet-fan-in",
        "same-id-fan-in",
        "unpruned",
        "chain",
        "wormhole",
    )
    shape = shapes[seed % len(shapes)]
    hops_on = rng.random() < 0.5
    if shape == "arbiters-hops-on":
        d, hops_on = (
            gen_unroutable_arbiters(rng, wide[seed % len(wide)], rng.random() < 0.5),
            True,
        )
    elif shape == "packet-fan-in":
        t = Target(dev)
        row = hub_row(t)
        d = Design(dev)
        recv = t.endpoints((0, row), False)
        k = rng.randint(min(5, len(recv)), min(6, len(recv)))
        above = [ep for r in range(row + 1, t.rows) for ep in t.endpoints((0, r), True)]
        for dst, src in zip(rng.sample(recv, k), rng.sample(above, k)):
            d.add_packet_flow(0, [src], [dst])
        assign_ids(rng, d, masks=False)
        d = canonical(d)
    elif shape == "same-id-fan-in":
        d = random_design(rng, dev)
        t = d.target
        eps = [
            ep
            for tile in sorted(t.kinds)
            if tile[0] < 2 and t.kind(tile) != "shim"
            for ep in t.endpoints(tile, True)
        ]
        a, b, x = rng.sample(eps, 3)
        pid = rng.randrange(32)
        d.flows = [f for f in d.flows if f[0] not in (a, b) and f[1] != x]
        d.packet_flows = [
            f for f in d.packet_flows if {a, b, x}.isdisjoint(f["srcs"] + f["dsts"])
        ]
        for src in (a, b):
            d.add_packet_flow(pid, [src], [x])
        d = canonical(d)
    elif shape == "unpruned":
        d = canonical(random_design(rng, dev))
    elif shape == "chain":
        d = gen_chain(rng, dev)
    else:
        d = gen_wormhole(rng, dev)
        hops_on = Target(dev).cols == 1 or hops_on
    return dict(
        tier="unknown",
        shape=shape,
        seed=seed,
        design=d,
        analysis=Analysis(d),
        hops_on=hops_on,
        truth=dict(routable=None),
    )


def generate(seed, params=None, device="npu2", tier="routable"):
    """One generated case: dict(tier, shape, seed, design, analysis, hops_on,
    truth). `device` is a family (npu1, npu2) or a device name; `params`
    updates PARAMS. Routable cases are None when the generator gave up."""
    if params:
        PARAMS.update(params)
    return {
        "routable": routable_case,
        "unroutable": unroutable_case,
        "unknown": unknown_case,
    }[tier](seed, device)


def verdict(d, hops_on=True, seed=0):
    """What the model says about routing `d`: routable True (the generator
    found a witness), False (the router's pre-routing check rejects it, with
    its exact reason), or None; and every pair of requested streams that can
    deadlock, with why and what the model assumed."""
    an = Analysis(d)
    reason = unroutable_arbiters(d, an, pins_hops_fn(d, hops_on))
    pairs = []
    for i in range(an.num_requested):
        for j in range(i + 1, an.num_requested):
            if an.conflict(i, j):
                f, g = (i, j) if an.can_block(i, j) else (j, i)
                pairs.append((i, j, an.explain(i, j), an.assumptions(f, g)))
    out = dict(
        routable=None,
        reason=reason,
        streams=[describe_stream(s) for s in an.streams[: an.num_requested]],
        pairs=pairs,
        witness=None,
    )
    if reason:
        out["routable"] = False
        return out
    c = d.copy()
    before = design_signature(c)
    built = construct(c, seed)
    if built is not None and design_signature(c) == before:
        out["routable"] = True
        out["witness"] = built[0].solution()
    return out
