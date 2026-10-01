#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Mutation testing of the router, and of the model's verifier.

    python utils/router_testing/router_mutation.py --device npu2 --seeds 40 [--first-seed 0]
        [--hops-off] [--only NAME ...] [--shrink] [--repros DIR] [--out DIR]
        [-v]

Each routable design the model generates is routed, then:

  mutators   change the design and say what the router now has to do: route
             it (the model built a witness, or the change cannot hurt), reject
             it (a certificate: more circuit trees must cross a row of links
             than it has channels, a duplicate flow, one port both circuit and
             packet switched, two circuit flows into one port), or follow the
             model's verdict. Whatever the router emits is verified.
  relations  change the design in ways that must not change the answer: flows,
             sources, destinations and DMA programs permuted; ids renamed; the
             routing lifted by aie-find-flows and routed again; a repeat run;
             a flow added on links nothing else uses.
  kill rate  corrupt the router's output (amsel, msel, rule mask and value,
             connect, masterset, shim mux, keep, ctrl, two conflicting streams
             on one arbiter) and count how often the verifier notices.

Failures are shrunk with router_shrink.py (--shrink) and written to --repros.
"""

import argparse
import random
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[2] / "test" / "create-packet-flows" / "nightly"
    ),
)
import router_properties as rp  # noqa: E402
import router_shrink as shrinker  # noqa: E402

# Design surgery.


def endpoints(d, sending):
    t = d.target
    return [ep for tile in sorted(t.kinds) for ep in t.endpoints(tile, sending)]


def busy(d):
    """Endpoints flows, switchboxes or shim muxes already use: (sending,
    receiving)."""
    srcs, dsts = map(set, rp.flow_endpoints(d))
    for tile, ops in d.boxes.items():
        for op in ops:
            if op[0] == "connect":
                srcs.add((*tile, *op[1]))
                dsts.add((*tile, *op[2]))
            elif op[0] == "masterset":
                dsts.add((*tile, *op[1]))
            elif op[0] == "rules":
                srcs.add((*tile, *op[1]))
    for tile, conns in d.muxes.items():
        for s, m in conns:
            srcs.add((*tile, *s))
            dsts.add((*tile, *m))
    return srcs, dsts


def free_endpoints(d):
    srcs, dsts = busy(d)
    return (
        [ep for ep in endpoints(d, True) if ep not in srcs],
        [ep for ep in endpoints(d, False) if ep not in dsts],
    )


def loopback(s, x):
    return s[1] == 0 and x[:2] == s[:2]


def circuit_ok(t, s, x):
    return rp.direct_ok(t, s, x) and not loopback(s, x)


def fresh_id(d, rng):
    claimed = [(f["mask"] or rp.PARAMS["max_id"], f["id"]) for f in d.packet_flows]
    claimed += [
        (m, v)
        for ops in d.boxes.values()
        for op in ops
        if op[0] == "rules"
        for m, v, _ in op[2]
    ]
    free = [
        i
        for i in range(rp.PARAMS["max_id"] + 1)
        if all(i & m != v & m for m, v in claimed)
    ]
    return rng.choice(free) if free else None


def resend(d, src, old, new):
    """What `src` sends with id `old` carries `new` instead."""
    c, r, b, ch = src
    if b != rp.DMA:
        return
    for p in d.programs:
        if p["tile"] == (c, r) and p["dir"] == rp.MM2S and p["ch"] == ch:
            p["seq"] = [
                [
                    ("bd", op[1], new) if op[0] == "bd" and op[2] == old else op
                    for op in block
                ]
                for block in p["seq"]
            ]
    syms = {
        s
        for s, a in d.allocs.items()
        if a["tile"] == (c, r) and a["dir"] == rp.MM2S and a["ch"] == ch
    }
    for s in syms:
        if d.allocs[s]["pkt"] == old:
            d.allocs[s]["pkt"] = new
    d.sequences = [
        [
            (
                ("memcpy", ev[1], new, *ev[3:])
                if ev[0] == "memcpy" and ev[1] in syms and ev[2] == old
                else ev
            )
            for ev in seq
        ]
        for seq in d.sequences
    ]


def drop_program(d, k):
    del d.programs[k]

    def fix(ev):
        if ev[0] not in ("start", "await"):
            return ev
        if ev[1] == k:
            return None
        return (ev[0], ev[1] - (ev[1] > k), *ev[2:])

    d.sequences = [[e for e in map(fix, s) if e is not None] for s in d.sequences]


def reorder_programs(d, order):
    where = {old: new for new, old in enumerate(order)}
    d.programs = [d.programs[k] for k in order]
    d.sequences = [
        [
            (ev[0], where[ev[1]], *ev[2:]) if ev[0] in ("start", "await") else ev
            for ev in s
        ]
        for s in d.sequences
    ]


def cut_capacity(t, r, up):
    """Channels from row r to r + 1 (up) or back, over all columns."""
    total = 0
    for c in range(t.cols):
        lo, hi = (c, r), (c, r + 1)
        if up:
            total += min(t.masters[lo][rp.NORTH], t.slaves[hi][rp.SOUTH])
        else:
            total += min(t.masters[hi][rp.SOUTH], t.slaves[lo][rp.NORTH])
    return total


def cut_load(d, r, up):
    """Channels the flows need across that cut: one per circuit tree that
    crosses it, and one for all packet flows that do."""
    trees = defaultdict(set)
    for s, x in d.flows:
        trees[s].add(x)

    def low(ep):
        return ep[1] <= r

    n = sum(
        1 for s, xs in trees.items() if low(s) == up and any(low(x) != up for x in xs)
    )
    packet = any(
        low(s) == up and low(x) != up
        for f in d.packet_flows
        for s in f["srcs"]
        for x in f["dsts"]
    )
    return n + packet


def must_reject(d):
    """Why no routing of `d` is legal, when that needs no search."""
    t = d.target
    if len(set(d.flows)) < len(d.flows):
        return "a flow twice"
    keys = [
        (f["id"], f["mask"], tuple(sorted(f["srcs"])), tuple(sorted(f["dsts"])))
        for f in d.packet_flows
    ]
    if len(set(keys)) < len(keys):
        return "a packet flow twice"
    into = defaultdict(set)
    for s, x in d.flows:
        into[x].add(s)
    for x, ss in into.items():
        if len(ss) > 1:
            return f"circuit flows from {len(ss)} sources into {rp.fmt_ep(x)}"
    csrc = {s for s, _ in d.flows}
    for f in d.packet_flows:
        for s in f["srcs"]:
            if s in csrc:
                return f"{rp.fmt_ep(s)} sends both circuit and packet flows"
        for x in f["dsts"]:
            if x in into:
                return f"{rp.fmt_ep(x)} receives both circuit and packet flows"
    for r in range(t.rows - 1):
        for up in (True, False):
            load, cap = cut_load(d, r, up), cut_capacity(t, r, up)
            if load > cap:
                return (
                    f"{load} trees must cross rows {r}/{r + 1} "
                    f"{'up' if up else 'down'} on {cap} channels"
                )
    return None


# Mutators: (rng, design, context) -> (what, design, expect) or None, where
# expect is "route", "reject" or "model".


def m_port_cap(rng, d, ctx):
    t = d.target
    k = rng.choice([-1, 0, 1])
    cuts = [(r, up) for r in range(t.rows - 1) for up in (True, False)]
    rng.shuffle(cuts)
    fs, fr = free_endpoints(d)
    for r, up in cuts:
        cap = cut_capacity(t, r, up)
        # The router takes longer than aie_opt's timeout to fill wider cuts.
        if cap > 64:
            continue
        need = cap + k - cut_load(d, r, up)
        ss = [ep for ep in fs if (ep[1] <= r) == up]
        xs = [ep for ep in fr if (ep[1] <= r) != up]
        if need <= 0 or need > min(len(ss), len(xs)):
            continue
        e = d.copy()
        e.flows += zip(rng.sample(ss, need), rng.sample(xs, need))
        what = f"{cap + k} trees {'up' if up else 'down'} across rows {r}/{r + 1}, {cap} channels"
        return what, e, "reject" if k > 0 else "model"
    return None


def m_add_flow(rng, d, ctx):
    t = d.target
    fs, fr = free_endpoints(d)
    packet = rng.random() < 0.5
    rng.shuffle(fs)
    for s in fs:
        xs = [x for x in fr if (rp.direct_ok if packet else circuit_ok)(t, s, x)]
        if not xs:
            continue
        x = rng.choice(xs)
        e = d.copy()
        if packet:
            pid = fresh_id(e, rng)
            if pid is None:
                return None
            e.add_packet_flow(pid, [s], [x])
        else:
            e.flows.append((s, x))
        return (
            f"{'packet' if packet else 'circuit'} {rp.fmt_ep(s)} -> {rp.fmt_ep(x)}",
            e,
            "model",
        )
    return None


def monotone(d, e):
    """Fewer streams cannot hurt, unless the keep a destination ends up with
    changes what its senders send, a prioritized source's route alone changes,
    or two streams no longer share a source or destination, so a hazard
    between them the router only warned of becomes one it has to avoid."""
    before, after = rp.last_keep(d), rp.last_keep(e)
    if any(before[x] != v for x, v in after.items()):
        return "model"
    prio = {s for f in d.packet_flows if f["priority"] for s in f["srcs"]}
    a, b = rp.Analysis(d), rp.Analysis(e)
    key = {(s.src, s.dst, s.pid): i for i, s in enumerate(a.streams)}
    old = [key.get((s.src, s.dst, s.pid)) for s in b.streams]
    if None in old or {k for k in key if k[0] in prio} != {
        (s.src, s.dst, s.pid) for s in b.streams if s.src in prio
    }:
        return "model"
    n = b.num_requested
    for s in range(n):
        for t in range(s + 1, n):
            if a.related(old[s], old[t]) and not b.related(s, t):
                return "model"
    return "route"


def m_remove_flow(rng, d, ctx):
    n = len(d.flows) + len(d.packet_flows)
    if n < 2:
        return None
    k = rng.randrange(n)
    e = d.copy()
    if k < len(d.flows):
        s, x = e.flows.pop(k)
        what = f"drop flow {rp.fmt_ep(s)} -> {rp.fmt_ep(x)}"
    else:
        f = e.packet_flows.pop(k - len(d.flows))
        what = f"drop packet flow id {f['id']}"
    return what, e, monotone(d, e)


def m_remove_end(rng, d, ctx):
    cands = [
        (k, key)
        for k, f in enumerate(d.packet_flows)
        for key in ("srcs", "dsts")
        if len(f[key]) > 1
    ]
    if not cands:
        return None
    k, key = rng.choice(cands)
    e = d.copy()
    ep = e.packet_flows[k][key].pop(rng.randrange(len(d.packet_flows[k][key])))
    return (
        f"drop {key[:-1]} {rp.fmt_ep(ep)} of id {d.packet_flows[k]['id']}",
        e,
        monotone(d, e),
    )


def m_add_dest(rng, d, ctx):
    t = d.target
    _, fr = free_endpoints(d)
    cands = [("packet", k) for k in range(len(d.packet_flows))]
    cands += [("circuit", s) for s in sorted({s for s, _ in d.flows})]
    rng.shuffle(cands)
    for kind, k in cands:
        e = d.copy()
        if kind == "packet":
            f = e.packet_flows[k]
            xs = [x for x in fr if all(rp.direct_ok(t, s, x) for s in f["srcs"])]
            if xs:
                f["dsts"].append(rng.choice(xs))
                return (
                    f"broadcast id {f['id']} to {rp.fmt_ep(f['dsts'][-1])}",
                    e,
                    "model",
                )
        else:
            xs = [x for x in fr if circuit_ok(t, k, x)]
            if xs:
                e.flows.append((k, rng.choice(xs)))
                return (
                    f"broadcast circuit {rp.fmt_ep(k)} to {rp.fmt_ep(e.flows[-1][1])}",
                    e,
                    "model",
                )
    return None


def m_id_reuse(rng, d, ctx):
    pf = d.packet_flows
    pairs = [
        (a, b)
        for a in range(len(pf))
        for b in range(len(pf))
        if a != b and pf[a]["id"] != pf[b]["id"]
    ]
    if not pairs:
        return None
    shared = [p for p in pairs if p in ctx["shared"]]
    on_link = bool(shared) and rng.random() < 0.7
    a, b = rng.choice(shared if on_link else pairs)
    e = d.copy()
    old, new = pf[a]["id"], pf[b]["id"]
    e.packet_flows[a]["id"] = new
    e.packet_flows[a]["mask"] = None
    for s in pf[a]["srcs"]:
        resend(e, s, old, new)
    return (
        f"id {old} -> {new}, the id of another flow"
        + (" on a shared link" if on_link else ""),
        e,
        "model",
    )


def m_multi_source(rng, d, ctx):
    t = d.target
    csrc = {s for s, _ in d.flows}
    fs, _ = free_endpoints(d)
    cands = list(range(len(d.packet_flows)))
    rng.shuffle(cands)
    for k in cands:
        f = d.packet_flows[k]
        ss = [
            s
            for s in fs
            if s not in csrc and all(rp.direct_ok(t, s, x) for x in f["dsts"])
        ]
        if ss:
            e = d.copy()
            s = rng.choice(ss)
            e.packet_flows[k]["srcs"].append(s)
            return f"id {f['id']} also from {rp.fmt_ep(s)}", e, "model"
    return None


def m_toggle(key):
    def mutate(rng, d, ctx):
        if not d.packet_flows:
            return None
        e = d.copy()
        k = rng.randrange(len(d.packet_flows))
        f = e.packet_flows[k]
        f[key] = True if not f[key] else rng.choice([False, None])
        return f"{key} of id {f['id']} -> {f[key]}", e, "model"

    return mutate


def m_ctrl(rng, d, ctx):
    t = d.target
    fs, _ = free_endpoints(d)
    _, dsts = busy(d)
    ctrl = [
        (*tile, rp.CTRL, 0)
        for tile in sorted(t.kinds)
        if t.masters[tile][rp.CTRL] and (*tile, rp.CTRL, 0) not in dsts
    ]
    rng.shuffle(fs)
    for s in fs:
        xs = [x for x in ctrl if rp.direct_ok(t, s, x)]
        if xs:
            e = d.copy()
            pid = fresh_id(e, rng)
            if pid is None:
                return None
            x = rng.choice(xs)
            e.add_packet_flow(pid, [s], [x], priority=rng.choice([True, None]))
            return f"control packets {rp.fmt_ep(s)} -> {rp.fmt_ep(x)}", e, "model"
    return None


def m_fixed(rng, d, ctx):
    con = ctx["witness"]()
    if con is None:
        return None
    e = rp.fix_ir(rng, d, con)
    return None if e is None else ("pin the witness of one flow", e, "model")


def free_box_ports(d, tile):
    ops = d.boxes.get(tile, [])
    slaves = {op[1] for op in ops if op[0] in ("connect", "rules")}
    masters = {op[2] for op in ops if op[0] == "connect"}
    masters |= {op[1] for op in ops if op[0] == "masterset"}
    t = d.target
    ss = [
        (b, i)
        for b in rp.DIRECTIONAL
        for i in range(t.slaves[tile][b])
        if (b, i) not in slaves
    ]
    ms = [
        (b, i)
        for b in rp.DIRECTIONAL
        for i in range(t.masters[tile][b])
        if (b, i) not in masters
    ]
    return ss, ms


def m_pin_config(rng, d, ctx):
    """A pre-placed connect, or packet rule, amsel and masterset, between
    ports no flow names."""
    t = d.target
    tiles = [x for x in d.used_tiles() if t.kind(x) != "shim"]
    if not tiles:
        return None
    tile = rng.choice(tiles)
    ss, ms = free_box_ports(d, tile)
    pairs = [(s, m) for s in ss for m in ms if t.legal(tile, s, m)]
    if not pairs:
        return None
    s, m = rng.choice(pairs)
    e = d.copy()
    box = e.boxes.setdefault(tile, [])
    if rng.random() < 0.5:
        box.append(("connect", s, m))
        return f"pin connect {rp.fmt_port(s)} -> {rp.fmt_port(m)} at {tile}", e, "model"
    used = {(op[2], op[3]) for op in box if op[0] == "amsel"}
    free = [
        (a, n)
        for a in range(rp.PARAMS["arbiters"])
        for n in range(rp.PARAMS["msels"])
        if (a, n) not in used
    ]
    if not free:
        return None
    a, n = rng.choice(free)
    pid = fresh_id(d, rng)
    if pid is None:
        return None
    box.insert(0, ("amsel", "pin", a, n))
    box.append(("masterset", m, ["pin"], None, False))
    box.append(("rules", s, [(rp.PARAMS["max_id"], pid, "pin")], False))
    return (
        f"pin amsel<{a}> ({n}) {rp.fmt_port(s)} -> {rp.fmt_port(m)} id {pid} at {tile}",
        e,
        "model",
    )


def m_neighbors(rng, d, ctx):
    """A circuit and a packet flow on neighboring ports at both ends."""
    t = d.target
    fs, fr = free_endpoints(d)
    by_tile = defaultdict(list)
    for s in fs:
        by_tile[("s", s[:2])].append(s)
    for x in fr:
        by_tile[("x", x[:2])].append(x)
    stiles = [k for k, v in by_tile.items() if k[0] == "s" and len(v) > 1]
    xtiles = [k for k, v in by_tile.items() if k[0] == "x" and len(v) > 1]
    rng.shuffle(stiles)
    for sk in stiles:
        s0, s1 = sorted(by_tile[sk])[:2]
        for xk in rng.sample(xtiles, len(xtiles)):
            x0, x1 = sorted(by_tile[xk])[:2]
            if rng.random() < 0.5:
                x0, x1 = x1, x0
            if circuit_ok(t, s0, x0) and rp.direct_ok(t, s1, x1):
                e = d.copy()
                pid = fresh_id(e, rng)
                if pid is None:
                    return None
                e.flows.append((s0, x0))
                e.add_packet_flow(pid, [s1], [x1])
                return (
                    f"circuit {rp.fmt_ep(s0)} -> {rp.fmt_ep(x0)} beside packet "
                    f"{rp.fmt_ep(s1)} -> {rp.fmt_ep(x1)}",
                    e,
                    "model",
                )
    return None


def m_same_port(rng, d, ctx):
    """One port both circuit and packet switched, or two circuit flows into
    one port."""
    t = d.target
    fs, fr = free_endpoints(d)
    csrc = sorted({s for s, _ in d.flows})
    cdst = sorted({x for _, x in d.flows})
    pdst = sorted({x for f in d.packet_flows for x in f["dsts"]})
    e = d.copy()
    ways = []
    if csrc:
        ways.append("packet from a circuit source")
    if pdst:
        ways.append("circuit into a packet destination")
    if cdst:
        ways += ["packet into a circuit destination", "second circuit into one port"]
    rng.shuffle(ways)
    for way in ways:
        pid = fresh_id(e, rng)
        if way == "packet from a circuit source":
            s = rng.choice(csrc)
            xs = [x for x in fr if rp.direct_ok(t, s, x)]
            if xs and pid is not None:
                e.add_packet_flow(pid, [s], [rng.choice(xs)])
                return way, e, "reject"
        else:
            x = rng.choice(pdst if way.startswith("circuit into") else cdst)
            ss = [s for s in fs if circuit_ok(t, s, x)]
            if not ss:
                continue
            if way.startswith("packet"):
                if pid is None:
                    continue
                e.add_packet_flow(pid, [rng.choice(ss)], [x])
            else:
                e.flows.append((rng.choice(ss), x))
            return way, e, "reject"
    return None


def m_duplicate(rng, d, ctx):
    n = len(d.flows) + len(d.packet_flows)
    if not n:
        return None
    k = rng.randrange(n)
    e = d.copy()
    if k < len(d.flows):
        e.flows.insert(rng.randrange(len(e.flows) + 1), d.flows[k])
        return "a flow twice", e, "reject"
    f = dict(e.packet_flows[k - len(d.flows)])
    f["srcs"], f["dsts"] = list(f["srcs"]), list(f["dsts"])
    rng.shuffle(f["srcs"])
    rng.shuffle(f["dsts"])
    e.packet_flows.append(f)
    return "a packet flow twice", e, "reject"


def m_drop_program(rng, d, ctx):
    cands = [k for k, p in enumerate(d.programs) if p["kind"] in ("start", "task")]
    if not cands:
        return None
    k = rng.choice(cands)
    e = d.copy()
    p = d.programs[k]
    drop_program(e, k)
    return f"drop the {rp.DIRS[p['dir']]} {p['ch']} program at {p['tile']}", e, "model"


def m_add_programs(rng, d, ctx):
    t = d.target
    busy_tiles = {p["tile"] for p in d.programs} | set(d.cores)
    tiles = [
        x
        for x in d.used_tiles()
        if x not in busy_tiles
        and t.kind(x) != "shim"
        and not any(v[:2] == x for v in d.locks.values())
    ]
    if not tiles:
        return None
    tile = rng.choice(tiles)
    e = d.copy()
    rp.add_programs(rng, e, [tile])
    if len(e.programs) == len(d.programs):
        return None
    return f"program the DMA at {tile}", e, "model"


def m_move_endpoint(rng, d, ctx):
    t = d.target
    fs, fr = free_endpoints(d)
    cands = [("flow", k, end) for k in range(len(d.flows)) for end in (0, 1)]
    cands += [
        ("packet", k, key, i)
        for k, f in enumerate(d.packet_flows)
        for key in ("srcs", "dsts")
        for i in range(len(f[key]))
    ]
    rng.shuffle(cands)
    csrc = {s for s, _ in d.flows}
    for c in cands:
        e = d.copy()
        if c[0] == "flow":
            s, x = d.flows[c[1]]
            here = (s, x)[c[2]]
            pool = fs if c[2] == 0 else fr
            pool = [p for p in pool if t.kind(p[:2]) != t.kind(here[:2])]
            if c[2] == 0:
                pool = [p for p in pool if circuit_ok(t, p, x)]
            else:
                pool = [p for p in pool if circuit_ok(t, s, p)]
            if not pool:
                continue
            new = rng.choice(pool)
            e.flows[c[1]] = (new, x) if c[2] == 0 else (s, new)
        else:
            f = e.packet_flows[c[1]]
            here = f[c[2]][c[3]]
            pool = fs if c[2] == "srcs" else fr
            pool = [p for p in pool if t.kind(p[:2]) != t.kind(here[:2])]
            if c[2] == "srcs":
                pool = [
                    p
                    for p in pool
                    if p not in csrc and all(rp.direct_ok(t, p, x) for x in f["dsts"])
                ]
            else:
                pool = [
                    p for p in pool if all(rp.direct_ok(t, s, p) for s in f["srcs"])
                ]
            if not pool:
                continue
            new = rng.choice(pool)
            f[c[2]][c[3]] = new
        return (
            f"move {rp.fmt_ep(here)} ({t.kind(here[:2])}) to {rp.fmt_ep(new)} "
            f"({t.kind(new[:2])})",
            e,
            "model",
        )
    return None


MUTATORS = {
    "port-cap": m_port_cap,
    "add-flow": m_add_flow,
    "remove-flow": m_remove_flow,
    "remove-end": m_remove_end,
    "add-dest": m_add_dest,
    "id-reuse": m_id_reuse,
    "multi-source": m_multi_source,
    "priority": m_toggle("priority"),
    "keep": m_toggle("keep"),
    "ctrl": m_ctrl,
    "fixed-witness": m_fixed,
    "pin-config": m_pin_config,
    "neighbors": m_neighbors,
    "same-port": m_same_port,
    "duplicate": m_duplicate,
    "drop-program": m_drop_program,
    "add-programs": m_add_programs,
    "move-endpoint": m_move_endpoint,
}


# Router bugs already reported, by the must_reject reason they show as.
REPORTED = {
    "circuit and packet flows from one port merged": "sends both circuit and packet flows",
}


def judge(e, expect, hops_on, seed):
    """Route `e` and hold the router to `expect`. Returns (outcome, detail),
    outcome one of ok, completeness, soundness, illegal, crash, known,
    model-limit."""
    why = must_reject(e)
    try:
        e = rp.canonical(e)
    except Exception as x:
        if not why:
            return "model-limit", f"emit: {x!r}"
        # The IR verifier rejects it before the router runs.
        p = rp.aie_opt(e.emit(), hops_on, timeout=120)
        if p.returncode == 0:
            return "soundness", f"router routes it, must reject: {why}"
        return "ok", f"rejected by the verifier ({why})"
    why = why or must_reject(e)
    if why:
        expect = "reject"
    elif expect == "model":
        try:
            v = rp.verdict(e, hops_on, seed)
        except Exception as x:
            return "model-limit", f"verdict: {x!r}"
        expect = {True: "route", False: "reject", None: None}[v["routable"]]
        why = v["reason"]
    p = rp.aie_opt(e.emit(), hops_on, timeout=120)
    if p.returncode < 0:
        return "crash", rp.first_error(p.stderr)[:300]
    if p.returncode:
        err = rp.first_error(p.stderr)
        if expect != "route":
            return "ok", f"rejected ({expect})"
        if rp.known_bug(e, [err]):
            return "known", err[:200]
        return "completeness", f"model routes it, router: {err[:300]}"
    if expect == "reject":
        label = next((k for k, v in REPORTED.items() if v in (why or "")), None)
        if label:
            return "known", label
        return "soundness", f"router routes it, must reject: {why}"
    try:
        problems, _ = rp.verify(e, rp.Analysis(e), p.stdout, hops_on)
    except Exception as x:
        return "model-limit", f"verify: {x!r}"
    if problems:
        if rp.known_bug(e, problems):
            return "known", problems[0][:200]
        return "illegal", problems[0][:300]
    return "ok", f"routed ({expect})"


# Metamorphic relations: (rng, design, routed output) -> (what, design,
# check) or None; check(output of the new design) returns a problem or None.


def routing_key(text):
    """The switch settings of a routed output, independent of op order and
    amsel names."""
    out = rp.load_design(text)
    key = set()
    for tile, ops in out.boxes.items():
        conns, amsels, ms, rules, _ = rp.box_view(ops)
        for s, m in conns.items():
            key |= {(tile, "c", s, x) for x in m}
        for port, op in ms.items():
            key.add(
                (
                    tile,
                    "m",
                    port,
                    tuple(sorted(amsels.get(n) for n in op[2])),
                    op[3],
                    op[4],
                )
            )
        for port, op in rules.items():
            key |= {(tile, "r", port, mk, v, amsels.get(n)) for mk, v, n in op[2]}
    for tile, conns in out.muxes.items():
        key |= {(tile, "x", s, m) for s, m in conns}
    return frozenset(key)


def r_permute(rng, d, routed):
    e = d.copy()
    rng.shuffle(e.flows)
    rng.shuffle(e.packet_flows)
    for f in e.packet_flows:
        rng.shuffle(f["srcs"])
        rng.shuffle(f["dsts"])
    order = list(range(len(e.programs)))
    rng.shuffle(order)
    reorder_programs(e, order)
    locks = list(e.locks.items())
    rng.shuffle(locks)
    e.locks = dict(locks)
    cores = list(e.cores.items())
    rng.shuffle(cores)
    e.cores = dict(cores)
    return "permute flows, ends, programs, locks and cores", e, None


def rename(d, c):
    """Every packet id x becomes x ^ c, which keeps every mask match."""
    e = d.copy()
    for f in e.packet_flows:
        f["id"] ^= c
        if f["mask"] is not None:
            f["id"] &= f["mask"]
    for p in e.programs:
        p["seq"] = [
            [
                ("bd", op[1], op[2] ^ c) if op[0] == "bd" and op[2] is not None else op
                for op in block
            ]
            for block in p["seq"]
        ]
    for a in e.allocs.values():
        if a["pkt"] is not None:
            a["pkt"] ^= c
    e.sequences = [
        [
            (
                ("memcpy", ev[1], ev[2] ^ c, *ev[3:])
                if ev[0] == "memcpy" and ev[2] is not None
                else ev
            )
            for ev in s
        ]
        for s in e.sequences
    ]
    for tile, ops in e.boxes.items():
        e.boxes[tile] = [
            (
                ("rules", op[1], [(m, (v ^ c) & m, n) for m, v, n in op[2]], op[3])
                if op[0] == "rules"
                else op
            )
            for op in ops
        ]
    return e


def r_rename(rng, d, routed):
    if not d.packet_flows:
        return None
    c = rng.randrange(1, rp.PARAMS["max_id"] + 1)
    return f"ids x -> x ^ {c}", rename(d, c), None


def find_flows(text):
    p = subprocess.run(
        [rp.aie_opt_path(), "--aie-find-flows", "-"],
        input=text,
        capture_output=True,
        text=True,
    )
    return p


def flow_set(d):
    return (
        frozenset(d.flows),
        frozenset(
            (f["id"], f["mask"], frozenset(f["srcs"]), frozenset(f["dsts"]))
            for f in d.packet_flows
        ),
    )


def packet_pairs(d):
    return {
        (s, x, f["id"]) for f in d.packet_flows for s in f["srcs"] for x in f["dsts"]
    }


def delivered(x, d):
    """(source, destination, id) the packet flows of `x` carry, for the ids
    the sources of `d` send and a flow of `d` claims. Where an id a source
    sends goes, when no flow claims it, is up to the router."""
    ids = defaultdict(set)
    claims = defaultdict(set)
    for f in d.packet_flows:
        m = rp.PARAMS["max_id"] if f["mask"] is None else f["mask"]
        for s in f["srcs"]:
            ids[s].add(f["id"])
            claims[s].add((m, f["id"] & m))
    for (c, r, dr, ch), v in rp.sent_packet_ids(d).items():
        s = (c, r, rp.DMA, ch)
        if dr == rp.MM2S:
            ids[s] |= {pid for pid in v if any(pid & m == x for m, x in claims[s])}
    out = set()
    for f in x.packet_flows:
        m = rp.PARAMS["max_id"] if f["mask"] is None else f["mask"]
        for s in f["srcs"]:
            for pid in ids[s]:
                if pid & m == f["id"] & m:
                    out |= {(s, t, pid) for t in f["dsts"]}
    return out


def r_lift(rng, d, routed):
    """aie-find-flows lifts the routing back into the flows the design asked
    for, and routing those again lifts to the same."""
    p = find_flows(routed)
    if p.returncode:
        return "lift", None, f"aie-find-flows fails: {rp.first_error(p.stderr)[:200]}"
    lifted = rp.load_design(p.stdout)
    out = rp.load_design(routed)
    left = {
        (tile, op[1])
        for tile, ops in lifted.boxes.items()
        for op in ops
        if op[0] != "amsel"
    }

    def partial(src, dst, pid):
        """find-flows leaves what it cannot lift as switchbox configuration."""
        hops = rp.trace_output(out, src, pid)[0].get(dst, [])
        return any((tile, s) in left or (tile, m) in left for tile, s, m, _ in hops)

    asked, got = delivered(d, d), delivered(lifted, d)
    lost = sorted(set(d.flows) - set(lifted.flows))
    lost += sorted(x for x in asked - got if not partial(*x))
    extra = []
    if not (d.boxes or d.muxes):
        extra = sorted(set(lifted.flows) - set(d.flows)) + sorted(got - asked)
    if lost or extra:
        return "lift", None, f"lifting the routing loses {lost}, adds {extra}"

    def check(text):
        q = find_flows(text)
        if q.returncode:
            return f"aie-find-flows fails on the rerouted output: {rp.first_error(q.stderr)[:200]}"
        again = rp.load_design(q.stdout)
        if left:
            if not set(lifted.flows) <= set(again.flows) or not got <= delivered(
                again, d
            ):
                return "lifting the rerouted output loses flows"
        elif set(again.flows) != set(lifted.flows) or delivered(again, d) != got:
            return "lifting the rerouted output gives other flows"
        return None

    check.partial = bool(left)
    return "lift and reroute", lifted, check


def r_repeat(rng, d, routed):
    return (
        "repeat",
        d.copy(),
        lambda text: None if text == routed else "output differs on a repeat run",
    )


def r_disjoint(rng, d, routed):
    """A circuit flow on a column whose switchboxes nothing uses, in the design
    or in its routing."""
    t = d.target
    out = rp.load_design(routed)
    used = {x[0] for x in d.used_tiles()} | {x[0] for x in out.boxes if out.boxes[x]}
    used |= {x[0] for x in out.muxes}
    cols = [c for c in range(t.cols) if c not in used]
    rng.shuffle(cols)
    for c in cols:
        ss = [ep for ep in endpoints(d, True) if ep[0] == c]
        xs = [ep for ep in endpoints(d, False) if ep[0] == c]
        pairs = [
            (s, x) for s in ss for x in xs if s[:2] != x[:2] and circuit_ok(t, s, x)
        ]
        if pairs:
            e = d.copy()
            e.flows.append(rng.choice(pairs))
            return (
                f"circuit {rp.fmt_ep(e.flows[-1][0])} -> {rp.fmt_ep(e.flows[-1][1])} on free column {c}",
                e,
                None,
            )
    return None


def dangling_amsel(d):
    """An amsel no masterset uses, which the router does not reserve."""
    for ops in d.boxes.values():
        used = {n for op in ops if op[0] == "masterset" for n in op[2]}
        if any(op[0] == "amsel" and op[1] not in used for op in ops):
            return True
    return False


RELATIONS = {
    "permute": r_permute,
    "rename": r_rename,
    "lift": r_lift,
    "repeat": r_repeat,
    "disjoint": r_disjoint,
}


# The verifier's kill rate: corruptions of a routed output that are wrong.


def op_sites(out, kind):
    return [
        (tile, i)
        for tile, ops in sorted(out.boxes.items())
        for i, op in enumerate(ops)
        if op[0] == kind
    ]


def k_msel_range(rng, out, ctx):
    sites = op_sites(out, "amsel")
    if not sites:
        return None
    tile, i = rng.choice(sites)
    op = out.boxes[tile][i]
    if rng.random() < 0.5:
        out.boxes[tile][i] = (op[0], op[1], op[2], rp.PARAMS["msels"])
        return f"amsel msel {rp.PARAMS['msels']} at {tile}"
    out.boxes[tile][i] = (op[0], op[1], rp.PARAMS["arbiters"], op[3])
    return f"amsel arbiter {rp.PARAMS['arbiters']} at {tile}"


def used_masters(out, tile):
    view = rp.box_view(out.boxes[tile])
    return {m for ms in view[0].values() for m in ms} | set(view[2])


def k_connect(rng, out, ctx):
    sites = op_sites(out, "connect")
    if not sites:
        return None
    tile, i = rng.choice(sites)
    op = out.boxes[tile][i]
    if rng.random() < 0.3:
        del out.boxes[tile][i]
        return f"drop connect {rp.fmt_port(op[1])} -> {rp.fmt_port(op[2])} at {tile}"
    t = out.target
    busy_m = used_masters(out, tile)
    ms = [
        m for m in t.master_ports(tile) if m not in busy_m and t.legal(tile, op[1], m)
    ]
    if not ms:
        return None
    m = rng.choice(ms)
    out.boxes[tile][i] = ("connect", op[1], m)
    return f"connect {rp.fmt_port(op[1])} -> {rp.fmt_port(m)}, not {rp.fmt_port(op[2])}, at {tile}"


def k_rule(rng, out, ctx):
    sites = [
        (tile, i, j)
        for tile, i in op_sites(out, "rules")
        for j in range(len(out.boxes[tile][i][2]))
    ]
    if not sites:
        return None
    tile, i, j = rng.choice(sites)
    op = out.boxes[tile][i]
    mask, value, name = op[2][j]
    rules = list(op[2])
    ids = ctx["ids_at"].get((tile, op[1]), set())

    def first(x):
        return next((k for k, r in enumerate(rules) if x & r[0] == r[1] & r[0]), None)

    # Only ids a later rule sends elsewhere: the rule is first in line for them.
    others = {
        x for x in ids if (k := first(x)) is not None and k > j and rules[k][2] != name
    }
    how = rng.random()
    if how < 0.4 and mask:
        b = rng.choice([b for b in range(5) if mask >> b & 1])
        rules[j] = (mask, value ^ (1 << b), name)
        what = f"rule value {value} -> {value ^ (1 << b)}"
    elif how < 0.8 and others:
        # Widen the mask until the rule catches an id another rule handles.
        x = rng.choice(sorted(others))
        new = mask & ~(x ^ value)
        rules[j] = (new, value & new, name)
        what = f"rule mask {mask} -> {new} catches id {x}"
    else:
        # Not an amsel that already takes these ids where this one does: that
        # merges two streams of one flow, which is legal.
        mine = {x for x in ids if first(x) == j}
        other = [
            n
            for n in {r[2] for r in rules} | set(ctx["amsels"][tile])
            if n != name
            and not (
                mine
                and all(
                    ctx["reach"][(tile, n, x)] == ctx["via"][(tile, op[1], x)]
                    for x in mine
                )
            )
        ]
        if not other:
            return None
        n = rng.choice(sorted(other))
        rules[j] = (mask, value, n)
        what = f"rule ({mask}, {value}) to another amsel"
    out.boxes[tile][i] = (op[0], op[1], rules, op[3])
    return f"{what} on {rp.fmt_port(op[1])} at {tile}"


def k_masterset(rng, out, ctx):
    sites = op_sites(out, "masterset")
    if not sites:
        return None
    tile, i = rng.choice(sites)
    op = out.boxes[tile][i]
    how = rng.random()
    t = out.target
    if how < 0.25:
        del out.boxes[tile][i]
        return f"drop masterset {rp.fmt_port(op[1])} at {tile}"
    if how < 0.5:
        busy_m = used_masters(out, tile)
        ms = [m for m in t.master_ports(tile) if m not in busy_m]
        if not ms:
            return None
        m = rng.choice(ms)
        out.boxes[tile].append(("masterset", m, list(op[2]), op[3], op[4]))
        return (
            f"extra masterset {rp.fmt_port(m)} copying {rp.fmt_port(op[1])} at {tile}"
        )
    if how < 0.75:
        keep = not rp.keeps_header(tile, op[1], op[3])
        out.boxes[tile][i] = (op[0], op[1], op[2], keep, op[4])
        return f"keep_pkt_header on {rp.fmt_port(op[1])} at {tile} -> {keep}"
    out.boxes[tile][i] = (op[0], op[1], op[2], op[3], not op[4])
    return f"is_ctrl_pkt_overlay on {rp.fmt_port(op[1])} at {tile} -> {not op[4]}"


def k_mux(rng, out, ctx):
    sites = [(tile, i) for tile, conns in out.muxes.items() for i in range(len(conns))]
    if not sites:
        return None
    tile, i = rng.choice(sites)
    s, m = out.muxes[tile][i]
    if rng.random() < 0.5:
        del out.muxes[tile][i]
        return f"drop shim mux {rp.fmt_port(s)} -> {rp.fmt_port(m)} at {tile}"
    t = out.target
    if m[0] == rp.NORTH:
        n = t.slaves[tile][rp.SOUTH]
    else:
        n = t.mux_masters[tile][m[0]]
    other = [
        c
        for c in range(n)
        if c != m[1] and (m[0], c) not in {x for _, x in out.muxes[tile]}
    ]
    if not other:
        return None
    c = rng.choice(other)
    out.muxes[tile][i] = (s, (m[0], c))
    return (
        f"shim mux {rp.fmt_port(s)} -> {rp.fmt_port((m[0], c))}, not {rp.fmt_port(m)}"
    )


def k_conflict_arbiter(rng, out, ctx):
    """Move a stream onto the arbiter of a stream it conflicts with."""
    for tile, a_name, b_name in ctx["conflict_sites"]:
        amsels = {op[1]: k for k, op in enumerate(out.boxes[tile]) if op[0] == "amsel"}
        if a_name not in amsels or b_name not in amsels:
            continue
        ka, kb = amsels[a_name], amsels[b_name]
        oa, ob = out.boxes[tile][ka], out.boxes[tile][kb]
        taken = {(op[2], op[3]) for op in out.boxes[tile] if op[0] == "amsel"}
        free = [m for m in range(rp.PARAMS["msels"]) if (ob[2], m) not in taken]
        if not free:
            continue
        out.boxes[tile][ka] = (oa[0], oa[1], ob[2], free[0])
        return f"conflicting streams on arbiter {ob[2]} at {tile}"
    return None


def k_five_msels(rng, out, ctx):
    """Five amsels on one arbiter: one more than it has msels."""
    for tile, ops in sorted(out.boxes.items()):
        am = [k for k, op in enumerate(ops) if op[0] == "amsel"]
        if len(am) > rp.PARAMS["msels"]:
            arb = ops[am[0]][2]
            for n, k in enumerate(am[: rp.PARAMS["msels"] + 1]):
                ops[k] = (ops[k][0], ops[k][1], arb, n)
            return f"{rp.PARAMS['msels'] + 1} amsels on arbiter {arb} at {tile}"
    return None


KILLERS = {
    "msel-range": k_msel_range,
    "connect": k_connect,
    "rule": k_rule,
    "masterset": k_masterset,
    "shim-mux": k_mux,
    "conflict-arbiter": k_conflict_arbiter,
    "five-msels": k_five_msels,
}


def kill_context(d, an, routed):
    """Where the routing's streams go, to aim corruptions that are wrong."""
    out = rp.load_design(routed)
    ids_at = defaultdict(set)
    hops_of = {}
    reach, via = defaultdict(set), defaultdict(set)
    names = {
        tile: {(op[2], op[3]): op[1] for op in ops if op[0] == "amsel"}
        for tile, ops in out.boxes.items()
    }
    for i, s in enumerate(an.streams[: an.num_requested]):
        ends, _ = rp.trace_output(out, s.src, s.pid)
        hops = ends.get(s.dst, [])
        hops_of[i] = hops
        for tile, slave, m, arb in hops:
            if s.pid is not None:
                ids_at[(tile, slave)].add(s.pid)
            if arb:
                reach[(tile, names[tile][arb], s.pid)].add(s.dst)
                via[(tile, slave, s.pid)].add(s.dst)
    conflict_sites = []
    for i in hops_of:
        for j in hops_of:
            if i < j and an.conflict(i, j):
                ai = {tile: arb for tile, _, _, arb in hops_of[i] if arb}
                aj = {tile: arb for tile, _, _, arb in hops_of[j] if arb}
                for tile in set(ai) & set(aj):
                    if ai[tile][0] != aj[tile][0]:
                        conflict_sites.append(
                            (tile, names[tile][ai[tile]], names[tile][aj[tile]])
                        )
    amsels = {
        tile: [op[1] for op in ops if op[0] == "amsel"]
        for tile, ops in out.boxes.items()
    }
    return dict(
        ids_at=ids_at,
        conflict_sites=conflict_sites,
        amsels=amsels,
        reach=reach,
        via=via,
    )


def shared_pairs(d, routed):
    """Packet flows whose streams share a link in the routing."""
    out = rp.load_design(routed)
    links = defaultdict(set)
    for k, f in enumerate(d.packet_flows):
        for s in f["srcs"]:
            ends, _ = rp.trace_output(out, s, f["id"])
            for hops in ends.values():
                for tile, _, m, _ in hops:
                    links[k].add((tile, m))
    return {(a, b) for a in links for b in links if a != b and links[a] & links[b]}


# Driver.


def run_seed(seed, args):
    rows = []
    hops_on = not args.hops_off
    try:
        case = rp.generate(seed, device=args.device)
    except Exception:
        return [
            ("base", "model-limit", "generate", traceback.format_exc(limit=2), None)
        ]
    if case is None:
        return rows
    d = case["design"]
    p = rp.aie_opt(d.emit(), hops_on, timeout=120)
    if p.returncode:
        return [("base", "base-fails", "", rp.first_error(p.stderr)[:200], d)]
    routed = p.stdout
    an = rp.Analysis(d)
    problems, _ = rp.verify(d, an, routed, hops_on)
    if problems:
        return [("base", "base-illegal", "", problems[0][:200], d)]

    witness = []

    def get_witness():
        if not witness:
            c = d.copy()
            before = rp.design_signature(c)
            built = rp.construct(c, seed)
            witness.append(
                built[0] if built and rp.design_signature(c) == before else None
            )
        return witness[0]

    ctx = dict(shared=shared_pairs(d, routed), witness=get_witness)
    for name, fn in MUTATORS.items():
        if args.only and name not in args.only:
            continue
        try:
            m = fn(random.Random(f"{seed}-{name}"), d, ctx)
        except Exception:
            rows.append(
                ("mutator", "model-limit", name, traceback.format_exc(limit=3), d)
            )
            continue
        if m is None:
            rows.append(("mutator", "n/a", name, "", None))
            continue
        what, e, expect = m
        outcome, detail = judge(e, expect, hops_on, seed)
        rows.append(("mutator", outcome, name, f"{what}: {detail}", (e, expect)))

    trees = {}
    for name, fn in RELATIONS.items():
        if args.only and name not in args.only:
            continue
        try:
            m = fn(random.Random(f"{seed}-{name}"), d, routed)
        except Exception:
            rows.append(
                ("relation", "model-limit", name, traceback.format_exc(limit=3), d)
            )
            continue
        if m is None:
            rows.append(("relation", "n/a", name, "", None))
            continue
        what, e, check = m
        if e is None:
            rows.append(("relation", "broken", name, f"{what}: {check}", d))
            continue
        q = rp.aie_opt(e.emit(), hops_on, timeout=120)
        # Splitting, merging or reordering flows can change the route a
        # prioritized source takes alone, which the rest has to route around.
        if q.returncode and rp.priority_trees(d, trees) != rp.priority_trees(e, trees):
            outcome, detail = judge(e, "model", hops_on, seed)
            rows.append(("relation", outcome, name, f"{what}: {detail}", e))
            continue
        if q.returncode:
            err = rp.first_error(q.stderr)
            kind = "known" if rp.known_bug(e, [err]) else "broken"
            if getattr(check, "partial", False):
                kind = "partial-lift"
            if name == "lift" and "which both match id" in err:
                kind = "lift-overlap"
            rows.append(
                ("relation", kind, name, f"{what}: router fails: {err[:200]}", e)
            )
            continue
        try:
            ce = rp.canonical(e)
            probs, _ = rp.verify(ce, rp.Analysis(ce), q.stdout, hops_on)
        except Exception as x:
            rows.append(("relation", "model-limit", name, f"{what}: {x!r}", e))
            continue
        bad = probs[0] if probs else (check(q.stdout) if check else None)
        same = name == "permute" and routing_key(q.stdout) == routing_key(routed)
        if bad:
            rows.append(("relation", "broken", name, f"{what}: {bad[:300]}", e))
        else:
            rows.append(
                (
                    "relation",
                    "ok",
                    name,
                    f"{what}" + (" (same routing)" if same else ""),
                    None,
                )
            )

    if not args.only or "kill" in args.only:
        kctx = kill_context(d, an, routed)
        for name, fn in KILLERS.items():
            out = rp.load_design(routed)
            try:
                what = fn(random.Random(f"{seed}-{name}"), out, kctx)
            except Exception:
                rows.append(
                    ("kill", "model-limit", name, traceback.format_exc(limit=3), None)
                )
                continue
            if what is None:
                rows.append(("kill", "n/a", name, "", None))
                continue
            try:
                probs, _ = rp.verify(d, an, out.emit(), hops_on)
            except Exception as x:
                rows.append(
                    ("kill", "killed", name, f"{what}: verifier raised {x!r}", None)
                )
                continue
            rows.append(
                ("kill", "killed" if probs else "survived", name, what, out.emit())
            )
    return rows


def shrink_failure(kind, name, payload, hops_on, seed, verbose):
    """The smallest design that still fails the same way."""
    if kind == "mutator":
        e, expect = payload
        # "route" rests on the router routing the unmutated design, which a
        # smaller design no longer has, so only the model can vouch for it.
        if expect == "route":
            expect = "model"
            if judge(e, expect, hops_on, seed)[0] == "ok":
                return e
        first, _ = judge(e, expect, hops_on, seed)

        def holds(x):
            try:
                return judge(x, expect, hops_on, seed)[0] == first
            except Exception:
                return False

        return shrinker.shrink(rp.canonical(e), holds)
    return payload


def main():
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cli.add_argument("--device", default="npu2")
    cli.add_argument("--seeds", type=int, default=20)
    cli.add_argument("--first-seed", type=int, default=0)
    cli.add_argument("--hops-off", action="store_true")
    cli.add_argument("--only", nargs="*", default=None)
    cli.add_argument("--jobs", type=int, default=4)
    cli.add_argument("--shrink", action="store_true")
    cli.add_argument("--repros", type=Path, default=Path("repros"))
    cli.add_argument("--out", type=Path, default=None, help="save failing cases")
    cli.add_argument("-v", action="store_true")
    args = cli.parse_args()
    t0 = time.monotonic()
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda s: (s, run_seed(s, args)), seeds))

    tally = defaultdict(Counter)
    failures = []
    bad = {
        "completeness",
        "soundness",
        "illegal",
        "crash",
        "broken",
        "survived",
        "base-fails",
        "base-illegal",
        "model-limit",
    }
    for seed, rows in results:
        for kind, outcome, name, detail, payload in rows:
            tally[(kind, name)][outcome] += 1
            if outcome in bad:
                failures.append((seed, kind, outcome, name, detail, payload))
    for (kind, name), c in sorted(tally.items()):
        print(
            f"{kind:9} {name:18} " + ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
        )
    kills = Counter()
    for (kind, _), c in tally.items():
        if kind == "kill":
            kills.update(c)
    total = kills["killed"] + kills["survived"]
    print(
        f"kill rate: {kills['killed']}/{total}"
        + (f" = {kills['killed'] / total:.3f}" if total else "")
    )
    seen = Counter()
    for seed, kind, outcome, name, detail, payload in failures:
        seen[(kind, outcome, name)] += 1
        if seen[(kind, outcome, name)] <= (10 if args.v else 3):
            print(f"FAIL seed {seed} {kind} {name} {outcome}: {detail.strip()[:400]}")
        if args.out and payload is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            text = (
                payload
                if isinstance(payload, str)
                else (
                    payload[0].emit() if isinstance(payload, tuple) else payload.emit()
                )
            )
            (args.out / f"{kind}-{name}-{outcome}-{seed}.mlir").write_text(text)
    if args.shrink:
        done = set()
        for seed, kind, outcome, name, detail, payload in failures:
            if (
                kind != "mutator"
                or outcome in ("model-limit",)
                or (name, outcome) in done
            ):
                continue
            done.add((name, outcome))
            small = shrink_failure(kind, name, payload, not args.hops_off, seed, args.v)
            args.repros.mkdir(parents=True, exist_ok=True)
            path = args.repros / f"{name}-{outcome}-{seed}.mlir"
            head = f"// router_mutation.py seed {seed} {args.device}, {name}: {outcome}\n// {detail.strip()[:300]}\n"
            path.write_text(head + small.emit())
            print(f"shrunk to {path} (size {shrinker.size(small)})")
    print(f"total: {time.monotonic() - t0:.1f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
