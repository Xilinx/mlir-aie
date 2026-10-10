#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The route space the router's tests reach."""

import itertools
import re
from collections import Counter, defaultdict

from .params import PARAMS
from .fabric import (
    BUNDLES,
    CTRL,
    DIRECTIONAL,
    DMA,
    EAST,
    MM2S,
    NORTH,
    SOUTH,
    TRACE,
    Target,
    WEST,
    phys_dst,
    phys_src,
)
from .design import load_design
from .deadlock import Analysis, program_ops, sent_packet_ids
from .output import box_view, trace_output
from .generate import devices_of, ids_by_source

# The route space: what a design asks of the router and what the router did,
# as values along a few dimensions, next to every value the TargetModel
# allows. route_space() places one design; space_domain() is the denominator.

PAIRWISE = ("tiles", "geometry", "kind", "load", "conflict")
EDGE_NAMES = ("lock", "stream", "host")
STALLS = ("unknown-volume", "unbuffered", "overrun")
SATURATION = ("<50%", "50-99%", "100%", ">100%")
KINDS = ("unicast", "multicast", "fan-in", "fan-in-multicast")
PRE_IR = (
    "none",
    "connect",
    "rules",
    "masked-rule",
    "amsel",
    "dangling-amsel",
    "masterset",
    "multi-amsel-masterset",
    "keep-masterset",
    "ctrl-masterset",
    "shim-mux",
    "pinned-stream",
    "partial-pinned-stream",
)
ID_SETS = (
    "no-packet-flows",
    "id-0",
    "id-max",
    "id-other",
    "masked",
    "id-reuse",
    "same-id-two-sources",
    "multi-id-source",
    "priority-shares-id",
    "sent-by-bd",
    "sent-by-alloc",
    "sent-by-memcpy",
    "sender-unknown-id",
    "sender-extra-id",
)
# Router outcomes, one per emitError site (AIECreatePathFindFlows.cpp,
# AIEPathFinder.cpp) or rejection reason; the first that matches names it.
OUTCOMES = (
    ("cover misses", r"packet rule cover misses"),
    ("cover claims", r"packet rule cover claims another"),
    ("overfull tile", r"can share an arbiter, and each takes one there"),
    ("same-id split", r"leave by different ports; a switchbox routes on the id"),
    ("rule slots (plan)", r"packet rules, and a slave port holds"),
    ("msels", r"need more arbiter msels than the switchbox"),
    ("keep apart", r"no routing found keeps them apart"),
    ("hold cycle", r"deadlock holding arbiters across switchboxes"),
    (
        "overlay masters",
        r"not by one of their master sets, which a control-packet reload keeps",
    ),
    ("overlay rule", r"a control-packet reload keeps their rules"),
    ("no path", r"no path leads from"),
    ("source unrouted", r"could not be routed to destination"),
    ("id exceeds", r"exceeds the maximum of"),
    ("false match", r"can lead to false packet id match"),
    ("claim rule", r"claim rule \(mask"),
    ("rule slots", r"slave port packet rules exceed"),
    ("fixed connections", r"cannot add the fixed connections"),
    ("not in device", r"must be contained within a device"),
    ("overused channel", r"the router found no routing that fits"),
    ("no legal routing", r"Unable to find a legal routing\s*$"),
)
INTERNALS = (
    "iterations:1",
    "iterations:2",
    "iterations:3-9",
    "iterations:10+",
    "max-iterations",
    "overcapacity",
    "illegal-edges",
    "hold-cycle",
    "no-arbiter-plan",
    "cover-avoid",
    "cover-multi-rule",
    "cover-masked",
    "flow-already-processed",
    *(f"rejected:{k}" for k, _ in OUTCOMES[2:11]),
)


def bucket(n, bounds=(1, 3, 7, 15, 31)):
    lo = bounds[0]
    for hi in bounds:
        if n <= hi:
            return str(hi) if lo == hi else f"{lo}-{hi}"
        lo = hi + 1
    return f"{lo}+"


def bucket_labels(bounds=(1, 3, 7, 15, 31)):
    return {bucket(n, bounds) for n in range(bounds[0], bounds[-1] + 2)}


# A detour over the Manhattan distance is even; counted in pairs of hops.
DETOUR, TURNS, BOX_AMSELS = (0, 1, 2, 4), (0, 1, 2, 3), (1, 3, 7, 15)


def outcome(rc, stderr):
    if rc == 0:
        return "routed"
    if rc == -1000:
        return "timeout"
    for key, rx in OUTCOMES:
        if re.search(rx, stderr, re.M):
            return key
    return "crash" if rc < 0 else "verifier"


def geometry(src, dst):
    dc, dr = dst[0] - src[0], dst[1] - src[1]
    if not dc and not dr:
        return "same-tile"
    heading = ("N" if dr > 0 else "S" if dr < 0 else "") + (
        "E" if dc > 0 else "W" if dc < 0 else ""
    )
    return f"{heading}:{bucket(abs(dc) + abs(dr), (1, 3, 7))}"


def tile_pair(t, src, dst):
    a, b = t.kind(src[:2]), t.kind(dst[:2])
    return f"same-{a}" if src[:2] == dst[:2] else f"{a}->{b}"


def space_endpoints(t, sending):
    """Endpoints flows may use: DMA and core ports, tile control, trace."""
    out = []
    for tile in sorted(t.kinds):
        out += t.endpoints(tile, sending)
        n = (t.slaves if sending else t.masters)[tile]
        for b in (CTRL, TRACE) if sending else (CTRL,):
            out += [(*tile, b, ch) for ch in range(n[b])]
    return out


def cut_capacity(t, axis, k, up):
    """Channels from row (axis 1) or column (axis 0) k to k + 1, or back."""
    fwd, back = (NORTH, SOUTH) if axis else (EAST, WEST)
    total = 0
    for j in range(t.cols if axis else t.rows):
        lo, hi = ((j, k), (j, k + 1)) if axis else ((k, j), (k + 1, j))
        if up:
            total += min(t.masters[lo][fwd], t.slaves[hi][back])
        else:
            total += min(t.masters[hi][back], t.slaves[lo][fwd])
    return total


def cut_saturation(d):
    """The fullest row or column cut: one channel per circuit tree across it,
    and one for all packet flows."""
    t = d.target
    trees = defaultdict(set)
    for s, x in d.flows:
        trees[s].add(x)
    pairs = [(s, x) for f in d.packet_flows for s in f["srcs"] for x in f["dsts"]]
    worst = 0.0
    for axis, n in ((0, t.cols), (1, t.rows)):
        for k in range(n - 1):
            for up in (True, False):

                def low(ep):
                    return ep[axis] <= k

                load = sum(
                    1
                    for s, xs in trees.items()
                    if low(s) == up and any(low(x) != up for x in xs)
                )
                load += any(low(s) == up and low(x) != up for s, x in pairs)
                cap = cut_capacity(t, axis, k, up)
                if load:
                    worst = max(worst, load / cap if cap else 2.0)
    return SATURATION[(worst >= 0.5) + (worst >= 1) + (worst > 1)]


def conflict_source(an, f, g):
    """Why stream f can fill its receiver, and the kinds of waits (lock,
    stream, host) that tie draining it to g."""
    fs = an.streams[f]
    if any(an.volumes.send_volume(s) is None for s in an.streams if s.dst == fs.dst):
        stall = STALLS[0]
    elif not an.volumes.receive_capacity(fs.dst):
        stall = STALLS[1]
    else:
        stall = STALLS[2]
    chain = an.blocking_chain(f, g)
    kinds = {
        EDGE_NAMES[k]
        for a, b in zip(chain, chain[1:])
        for x, k in an.graph.edges[a]
        if x == b
    }
    return f"{stall}/{'+'.join(sorted(kinds)) or 'direct'}"


def id_sets(d):
    if not d.packet_flows:
        return {"no-packet-flows"}
    out = set()
    ids = Counter(f["id"] for f in d.packet_flows)
    top = PARAMS["max_id"]
    out |= {"id-0" if i == 0 else "id-max" if i == top else "id-other" for i in ids}
    if any(f["mask"] is not None and f["mask"] != top for f in d.packet_flows):
        out.add("masked")
    if max(ids.values()) > 1:
        out.add("id-reuse")
    srcs_of = defaultdict(set)
    for f in d.packet_flows:
        srcs_of[f["id"]] |= set(f["srcs"])
    if any(len(s) > 1 for s in srcs_of.values()):
        out.add("same-id-two-sources")
    by_src = ids_by_source(d)
    if any(len(set(v)) > 1 for v in by_src.values()):
        out.add("multi-id-source")
    prio = {f["id"] for f in d.packet_flows if f["priority"]}
    if any(not f["priority"] and f["id"] in prio for f in d.packet_flows):
        out.add("priority-shares-id")
    for p in d.programs:
        if p["dir"] == MM2S and any(
            op[0] == "bd" and op[2] is not None for op in program_ops(p)
        ):
            out.add("sent-by-bd")
    if any(a["pkt"] is not None for a in d.allocs.values()):
        out.add("sent-by-alloc")
    if any(ev[0] == "memcpy" and ev[2] is not None for s in d.sequences for ev in s):
        out.add("sent-by-memcpy")
    sent = sent_packet_ids(d)
    for src, flow_ids in by_src.items():
        if src[2] != DMA:
            continue
        got = sent.get((src[0], src[1], MM2S, src[3]))
        if got is None:
            out.add("sender-unknown-id")
        elif set(got) - set(flow_ids):
            out.add("sender-extra-id")
    return out


def pre_ir(d, an):
    out = set()
    for ops in d.boxes.values():
        _, amsels, ms, rules, _ = box_view(ops)
        used = {n for op in ms.values() for n in op[2]}
        kinds = {op[0] for op in ops}
        out |= kinds & {"connect", "rules", "amsel", "masterset"}
        if set(amsels) - used:
            out.add("dangling-amsel")
        if any(m != PARAMS["max_id"] for op in rules.values() for m, _, _ in op[2]):
            out.add("masked-rule")
        for op in ms.values():
            if len(op[2]) > 1:
                out.add("multi-amsel-masterset")
            if op[3]:
                out.add("keep-masterset")
            if op[4]:
                out.add("ctrl-masterset")
    if any(d.muxes.values()):
        out.add("shim-mux")
    for s in an.streams[an.num_requested :]:
        out.add("partial-pinned-stream" if s.dst[2] in DIRECTIONAL else "pinned-stream")
    return out or {"none"}


def transitions(out):
    """(tile kind, slave bundle, master bundle) the routing uses."""
    t = out.target
    seen = set()
    for tile, ops in out.boxes.items():
        if not t.exists(tile):
            continue
        conns, _, ms, rules, _ = box_view(ops)
        kind = t.kind(tile)
        for s, masters in conns.items():
            seen |= {f"{kind} {BUNDLES[s[0]]}->{BUNDLES[m[0]]}" for m in masters}
        for s, op in rules.items():
            names = {r[2] for r in op[2]}
            for m, mop in ms.items():
                if names & set(mop[2]):
                    seen.add(f"{kind} {BUNDLES[s[0]]}->{BUNDLES[m[0]]}")
    for tile, conns in out.muxes.items():
        seen |= {f"shim-mux {BUNDLES[s[0]]}->{BUNDLES[m[0]]}" for s, m in conns}
    return seen


def routed_stats(d, an, out, hops_on):
    """Shapes of the router's output: amsels, msels and arbiters per
    switchbox, mastersets, packet rules, and per stream its detour over the
    Manhattan distance, its turns, and hops packet or circuit switched."""
    dims = defaultdict(set)
    for tile, ops in out.boxes.items():
        conns, amsels, ms, rules, _ = box_view(ops)
        if amsels:
            dims["amsels/box"].add(bucket(len(amsels), BOX_AMSELS))
            dims["arbiters/box"].add(str(len({a for a, _ in amsels.values()})))
            dims["msel"] |= {str(m) for _, m in amsels.values()}
        for op in ms.values():
            dims["masterset"].add(f"{bucket(len(op[2]), (1, 2, 3))} amsels")
            if op[3] is not None:
                dims["masterset"].add(f"keep={bool(op[3])}")
            if op[4]:
                dims["masterset"].add("ctrl")
        users = Counter(n for op in ms.values() for n in op[2])
        if any(v > 1 for v in users.values()):
            dims["branch"].add("packet")
        if any(len(m) > 1 for m in conns.values()):
            dims["branch"].add("circuit")
        for op in rules.values():
            dims["rules/port"].add(str(len(op[2])))
            dims["rule-mask"] |= {f"{bin(m).count('1')} bits" for m, _, _ in op[2]}
    want = defaultdict(set)
    for s in an.streams[: an.num_requested]:
        want[(s.src, s.pid)].add(s.dst)
    for (src, pid), dsts in want.items():
        ends, _ = trace_output(out, src, pid)
        for dst, hops in ends.items():
            dist = abs(src[0] - dst[0]) + abs(src[1] - dst[1])
            extra = max(0, len(hops) - dist - 1)
            dims["detour"].add(f"+{bucket(extra // 2, DETOUR)} pairs")
            heads = [m[0] for _, _, m, _ in hops if m[0] in DIRECTIONAL]
            dims["turns"].add(
                bucket(sum(a != b for a, b in zip(heads, heads[1:])), TURNS)
            )
            if pid is None:
                dims["switching"].add("circuit")
            elif any(arb is None for *_, arb in hops):
                dims["switching"].add("promoted" if hops_on else "circuit-hop")
            else:
                dims["switching"].add("packet")
    return dims


def internals(debug):
    """What the router's debug output (DEBUG_ONLY) shows it did."""
    out = set()
    iters = [int(n) for n in re.findall(r"End findPaths iteration #(\d+)", debug)]
    if iters:
        out.add(f"iterations:{bucket(max(iters) + 1, (1, 2, 9))}")
    if "maxIterations has been exceeded" in debug:
        out.add("max-iterations")
    if "Too much capacity" in debug:
        out.add("overcapacity")
    if re.search(r"illegal edges count = [1-9]", debug):
        out.add("illegal-edges")
    if "Hold cycle:" in debug:
        out.add("hold-cycle")
    if "No arbiter plan at tile" in debug:
        out.add("no-arbiter-plan")
    if "Flow already processed" in debug:
        out.add("flow-already-processed")
    for line in re.findall(r"^Routing rejected: (.*)$", debug, re.M):
        out.add(f"rejected:{outcome(1, line)}")
    for avoid, rules in re.findall(r"avoid \{(.*?)\} ->(.*)$", debug, re.M):
        if avoid.strip():
            out.add("cover-avoid")
        cover = re.findall(r"rule\((\d+), \d+\)", rules)
        if len(cover) > 1:
            out.add("cover-multi-rule")
        if any(int(m) != PARAMS["max_id"] for m in cover):
            out.add("cover-masked")
    return out


def route_space(d, rc=None, text="", stderr="", debug="", hops_on=True):
    """Where `d` and what the router made of it (return code, output, stderr,
    DEBUG_ONLY output) sit: dict(device, dims {dim: set of values}, streams
    [one value per PAIRWISE dim, per requested stream])."""
    t = d.target
    an = Analysis(d)
    an.graph
    nreq = an.num_requested
    dims = defaultdict(set)
    dims["device"].add(d.dev)
    load = f"{bucket(max(nreq, 1))}/{cut_saturation(d)}"
    dims["load"].add(load)
    flow_of = [None] * len(d.flows)
    for f in d.packet_flows:
        flow_of += [f] * (len(f["srcs"]) * len(f["dsts"]))
    streams = []
    for i in range(nreq):
        s, f = an.streams[i], flow_of[i]
        tiles, geo = tile_pair(t, s.src, s.dst), geometry(s.src, s.dst)
        if f is None:
            coarse = full = "circuit"
        else:
            coarse = KINDS[2 * (len(f["srcs"]) > 1) + (len(f["dsts"]) > 1)]
            full = coarse + ("+prio" if f["priority"] else "")
            if f["keep"] is not None:
                full += "+keep" if f["keep"] else "+nokeep"
        conflicts = []
        for j in range(len(an.streams)):
            if j != i and an.conflict(i, j):
                f_, g_ = (i, j) if an.can_block(i, j) else (j, i)
                conflicts.append(conflict_source(an, f_, g_))
                if j >= nreq:
                    dims["conflict"].add("pinned-partner")
                if any("Nothing in the design" in a for a in an.assumptions(f_, g_)):
                    dims["conflict"].add("unmodeled-agent")
        dims["conflict"] |= set(conflicts) or {"none"}
        dims["tiles"].add(tiles)
        dims["bundle"].add(f"{BUNDLES[s.src[2]]}->{BUNDLES[s.dst[2]]}")
        dims["geometry"].add(geo)
        dims["kind"].add(full)
        streams.append((tiles, geo, coarse, load, (conflicts or ["none"])[0]))
    dims["pre-ir"] = pre_ir(d, an)
    dims["ids"] = id_sets(d)
    if rc is not None:
        dims["outcome"].add(outcome(rc, stderr))
    if rc == 0 and text:
        out = load_design(text)
        dims["transitions"] = transitions(out)
        for k, v in routed_stats(d, an, out, hops_on).items():
            dims[k] |= v
    if debug:
        dims["internals"] = internals(debug)
    return dict(device=d.dev, dims=dict(dims), streams=streams)


_domains = {}


def space_domain(device):
    """Every value the TargetModel allows per dimension, for a family or a
    device: dict(dims, pairs {(dim, dim): feasible value pairs}, transitions
    {dev: set})."""
    if device in _domains:
        return _domains[device]
    devs = devices_of(device)
    dims = defaultdict(set)
    feasible = set()
    per_dev = {}
    for dev in devs:
        t = Target(dev)
        dims["device"].add(dev)
        srcs, dsts = space_endpoints(t, True), space_endpoints(t, False)
        for s in srcs:
            for x in dsts:
                if s[:2] == x[:2] and not t.legal(
                    s[:2], phys_src(s)[2:], phys_dst(x)[2:]
                ):
                    continue
                tp, geo = tile_pair(t, s, x), geometry(s, x)
                dims["tiles"].add(tp)
                dims["geometry"].add(geo)
                dims["bundle"].add(f"{BUNDLES[s[2]]}->{BUNDLES[x[2]]}")
                feasible.add((tp, geo))
        seen = set()
        for tile, kind in t.kinds.items():
            for sb, ns in t.slaves[tile].items():
                for db, nm in t.masters[tile].items():
                    if any(
                        t.legal(tile, (sb, i), (db, j))
                        for i in range(ns)
                        for j in range(nm)
                    ):
                        seen.add(f"{kind} {BUNDLES[sb]}->{BUNDLES[db]}")
            if kind == "shim":
                for b, n in t.mux_slaves[tile].items():
                    if n and b != NORTH:
                        seen.add(f"shim-mux {BUNDLES[b]}->North")
                for b, n in t.mux_masters[tile].items():
                    if n and b != NORTH:
                        seen.add(f"shim-mux North->{BUNDLES[b]}")
        per_dev[dev] = seen
        dims["transitions"] |= seen
    dims["kind"] = {"circuit"} | {
        k + p + e
        for k in KINDS
        for p in ("", "+prio")
        for e in ("", "+keep", "+nokeep")
    }
    dims["load"] = {f"{b}/{s}" for b in bucket_labels() for s in SATURATION}
    dims["conflict"] = {
        f"{s}/{'+'.join(e) or 'direct'}"
        for s in STALLS
        for r in range(4)
        for e in itertools.combinations(sorted(EDGE_NAMES), r)
    } | {"none", "pinned-partner", "unmodeled-agent"}
    dims["pre-ir"] = set(PRE_IR)
    dims["ids"] = set(ID_SETS)
    dims["outcome"] = {"routed", "verifier", "crash", "timeout"} | {
        k for k, _ in OUTCOMES
    }
    dims["amsels/box"] = {
        bucket(n, BOX_AMSELS)
        for n in range(1, PARAMS["arbiters"] * PARAMS["msels"] + 1)
    }
    dims["arbiters/box"] = {str(n) for n in range(1, PARAMS["arbiters"] + 1)}
    dims["msel"] = {str(m) for m in range(PARAMS["msels"])}
    dims["masterset"] = {
        f"{bucket(n, (1, 2, 3))} amsels" for n in range(1, PARAMS["msels"] + 1)
    } | {"keep=True", "keep=False", "ctrl"}
    dims["branch"] = {"packet", "circuit"}
    dims["rules/port"] = {str(n) for n in range(1, PARAMS["rule_slots"] + 1)}
    width = PARAMS["max_id"].bit_length()
    dims["rule-mask"] = {f"{n} bits" for n in range(width + 1)}
    dims["detour"] = {f"+{b} pairs" for b in bucket_labels(DETOUR)}
    dims["turns"] = bucket_labels(TURNS)
    dims["switching"] = {"circuit", "packet", "promoted", "circuit-hop"}
    dims["internals"] = set(INTERNALS)
    coarse = {
        "tiles": dims["tiles"],
        "geometry": dims["geometry"],
        "kind": {"circuit", *KINDS},
        "load": dims["load"],
        "conflict": dims["conflict"] - {"pinned-partner", "unmodeled-agent"},
    }
    pairs = {
        (a, b): (
            feasible
            if (a, b) == ("tiles", "geometry")
            else set(itertools.product(coarse[a], coarse[b]))
        )
        for a, b in itertools.combinations(PAIRWISE, 2)
    }
    _domains[device] = dict(dims=dict(dims), pairs=pairs, transitions=per_dev)
    return _domains[device]


def space_coverage(spaces, device):
    """Union of route_space() results: ({dim: (hit, total)}, {pair: (hit,
    total)}, hit values per dim, transitions per device)."""
    dom = space_domain(device)
    hit = defaultdict(set)
    pair_hit = defaultdict(set)
    per_dev = defaultdict(set)
    for sp in spaces:
        for k, v in sp["dims"].items():
            hit[k] |= v
        per_dev[sp["device"]] |= sp["dims"].get("transitions", set())
        for row in sp["streams"]:
            for (i, a), (j, b) in itertools.combinations(enumerate(PAIRWISE), 2):
                pair_hit[(a, b)].add((row[i], row[j]))
    dims = {k: (len(hit[k] & v), len(v)) for k, v in sorted(dom["dims"].items())}
    pairs = {k: (len(pair_hit[k] & v), len(v)) for k, v in dom["pairs"].items()}
    return dims, pairs, hit, per_dev
