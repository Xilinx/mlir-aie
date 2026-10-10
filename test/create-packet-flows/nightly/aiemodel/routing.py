#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The router's own rules, mirrored from AIECreatePathFindFlows.cpp, and a
witness routing built with them."""

import random
from collections import defaultdict, deque

from .params import PARAMS
from .fabric import (
    BUNDLES,
    CTRL,
    DIRECTIONAL,
    EAST,
    NORTH,
    SOUTH,
    STEP,
    WEST,
    fmt_port,
    linked_input,
    phys_dst,
    phys_src,
)
from .design import Design, agree_overlay_keep, keeps_header, load_design
from .deadlock import Analysis, describe_stream, requested_streams
from .run import aie_opt
from .output import box_view, trace_output

# The router's own rules, mirrored from AIECreatePathFindFlows.cpp: arbiter
# planning per switchbox, the pre-routing arbiter count, and runOnPacketFlow's
# checks on one routing, including the hold-cycle search.


def cubes_intersect(a, b):
    return ((a[1] ^ b[1]) & a[0] & b[0]) == 0


def plan_arbiters(flows, conflict, excluded, reserved):
    """planArbiters. flows are (slave, id, masters sorted, is_ctrl). Returns
    (plan or None, blocking pairs); a plan is ({(slave, id): amsel},
    {amsel: masters})."""
    na, nm = PARAMS["arbiters"], PARAMS["msels"]
    leader = {}

    def find(p):
        leader.setdefault(p, p)
        if leader[p] == p:
            return p
        leader[p] = find(leader[p])
        return leader[p]

    for f in flows:
        for m in f[2]:
            leader[find(m)] = find(f[2][0])
    units, unit_of, flow_unit = [], {}, []
    for i, f in enumerate(flows):
        root = find(f[2][0])
        if root not in unit_of:
            unit_of[root] = len(units)
            units.append(dict(flows=[], sets=[], ctrl=False))
        u = units[unit_of[root]]
        flow_unit.append(unit_of[root])
        u["flows"].append(i)
        u["ctrl"] |= f[3]
        if list(f[2]) not in u["sets"]:
            u["sets"].append(list(f[2]))

    def clash(a, b):
        return flows[a][0] != flows[b][0] and conflict(a, b)

    blocking = []
    for u in units:
        for k, a in enumerate(u["flows"]):
            for b in u["flows"][k + 1 :]:
                if clash(a, b):
                    blocking.append((a, b))
    if blocking:
        return None, blocking
    neighbors = [set() for _ in units]
    cross = []
    for a in range(len(flows)):
        for b in range(a + 1, len(flows)):
            if flow_unit[a] != flow_unit[b] and clash(a, b):
                neighbors[flow_unit[a]].add(flow_unit[b])
                neighbors[flow_unit[b]].add(flow_unit[a])
                cross.append((a, b))
    free = [[m for m in range(nm) if a + m * na not in reserved] for a in range(na)]
    excluded_units = [[] for _ in range(na)]
    for a in range(na):
        for f in range(len(flows)):
            if excluded(f, a) and flow_unit[f] not in excluded_units[a]:
                excluded_units[a].append(flow_unit[f])
    order = sorted(
        range(len(units)),
        key=lambda u: (-len(neighbors[u]), -len(units[u]["sets"])),
    )
    arbiter_of = [-1] * len(units)
    load = [0] * na
    steps = [0]

    def place(depth):
        if depth == len(order):
            return True
        steps[0] += 1
        if steps[0] > PARAMS["plan_budget"]:
            return False
        u = order[depth]
        n = len(units[u]["sets"])
        if units[u]["ctrl"]:
            cands = list(range(na - 1, -1, -1))
        else:
            cands = sorted(range(na), key=lambda a: load[a])
        tried = set()
        for a in cands:
            if load[a] + n > len(free[a]):
                continue
            if u in excluded_units[a] or any(arbiter_of[v] == a for v in neighbors[u]):
                continue
            if load[a] == 0:
                k = (len(free[a]), tuple(excluded_units[a]))
                if k in tried:
                    continue
                tried.add(k)
            arbiter_of[u] = a
            load[a] += n
            if place(depth + 1):
                return True
            load[a] -= n
            arbiter_of[u] = -1
        return False

    if not place(0):
        return None, cross
    slave_amsels, amsel_masters = {}, {}
    low, high = [0] * na, [0] * na
    for ui, u in enumerate(units):
        a = arbiter_of[ui]
        set_amsel = {}
        for masters in u["sets"]:
            if u["ctrl"]:
                msel = free[a][len(free[a]) - 1 - high[a]]
                high[a] += 1
            else:
                msel = free[a][low[a]]
                low[a] += 1
            set_amsel[tuple(masters)] = a + msel * na
            amsel_masters[a + msel * na] = list(masters)
        for f in u["flows"]:
            slave_amsels[(flows[f][0], flows[f][1])] = set_amsel[tuple(flows[f][2])]
    return (slave_amsels, amsel_masters), []


def cut_tiles(target, src, dst):
    """cutTiles: tiles other than src and dst every route between them passes."""

    def neighbors(t):
        out = []
        for b, (dc, dr, _) in (
            (NORTH, STEP[NORTH]),
            (SOUTH, STEP[SOUTH]),
            (EAST, STEP[EAST]),
            (WEST, STEP[WEST]),
        ):
            n = (t[0] + dc, t[1] + dr)
            if (
                0 <= n[0] < target.cols
                and 0 <= n[1] < target.rows
                and target.masters[t][b] > 0
            ):
                out.append(n)
        return out

    def path(avoid):
        via = {src: src}
        queue = deque([src])
        while queue and dst not in via:
            t = queue.popleft()
            for n in neighbors(t):
                if n != avoid and n not in via:
                    via[n] = t
                    queue.append(n)
        tiles = []
        if dst in via:
            t = via[dst]
            while t != src:
                tiles.append(t)
                t = via[t]
        return dst in via, tiles

    reachable, interior = path(None)
    if not reachable:
        return []
    return [t for t in interior if not path(t)[0]]


def reserved_amsels(d):
    """Per tile, arbiter + 6 * msel of every declared amsel. An amsel no
    masterset uses can still have rules steering packets to it."""
    out = defaultdict(set)
    for tile, ops in d.boxes.items():
        out[tile] |= {
            op[2] + op[3] * PARAMS["arbiters"] for op in ops if op[0] == "amsel"
        }
    return out


def free_arbiters(reserved, wholly=False):
    """Arbiters with a free msel, as unroutableArbiters counts them, or with
    every msel free, as the circuit-switched hop promotion does."""
    na, nm = PARAMS["arbiters"], PARAMS["msels"]
    test = all if wholly else any
    return sum(
        1 for a in range(na) if test(a + m * na not in reserved for m in range(nm))
    )


def unroutable_arbiters(d, an, pins_hops):
    """unroutableArbiters: the reason no routing can plan its arbiters, found
    before routing, or None."""
    target = d.target
    prioritized = {
        (src, f["id"]) for f in d.packet_flows if f["priority"] for src in f["srcs"]
    }
    pinned = defaultdict(list)
    for i, s in enumerate(an.streams[: an.num_requested]):
        if s.pid is None:
            continue
        pinned[s.dst[:2]].append(i)
        if s.src[:2] == s.dst[:2]:
            continue
        circuitless = (s.src, s.pid) in prioritized
        if circuitless or pins_hops(s.src[:2]):
            pinned[s.src[:2]].append(i)
        for t in cut_tiles(target, s.src[:2], s.dst[:2]):
            if circuitless or pins_hops(t):
                pinned[t].append(i)
    reserved = reserved_amsels(d)
    for tile in sorted(pinned):
        cands = pinned[tile]
        free = free_arbiters(reserved[tile])
        srcs = {an.streams[s].src for s in cands}
        dsts = {an.streams[s].dst for s in cands}
        if min(len(srcs), len(dsts)) <= free:
            continue
        clique, best, steps = [], [], [0]

        def grow(cs):
            if len(clique) > len(best):
                best[:] = clique
            for k, s in enumerate(cs):
                if len(best) > free:
                    return
                steps[0] += 1
                if steps[0] > PARAMS["clique_budget"] or len(clique) + len(
                    cs
                ) - k <= len(best):
                    return
                nxt = [t for t in cs[k + 1 :] if an.must_separate(s, t)]
                clique.append(s)
                grow(nxt)
                clique.pop()

        grow(cands)
        if len(best) <= free:
            continue
        an.clique = list(best)
        return f"at tile ({tile[0]}, {tile[1]}), no two of " + ", ".join(
            describe_stream(an.streams[s]) for s in best
        ) + " can share an arbiter, and each takes one there whatever the routing," f" but the switchbox has {free} free. For example, " + an.explain(
            best[0], best[1]
        )
    return None


def find_path_to_dest(settings, tile, port, dst):
    if tile == dst[:2] and port == dst[2:]:
        return True
    nxt = linked_input(tile, port)
    if nxt is None:
        return False
    ntile, nport = nxt
    for s, t in settings.get(ntile, ()):
        if s == nport and find_path_to_dest(settings, ntile, t, dst):
            return True
    return False


def plan_routing(d, an, solution, hops_on):
    """runOnPacketFlow up to emission, on `solution` ({source endpoint: {tile:
    [(slave port, master port)]}}, logical shim DMA ports as the pathfinder
    keeps them). Returns a dict: reason (None if the routing is legal), blame
    (requested streams in the way), plans, circuit_hops, routes."""
    target = d.target
    na = PARAMS["arbiters"]
    nreq = an.num_requested
    index = {}
    for i, s in enumerate(an.streams):
        if s.pid is not None:
            index.setdefault((s.src, s.dst, s.pid), i)
    routes = [list(s.hops) for s in an.streams]
    circuit_sources = {s for s, _ in d.flows}
    switchboxes = defaultdict(list)
    slave_streams = defaultdict(list)
    ctrl_flows = {}
    out = dict(
        reason=None,
        blame=[],
        plans={},
        circuit_hops={},
        routes=routes,
        shared_receiver_cycle=None,
    )

    def fail(reason, blame):
        out["reason"], out["blame"] = reason, sorted(set(blame))
        return out

    for f in d.packet_flows:
        pid = f["id"]
        for dst in f["dsts"]:
            for src in f["srcs"]:
                if src in circuit_sources:
                    continue
                settings = solution.get(src, {})
                k = index.get((src, dst, pid))
                if k is not None:
                    hops, at = [], (src[:2], src[2:])
                    while at and len(hops) <= len(settings):
                        tile, inp = at
                        hops.append((tile, inp, None))
                        at = None
                        if tile not in settings:
                            break
                        for s, t in settings[tile]:
                            if (
                                s == inp
                                and not (tile == dst[:2] and t == dst[2:])
                                and find_path_to_dest(settings, tile, t, dst)
                            ):
                                at = linked_input(tile, t)
                                break
                    routes[k] = hops
                src_routed = False
                for tile in sorted(settings):
                    for s, t in settings[tile]:
                        if not find_path_to_dest(settings, tile, t, dst):
                            continue
                        if tile == src[:2] and s == src[2:]:
                            src_routed = True
                        if ((s, t), pid) not in switchboxes[tile]:
                            switchboxes[tile].append(((s, t), pid))
                        if k is not None and k not in slave_streams[((tile, s), pid)]:
                            slave_streams[((tile, s), pid)].append(k)
                        ctrl_flows[((tile, t), pid)] = ctrl_flows.get(
                            ((tile, t), pid), False
                        ) or bool(f["priority"])
                if not src_routed:
                    return fail(
                        f"packet flow source ({src[0]}, {src[1]}) {BUNDLES[src[2]]}{src[3]}"
                        f" could not be routed to destination ({dst[0]}, {dst[1]}) "
                        f"{BUNDLES[dst[2]]}{dst[3]}; the pathfinder produced an "
                        "incomplete routing for this placement.",
                        [] if k is None else [k],
                    )

    reserved = reserved_amsels(d)
    circuit_ports = set()
    for tile, ops in d.boxes.items():
        for op in ops:
            if op[0] == "connect":
                circuit_ports |= {(tile, op[1]), (tile, op[2])}
    for src, _ in d.flows:
        for tile, pairs in solution.get(src, {}).items():
            for s, t in pairs:
                circuit_ports |= {(tile, s), (tile, t)}
    circuit_hops = defaultdict(list)
    for tile in sorted(switchboxes):
        if not hops_on or target.kind(tile) == "shim":
            continue
        connects = switchboxes[tile]
        masters = {c[1] for c, _ in connects}
        slaves = list(dict.fromkeys(c[0] for c, _ in connects))
        free = sum(
            1
            for a in range(na)
            if not any(a + m * na in reserved[tile] for m in range(PARAMS["msels"]))
        )
        for slave in slaves:
            if len(masters) <= free:
                break
            reached, by_id = set(), defaultdict(set)
            for c, pid in connects:
                if c[0] == slave:
                    reached.add(c[1])
                    by_id[pid].add(c[1])
            exclusive = (
                (tile, slave) not in circuit_ports
                and all(
                    m[0] in DIRECTIONAL and (tile, m) not in circuit_ports
                    for m in reached
                )
                and all(v == reached for v in by_id.values())
                and all(
                    c[1] not in reached
                    or (
                        c[0] == slave and not ctrl_flows.get(((tile, c[1]), pid), False)
                    )
                    for c, pid in connects
                )
            )
            if not exclusive:
                continue
            for m in sorted(reached):
                circuit_hops[tile].append((slave, m))
                masters.discard(m)
            connects[:] = [e for e in connects if e[0][0] != slave]
    out["circuit_hops"] = dict(circuit_hops)

    packet_flows, ctrl_packet_flows = defaultdict(list), defaultdict(list)
    for tile in sorted(switchboxes):
        for (s, t), pid in switchboxes[tile]:
            key = ((tile, s), pid)
            (ctrl_packet_flows if ctrl_flows[((tile, t), pid)] else packet_flows)[
                key
            ].append(t)
    tile_slave_flows = defaultdict(dict)
    for m, is_ctrl in ((ctrl_packet_flows, True), (packet_flows, False)):
        for key in sorted(m):
            (tile, slave), pid = key
            f = tile_slave_flows[tile].setdefault((slave, pid), [slave, pid, [], False])
            f[3] |= is_ctrl
            for t in m[key]:
                if t not in f[2]:
                    f[2].append(t)
            f[2].sort()
    tile_flows = {
        tile: [tuple(v[:2]) + (tuple(v[2]), v[3]) for _, v in sorted(byf.items())]
        for tile, byf in sorted(tile_slave_flows.items())
    }
    out["tile_flows"] = tile_flows

    def conflicting_streams(tile, a, b):
        as_ = slave_streams.get(((tile, a[0]), a[1]))
        bs = slave_streams.get(((tile, b[0]), b[1]))
        if not as_ or not bs:
            return None
        for s in as_:
            for t in bs:
                if an.conflict(s, t):
                    return s, t
        return None

    apart, off = set(), set()

    def plan_tile(tile):
        flows = tile_flows[tile]

        def key(f):
            return (flows[f][0], flows[f][1])

        return plan_arbiters(
            flows,
            lambda a, b: (tile, min(key(a), key(b)), max(key(a), key(b))) in apart
            or conflicting_streams(tile, flows[a], flows[b]) is not None,
            lambda f, a: (tile, key(f), a) in off,
            reserved[tile],
        )

    plans = {}
    failure = None
    for tile, flows in tile_flows.items():
        plan, blocking = plan_tile(tile)
        if plan is not None:
            plans[tile] = plan
            continue
        if failure:
            continue
        if not blocking:
            failure = (
                f"at tile ({tile[0]}, {tile[1]}), the packet flows need more arbiter "
                "msels than the switchbox has free.",
                [k for f in flows for k in slave_streams.get(((tile, f[0]), f[1]), [])],
            )
            continue
        s, t = conflicting_streams(tile, flows[blocking[0][0]], flows[blocking[0][1]])
        failure = (
            f"{describe_stream(an.streams[s])} and {describe_stream(an.streams[t])} can "
            "deadlock if they share an arbiter, and no routing found keeps them apart "
            f"(last tried: tile ({tile[0]}, {tile[1]})). " + an.explain(s, t),
            [x for x in (s, t) if x < nreq],
        )
    out["plans"] = plans
    if failure:
        return fail(*failure)

    def arbitrate():
        for s in range(nreq):
            pid = an.streams[s].pid
            if pid is None:
                continue
            hops = []
            for tile, inp, _ in routes[s]:
                amsel = plans[tile][0].get((inp, pid)) if tile in plans else None
                hops.append((tile, inp, None if amsel is None else amsel % na))
            routes[s] = hops

    first = []
    budget = [PARAMS["hold_budget"]]

    def search(forced_waits, definite=False):
        arbitrate()
        cycle = an.hold_cycle(routes, forced_waits, definite)
        if cycle is None and definite:
            cycle = an.hold_cycle(routes)
        if cycle is None:
            return True
        if not first:
            first.append(cycle)
        for wait, waiting, sharer, holding, tile, sin, hin, arb, forced in cycle:
            if wait != "arbiter" or forced:
                continue
            sk = (sin, an.streams[sharer].pid)
            hk = (hin, an.streams[holding].pid)
            sn, hn = sharer < nreq, holding < nreq
            pair = o = None
            if sn and hn:
                pair = (tile, min(sk, hk), max(sk, hk))
            elif sn or hn:
                o = (tile, sk if sn else hk, arb)
            else:
                continue
            if budget[0] <= 0:
                return False
            if (pair and pair in apart) or (o and o in off):
                continue
            if pair:
                apart.add(pair)
            if o:
                off.add(o)
            budget[0] -= 1
            saved = plans[tile]
            plan, _ = plan_tile(tile)
            if plan is not None:
                plans[tile] = plan
                if search(forced_waits, definite):
                    return True
                plans[tile] = saved
            if pair:
                apart.discard(pair)
            if o:
                off.discard(o)
        return False

    if not search(False):
        arbitrate()
        cycle = first[0]
        blame = [
            x
            for st in cycle
            if st[0] != "drain"
            for x in (st[1], st[2], st[3])
            if x < nreq
        ]
        return fail(
            "packet flows can deadlock holding arbiters across switchboxes, and no "
            "arbiter assignment found avoids it. " + an.explain_cycle(cycle),
            blame,
        )
    # Trees into a receiver they share wait on each other at its master port
    # whatever the plan; where no plan breaks a cycle through such waits, the
    # plan stands, failing unless every such plan has the cycle only by
    # assumption.
    first.clear()
    budget[0] = PARAMS["hold_budget"]
    if not search(True):
        first.clear()
        budget[0] = PARAMS["hold_budget"]
        if not search(True, definite=True):
            out["shared_receiver_cycle"] = first[0]
        arbitrate()
    return out


class Construction:
    """Routes flows on exclusive links, as a witness that a routing exists.

    Ports here are physical (c, r, bundle, channel): a shim DMA is its South
    port. A master port belongs to one tree, except the port into a packet
    destination, which packet trees from several sources may share. Ports the
    design's switchboxes already use are taken.
    """

    def __init__(self, d, rng):
        self.t, self.rng = d.target, rng
        self.owner_m, self.owner_s = {}, {}
        self.trees = {}
        # A control-packet reload keeps the pinned trees' master sets and
        # arbiters, so a packet tree ending at one of their ports would share
        # an arbiter it cannot be moved off; the witness keeps clear of them.
        self.overlay_ports = set()
        for tile, ops in d.boxes.items():
            for op in ops:
                if op[0] == "connect":
                    self.owner_s[(*tile, *op[1])] = "fixed"
                    self.owner_m[(*tile, *op[2])] = "fixed"
                elif op[0] == "masterset":
                    self.owner_m[(*tile, *op[1])] = "fixed"
                elif op[0] == "rules":
                    self.owner_s[(*tile, *op[1])] = "fixed"

    def route(self, src, dsts, packet):
        root = phys_src(src)
        tree = dict(src=src, root=root, children=defaultdict(list), next={}, ends={})
        self.trees[src] = tree
        if root in self.owner_s:
            return []
        self.owner_s[root] = src
        routed = []
        for d in dsts:
            found = self._search(tree, d, packet)
            if found is None:
                continue
            parent, s, m = found
            tree["children"][s].append(m)
            tree["ends"][m] = d
            self.owner_m[m] = ("dst", d) if packet else src
            while parent[s] is not None:
                ps, pm = parent[s]
                tree["children"][ps].append(pm)
                tree["next"][pm] = s
                self.owner_m[pm] = src
                self.owner_s[s] = src
                s = ps
            routed.append(d)
        return routed

    def pin(self, src, ends):
        """Take the tree from `src` the router emitted: `ends` as
        trace_output has them."""
        root = phys_src(src)
        tree = dict(src=src, root=root, children=defaultdict(list), next={}, ends={})
        self.trees[src] = tree
        self.owner_s[root] = src
        for d, hops in ends.items():
            for k, (tile, slave, master, _) in enumerate(hops):
                s, m = (*tile, *slave), (*tile, *master)
                if m not in tree["children"][s]:
                    tree["children"][s].append(m)
                self.owner_s[s] = src
                if k + 1 < len(hops):
                    tree["next"][m] = (*hops[k + 1][0], *hops[k + 1][1])
                    self.owner_m[m] = src
                else:
                    tree["ends"][m] = d
                    self.owner_m[m] = ("dst", d)
        for ms in tree["children"].values():
            self.overlay_ports.update(ms)

    def _search(self, tree, d, packet):
        t = self.t
        target = phys_dst(d)
        starts = [tree["root"], *tree["next"].values()]
        parent = {s: None for s in starts}
        queue = deque(starts)
        while queue:
            s = queue.popleft()
            tile = s[:2]
            cands = t.master_ports(tile)
            self.rng.shuffle(cands)
            for port in cands:
                m = (*tile, *port)
                if not t.legal(tile, s[2:], port):
                    continue
                if m == target:
                    owner = self.owner_m.get(m)
                    if m in self.overlay_ports:
                        continue
                    if owner is None or (packet and owner == ("dst", d)):
                        return parent, s, m
                    continue
                if m in self.owner_m or port[0] not in DIRECTIONAL:
                    continue
                if tile[1] == 0 and port[0] == SOUTH:
                    continue
                ntile, nport = linked_input(tile, port)
                if not t.exists(ntile) or nport[1] >= t.slaves[ntile][nport[0]]:
                    continue
                nxt = (*ntile, *nport)
                if nxt in self.owner_s or nxt in parent:
                    continue
                parent[nxt] = (s, m)
                queue.append(nxt)
        return None

    def settings(self, src):
        """The tree from `src` as the pathfinder's switch settings."""
        tree = self.trees.get(src)
        out = defaultdict(list)
        if tree is None:
            return {}
        for s, ms in tree["children"].items():
            sp = src[2:] if s == tree["root"] else s[2:]
            for m in ms:
                mp = tree["ends"][m][2:] if m in tree["ends"] else m[2:]
                out[s[:2]].append((sp, mp))
        return dict(out)

    def solution(self):
        return {src: self.settings(src) for src in self.trees}

    def hops(self):
        return sum(
            len(ms) for tr in self.trees.values() for ms in tr["children"].values()
        )


def drop_stream(d, stream):
    """Remove one requested stream, or the least more that removing it takes,
    as where it was the last into a port the prioritized flows end at."""
    if stream.pid is None:
        d.flows.remove((stream.src, stream.dst))
        return
    for f in d.packet_flows:
        if (
            f["id"] == stream.pid
            and stream.src in f["srcs"]
            and stream.dst in f["dsts"]
        ):
            if len(f["dsts"]) == 1 and len(f["srcs"]) > 1:
                f["srcs"].remove(stream.src)
            else:
                f["dsts"].remove(stream.dst)
            break
    d.packet_flows = [f for f in d.packet_flows if f["srcs"] and f["dsts"]]
    agree_overlay_keep(d)


def pins_hops_fn(d, hops_on):
    return lambda tile: not hops_on or d.target.kind(tile) == "shim"


def priority_trees(d, cache):
    """The trees from `d`'s prioritized sources, {source: trace_output ends},
    as the router routes the prioritized packets without the rest of the
    design: priority_route promises they keep that routing whatever else the
    design asks for, so a witness has to route around them. A source's other
    ids are no part of it. None if the router cannot route them."""
    prio = {(s, f["id"]) for f in d.packet_flows if f["priority"] for s in f["srcs"]}
    if not prio:
        return {}
    c = d.copy()
    c.flows = []
    c.packet_flows = [
        dict(f, srcs=srcs)
        for f in c.packet_flows
        if (srcs := [s for s in f["srcs"] if (s, f["id"]) in prio])
    ]
    text = c.emit()
    if text not in cache:
        p = aie_opt(text, hops_on=False, timeout=120)
        trees = None
        if p.returncode == 0:
            out, trees = load_design(p.stdout), defaultdict(dict)
            for f in c.packet_flows:
                for s in f["srcs"]:
                    trees[s].update(trace_output(out, s, f["id"])[0])
        cache[text] = trees
    return cache[text]


def overlay_design(d, reload=False):
    """`d` with only the flows the router routes as the control overlay: the
    prioritized ones, and the other flows from a prioritized source with
    their id. A control-packet reload configures the prioritized flows with
    no TileControl end, so with `reload` they are not in it. None if that is
    all of `d` or none of it."""
    prio = {
        (s, f["id"])
        for f in d.packet_flows
        if f["priority"]
        and (not reload or any(e[2] == CTRL for e in f["srcs"] + f["dsts"]))
        for s in f["srcs"]
    }
    kept = [f for f in d.packet_flows if any((s, f["id"]) in prio for s in f["srcs"])]
    if not kept or (not d.flows and len(kept) == len(d.packet_flows)):
        return None
    c = d.copy()
    c.flows = []
    c.packet_flows = kept
    return c


def standalone_overlay(d):
    """The @ctrl_pkt_overlay a control-packet reload of `d` installs: `d`'s
    tiles and the overlay's flows, without `d`'s own switchboxes."""
    o = overlay_design(d, reload=True)
    if o is None:
        return None
    c = Design(d.dev)
    c.tiles = d.used_tiles()
    c.packet_flows = o.packet_flows
    c.reload = True
    return c


def overlay_ops(out):
    """What a control-packet reload of the routed `out` skips, by tile:
    ({master: (msels, keep)}, {slave: rules}), amsels as (arbiter, msel),
    and problems where a skipped rule follows one the reload writes."""
    ops, problems = {}, []
    for tile, box in out.boxes.items():
        _, amsels, ms, rules, _ = box_view(box)
        masters = {
            port: (
                sorted(amsels.get(n, (-1, -1)) for n in op[2]),
                keeps_header(tile, port, op[3]),
            )
            for port, op in ms.items()
            if op[4]
        }
        slaves = {}
        for port, op in rules.items():
            tagged = [
                op[3] or t for t in (op[4] if len(op) > 4 else [False] * len(op[2]))
            ]
            if any(t and not u for u, t in zip(tagged, tagged[1:])):
                problems.append(
                    f"{tile} {fmt_port(port)} has an overlay rule after a design rule"
                )
            kept = [(m, v, amsels.get(n)) for (m, v, n), t in zip(op[2], tagged) if t]
            if kept:
                slaves[port] = kept
        if masters or slaves:
            ops[tile] = (masters, slaves)
    return ops, problems


def overlay_problems(out, alone):
    """Where the routed design `out` sets the switches a control-packet
    reload skips other than the overlay `alone` routed by itself does."""
    got, problems = overlay_ops(out)
    want, _ = overlay_ops(alone)
    for tile in sorted(set(got) | set(want)):
        g, w = got.get(tile, ({}, {})), want.get(tile, ({}, {}))
        for k, what in ((0, "master"), (1, "slave")):
            for port in sorted(set(g[k]) | set(w[k])):
                if g[k].get(port) != w[k].get(port):
                    problems.append(
                        f"{tile} {what} {fmt_port(port)}: overlay ops "
                        f"{g[k].get(port)} in the design, {w[k].get(port)} alone"
                    )
    return problems


def construct(d, seed, attempts=40):
    """Route `d` on exclusive links and plan its arbiters with the router's
    rules, dropping streams until both hop modes accept the witness. Returns
    (construction, analysis, plan) or None; `d` is pruned in place."""
    cache = {}
    for attempt in range(attempts):
        con = Construction(d, random.Random(f"route-{seed}-{attempt}"))
        pinned = priority_trees(d, cache)
        if pinned is None:
            prio = {s for f in d.packet_flows if f["priority"] for s in f["srcs"]}
            blame = [st for st in requested_streams(d) if st.src in prio]
            drop_stream(d, random.Random(f"drop-{seed}-{attempt}").choice(blame))
            if not d.flows and not d.packet_flows:
                return None
            continue
        for src, ends in pinned.items():
            con.pin(src, ends)
        groups = defaultdict(list)
        for s, t in d.flows:
            groups[(s, False)].append(t)
        for f in d.packet_flows:
            for s in f["srcs"]:
                groups[(s, True)] += f["dsts"]
        order = sorted(groups)
        con.rng.shuffle(order)
        missing = []
        for src, packet in order:
            wanted = list(dict.fromkeys(groups[(src, packet)]))
            if packet and src in pinned:
                # A pinned source's other ids follow its tree only where they
                # go everywhere it goes; the witness has no second tree for
                # those that do not.
                dsts = defaultdict(set)
                for f in d.packet_flows:
                    if src in f["srcs"]:
                        dsts[f["id"]] |= set(f["dsts"])
                prio = {
                    f["id"]
                    for f in d.packet_flows
                    if f["priority"] and src in f["srcs"]
                }
                tree = set().union(*(dsts[i] for i in prio))
                missing += [
                    (src, t, packet, None) for t in wanted if t not in pinned[src]
                ]
                missing += [
                    (src, t, packet, i)
                    for i, ts in dsts.items()
                    if i not in prio and ts != tree
                    for t in ts
                ]
                continue
            routed = con.route(src, wanted, packet)
            missing += [(src, t, packet, None) for t in wanted if t not in routed]
        if missing:
            for src, t, packet, pid in missing:
                for st in requested_streams(d):
                    if (
                        st.src == src
                        and st.dst == t
                        and (st.pid is not None) == packet
                        and pid in (None, st.pid)
                    ):
                        drop_stream(d, st)
            if not d.flows and not d.packet_flows:
                return None
            continue
        if not d.flows and not d.packet_flows:
            return None
        an = Analysis(d)
        blame = None
        if unroutable_arbiters(d, an, pins_hops_fn(d, False)):
            blame = [i for i in an.clique if i < an.num_requested]
        if blame is None:
            res = plan_routing(d, an, con.solution(), hops_on=False)
            if res["reason"] is None:
                return con, an, res
            blame = res["blame"] or list(range(an.num_requested))
        drop_stream(
            d, an.streams[random.Random(f"drop-{seed}-{attempt}").choice(blame)]
        )
        if not d.flows and not d.packet_flows:
            return None
    return None
