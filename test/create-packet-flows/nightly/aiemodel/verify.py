#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""What any routing the router emits has to satisfy."""

from collections import defaultdict

from .params import PARAMS
from .fabric import DIRECTIONAL, NORTH, SOUTH, fmt_ep, fmt_port
from .design import keeps_header, last_keep, load_design
from .deadlock import describe_stream
from .output import box_view, trace_output
from .routing import free_arbiters, reserved_amsels

# What any routing the router emits has to satisfy.


def verify(d, an, text, hops_on):
    """Check a routing of `d` (analysis `an`) the router emitted. Returns
    (problems, stats)."""
    t = d.target
    out = load_design(text)
    problems = []
    if out.flows or out.packet_flows:
        problems.append("flows left unrouted in the output")
    reserved = reserved_amsels(d)
    for tile, ops in sorted(out.boxes.items()):
        if not t.exists(tile):
            problems.append(f"switchbox at missing tile {tile}")
            continue
        connects, amsels, mastersets, rules, bad = box_view(ops)
        problems += [f"{tile} {b}" for b in bad]
        driven = defaultdict(int)
        for slave, ms in connects.items():
            for m in ms:
                driven[m] += 1
                if not t.legal(tile, slave, m):
                    problems.append(
                        f"{tile} connects {fmt_port(slave)} to {fmt_port(m)} illegally"
                    )
        for port, op in mastersets.items():
            driven[port] += 1
            names = op[2]
            arbs = {amsels[n][0] for n in names if n in amsels}
            if (
                len(arbs) != 1
                or len(names) != len(set(names))
                or any(n not in amsels for n in names)
            ):
                problems.append(
                    f"{tile} masterset {fmt_port(port)} spans arbiters {sorted(arbs)}"
                )
        for port, n in driven.items():
            if n > 1:
                problems.append(f"{tile} {fmt_port(port)} is driven {n} times")
            if port[1] >= t.masters[tile][port[0]]:
                problems.append(f"{tile} has no master port {fmt_port(port)}")
        for port, op in rules.items():
            if len(op[2]) > PARAMS["rule_slots"]:
                problems.append(f"{tile} {fmt_port(port)} holds {len(op[2])} rules")
            if port[1] >= t.slaves[tile][port[0]]:
                problems.append(f"{tile} has no slave port {fmt_port(port)}")
            for mask, value, name in op[2]:
                if name not in amsels:
                    problems.append(f"{tile} rule on {fmt_port(port)} names no amsel")
                if value & ~mask & PARAMS["max_id"]:
                    problems.append(
                        f"{tile} rule ({mask}, {value}) sets bits outside its mask"
                    )
        for name, (a, m) in amsels.items():
            if a >= PARAMS["arbiters"] or m >= PARAMS["msels"]:
                problems.append(f"{tile} amsel<{a}> ({m}) out of range")
        fresh = [op for op in ops if op[0] == "amsel"]
        if len({(op[2], op[3]) for op in fresh}) != len(fresh):
            problems.append(f"{tile} declares one amsel twice")
    # The shim mux's North side is the switchbox's South side, which the
    # TargetModel counts on the switchbox, not on the mux.
    for tile, conns in out.muxes.items():
        for s, m in conns:
            if t.kind(tile) != "shim":
                problems.append(f"shim mux at {tile}, not a shim tile")
                continue
            ns = t.masters[tile][SOUTH] if s[0] == NORTH else t.mux_slaves[tile][s[0]]
            nm = t.slaves[tile][SOUTH] if m[0] == NORTH else t.mux_masters[tile][m[0]]
            if s[1] >= ns or m[1] >= nm:
                problems.append(
                    f"{tile} shim mux {fmt_port(s)} -> {fmt_port(m)} out of range"
                )

    # The switchbox configuration the input already had stays.
    for tile, ops in d.boxes.items():
        before = box_view(ops)
        after = box_view(out.boxes.get(tile, []))
        for s, ms in before[0].items():
            for m in ms:
                if m not in after[0].get(s, []):
                    problems.append(
                        f"{tile} lost connect {fmt_port(s)} -> {fmt_port(m)}"
                    )
        for port, op in before[2].items():
            new = after[2].get(port)
            if (
                new is None
                or sorted(before[1][n] for n in op[2])
                != sorted(after[1].get(n) for n in new[2])
                or keeps_header(tile, port, op[3]) != keeps_header(tile, port, new[3])
                or op[4] != new[4]
            ):
                problems.append(f"{tile} changed masterset {fmt_port(port)}")
        for port, op in before[3].items():
            new = after[3].get(port)
            old_rules = [(m, v, before[1].get(n)) for m, v, n in op[2]]
            new_rules = (
                [] if new is None else [(m, v, after[1].get(n)) for m, v, n in new[2]]
            )
            if any(x not in new_rules for x in old_rules):
                problems.append(f"{tile} changed packet_rules {fmt_port(port)}")
    for tile, conns in d.muxes.items():
        for conn in conns:
            if conn not in out.muxes.get(tile, []):
                problems.append(
                    f"{tile} lost shim mux {fmt_port(conn[0])} -> {fmt_port(conn[1])}"
                )

    nreq = an.num_requested
    want = defaultdict(set)
    for s in an.streams[:nreq]:
        want[(s.src, s.pid)].add(s.dst)
    paths = {}
    for (src, pid), dsts in sorted(want.items(), key=str):
        ends, bad = trace_output(out, src, pid)
        problems += bad
        if set(ends) != dsts:
            problems.append(
                f"{'circuit' if pid is None else f'id {pid}'} from {fmt_ep(src)} reaches "
                f"[{', '.join(sorted(map(fmt_ep, ends)))}], wants "
                f"[{', '.join(sorted(map(fmt_ep, dsts)))}]"
            )
        for dst, hops in ends.items():
            paths[(src, dst, pid)] = hops

    # Each masterset the router added carries a requested stream.
    carried = {(tile, m) for hops in paths.values() for tile, _, m, arb in hops if arb}
    for tile, ops in sorted(out.boxes.items()):
        had = box_view(d.boxes.get(tile, []))[2]
        for port in box_view(ops)[2]:
            if port not in had and (tile, port) not in carried:
                problems.append(f"{tile} masterset {fmt_port(port)} carries no stream")

    # What the input's own switchbox configuration delivered, it still does.
    pinned = defaultdict(set)
    for s in an.streams[nreq:]:
        if s.dst[2] not in DIRECTIONAL:
            pinned[(s.src, s.pid)].add(s.dst)
    for (src, pid), dsts in sorted(pinned.items(), key=str):
        ends = set(trace_output(out, src, pid)[0])
        if not dsts <= ends <= dsts | want.get((src, pid), set()):
            problems.append(
                f"pinned {'circuit' if pid is None else f'id {pid}'} from "
                f"{fmt_ep(src)} reaches [{', '.join(sorted(map(fmt_ep, ends)))}], "
                f"had [{', '.join(sorted(map(fmt_ep, dsts)))}]"
            )

    packet_hops, circuit_hops = set(), set()
    promoted = defaultdict(set)
    for (src, dst, pid), hops in paths.items():
        for tile, slave, m, arb in hops:
            if arb is None:
                (circuit_hops if pid is None else packet_hops).add((tile, slave, m))
    for conn in sorted(packet_hops & circuit_hops):
        problems.append(f"circuit and packet streams share {conn}")
    for tile, slave, m in sorted(packet_hops):
        promoted[tile].add(m)
        if not hops_on or t.kind(tile) == "shim" or m[0] not in DIRECTIONAL:
            problems.append(
                f"packet hop {fmt_port(slave)} -> {fmt_port(m)} at {tile} is circuit switched"
            )
    for tile, ms in promoted.items():
        view = box_view(out.boxes[tile])
        packet_masters = set(view[2]) - set(box_view(d.boxes.get(tile, []))[2])
        if len(packet_masters | ms) <= free_arbiters(reserved[tile], wholly=True):
            problems.append(
                f"{tile} circuit switches packet hops with only "
                f"{len(packet_masters | ms)} packet masters"
            )

    # Streams that can deadlock never share an arbiter or a link.
    routes = [list(s.hops) for s in an.streams]
    marks = {}
    for i, s in enumerate(an.streams[:nreq]):
        hops = paths.get((s.src, s.dst, s.pid), [])
        routes[i] = [
            (tile, slave, None if arb is None else arb[0])
            for tile, slave, _, arb in hops
        ]
        marks[i] = (
            {(tile, arb[0]) for tile, _, _, arb in hops if arb is not None},
            {(tile, m) for tile, _, m, _ in hops},
        )
    for i in range(nreq):
        for j in range(i + 1, nreq):
            if not an.conflict(i, j):
                continue
            arbs = marks[i][0] & marks[j][0]
            links = marks[i][1] & marks[j][1]
            if arbs or links:
                where = f"arbiter {min(arbs)}" if arbs else f"link {min(links)}"
                problems.append(
                    f"conflicting {describe_stream(an.streams[i])} and "
                    f"{describe_stream(an.streams[j])} share {where}"
                )
    cycle = an.hold_cycle(routes)
    if cycle is not None:
        problems.append("hold cycle: " + an.explain_cycle(cycle))
    # A cycle through waits at receivers trees share fails the router, unless
    # it rests on assumptions.
    shared = (
        an.hold_cycle(routes, forced_waits=True, definite=True)
        if cycle is None
        else None
    )

    keeps = last_keep(d)
    mixed_ctrl, low_priority, ctrl_used = set(), set(), set()
    # Without a control-packet reload, a routing that cannot keep the overlay
    # routes the prioritized flows as the others.
    prioritized = d.reload or any(
        op[4] for ops in out.boxes.values() for op in box_view(ops)[2].values()
    )
    # A source's packets with a priority flow's id route as that flow does.
    prio_sent = {
        (src, f["id"])
        for f in d.packet_flows
        if f["priority"] and prioritized
        for src in f["srcs"]
    }
    for f in d.packet_flows:
        for dst in f["dsts"]:
            for src in f["srcs"]:
                hops = paths.get((src, dst, f["id"]))
                if not hops:
                    continue
                tile, _, m, _ = hops[-1]
                ms = box_view(out.boxes[tile])[2].get(m)
                if ms is not None and ms[3] != keeps[dst]:
                    problems.append(
                        f"packet flow {fmt_ep(src)} -> {fmt_ep(dst)} (id {f['id']}) ends "
                        f"with keep_pkt_header {ms[3]}, wants {keeps[dst]}"
                    )
                for k, (tile, _, m, arb) in enumerate(hops):
                    if arb is None:
                        continue
                    view = box_view(out.boxes[tile])
                    ctrl = view[2][m][4]
                    if k < len(hops) - 1 and not keeps_header(tile, m, view[2][m][3]):
                        problems.append(
                            f"id {f['id']} loses its header at {tile} {fmt_port(m)}"
                        )
                    if (src, f["id"]) in prio_sent:
                        ctrl_used.add((tile, m))
                    if not (f["priority"] and prioritized):
                        if ctrl:
                            mixed_ctrl.add((tile, m))
                        continue
                    if not ctrl:
                        problems.append(
                            f"priority id {f['id']} at {tile} {fmt_port(m)} not ctrl"
                        )
                    if any(
                        view[1][n][0] == arb[0] and view[1][n][1] > arb[1]
                        for op in view[2].values()
                        if not op[4]
                        for n in op[2]
                        if n in view[1]
                    ):
                        low_priority.add((tile, m))
    # A reload that skips the control overlay skips a ctrl masterset, so one
    # no priority flow takes leaves its flows unconfigured.
    for tile, ops in out.boxes.items():
        given = box_view(d.boxes.get(tile, []))[2]
        for m, op in box_view(ops)[2].items():
            if (
                op[4]
                and (tile, m) not in ctrl_used
                and not (m in given and given[m][4])
            ):
                problems.append(
                    f"ctrl masterset {fmt_port(m)} at {tile} carries no priority flow"
                )

    groups, bound = defaultdict(set), defaultdict(int)
    for (src, dst, pid), hops in paths.items():
        groups[(src, pid is None)] |= {(h[0], h[2]) for h in hops}
    for s in an.streams[:nreq]:
        dist = abs(s.src[0] - s.dst[0]) + abs(s.src[1] - s.dst[1]) + 1
        bound[(s.src, s.pid is None)] = max(bound[(s.src, s.pid is None)], dist)

    def amsel_count(boxes):
        return sum(1 for ops in boxes.values() for op in ops if op[0] == "amsel")

    def arbiter_count(boxes):
        return len(
            {
                (tile, op[2])
                for tile, ops in boxes.items()
                for op in ops
                if op[0] == "amsel"
            }
        )

    stats = dict(
        hops=sum(len(v) for v in groups.values()),
        bound=sum(bound.values()),
        arbiters=arbiter_count(out.boxes) - arbiter_count(d.boxes),
        amsels=amsel_count(out.boxes) - amsel_count(d.boxes),
        promoted=sum(len(v) for v in promoted.values()),
        mixed_ctrl=len(mixed_ctrl),
        low_priority=len(low_priority),
        shared_receiver_cycle=shared and an.explain_cycle(shared),
    )
    return problems, stats


# Router bugs reported and not yet fixed: (label, test on the design and the
# verifier's problems). Matching cases are counted, not failed.
KNOWN_BUGS = []


def known_bug(d, problems):
    return next((label for label, test in KNOWN_BUGS if test(d, problems)), None)
