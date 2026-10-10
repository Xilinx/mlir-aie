#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Reading the switchbox settings the router emitted."""

from collections import defaultdict, deque

from .fabric import (
    CORE,
    CTRL,
    DIRECTIONAL,
    DMA,
    NORTH,
    SOUTH,
    TRACE,
    fmt_ep,
    fmt_port,
    linked_input,
)


def box_view(ops):
    """(connects {slave: [master]}, amsels {name: (arbiter, msel)},
    mastersets {port: op}, rules {slave: [rule]}, problems)."""
    connects, amsels, mastersets, rules, problems = defaultdict(list), {}, {}, {}, []
    for op in ops:
        if op[0] == "connect":
            if op[2] in connects[op[1]]:
                problems.append(f"connect {fmt_port(op[1])} -> {fmt_port(op[2])} twice")
            connects[op[1]].append(op[2])
        elif op[0] == "amsel":
            amsels[op[1]] = (op[2], op[3])
        elif op[0] == "masterset":
            if op[1] in mastersets:
                problems.append(f"two mastersets on {fmt_port(op[1])}")
            mastersets[op[1]] = op
        else:
            if op[1] in rules:
                problems.append(f"two packet_rules on {fmt_port(op[1])}")
            rules[op[1]] = op
    return connects, amsels, mastersets, rules, problems


def trace_output(out, src, pid):
    """Follow what leaves `src` with id `pid` (None: circuit) through the
    emitted switchboxes and shim muxes. Returns ({endpoint: hops}, problems);
    a hop is (tile, slave, master, (arbiter, msel) or None), shim DMA inputs
    as their South port, as StreamTracer has them."""
    t = out.target
    views = {tile: box_view(ops) for tile, ops in out.boxes.items()}
    what = f"{'circuit' if pid is None else f'id {pid}'} from {fmt_ep(src)}"
    ends, problems, seen = {}, [], set()
    c, r = src[:2]
    if t.kind(src[:2]) == "shim" and src[2] not in (TRACE, CTRL):
        outs = [m for s, m in out.muxes.get((c, r), []) if s == src[2:]]
        if len(outs) != 1 or outs[0][0] != NORTH:
            return ends, [f"{fmt_ep(src)} has shim mux outputs {outs}"]
        start = ((c, r), (SOUTH, outs[0][1]), [])
    else:
        start = ((c, r), src[2:], [])
    queue = deque([start])
    while queue:
        tile, slave, path = queue.popleft()
        if (tile, slave) in seen:
            problems.append(f"{what} reaches {tile} {fmt_port(slave)} twice")
            continue
        seen.add((tile, slave))
        view = views.get(tile)
        conn = view[0].get(slave, []) if view else []
        rules = view[3].get(slave) if view else None
        if conn and rules:
            problems.append(
                f"{tile} {fmt_port(slave)} is both circuit and packet switched"
            )
        if conn:
            masters, arb = conn, None
        elif rules and pid is not None:
            hits = [x for x in rules[2] if pid & x[0] == x[1] & x[0]]
            if not hits:
                problems.append(f"{what} matches no rule at {tile} {fmt_port(slave)}")
                continue
            arb = view[1].get(hits[0][2])
            masters = [p for p, ms in view[2].items() if hits[0][2] in ms[2]]
            if arb is None or not masters:
                problems.append(f"{what} selects a dangling amsel at {tile}")
                continue
        else:
            problems.append(f"{what} stops at {tile} {fmt_port(slave)}")
            continue
        for m in masters:
            hop_path = path + [(tile, slave, m, arb)]
            if t.kind(tile) == "shim" and m[0] == SOUTH:
                outs = [d for s, d in out.muxes.get(tile, []) if s == (NORTH, m[1])]
                if outs and (len(outs) != 1 or outs[0][0] != DMA):
                    problems.append(f"{what} leaves {tile} South:{m[1]} to {outs}")
                    continue
                ep = (*tile, DMA, outs[0][1]) if outs else (*tile, *m)
            elif m[0] in DIRECTIONAL:
                ntile, nport = linked_input(tile, m)
                if not t.exists(ntile):
                    problems.append(f"{what} leaves the device at {tile} {fmt_port(m)}")
                    continue
                queue.append((ntile, nport, hop_path))
                continue
            elif m[0] == CTRL or (m[0] in (DMA, CORE) and t.kind(tile) != "shim"):
                ep = (*tile, *m)
            else:
                problems.append(f"{what} leaves {tile} by {fmt_port(m)}")
                continue
            if ep in ends:
                problems.append(f"{what} reaches {fmt_ep(ep)} twice")
            ends[ep] = hop_path
    return ends, problems
