#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %python %s --device npu1 | FileCheck %s
# RUN: %python %s --device npu2 | FileCheck %s

# Model-based property test for --aie-create-pathfinder-flows.
#
# The .mlir tests in this directory pin exact switchbox settings on a handful of
# designs. This file checks what any routing must satisfy on many generated
# ones, against a model of the fabric and of the router's deadlock rules:
#
#  * the fabric: port counts per tile from the TargetModel bindings, legal
#    crossbar connections (AIE2TargetModel::isLegalTileConnection), six
#    arbiters of four msels per switchbox, four packet rules per slave port;
#  * which streams can deadlock (AIEStreamDependencyAnalysis, mirrored here
#    line for line: stream volumes, receiver capacities, the waits-for graph
#    of cores, DMA channels and the host, and the global hold-cycle search);
#  * the router's own arbiter planning and pre-routing rejection
#    (planArbiters, cutTiles, unroutableArbiters, the hold-cycle search), run
#    on a routing the generator builds itself as a witness.
#
# Designs come in three tiers:
#
#  * routable: the generator routes every flow on exclusive links, plans its
#    arbiters with the router's own rules, and emits only the flows, programs,
#    runtime sequence and pre-placed switchbox configuration. The router has
#    to succeed, and its output is checked hop by hop.
#  * unroutable: more pairwise conflicting streams must take an arbiter at a
#    tile than it has (the router's message is predicted exactly), more
#    circuit flows must cross a port than it has channels, or conflicting
#    streams must merge. The router has to fail.
#  * unknown: recorded; whatever the router emits is still checked, with the
#    hold-cycle rule applied to the output.
#
# The model is importable: Target, Design, load_design, generate, verdict and
# verify.

import argparse
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aie.ir import Context, Location, Module
import aie.dialects.aie  # noqa: F401
import aie.dialects.aiex  # noqa: F401
from aie.dialects.aie import AIEDevice, WireBundle, get_target_model

PARAMS = dict(
    # numArbiters, numMselsPerArbiter: AIECreatePathFindFlows.cpp
    arbiters=6,
    msels=4,
    # AIETargetModel::getNumSlaveSlots, getMaxPacketId
    rule_slots=4,
    max_id=31,
    # pktHeaderBytes, maxBDs: AIEStreamDependencyAnalysis.cpp
    header_bytes=4,
    max_bd_steps=1024,
    # stepBudget in planArbiters and unroutableArbiters, budget in
    # runOnPacketFlow's hold-cycle search: AIECreatePathFindFlows.cpp
    plan_budget=100000,
    clique_budget=100000,
    hold_budget=256,
    # Designs per tier and per aie-opt run.
    routable=170,
    unroutable=40,
    unknown=50,
    batch=40,
    # Ratchets on the router's output.
    min_routed_rate=1.0,
    max_hop_ratio=1.10,
    max_amsels=19.5,
    max_low_priority=0.16,
    max_ms_per_design=250,
    # Generator knobs: share of routable designs given pre-placed switchbox
    # configuration, and parallel aie-opt runs.
    fixed_rate=0.25,
    jobs=4,
)
# Ratchet on the route space the test reaches: values hit per dimension
# (route_space, space_domain). Raise as the generator reaches more.
SPACE_FLOORS = {}

FAMILIES = {
    "npu1": ("npu1_1col", "npu1_2col", "npu1_3col", "npu1"),
    "npu2": ("npu2_1col", "npu2_3col", "npu2_4col", "npu2"),
}
DEVICE_IDS = {
    4: "npu1",
    5: "npu1_1col",
    6: "npu1_2col",
    7: "npu1_3col",
    8: "npu2",
    **{9 + k: f"npu2_{k + 1}col" for k in range(7)},
}

BUNDLES = [
    "Core",
    "DMA",
    "FIFO",
    "South",
    "West",
    "North",
    "East",
    "PLIO",
    "NOC",
    "Trace",
    "TileControl",
]
CORE, DMA, FIFO, SOUTH, WEST, NORTH, EAST, PLIO, NOC, TRACE, CTRL = range(11)
BUNDLE = {n: i for i, n in enumerate(BUNDLES)}
DIRECTIONAL = (SOUTH, WEST, NORTH, EAST)
STEP = {
    NORTH: (0, 1, SOUTH),
    SOUTH: (0, -1, NORTH),
    EAST: (1, 0, WEST),
    WEST: (-1, 0, EAST),
}
S2MM, MM2S = 0, 1
DIRS = ("S2MM", "MM2S")
ACTIONS = ("Acquire", "Release", "AcquireGreaterEqual")


def fmt_port(p):
    return f"{BUNDLES[p[0]]}:{p[1]}"


def fmt_ep(ep):
    return f"({ep[0]}, {ep[1]}) {BUNDLES[ep[2]]}:{ep[3]}"


def linked_input(tile, port):
    if port[0] not in STEP:
        return None
    dc, dr, into = STEP[port[0]]
    return (tile[0] + dc, tile[1] + dr), (into, port[1])


class Target:
    """What the TargetModel says about a device, plus the AIE2 crossbar."""

    _cache = {}

    def __new__(cls, dev):
        if dev not in cls._cache:
            self = super().__new__(cls)
            self._load(dev)
            cls._cache[dev] = self
        return cls._cache[dev]

    def _load(self, dev):
        self.dev = dev
        with Context():
            tm = get_target_model(getattr(AIEDevice, dev))
            self.cols, self.rows = tm.columns(), tm.rows()
            wb = {i: getattr(WireBundle, n) for i, n in enumerate(BUNDLES)}
            self.kinds, self.masters, self.slaves = {}, {}, {}
            self.mux_masters, self.mux_slaves, self.num_locks = {}, {}, {}
            for c in range(self.cols):
                for r in range(self.rows):
                    t = (c, r)
                    if tm.is_shim_noc_or_pl_tile(c, r):
                        self.kinds[t] = "shim"
                    elif tm.is_mem_tile(c, r):
                        self.kinds[t] = "mem"
                    else:
                        self.kinds[t] = "core"
                    self.masters[t] = {
                        b: tm.get_num_dest_switchbox_connections(c, r, wb[b])
                        for b in range(len(BUNDLES))
                    }
                    self.slaves[t] = {
                        b: tm.get_num_source_switchbox_connections(c, r, wb[b])
                        for b in range(len(BUNDLES))
                    }
                    self.num_locks[t] = tm.get_num_locks(c, r)
                    if self.kinds[t] == "shim":
                        self.mux_masters[t] = {
                            b: tm.get_num_dest_shim_mux_connections(c, r, wb[b])
                            for b in range(len(BUNDLES))
                        }
                        self.mux_slaves[t] = {
                            b: tm.get_num_source_shim_mux_connections(c, r, wb[b])
                            for b in range(len(BUNDLES))
                        }
                    self.shim_noc = {
                        c: tm.is_shim_noc_tile(c, 0) for c in range(self.cols)
                    }

    def kind(self, tile):
        return self.kinds[tile]

    def exists(self, tile):
        return tile in self.kinds

    def legal(self, tile, sp, mp):
        """AIE2TargetModel::isLegalTileConnection."""
        (sb, si), (db, di) = sp, mp
        if si >= self.slaves[tile][sb] or di >= self.masters[tile][db]:
            return False
        kind = self.kinds[tile]
        if kind == "mem":
            if sb == DMA:
                if db == DMA:
                    return si == di
                if db in (CTRL, SOUTH, NORTH):
                    return True
            if sb == CTRL:
                if db == DMA:
                    return di == 5
                if db in (SOUTH, NORTH):
                    return True
            if sb in (SOUTH, NORTH):
                if db in (DMA, CTRL):
                    return True
                if db in (SOUTH, NORTH):
                    return si == di
            if sb == TRACE:
                if db == DMA:
                    return di == 5
                if db == SOUTH:
                    return True
            return False
        if kind == "shim":
            fabric = (CTRL, FIFO, SOUTH, WEST, NORTH, EAST)
            if sb == CTRL:
                return db != CTRL
            if sb in (FIFO, SOUTH):
                return db in fabric
            if sb in (WEST, NORTH, EAST):
                return si == di if sb == db else db in fabric
            if sb == TRACE:
                if db in (FIFO, SOUTH):
                    return True
                if db in (WEST, EAST):
                    return di == 0
            return False
        if sb in (DMA, FIFO, SOUTH, WEST, NORTH, EAST):
            if db in (CORE, DMA, CTRL, FIFO, SOUTH, WEST, NORTH, EAST):
                return si == di if sb == db else True
        if sb == CORE:
            return db != CORE
        if sb == CTRL:
            return db not in (CTRL, DMA)
        if sb == TRACE:
            if db == DMA:
                return di == 0
            if db in (FIFO, SOUTH):
                return True
        return False

    def master_ports(self, tile):
        return [(b, i) for b, n in self.masters[tile].items() for i in range(n)]

    def endpoints(self, tile, sending):
        """Tile ports flows may start or end at."""
        c, r = tile
        kind = self.kinds[tile]
        if kind == "shim":
            return [(c, r, DMA, 0), (c, r, DMA, 1)] if self.shim_noc[c] else []
        n = (self.slaves if sending else self.masters)[tile]
        return [(c, r, DMA, ch) for ch in range(n[DMA])] + [
            (c, r, CORE, ch) for ch in range(n[CORE])
        ]


# A shim DMA reaches its switchbox through the shim mux: MM2S 0/1 arrive on
# South 3/7, and S2MM 0/1 leave from South 2/3.
def phys_src(ep):
    c, r, b, ch = ep
    if r == 0 and b == DMA:
        return (c, 0, SOUTH, 3 if ch == 0 else 7)
    return ep


def phys_dst(ep):
    c, r, b, ch = ep
    if r == 0 and b == DMA:
        return (c, 0, SOUTH, 2 if ch == 0 else 3)
    return ep


class Design:
    """Everything in a device the router and its deadlock rules read.

    Ports are (bundle, channel) and endpoints (col, row, bundle, channel),
    bundles as WireBundle numbers so tuples order as the C++ Port and TileID
    do. A DMA program's `seq` is its BD chain as the analysis walks it: for
    aie.dma_start the blocks from its first BD, including the terminal block
    when the chain ends; its ops are ("lock", action, lock, amount) and
    ("bd", bytes, packet id).
    """

    def __init__(self, dev):
        self.dev = dev
        self.tiles = []
        self.locks = {}  # name -> (col, row, id, init)
        self.programs = []  # document order
        self.cores = {}  # tile -> [(action, lock, amount)]
        self.boxes = {}  # tile -> [op]
        self.muxes = {}  # tile -> [(src port, dst port)]
        self.flows = []  # (src, dst)
        self.packet_flows = []
        self.allocs = {}  # symbol -> dict(tile, dir, ch, pkt)
        self.sequences = []  # [[event]]
        self.unsupported = []

    @property
    def target(self):
        return Target(self.dev)

    def copy(self):
        d = Design(self.dev)
        d.tiles = list(self.tiles)
        d.locks = dict(self.locks)
        d.programs = [
            dict(p, seq=[list(b) for b in p["seq"]], users=list(p.get("users", [])))
            for p in self.programs
        ]
        d.cores = {k: list(v) for k, v in self.cores.items()}
        d.boxes = {k: [tuple(o) for o in v] for k, v in self.boxes.items()}
        d.muxes = {k: list(v) for k, v in self.muxes.items()}
        d.flows = list(self.flows)
        d.packet_flows = [
            dict(f, srcs=list(f["srcs"]), dsts=list(f["dsts"]))
            for f in self.packet_flows
        ]
        d.allocs = {k: dict(v) for k, v in self.allocs.items()}
        d.sequences = [list(s) for s in self.sequences]
        d.unsupported = list(self.unsupported)
        return d

    def add_packet_flow(self, pid, srcs, dsts, keep=None, priority=None, mask=None):
        self.packet_flows.append(
            dict(
                id=pid,
                mask=mask,
                keep=keep,
                priority=priority,
                srcs=list(srcs),
                dsts=list(dsts),
            )
        )

    def lock(self, tile, init):
        n = sum(1 for v in self.locks.values() if v[:2] == tile)
        name = f"l_{tile[0]}_{tile[1]}_{n}"
        self.locks[name] = (tile[0], tile[1], n, init)
        return name

    def program_key(self, p):
        return (*p["tile"], p["dir"], p["ch"])

    def used_tiles(self):
        tiles = set(self.tiles)
        for s, d in self.flows:
            tiles |= {s[:2], d[:2]}
        for f in self.packet_flows:
            tiles |= {e[:2] for e in f["srcs"] + f["dsts"]}
        tiles |= {p["tile"] for p in self.programs}
        tiles |= set(self.cores) | set(self.boxes) | set(self.muxes)
        tiles |= {v[:2] for v in self.locks.values()}
        tiles |= {a["tile"] for a in self.allocs.values()}
        return sorted(tiles)

    def emit(self, tag=None):
        out = [
            "module {" if tag is None else f"module @s{tag} {{",
            f"  aie.device({self.dev}) {{",
        ]
        for c, r in self.used_tiles():
            out.append(f"    %t_{c}_{r} = aie.tile({c}, {r})")
        for name, (c, r, lid, init) in self.locks.items():
            out.append(
                f"    %{name} = aie.lock(%t_{c}_{r}, {lid}) "
                f'{{init = {init} : i32, sym_name = "{name}"}}'
            )
        nbuf = [0]

        def buffer(tile, nbytes):
            name = f"b_{tile[0]}_{tile[1]}_{nbuf[0]}"
            nbuf[0] += 1
            elem, n = ("i32", nbytes // 4) if nbytes % 4 == 0 else ("i8", nbytes)
            out.append(
                f"    %{name} = aie.buffer(%t_{tile[0]}_{tile[1]}) "
                f'{{sym_name = "{name}"}} : memref<{n}x{elem}>'
            )
            return f"%{name} : memref<{n}x{elem}> offset = 0 len = {n}"

        def lock_ops(ops, indent, consts):
            lines = []
            for op in ops:
                if op[0] == "lock":
                    _, action, lock, n = op
                    consts.add(n)
                    lines.append(
                        f"{indent}aie.use_lock(%{lock}, {ACTIONS[action]}, %c{n})"
                    )
            return lines

        def pkt_attr(pkt, bd_id=None):
            attrs = []
            if bd_id is not None:
                attrs.append(f"bd_id = {bd_id} : i32")
            if pkt is not None:
                attrs.append(f"packet = #aie.packet_info<pkt_type = 0, pkt_id = {pkt}>")
            return f" {{{', '.join(attrs)}}}" if attrs else ""

        by_tile = {}
        for p in self.programs:
            if p["kind"] == "start":
                by_tile.setdefault(p["tile"], []).append(p)
        for tile in by_tile:
            progs = by_tile[tile]
            body, consts = [], set()
            for n, p in enumerate(progs):
                blocks = p["seq"] if p["loops"] else p["seq"][:-1]
                head = f"^p{n}b0" if blocks else "^end"
                nxt = f"^p{n + 1}" if n + 1 < len(progs) else "^end"
                if n:
                    body.append(f"    ^p{n}:")
                rep = f", repeat_count = {p['repeat']}" if p["repeat"] else ""
                body.append(
                    f"      %d{n} = aie.dma_start({DIRS[p['dir']]}, {p['ch']}, "
                    f"{head}, {nxt}{rep})"
                )
                for i, ops in enumerate(blocks):
                    body.append(f"    ^p{n}b{i}:")
                    for op in ops:
                        if op[0] == "lock":
                            body += lock_ops([op], "      ", consts)
                        else:
                            body.append(
                                f"      aie.dma_bd({buffer(tile, op[1])})"
                                + pkt_attr(op[2])
                            )
                    if i + 1 < len(blocks):
                        body.append(f"      aie.next_bd ^p{n}b{i + 1}")
                    elif p["loops"]:
                        body.append(f"      aie.next_bd ^p{n}b{p.get('loop_to', 0)}")
                    else:
                        body.append("      aie.next_bd ^end")
            op = "aie.memtile_dma" if self.target.kind(tile) == "mem" else "aie.mem"
            out.append(
                f"    %dma_{tile[0]}_{tile[1]} = {op}(%t_{tile[0]}_{tile[1]}) {{"
            )
            out += [f"      %c{n} = arith.constant {n} : i32" for n in sorted(consts)]
            out += body
            out += ["    ^end:", "      aie.end", "    }"]
        for (c, r), uses in self.cores.items():
            out.append(f"    %core_{c}_{r} = aie.core(%t_{c}_{r}) {{")
            consts = {n for _, _, n in uses}
            out += [f"      %c{n} = arith.constant {n} : i32" for n in sorted(consts)]
            for action, lock, n in uses:
                out.append(f"      aie.use_lock(%{lock}, {ACTIONS[action]}, %c{n})")
            out += ["      aie.end", "    }"]
        for (c, r), ops in sorted(self.boxes.items()):
            out.append(f"    %sb_{c}_{r} = aie.switchbox(%t_{c}_{r}) {{")
            for op in ops:
                if op[0] == "connect":
                    (sb, si), (db, di) = op[1], op[2]
                    out.append(
                        f"      aie.connect<{BUNDLES[sb]} : {si}, {BUNDLES[db]} : {di}>"
                    )
                elif op[0] == "amsel":
                    out.append(f"      %{op[1]} = aie.amsel<{op[2]}> ({op[3]})")
                elif op[0] == "masterset":
                    _, port, names, keep, ctrl = op
                    attrs = []
                    if keep is not None:
                        attrs.append(f"keep_pkt_header = {str(keep).lower()}")
                    if ctrl:
                        attrs.append("is_ctrl_pkt_overlay")
                    out.append(
                        f"      %m_{BUNDLES[port[0]]}_{port[1]} = aie.masterset("
                        f"{BUNDLES[port[0]]} : {port[1]}, "
                        + ", ".join(f"%{n}" for n in names)
                        + ")"
                        + (f" {{{', '.join(attrs)}}}" if attrs else "")
                    )
                else:
                    _, port, rules, ctrl = op
                    out.append(
                        f"      aie.packet_rules({BUNDLES[port[0]]} : {port[1]}) {{"
                    )
                    for mask, value, name in rules:
                        out.append(f"        aie.rule({mask}, {value}, %{name})")
                    out.append("      }" + (" {is_ctrl_pkt_overlay}" if ctrl else ""))
            out.append("    }")
        for (c, r), conns in sorted(self.muxes.items()):
            out.append(f"    %mux_{c}_{r} = aie.shim_mux(%t_{c}_{r}) {{")
            for (sb, si), (db, di) in conns:
                out.append(
                    f"      aie.connect<{BUNDLES[sb]} : {si}, {BUNDLES[db]} : {di}>"
                )
            out.append("    }")
        for s, d in self.flows:
            out.append(
                f"    aie.flow(%t_{s[0]}_{s[1]}, {BUNDLES[s[2]]} : {s[3]}, "
                f"%t_{d[0]}_{d[1]}, {BUNDLES[d[2]]} : {d[3]})"
            )
        for f in self.packet_flows:
            ports = [
                f"aie.packet_source<%t_{s[0]}_{s[1]}, {BUNDLES[s[2]]} : {s[3]}>"
                for s in f["srcs"]
            ] + [
                f"aie.packet_dest<%t_{d[0]}_{d[1]}, {BUNDLES[d[2]]} : {d[3]}>"
                for d in f["dsts"]
            ]
            attrs = [
                f"{k} = {str(f[a]).lower()}"
                for k, a in (
                    ("keep_pkt_header", "keep"),
                    ("priority_route", "priority"),
                )
                if f[a] is not None
            ]
            head = f"{f['id']}" + (
                f", mask = {f['mask']}" if f["mask"] is not None else ""
            )
            out.append(
                f"    aie.packet_flow({head}) {{ {' '.join(ports)} }}"
                + (f" {{{', '.join(attrs)}}}" if attrs else "")
            )
        for sym, a in self.allocs.items():
            c, r = a["tile"]
            pkt = (
                f", <pkt_id = {a['pkt']}, pkt_type = 0>" if a["pkt"] is not None else ""
            )
            out.append(
                f"    aie.shim_dma_allocation @{sym}(%t_{c}_{r}, "
                f"{DIRS[a['dir']]}, {a['ch']}{pkt})"
            )
        if any(ev[0] == "chain" for events in self.sequences for ev in events):
            out += [
                "    aie.bd_chain @chain(%x: memref<16xi32>) {",
                "      aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)",
                "      aie.end",
                "    }",
            ]
        tasks = [p for p in self.programs if p["kind"] in ("task", "task_for")]
        for k, events in enumerate(self.sequences):
            args, body = [], []
            ntask = {}
            for p in tasks:
                if p.get("sequence", 0) != k:
                    continue
                t = len(ntask)
                ntask[id(p)] = t
                if p["kind"] == "task":
                    head = (
                        f"aiex.dma_configure_task(%t_{p['tile'][0]}_{p['tile'][1]}, "
                        f"{DIRS[p['dir']]}, {p['ch']})"
                    )
                else:
                    head = f"aiex.dma_configure_task_for @{p['alloc']}"
                body.append(f"      %task{t} = {head} {{")
                blocks = p["seq"]
                for i, ops in enumerate(blocks):
                    if i:
                        body.append(f"      ^bb{i}:")
                    for op in ops:
                        if op[0] == "bd":
                            a = len(args)
                            n = op[1] // 4
                            args.append(f"%a{a}: memref<{n}xi32>")
                            body.append(
                                f"        aie.dma_bd(%a{a} : memref<{n}xi32> offset = 0 "
                                f"len = {n})" + pkt_attr(op[2], i)
                            )
                    body.append(
                        f"        aie.next_bd ^bb{i + 1}"
                        if i + 1 < len(blocks)
                        else "        aie.end"
                    )
                attrs = ["issue_token = true"]
                if p["repeat"]:
                    attrs.append(f"repeat_count = {p['repeat']} : i32")
                body.append(f"      }} {{{', '.join(attrs)}}}")
            loops = 0
            by_prog = {i: p for i, p in enumerate(self.programs)}
            for ei, ev in enumerate(events):
                indent = "      "
                lines = []
                if ev[0] == "memcpy":
                    _, sym, pkt, nbytes, in_loop = ev
                    a = len(args)
                    n = nbytes // 4
                    args.append(f"%a{a}: memref<{n}xi32>")
                    pk = (
                        f", packet = <pkt_id = {pkt}, pkt_type = 0>"
                        if pkt is not None
                        else ""
                    )
                    lines.append(
                        f"aiex.npu.dma_memcpy_nd(%a{a}[0, 0, 0, 0][1, 1, 1, {n}]"
                        f"[0, 0, 0, 1]{pk}) {{ metadata = @{sym}, id = {a} : i64, "
                        f"issue_token = true }} : memref<{n}xi32>"
                    )
                elif ev[0] == "start":
                    in_loop = ev[2]
                    lines.append(
                        f"aiex.dma_start_task(%task{ntask[id(by_prog[ev[1]])]})"
                    )
                elif ev[0] == "await":
                    in_loop = False
                    lines.append(
                        f"aiex.dma_await_task(%task{ntask[id(by_prog[ev[1]])]})"
                    )
                elif ev[0] == "wait":
                    in_loop = False
                    lines.append(f"aiex.npu.dma_wait {{symbol = @{ev[1]}}}")
                elif ev[0] == "chain":
                    _, key, alloc, in_loop = ev
                    a = len(args)
                    args.append(f"%a{a}: memref<16xi32>")
                    on = (
                        f"for @{alloc}"
                        if alloc
                        else f"on (%t_{key[0]}_{key[1]}, {DIRS[key[2]]}, {key[3]})"
                    )
                    lines.append(
                        f"%chain{ei} = aiex.dma_start_bd_chain{'_for' if alloc else ''} "
                        f"@chain(%a{a}) : "
                        f"(memref<16xi32>) {on}"
                    )
                elif ev[0] == "await_chain":
                    in_loop = False
                    lines.append(f"aiex.dma_await_task(%chain{ev[1]})")
                else:
                    continue
                if in_loop:
                    body += [
                        f"{indent}%lb{loops} = arith.constant 0 : index",
                        f"{indent}%ub{loops} = arith.constant 2 : index",
                        f"{indent}%st{loops} = arith.constant 1 : index",
                        f"{indent}scf.for %i{loops} = %lb{loops} to %ub{loops} "
                        f"step %st{loops} {{",
                    ]
                    body += [f"{indent}  {l}" for l in lines]
                    body.append(f"{indent}}}")
                    loops += 1
                else:
                    body += [f"{indent}{l}" for l in lines]
            out.append(f"    aie.runtime_sequence @seq{k}({', '.join(args)}) {{")
            out += body
            out.append("    }")
        out += ["  }", "}", ""]
        return "\n".join(out)


def _int(attr):
    s = str(attr)
    if s in ("true", "false"):
        return s == "true"
    return int(s.split(":")[0].strip())


def _pkt(attr):
    m = re.search(r"pkt_id = (\d+)", str(attr))
    return int(m.group(1)) if m else None


def _elem_bits(type_str):
    m = re.search(r"x(i|f|bf|ui|si)(\d+)>", type_str)
    if m:
        return int(m.group(2))
    return None


def _num_elems(type_str):
    m = re.match(r"memref<([\dx]+)x[a-z]", type_str)
    if not m:
        return None
    n = 1
    for d in m.group(1).split("x"):
        n *= int(d)
    return n


def _attrs(op):
    return {
        op.attributes[i].name: op.attributes[i].attr for i in range(len(op.attributes))
    }


def _ops(block):
    return [x.operation for x in block.operations]


def load_design(text):
    """Parse MLIR text (one aie.device) into a Design, runtime sequence
    included. Ops the model does not read are listed in design.unsupported."""
    with Context(), Location.unknown():
        module = Module.parse(text)
        device = None
        for op in module.body.operations:
            op = op.operation
            if op.name == "aie.device":
                device = op
                break
        if device is None:
            raise ValueError("no aie.device")
        attrs = _attrs(device)
        d = Design(DEVICE_IDS[_int(attrs["device"])])
        tiles, locks, amsels_of = {}, {}, {}
        consts = {}
        results = {}

        def value_key(v):
            return hash(v)

        def const_of(v):
            return consts.get(value_key(v))

        def tile_of(v):
            return tiles.get(value_key(v))

        def lock_ops(op, out):
            if op.name == "aie.use_lock":
                a = _attrs(op)
                lock = locks.get(value_key(op.operands[0]))
                n = const_of(op.operands[1]) if len(op.operands) > 1 else None
                if n is None and "value" in a:
                    n = _int(a["value"])
                out.append(("lock", _int(a["action"]), lock, n))
            elif op.name == "aie.dma_bd":
                a = _attrs(op)
                ty = str(op.operands[0].type)
                bits = _elem_bits(ty) or 32
                if "static_len" in a:
                    n = _int(a["static_len"])
                else:
                    n = _num_elems(ty) or 0
                pkt = _pkt(a["packet"]) if "packet" in a else None
                out.append(("bd", n * max(bits // 8, 1), pkt))
            elif op.name == "arith.constant":
                try:
                    consts[value_key(op.results[0])] = _int(_attrs(op)["value"])
                except (ValueError, KeyError):
                    pass

        def walk_ops(op, out):
            lock_ops(op, out)
            for region in op.regions:
                for b in region.blocks:
                    for x in _ops(b):
                        walk_ops(x, out)

        def dma_programs(op, tile):
            blocks = list(op.regions[0].blocks)
            for b in blocks:
                for x in _ops(b):
                    if x.name == "arith.constant":
                        lock_ops(x, [])
            for b in blocks:
                for x in _ops(b):
                    if x.name == "aie.dma_start":
                        a = _attrs(x)
                        seq, seen, loops, loop_to = [], [], False, 0
                        blk = list(x.successors)[0]
                        while blk is not None:
                            idx = blocks.index(blk)
                            if idx in seen:
                                loops, loop_to = True, seen.index(idx)
                                break
                            seen.append(idx)
                            ops = []
                            nxt = None
                            for y in _ops(blocks[idx]):
                                lock_ops(y, ops)
                                if y.name == "aie.next_bd":
                                    nxt = list(y.successors)[0]
                            seq.append(ops)
                            blk = nxt
                        d.programs.append(
                            dict(
                                tile=tile,
                                kind="start",
                                dir=_int(a["channel_dir"]),
                                ch=_int(a["channel_index"]),
                                seq=seq,
                                loops=loops,
                                loop_to=loop_to,
                                repeat=(
                                    _int(a["repeat_count"])
                                    if "repeat_count" in a
                                    else 0
                                ),
                                dyn_repeat=False,
                                users=[],
                            )
                        )
                    elif x.name == "aie.dma":
                        d.unsupported.append(x.name)

        def box_ops(op):
            out = []
            names = {}
            for x in _ops(op.regions[0].blocks[0]):
                a = _attrs(x)
                if x.name == "aie.connect":
                    out.append(
                        (
                            "connect",
                            (_int(a["source_bundle"]), _int(a["source_channel"])),
                            (_int(a["dest_bundle"]), _int(a["dest_channel"])),
                        )
                    )
                elif x.name == "aie.amsel":
                    name = f"a{_int(a['arbiterID'])}_{_int(a['msel'])}_{len(amsels_of)}"
                    amsels_of[name] = None
                    names[value_key(x.results[0])] = name
                    out.append(("amsel", name, _int(a["arbiterID"]), _int(a["msel"])))
                elif x.name == "aie.masterset":
                    out.append(
                        (
                            "masterset",
                            (_int(a["dest_bundle"]), _int(a["dest_channel"])),
                            [names.get(value_key(v)) for v in x.operands],
                            (
                                _int(a["keep_pkt_header"])
                                if "keep_pkt_header" in a
                                else None
                            ),
                            "is_ctrl_pkt_overlay" in a,
                        )
                    )
                elif x.name == "aie.packet_rules":
                    rules = []
                    for y in _ops(x.regions[0].blocks[0]):
                        if y.name == "aie.rule":
                            ya = _attrs(y)
                            rules.append(
                                (
                                    _int(ya["mask"]),
                                    _int(ya["value"]),
                                    names.get(value_key(y.operands[0])),
                                )
                            )
                    out.append(
                        (
                            "rules",
                            (_int(a["source_bundle"]), _int(a["source_channel"])),
                            rules,
                            "is_ctrl_pkt_overlay" in a,
                        )
                    )
                elif x.name != "aie.end":
                    d.unsupported.append(x.name)
            return out

        def sequence(op):
            events = []
            k = len(d.sequences)

            def walk(x, in_loop):
                a = _attrs(x)
                if x.name == "aiex.npu.dma_memcpy_nd":
                    sym = str(a["metadata"]).lstrip("@")
                    ty = str(x.operands[0].type)
                    bits = _elem_bits(ty)
                    sizes = [
                        int(v)
                        for v in re.findall(
                            r"-?\d+", str(a["static_sizes"]).split(":")[-1]
                        )
                    ]
                    dynamic = any(v < 0 for v in sizes) or bits is None
                    n = 1
                    for v in sizes:
                        n *= v
                    nbytes = None if dynamic else n * bits // 8
                    pkt = _pkt(a["packet"]) if "packet" in a else None
                    events.append(("memcpy", sym, pkt, nbytes, in_loop))
                elif x.name in (
                    "aiex.dma_configure_task",
                    "aiex.dma_configure_task_for",
                ):
                    if x.name == "aiex.dma_configure_task":
                        tile = tile_of(x.operands[0])
                        dr, ch, alloc = _int(a["direction"]), _int(a["channel"]), None
                        dyn = len(x.operands) > 1
                    else:
                        alloc = str(a["alloc"]).lstrip("@")
                        al = d.allocs.get(alloc)
                        tile = al["tile"] if al else None
                        dr, ch = (al["dir"], al["ch"]) if al else (None, None)
                        dyn = len(x.operands) > 0
                    blocks = []
                    for b in x.regions[0].blocks:
                        ops = []
                        for y in _ops(b):
                            walk_ops(y, ops)
                        blocks.append(ops)
                    if tile is not None:
                        results[value_key(x.results[0])] = len(d.programs)
                        d.programs.append(
                            dict(
                                tile=tile,
                                kind="task" if alloc is None else "task_for",
                                alloc=alloc,
                                dir=dr,
                                ch=ch,
                                seq=blocks,
                                loops=False,
                                repeat=(
                                    _int(a["repeat_count"])
                                    if "repeat_count" in a
                                    else 0
                                ),
                                dyn_repeat=dyn,
                                users=[],
                                sequence=k,
                            )
                        )
                elif x.name in ("aiex.dma_start_task", "aiex.dma_await_task"):
                    p = results.get(value_key(x.operands[0]))
                    if isinstance(p, int):
                        if x.name == "aiex.dma_start_task":
                            d.programs[p]["users"].append(in_loop)
                            events.append(("start", p, in_loop))
                        else:
                            events.append(("await", p))
                    elif p is not None and x.name == "aiex.dma_await_task":
                        events.append(("await_chain", p[1]))
                elif x.name == "aiex.npu.dma_wait":
                    events.append(("wait", str(a["symbol"]).lstrip("@")))
                elif x.name in (
                    "aiex.dma_start_bd_chain",
                    "aiex.dma_start_bd_chain_for",
                ):
                    if x.name == "aiex.dma_start_bd_chain":
                        tile = next(
                            (tile_of(v) for v in x.operands if tile_of(v) is not None),
                            None,
                        )
                        key = (
                            None
                            if tile is None
                            else (*tile, _int(a["direction"]), _int(a["channel"]))
                        )
                        alloc = None
                    else:
                        key, alloc = None, str(a["alloc"]).lstrip("@")
                    results[value_key(x.results[0])] = ("chain", len(events))
                    events.append(("chain", key, alloc, in_loop))
                elif x.name == "arith.constant":
                    lock_ops(x, [])
                elif x.name in ("scf.for", "scf.while", "scf.parallel", "affine.for"):
                    for region in x.regions:
                        for b in region.blocks:
                            for y in _ops(b):
                                walk(y, True)
                elif x.name not in ("aie.end", "scf.yield", "aie.next_bd"):
                    if x.regions:
                        for region in x.regions:
                            for b in region.blocks:
                                for y in _ops(b):
                                    walk(y, in_loop)

            for b in op.regions[0].blocks:
                for x in _ops(b):
                    walk(x, False)
            d.sequences.append(events)

        for op in _ops(device.regions[0].blocks[0]):
            a = _attrs(op)
            name = op.name
            if name == "aie.tile":
                t = (_int(a["col"]), _int(a["row"]))
                tiles[value_key(op.results[0])] = t
                d.tiles.append(t)
            elif name == "aie.lock":
                t = tile_of(op.operands[0])
                sym = (
                    str(a["sym_name"]).strip('"')
                    if "sym_name" in a
                    else f"lock{len(locks)}"
                )
                lid = _int(a["lockID"]) if "lockID" in a else -1
                init = _int(a["init"]) if "init" in a else 0
                locks[value_key(op.results[0])] = sym
                d.locks[sym] = (t[0], t[1], lid, init)
            elif name in ("aie.mem", "aie.memtile_dma", "aie.shim_dma"):
                dma_programs(op, tile_of(op.operands[0]))
            elif name == "aie.core":
                uses = []
                for region in op.regions:
                    for b in region.blocks:
                        for x in _ops(b):
                            walk_ops(x, uses)
                d.cores[tile_of(op.operands[0])] = [
                    (u[1], u[2], u[3]) for u in uses if u[0] == "lock"
                ]
            elif name == "aie.switchbox":
                d.boxes.setdefault(tile_of(op.operands[0]), []).extend(box_ops(op))
            elif name == "aie.shim_mux":
                d.muxes.setdefault(tile_of(op.operands[0]), []).extend(
                    (o[1], o[2]) for o in box_ops(op) if o[0] == "connect"
                )
            elif name == "aie.flow":
                s, t = tile_of(op.operands[0]), tile_of(op.operands[1])
                d.flows.append(
                    (
                        (*s, _int(a["source_bundle"]), _int(a["source_channel"])),
                        (*t, _int(a["dest_bundle"]), _int(a["dest_channel"])),
                    )
                )
            elif name == "aie.packet_flow":
                srcs, dsts = [], []
                for x in _ops(op.regions[0].blocks[0]):
                    xa = _attrs(x)
                    if x.name in ("aie.packet_source", "aie.packet_dest"):
                        ep = (
                            *tile_of(x.operands[0]),
                            _int(xa["bundle"]),
                            _int(xa["channel"]),
                        )
                        (srcs if x.name == "aie.packet_source" else dsts).append(ep)
                d.add_packet_flow(
                    _int(a["ID"]),
                    srcs,
                    dsts,
                    keep=_int(a["keep_pkt_header"]) if "keep_pkt_header" in a else None,
                    priority=(
                        _int(a["priority_route"]) if "priority_route" in a else None
                    ),
                    mask=_int(a["mask"]) if "mask" in a else None,
                )
            elif name == "aie.shim_dma_allocation":
                d.allocs[str(a["sym_name"]).strip('"')] = dict(
                    tile=tile_of(op.operands[0]),
                    dir=_int(a["channel_dir"]),
                    ch=_int(a["channel_index"]),
                    pkt=_pkt(a["packet"]) if "packet" in a else None,
                )
            elif name == "aie.runtime_sequence":
                sequence(op)
            elif name in ("aie.buffer", "aie.wire", "aie.end", "aie.bd_chain"):
                pass
            elif name == "arith.constant":
                lock_ops(op, [])
            else:
                d.unsupported.append(name)
        for p in d.programs:
            if p["kind"] == "start" and not p["loops"]:
                pass
        return d


def design_signature(d):
    """What the model reads, for comparing two Designs: a snapshot, as the
    model prunes Designs in place."""
    return repr(
        (
            d.dev,
            sorted(d.tiles),
            [(k, v[:2], v[3]) for k, v in d.locks.items()],
            [
                (
                    p["tile"],
                    p["kind"],
                    p["dir"],
                    p["ch"],
                    p["seq"],
                    p["loops"],
                    p["repeat"],
                    p["dyn_repeat"],
                    p["users"],
                )
                for p in d.programs
            ],
            sorted(d.cores.items()),
            sorted(
                (
                    t,
                    [
                        o if o[0] != "masterset" else o[:2] + (len(o[2]),) + o[3:]
                        for o in ops
                    ],
                )
                for t, ops in d.boxes.items()
            ),
            sorted(d.muxes.items()),
            d.flows,
            d.packet_flows,
            sorted(d.allocs.items(), key=lambda kv: kv[0]),
            d.sequences,
        )
    )


# The deadlock rules, mirrored from AIEStreamDependencyAnalysis.cpp. Names
# follow the C++ so the two read side by side.


class Stream:
    __slots__ = ("src", "dst", "pid", "keep", "hops")

    def __init__(self, src, dst, pid=None, keep=False, hops=()):
        self.src, self.dst, self.pid, self.keep = src, dst, pid, keep
        self.hops = list(hops)  # [(tile, input port, arbiter or None)]

    def key(self):
        return (self.src, self.dst, self.pid)

    def __repr__(self):
        return describe_stream(self)


def describe_stream(s):
    text = ("packet flow " if s.pid is not None else "flow ") + (
        f"{fmt_ep(s.src)} -> {fmt_ep(s.dst)}"
    )
    return text + (f" (id {s.pid})" if s.pid is not None else "")


def program_ops(p):
    return [op for block in p["seq"] for op in block]


def requested_streams(d):
    streams = [Stream(s, t) for s, t in d.flows]
    for f in d.packet_flows:
        for s in f["srcs"]:
            for t in f["dsts"]:
                streams.append(Stream(s, t, f["id"], bool(f["keep"])))
    return streams


def sent_packet_ids(d):
    ids = defaultdict(set)
    for p in d.programs:
        if p["dir"] != MM2S:
            continue
        for op in program_ops(p):
            if op[0] == "bd" and op[2] is not None:
                ids[d.program_key(p)].add(op[2])
    for a in d.allocs.values():
        if a["pkt"] is not None:
            ids[(*a["tile"], a["dir"], a["ch"])].add(a["pkt"])
    for events in d.sequences:
        for ev in events:
            if ev[0] == "memcpy" and ev[2] is not None and ev[1] in d.allocs:
                a = d.allocs[ev[1]]
                ids[(*a["tile"], a["dir"], a["ch"])].add(ev[2])
    return {k: sorted(v) for k, v in ids.items()}


def trace_routed_streams(d):
    """StreamTracer: every stream the switchboxes and shim muxes carry."""
    boxes = {t: d.boxes[t] for t in sorted(d.boxes)}
    muxes = {t: [("connect", s, m) for s, m in d.muxes[t]] for t in sorted(d.muxes)}
    sent = sent_packet_ids(d)
    streams = []

    def input_ports(ops):
        ports = []
        for op in ops:
            p = op[1] if op[0] in ("connect", "rules") else None
            if p is not None and p not in ports:
                ports.append(p)
        return ports

    def follow(here, is_mux, out):
        if is_mux:
            if out[0] == NORTH and here in boxes:
                return ("hop", here, False, (SOUTH, out[1]))
            return ("end", (*here, *out))
        if out[0] not in DIRECTIONAL:
            return ("end", (*here, *out))
        if out[0] == SOUTH and here in muxes:
            return ("hop", here, True, (NORTH, out[1]))
        dc, dr, into = STEP[out[0]]
        nxt = (here[0] + dc, here[1] + dr)
        if nxt in boxes:
            return ("hop", nxt, False, (into, out[1]))
        return ("end", (*here, *out))

    def step(src, tile, is_mux, inp, pid, visited, path):
        key = (tile, is_mux, inp)
        if key in visited:
            return
        visited = visited | {key}
        ops = muxes[tile] if is_mux else boxes[tile]

        def nxt(out, next_id, arbiter, keep=False):
            next_path = path if is_mux else path + [(tile, inp, arbiter)]
            to = follow(tile, is_mux, out)
            if to[0] == "end":
                streams.append(Stream(src, to[1], next_id, keep, next_path))
                return
            step(src, to[1], to[2], to[3], next_id, visited, next_path)

        for op in ops:
            if op[0] == "connect" and op[1] == inp:
                nxt(op[2], pid, None)
        amsels = {op[1]: op[2] for op in ops if op[0] == "amsel"}
        for op in ops:
            if op[0] != "rules" or op[1] != inp:
                continue

            def route(rule, rule_id):
                arbiter = amsels.get(rule[2])
                for ms in ops:
                    if ms[0] == "masterset" and rule[2] in ms[2]:
                        nxt(ms[1], rule_id, arbiter, bool(ms[3]))

            for rule in op[2]:
                mask, value = rule[0], rule[1]
                if pid is None:
                    route(rule, value)
                    continue
                if (pid & mask) == (value & mask):
                    route(rule, pid)
                    break

    def trace_from(src, tile, is_mux, inp):
        ids = None
        if src[2] == DMA:
            ids = sent.get((src[0], src[1], MM2S, src[3]))
        if ids is None:
            step(src, tile, is_mux, inp, None, frozenset(), [])
            return
        for pid in ids:
            step(src, tile, is_mux, inp, pid, frozenset(), [])

    for tile, ops in muxes.items():
        for p in input_ports(ops):
            if p[0] != NORTH:
                trace_from((*tile, *p), tile, True, p)
    for tile, ops in boxes.items():
        for p in input_ports(ops):
            if p[0] not in DIRECTIONAL:
                trace_from((*tile, *p), tile, False, p)
    return streams


class Volumes:
    """StreamVolumeAnalysis."""

    def __init__(self, d):
        self.d = d
        self.programs = defaultdict(list)
        for p in d.programs:
            self.programs[d.program_key(p)].append(p)
        self.memcpys = []
        for events in d.sequences:
            for ev in events:
                if ev[0] == "memcpy":
                    self.memcpys.append(ev)

    def send_volume(self, s):
        if s.src[2] != DMA:
            return None
        key = (s.src[0], s.src[1], MM2S, s.src[3])

        def carries(pkt):
            return pkt is None or s.pid is None or pkt == s.pid

        def header(pkt):
            return PARAMS["header_bytes"] if pkt is not None and s.keep else 0

        known, total = False, 0
        for p in self.programs.get(key, []):
            if p["loops"]:
                return None
            nbytes = sum(
                op[1] + header(op[2])
                for op in program_ops(p)
                if op[0] == "bd" and carries(op[2])
            )
            if p["kind"] == "start":
                runs = p["repeat"] + 1
            else:
                if p["dyn_repeat"]:
                    return None
                runs = 0
                for in_loop in p["users"]:
                    if in_loop:
                        return None
                    runs += p["repeat"] + 1
            known = True
            total += nbytes * runs
        for _, sym, pkt, nbytes, in_loop in self.memcpys:
            a = self.d.allocs.get(sym)
            if a is None or (*a["tile"], a["dir"], a["ch"]) != key:
                continue
            if pkt is None:
                pkt = a["pkt"]
            if not carries(pkt):
                continue
            if in_loop or nbytes is None:
                return None
            known = True
            total += nbytes + header(pkt)
        return total if known else None

    def receive_capacity(self, ep):
        if ep[2] != DMA:
            return 0
        progs = self.programs.get((ep[0], ep[1], S2MM, ep[3]))
        if not progs:
            return 0
        p = progs[0]
        if p["kind"] != "start":
            return 0
        passes = 1 + p["repeat"]
        seq = p["seq"]
        if not seq:
            return 0
        values = {}
        nbytes = 0
        for step in range(PARAMS["max_bd_steps"]):
            i = step % len(seq)
            if not p["loops"] and step >= passes * len(seq):
                return nbytes
            for op in seq[i]:
                if op[0] == "lock":
                    _, action, lock, n = op
                    if lock is None or n is None:
                        return 0
                    v = values.setdefault(lock, self.d.locks[lock][3])
                    if action == 1:
                        values[lock] = v + n
                    elif v < n:
                        return nbytes
                    else:
                        values[lock] = v - n
                else:
                    nbytes += op[1]
        return None


class WaitGraph:
    """StreamWaitGraph. Agents are (col, row, is_core, dir, channel)."""

    LOCK, STREAM, HOST = range(3)

    def __init__(self, d, streams):
        self.agents, self.edges, self.ids, self.modeled = [], [], {}, set()
        acquirers, releasers = {}, {}

        def note(use, agent):
            _, action, lock, _ = use
            if lock is None:
                return
            m = releasers if action == 1 else acquirers
            s = m.setdefault(lock, [])
            if agent not in s:
                s.append(agent)

        for tile, uses in d.cores.items():
            agent = self.get_or_create(tile, True, MM2S, 0)
            self.modeled.add(agent)
            for u in uses:
                note(("lock", *u), agent)
        for p in d.programs:
            agent = self.get_or_create(p["tile"], False, p["dir"], p["ch"])
            self.modeled.add(agent)
            for op in program_ops(p):
                if op[0] == "lock":
                    note(op, agent)
        for lock in d.locks:
            if lock not in acquirers or lock not in releasers:
                continue
            for p in acquirers[lock]:
                for q in releasers[lock]:
                    if p != q:
                        self.add_edge(p, q, self.LOCK)

        def endpoint_agent(ep, sending):
            if ep[2] == CORE:
                return self.get_or_create(ep[:2], True, MM2S, 0)
            if ep[2] == DMA:
                return self.get_or_create(
                    ep[:2], False, MM2S if sending else S2MM, ep[3]
                )
            return None

        for s in streams:
            a = endpoint_agent(s.src, True)
            b = endpoint_agent(s.dst, False)
            if a is None or b is None or a == b:
                continue
            self.add_edge(a, b, self.STREAM)
            self.add_edge(b, a, self.STREAM)

        def by_symbol(sym):
            a = d.allocs.get(sym)
            return None if a is None else (*a["tile"], a["dir"], a["ch"])

        for events in d.sequences:
            waited, issued = [], set()

            def agent_of(key):
                if key is None or key[2] is None:
                    return None
                return self.get_or_create(key[:2], False, key[2], key[3])

            def chain_key(ev):
                return by_symbol(ev[2]) if ev[2] else ev[1]

            for ev in events:
                if ev[0] in ("memcpy", "start", "chain"):
                    if ev[0] == "memcpy":
                        key = by_symbol(ev[1])
                    elif ev[0] == "start":
                        key = d.program_key(d.programs[ev[1]])
                    else:
                        key = chain_key(ev)
                    agent = agent_of(key)
                    if agent is None or agent in issued:
                        continue
                    issued.add(agent)
                    self.modeled.add(agent)
                    for w in waited:
                        if w != agent:
                            self.add_edge(agent, w, self.HOST)
                elif ev[0] in ("wait", "await", "await_chain"):
                    if ev[0] == "wait":
                        key = by_symbol(ev[1])
                    elif ev[0] == "await":
                        key = d.program_key(d.programs[ev[1]])
                    else:
                        key = chain_key(events[ev[1]])
                    agent = agent_of(key)
                    if agent is not None and agent not in waited:
                        waited.append(agent)
        for a in range(len(self.agents)):
            if a in self.modeled:
                continue
            for b in range(len(self.agents)):
                if b != a and self.agents[b][:2] == self.agents[a][:2]:
                    self.add_edge(a, b, self.LOCK)

    def _key(self, tile, is_core, dr, ch):
        return (tile[0], tile[1], is_core, 0 if is_core else dr, 0 if is_core else ch)

    def get_or_create(self, tile, is_core, dr, ch):
        k = self._key(tile, is_core, dr, ch)
        if k not in self.ids:
            self.ids[k] = len(self.agents)
            self.agents.append(k)
            self.edges.append([])
        return self.ids[k]

    def add_edge(self, a, b, kind):
        if (b, kind) not in self.edges[a]:
            self.edges[a].append((b, kind))

    def agent_at(self, ep, sending):
        if ep[2] == CORE:
            return self.ids.get(self._key(ep[:2], True, MM2S, 0))
        if ep[2] == DMA:
            return self.ids.get(
                self._key(ep[:2], False, MM2S if sending else S2MM, ep[3])
            )
        return None

    def drainers_of(self, a):
        if self.agents[a][2]:
            return [a]
        return [b for b, kind in self.edges[a] if kind != self.STREAM]

    def wait_chain(self, frm, targets, avoid):
        parent = {a: None for a in avoid}
        work = deque()
        for a in frm:
            if a not in parent:
                parent[a] = None
                work.append(a)
        while work:
            a = work.popleft()
            if a in targets:
                chain = []
                at = a
                while at is not None:
                    chain.append(at)
                    at = parent[at]
                return chain[::-1]
            for b, _ in self.edges[a]:
                if b not in parent:
                    parent[b] = a
                    work.append(b)
        return []

    def describe(self, a):
        c, r, is_core, dr, ch = self.agents[a]
        return f"({c}, {r}) core" if is_core else f"({c}, {r}) {DIRS[dr]} {ch}"


class Analysis:
    """StreamConflicts: requested streams first, then those already routed."""

    def __init__(self, d):
        self.d = d
        self.streams = requested_streams(d)
        self.num_requested = len(self.streams)
        self.streams += trace_routed_streams(d)
        self._graph = None
        self._stalls, self._blocks, self._conflicts = {}, {}, {}

    @property
    def graph(self):
        if self._graph is None:
            self.volumes = Volumes(self.d)
            self._graph = WaitGraph(self.d, self.streams)
        return self._graph

    def can_stall(self, f):
        if f in self._stalls:
            return self._stalls[f]
        dst = self.streams[f].dst
        cap = self.volumes.receive_capacity(dst)
        result = False
        if cap is not None:
            sent = 0
            for s in self.streams:
                if s.dst != dst:
                    continue
                v = self.volumes.send_volume(s)
                if v is None:
                    result = True
                    break
                sent += v
            else:
                result = sent > cap
        self._stalls[f] = result
        return result

    def blocking_chain(self, f, g):
        g_ = self.graph
        fs, gs = self.streams[f], self.streams[g]
        f_dst = g_.agent_at(fs.dst, False)
        if f_dst is None:
            return []
        own = []
        f_src = g_.agent_at(fs.src, True)
        if f_src is not None:
            own.append(f_src)
        own.append(f_dst)
        drainers = g_.drainers_of(f_dst)
        avoid = [a for a in own if a not in drainers]
        targets = [
            a
            for a in (g_.agent_at(gs.src, True), g_.agent_at(gs.dst, False))
            if a is not None and a not in own
        ]
        if not targets:
            return []
        return g_.wait_chain(drainers, targets, avoid)

    def can_block(self, f, g):
        if (f, g) not in self._blocks:
            self.graph
            self._blocks[(f, g)] = self.can_stall(f) and bool(self.blocking_chain(f, g))
        return self._blocks[(f, g)]

    def assumptions(self, f, g):
        fs = self.streams[f]
        out = []
        for other in self.streams:
            if other.dst == fs.dst and self.volumes.send_volume(other) is None:
                out.append(
                    f"The volume {describe_stream(other)} carries is unknown, so it "
                    "is assumed to overrun its receiver."
                )
                break
        chain = self.blocking_chain(f, g)
        waiters = [self.graph.agent_at(fs.dst, False)] + chain[:-1]
        for i, a in enumerate(waiters):
            if a not in self.graph.modeled and a not in waiters[:i]:
                out.append(
                    f"Nothing in the design programs {self.graph.describe(a)}, so it "
                    "is assumed to wait on anything on its tile."
                )
        return out

    def explain_block(self, f, g):
        fs, gs = self.streams[f], self.streams[g]
        chain = self.blocking_chain(f, g)
        s = (
            describe_stream(fs)
            + " can fill its receiver, and draining that waits on "
            + ", then ".join(self.graph.describe(a) for a in chain)
        )
        s += (
            ", which receives "
            if self.graph.agent_at(gs.dst, False) == chain[-1]
            else ", which sends "
        )
        s += describe_stream(gs) + "."
        for a in self.assumptions(f, g):
            s += " " + a
        return s

    def blocks(self, s, t):
        a, b = self.streams[s], self.streams[t]
        if a.src == b.src or a.dst == b.dst:
            return False
        return self.can_block(s, t)

    def conflict(self, s, t):
        if (s, t) not in self._conflicts:
            self._conflicts[(s, t)] = self.blocks(s, t) or self.blocks(t, s)
        return self._conflicts[(s, t)]

    def explain(self, s, t):
        self.graph
        return (
            self.explain_block(s, t)
            if self.can_block(s, t)
            else self.explain_block(t, s)
        )

    def hold_cycle(self, routes):
        """StreamConflicts::holdCycle. routes[i] is [(tile, input, arbiter)]
        for streams[i]. Steps are (wait, waiting, sharer, holding, tile,
        sharer input, holder input, arbiter)."""
        streams, nreq = self.streams, self.num_requested
        trees, tree_ids = [], {}
        for i, s in enumerate(streams):
            if s.pid is None:
                continue
            k = (s.src[:2], s.src[2:], s.pid, i < nreq)
            if k not in tree_ids:
                tree_ids[k] = len(trees)
                trees.append(dict(members=[], hops=[], parent=[], arbiter=[], base=0))
            tree = trees[tree_ids[k]]
            tree["members"].append(i)
            prev = -1
            for tile, inp, arb in routes[i]:
                key = (tile, inp)
                if key in tree["hops"]:
                    h = tree["hops"].index(key)
                else:
                    h = len(tree["hops"])
                    tree["hops"].append(key)
                    tree["parent"].append(prev)
                    tree["arbiter"].append(arb)
                prev = h

        def related(a, b):
            x, y = trees[a], trees[b]
            if streams[x["members"][0]].src == streams[y["members"][0]].src:
                return True
            return any(
                streams[m].dst == streams[n].dst
                for m in x["members"]
                for n in y["members"]
            )

        nodes = []
        for t, tree in enumerate(trees):
            tree["base"] = len(nodes)
            nodes += [(t, "hop", h) for h in range(len(tree["hops"]))]
            nodes += [(t, "recv", m) for m in range(len(tree["members"]))]
            nodes.append((t, "any", 0))
            nodes += [(t, "past", h) for h in range(len(tree["hops"]))]

        def receiver_node(t, m):
            return trees[t]["base"] + len(trees[t]["hops"]) + m

        def anywhere_node(t):
            return receiver_node(t, len(trees[t]["members"]))

        def past_node(t, h):
            return anywhere_node(t) + 1 + h

        entering, passing = defaultdict(list), defaultdict(list)
        for t, tree in enumerate(trees):
            for h, hop in enumerate(tree["hops"]):
                entering[hop].append((t, h))
                passing[hop[0]].append((t, h))

        cache = {}

        def successors(n):
            if n in cache:
                return cache[n]
            out = []
            t0, kind, index = nodes[n]
            u = trees[t0]
            if kind == "hop":
                tile, inp = u["hops"][index]
                waiting = u["members"][0]
                for t, ht in entering[(tile, inp)]:
                    if t != t0 and related(t0, t):
                        continue
                    sharer = trees[t]["members"][0]
                    if t != t0:
                        out.append(
                            (
                                past_node(t, ht),
                                ("link", waiting, sharer, sharer, tile, inp, inp, -1),
                            )
                        )
                    arbiter = trees[t]["arbiter"][ht]
                    if arbiter is None:
                        continue
                    for v, hv in passing[tile]:
                        holder_input = trees[v]["hops"][hv][1]
                        if (
                            v == t0
                            or v == t
                            or holder_input == inp
                            or trees[v]["arbiter"][hv] != arbiter
                            or related(t, v)
                        ):
                            continue
                        out.append(
                            (
                                past_node(v, hv),
                                (
                                    "arbiter",
                                    waiting,
                                    sharer,
                                    trees[v]["members"][0],
                                    tile,
                                    inp,
                                    holder_input,
                                    arbiter,
                                ),
                            )
                        )
            elif kind == "recv":
                s = u["members"][index]
                for g in range(len(trees)):
                    if g == t0:
                        continue
                    for m in trees[g]["members"]:
                        if self.blocks(s, m):
                            out.append(
                                (
                                    anywhere_node(g),
                                    ("drain", s, s, m, None, None, None, -1),
                                )
                            )
                            break
            else:
                behind = set()
                if kind == "past":
                    h = index
                    while h >= 0:
                        behind.add(h)
                        h = u["parent"][h]
                for h in range(len(u["hops"])):
                    if h not in behind:
                        out.append((u["base"] + h, None))
                for m in range(len(u["members"])):
                    out.append((receiver_node(t0, m), None))
            cache[n] = out
            return out

        def counts(e):
            st = e[1]
            if st is None or st[0] == "drain":
                return False
            return st[1] < nreq or st[2] < nreq or st[3] < nreq

        roots = [
            n
            for n in range(len(nodes))
            if nodes[n][1] == "hop" and any(counts(e) for e in successors(n))
        ]
        index = [-1] * len(nodes)
        low = [0] * len(nodes)
        comp = [-1] * len(nodes)
        on_stack = [False] * len(nodes)
        stack, frames = [], []
        counter = [0]
        ncomp = 0

        def visit(n):
            index[n] = low[n] = counter[0]
            counter[0] += 1
            stack.append(n)
            on_stack[n] = True
            frames.append([n, 0])

        for root in roots:
            if index[root] >= 0:
                continue
            visit(root)
            while frames:
                n, nx = frames[-1]
                out = successors(n)
                if nx < len(out):
                    frames[-1][1] += 1
                    w = out[nx][0]
                    if index[w] < 0:
                        visit(w)
                    elif on_stack[w]:
                        low[n] = min(low[n], index[w])
                    continue
                frames.pop()
                if frames:
                    low[frames[-1][0]] = min(low[frames[-1][0]], low[n])
                if low[n] != index[n]:
                    continue
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp[w] = ncomp
                    if w == n:
                        break
                ncomp += 1

        for x in roots:
            for e in successors(x):
                if not counts(e) or comp[e[0]] != comp[x]:
                    continue
                via = {e[0]: (e[0], None)}
                work = deque([e[0]])
                while x not in via:
                    n = work.popleft()
                    for nxt in successors(n):
                        if comp[nxt[0]] == comp[x] and nxt[0] not in via:
                            via[nxt[0]] = (n, nxt)
                            work.append(nxt[0])
                path = []
                n = x
                while n != e[0]:
                    path.append(via[n][1])
                    n = via[n][0]
                path.append(e)
                return [edge[1] for edge in reversed(path) if edge[1] is not None]
        return None

    def explain_cycle(self, steps):
        out = []
        for wait, waiting, sharer, holding, tile, sin, hin, arb in steps:
            if wait == "link":
                out.append(
                    f"{describe_stream(self.streams[waiting])} can queue behind "
                    f"{describe_stream(self.streams[holding])} on {fmt_port(sin)} into "
                    f"tile ({tile[0]}, {tile[1]})."
                )
            elif wait == "arbiter":
                s = (
                    f"{describe_stream(self.streams[holding])} can hold arbiter {arb} at "
                    f"tile ({tile[0]}, {tile[1]}) that "
                    f"{describe_stream(self.streams[sharer])} needs"
                )
                if waiting != sharer:
                    s += f", and {describe_stream(self.streams[waiting])} can queue behind it"
                out.append(s + ".")
            else:
                self.graph
                out.append(self.explain_block(waiting, holding))
        return " ".join(out)


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
    """Per tile, arbiter + 6 * msel of every amsel an existing masterset uses."""
    out = defaultdict(set)
    for tile, ops in d.boxes.items():
        amsels = {
            op[1]: op[2] + op[3] * PARAMS["arbiters"] for op in ops if op[0] == "amsel"
        }
        for op in ops:
            if op[0] == "masterset":
                out[tile] |= {amsels[n] for n in op[2] if n in amsels}
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
    pinned = defaultdict(list)
    for i, s in enumerate(an.streams[: an.num_requested]):
        if s.pid is None:
            continue
        pinned[s.dst[:2]].append(i)
        if s.src[:2] == s.dst[:2]:
            continue
        if pins_hops(s.src[:2]):
            pinned[s.src[:2]].append(i)
        for t in cut_tiles(target, s.src[:2], s.dst[:2]):
            if pins_hops(t):
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
                nxt = [t for t in cs[k + 1 :] if an.conflict(s, t)]
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
    out = dict(reason=None, blame=[], plans={}, circuit_hops={}, routes=routes)

    def fail(reason, blame):
        out["reason"], out["blame"] = reason, sorted(set(blame))
        return out

    for f in d.packet_flows:
        if not f["srcs"]:
            return fail("packet_flow has no packet_source", [])
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
                        ctrl_flows[((tile, t), pid)] = bool(f["priority"])
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

    def search():
        arbitrate()
        cycle = an.hold_cycle(routes)
        if cycle is None:
            return True
        if not first:
            first.append(cycle)
        for wait, waiting, sharer, holding, tile, sin, hin, arb in cycle:
            if wait != "arbiter":
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
                if search():
                    return True
                plans[tile] = saved
            if pair:
                apart.discard(pair)
            if o:
                off.discard(o)
        return False

    if not search():
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
    """Remove one requested stream, or the least more that removing it takes."""
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


def pins_hops_fn(d, hops_on):
    return lambda tile: not hops_on or d.target.kind(tile) == "shim"


def construct(d, seed, attempts=40):
    """Route `d` on exclusive links and plan its arbiters with the router's
    rules, dropping streams until both hop modes accept the witness. Returns
    (construction, analysis, plan) or None; `d` is pruned in place."""
    for attempt in range(attempts):
        con = Construction(d, random.Random(f"route-{seed}-{attempt}"))
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
            routed = con.route(src, wanted, packet)
            missing += [(src, t, packet) for t in wanted if t not in routed]
        if missing:
            for src, t, packet in missing:
                for st in requested_streams(d):
                    if st.src == src and st.dst == t and (st.pid is not None) == packet:
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


# What any routing the router emits has to satisfy.


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
    if t.kind(src[:2]) == "shim":
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
            if len({view[1].get(h[2]) for h in hits}) > 1:
                problems.append(
                    f"{what} matches {len(hits)} rules at {tile} {fmt_port(slave)}"
                )
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
                if len(outs) != 1 or outs[0][0] != DMA:
                    problems.append(f"{what} leaves {tile} South:{m[1]} to {outs}")
                    continue
                ep = (*tile, DMA, outs[0][1])
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


def keeps_header(tile, port, keep):
    """AIERT: packets leave a master with their header unless it is a DMA or
    the shim's South; keep_pkt_header overrides."""
    if keep is not None:
        return keep
    return port[0] != DMA and not (tile[1] == 0 and port[0] == SOUTH)


def last_keep(d):
    """keep_pkt_header per destination port: the last packet flow into it
    decides, as keepPktHeaderAttr is written per destination."""
    keep = {}
    for f in d.packet_flows:
        for t in f["dsts"]:
            keep[t] = f["keep"]
    return keep


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
            if new is None or sorted(before[1][n] for n in op[2]) != sorted(
                after[1].get(n) for n in new[2]
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

    keeps = last_keep(d)
    mixed_ctrl, low_priority = set(), set()
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
                    if not f["priority"]:
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
    )
    return problems, stats


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
    lock-bounded receivers as (endpoint, program, lock it acquires)."""
    t = d.target
    srcs, dsts = flow_endpoints(d)
    ids_of = ids_by_source(d)
    bounded = []
    for tile in sorted(tiles):
        if t.kind(tile) == "shim":
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


def add_host(rng, d):
    """A runtime sequence driving the shim DMA endpoints: memcpys (some with
    packet ids, some in loops), configured tasks, bd chains, and waits after
    them in varied orders."""
    t = d.target
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
                    )
                )
        waits.append(("wait", sym))
    for ep in sorted(dsts):
        if t.kind(ep[:2]) != "shim":
            continue
        sym = f"out{ep[0]}_{ep[3]}"
        d.allocs[sym] = dict(tile=ep[:2], dir=S2MM, ch=ep[3], pkt=None)
        events.append(("memcpy", sym, None, rng.choice([64, 128, 256]), False))
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
    send, that without packet headers kept, or less. Returns the modes."""
    an = Analysis(d)
    an.graph
    modes = []
    for ep, p, prod in bounded:
        streams = [s for s in an.streams if s.dst == ep]
        v = [an.volumes.send_volume(s) for s in streams]
        if not streams or None in v:
            continue
        v0 = sum(an.volumes.send_volume(Stream(s.src, s.dst, s.pid)) for s in streams)
        if v0 < 8:
            continue
        mode = rng.choice(["exact", "header", "overrun"])
        if mode == "header" and sum(v) == v0:
            mode = "exact"
        cap = {"exact": sum(v), "header": v0, "overrun": v0 // 2 // 4 * 4}[mode]
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
    d.sized = size_receivers(rng, d, bounded)
    return d


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


def memtile_row(t):
    return next(r for r in range(t.rows) if t.kind((0, r)) == "mem")


def gen_unroutable_arbiters(rng, dev, programmed):
    """More pairwise conflicting packet streams take an arbiter at memtile T
    than it has: streams into T's S2MM channels, and with circuit switched
    hops off, streams out of its MM2S channels. Unprogrammed, every pair
    conflicts; programmed, the model decides."""
    t = Target(dev)
    row = memtile_row(t)
    tc, yc = rng.sample(range(t.cols), 2)
    d = Design(dev)
    ins = rng.randint(1, 6)
    outs = rng.randint(max(1, 7 - ins), 6)
    others = [
        ep
        for tile in sorted(t.kinds)
        if tile not in ((tc, row), (yc, row))
        for ep in t.endpoints(tile, True)
    ]
    for ch, src in zip(rng.sample(range(6), ins), rng.sample(others, ins)):
        d.add_packet_flow(0, [src], [(tc, row, DMA, ch)])
    for a, b in zip(rng.sample(range(6), outs), rng.sample(range(6), outs)):
        d.add_packet_flow(
            0,
            [(tc, row, DMA, a)],
            [(yc, row, DMA, b)],
            priority=True if rng.random() < 0.1 else None,
        )
    assign_ids(rng, d, masks=False)
    if programmed:
        for f in d.packet_flows:
            f["keep"] = rng.choice([None, True, False])
        srcs, dsts = flow_endpoints(d)
        bounded = add_programs(rng, d, {ep[:2] for ep in srcs | dsts})
        add_host(rng, d)
        size_receivers(rng, d, bounded)
    return canonical(d)


def gen_unroutable_ports(rng, dev):
    """More circuit flows from above a one-column memtile into its S2MM
    channels than its North inputs carry."""
    t = Target(dev)
    row = memtile_row(t)
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
    row = memtile_row(t)
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
        prod, cons = d.lock(x, 1), d.lock(x, 0)
        add_program(d, x, S2MM, ch, [bd_block(64, None, prod, cons)], True)
        locks.append((prod, cons))
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
        row = memtile_row(t)
        d = Design(dev)
        k = rng.randint(5, 6)
        above = [ep for r in range(row + 1, t.rows) for ep in t.endpoints((0, r), True)]
        for ch, src in zip(rng.sample(range(6), k), rng.sample(above, k)):
            d.add_packet_flow(0, [src], [(0, row, DMA, ch)])
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


# The route space: what a design asks of the router and what the router did,
# as values along a few dimensions, next to every value the TargetModel
# allows. route_space() places one design; space_domain() is the denominator.

DEBUG_ONLY = "-debug-only=aie-create-pathfinder-flows,aie-pathfinder"
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
    ("no path", r"no path leads from"),
    ("source unrouted", r"could not be routed to destination"),
    ("id exceeds", r"exceeds the maximum of"),
    ("false match", r"can lead to false packet id match"),
    ("claim rule", r"claim rule \(mask"),
    ("rule slots", r"slave port packet rules exceed"),
    ("fixed connections", r"Unable to add fixed connections"),
    ("no packet_source", r"packet_flow has no packet_source"),
    ("not in device", r"must be contained within a device"),
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
    *(f"rejected:{k}" for k, _ in OUTCOMES[2:9]),
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


# Running the router.


def aie_opt_path():
    near = (p / "build" / "bin" / "aie-opt" for p in Path(__file__).resolve().parents)
    here = next((p for p in near if p.exists()), "aie-opt")
    return os.environ.get("AIE_OPT") or shutil.which("aie-opt") or str(here)


def aie_opt(text, hops_on=True, split=False, extra=(), timeout=None):
    """Run the router. A timeout comes back as returncode -1000."""
    opt = "--aie-create-pathfinder-flows"
    if not hops_on:
        opt += "=circuit-switch-hops=false"
    cmd = (
        [aie_opt_path(), opt, *extra]
        + (["--split-input-file"] if split else [])
        + ["-"]
    )
    for _ in range(2):
        try:
            p = subprocess.run(
                cmd, input=text, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                cmd, -1000, "", f"timeout after {timeout}s"
            )
        # The binary may be relinked under us; a crash is worth one retry.
        if p.returncode >= 0:
            break
    return p


MODULE_TAG_RE = re.compile(r"^module @s(\d+)\b", re.M)


def route_batch(tagged, hops_on, extra=()):
    """Route many tagged designs in one run. A failure drops that design's
    output, so designs missing from it are rerun alone. Returns ({tag:
    output}, {tag: (returncode, stderr)}, {tag: debug output})."""
    p = aie_opt(
        "\n// -----\n".join(m for _, m in tagged), hops_on, split=True, extra=extra
    )
    out, errs, debug = {}, {}, {}
    for chunk in p.stdout.split("\n// -----\n"):
        if m := MODULE_TAG_RE.search(chunk):
            out[int(m.group(1))] = chunk
    logs = p.stderr.split(PASS_BEGIN)[1:]
    if len(logs) == len(tagged):
        debug = {tag: log for (tag, _), log in zip(tagged, logs)}
    for tag, text in tagged:
        if tag not in out:
            q = aie_opt(text, hops_on, extra=extra)
            debug[tag] = q.stderr
            if q.returncode == 0:
                out[tag] = q.stdout
            else:
                errs[tag] = (q.returncode, q.stderr)
    return out, errs, debug


PASS_BEGIN = "---Begin AIEPathfinderPass---"
_debug_ok = []


def debug_flags():
    """DEBUG_ONLY when aie-opt was built with assertions, else nothing."""
    if not _debug_ok:
        p = aie_opt("module {}", extra=(DEBUG_ONLY,))
        _debug_ok.append(p.returncode == 0)
    return (DEBUG_ONLY,) if _debug_ok[0] else ()


def first_error(stderr):
    return next(
        (l.strip() for l in stderr.splitlines() if "error" in l), stderr.strip()[:300]
    )


# Self checks.


def check_model():
    """Answers the model owes on designs worked by hand."""
    out = []

    def expect(label, got, want):
        if got != want:
            out.append(f"{label}: got {got!r}, want {want!r}")

    # arbiter_deadlock_exhausted.mlir: nothing programs memtile (1,1), so the
    # six flows into it wait on each other, and flow 6 into (0,1) waits on all.
    d = Design("npu2")
    for ch in range(6):
        d.add_packet_flow(ch, [(0, 1, DMA, ch)], [(1, 1, DMA, ch)])
    d.add_packet_flow(6, [(0, 0, DMA, 0)], [(0, 1, DMA, 0)])
    an = Analysis(d)
    reason = unroutable_arbiters(d, an, pins_hops_fn(d, False))
    expect("unprogrammed tile", (reason or "")[:14], "at tile (0, 1)")
    expect("unprogrammed clique", len(getattr(an, "clique", [])), 7)
    expect(
        "unprogrammed hops on",
        unroutable_arbiters(d, Analysis(d), pins_hops_fn(d, True)),
        None,
    )
    expect("unprogrammed verdict", verdict(d, hops_on=False)["routable"], False)
    # Its second design: (1,1) drains each channel into a buffer nothing
    # frees, so none of the six waits on another; flow 6 still waits on all.
    for ch in range(6):
        prod, cons = d.lock((1, 1), 1), d.lock((1, 1), 0)
        add_program(d, (1, 1), S2MM, ch, [bd_block(1024, None, prod, cons)], True)
    an = Analysis(d)
    an.graph
    expect("buffered capacity", an.volumes.receive_capacity((1, 1, DMA, 0)), 1024)
    expect("buffered stalls", an.can_stall(0), True)
    expect(
        "buffered conflicts",
        [(i, j) for i in range(7) for j in range(i + 1, 7) if an.conflict(i, j)],
        [(i, 6) for i in range(6)],
    )
    # A receiver that never blocks cannot stall, whatever is sent to it.
    d.programs[0]["seq"] = [bd_block(64)]
    an = Analysis(d)
    an.graph
    expect("sink capacity", an.volumes.receive_capacity((1, 1, DMA, 0)), None)
    expect("sink stalls", an.can_stall(0), False)
    # Volume counts only the BDs carrying the stream's id, a finite send that
    # fits the receiver does not stall it, a kept header does, and repeats
    # multiply.
    d = Design("npu2")
    d.add_packet_flow(3, [(0, 2, DMA, 0)], [(0, 3, DMA, 0)])
    send = add_program(d, (0, 2), MM2S, 0, [bd_block(64, 3), bd_block(128, 4)], False)
    p, c = d.lock((0, 3), 2), d.lock((0, 3), 0)
    add_program(d, (0, 3), S2MM, 0, [bd_block(32, None, p, c)], True)
    an = Analysis(d)
    an.graph
    expect("volume by id", an.volumes.send_volume(an.streams[0]), 64)
    expect(
        "finite fits",
        (an.volumes.receive_capacity((0, 3, DMA, 0)), an.can_stall(0)),
        (64, False),
    )
    d.packet_flows[0]["keep"] = True
    an = Analysis(d)
    an.graph
    expect(
        "kept header",
        (an.volumes.send_volume(an.streams[0]), an.can_stall(0)),
        (68, True),
    )
    d.packet_flows[0]["keep"] = None
    send["repeat"] = 2
    an = Analysis(d)
    an.graph
    expect("repeats", an.volumes.send_volume(an.streams[0]), 192)
    # A chain through a core: f stalls at (0,3) S2MM 0, which the core
    # drains; the core feeds MM2S 1, which sends g. g does not stall at a sink.
    send.update(repeat=0, loops=True, seq=send["seq"][:-1])
    full, empty = d.lock((0, 3), 0), d.lock((0, 3), 1)
    d.cores[(0, 3)] = [(2, c, 1), (1, p, 1), (1, full, 1), (2, empty, 1)]
    add_program(d, (0, 3), MM2S, 1, [bd_block(32, None, full, empty)], True)
    d.flows.append(((0, 3, DMA, 1), (0, 4, DMA, 0)))
    add_program(d, (0, 4), S2MM, 0, [bd_block(32)], True)
    an = Analysis(d)
    g, f = 0, 1
    expect("relayed through core", (an.blocks(f, g), an.blocks(g, f)), (True, False))
    expect("relayed conflict", an.conflict(f, g), True)
    # arbiter_hold_cycle_wormhole.mlir: no pair conflicts, but with every hop
    # packet switched the arbiters close a cycle.
    d = gen_wormhole(random.Random(0), "npu1_1col")
    an = Analysis(d)
    expect(
        "wormhole pre-routing", unroutable_arbiters(d, an, pins_hops_fn(d, False)), None
    )
    con = Construction(d, random.Random(0))
    for fl in d.packet_flows:
        con.route(fl["srcs"][0], fl["dsts"], True)
    reason = plan_routing(d, an, con.solution(), hops_on=False)["reason"] or ""
    want = "packet flows can deadlock holding arbiters across switchboxes"
    expect("wormhole", reason[: len(want)], want)
    # What the model reads survives printing and parsing.
    for seed in range(8):
        d = canonical(
            random_design(random.Random(seed), ("npu1_2col", "npu2")[seed % 2])
        )
        again = canonical(d)
        expect(
            f"roundtrip {seed}", design_signature(again) == design_signature(d), True
        )
    return out


def check_verifier():
    """The verifier has to notice a routing that lost a hop: delete each
    connect and rule of a routed design in turn, and every deletion must be
    reported."""
    d = Design("npu1_2col")
    d.flows.append(((0, 0, DMA, 0), (1, 3, DMA, 0)))
    d.add_packet_flow(5, [(1, 2, DMA, 1)], [(0, 1, DMA, 2), (0, 4, CORE, 0)])
    d.add_packet_flow(6, [(0, 1, DMA, 3)], [(0, 0, DMA, 1)])
    an = Analysis(d)
    p = aie_opt(d.emit())
    if p.returncode:
        return [f"routing the verifier's design failed: {first_error(p.stderr)}"]
    out = []
    problems, _ = verify(d, an, p.stdout, True)
    out += [f"clean routing flagged: {x}" for x in problems]
    lines = p.stdout.splitlines()
    mutable = [
        i for i, l in enumerate(lines) if "aie.connect<" in l or "aie.rule(" in l
    ]
    for i in mutable:
        broken = "\n".join(lines[:i] + lines[i + 1 :])
        if not verify(d, an, broken, True)[0]:
            out.append(f"missed deletion of: {lines[i].strip()}")
    if len(mutable) < 5:
        out.append(f"only {len(mutable)} lines to delete")
    return out


# The test.


# Router bugs reported and not yet fixed: (label, test on the design and the
# verifier's problems). Matching cases are counted, not failed.
KNOWN_BUGS = [
    (
        "unroutable same-id sources on one tile",
        lambda d, problems: any(
            len({s[:2] for s in f["srcs"]}) < len(f["srcs"]) for f in d.packet_flows
        )
        and len(problems) == 1
        and problems[0].endswith("error: Unable to find a legal routing"),
    ),
]


def known_bug(d, problems):
    return next((label for label, test in KNOWN_BUGS if test(d, problems)), None)


def run_cases(cases, route):
    with ThreadPoolExecutor(max_workers=PARAMS["jobs"]) as pool:
        return list(pool.map(route, cases))


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--device", default="npu2", help="npu1, npu2 or a device name")
    cli.add_argument("--seeds", type=int, default=None, help="routable designs")
    cli.add_argument("--first-seed", type=int, default=0)
    cli.add_argument("--out", type=Path, default=None, help="save failing cases here")
    cli.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    cli.add_argument(
        "--space-out", type=Path, default=None, help="write route spaces (JSON)"
    )
    args = cli.parse_args(argv)
    for kv in args.param:
        k, v = kv.split("=", 1)
        PARAMS[k] = type(PARAMS[k])(v)
    if args.seeds is not None:
        PARAMS["routable"] = args.seeds
        PARAMS["unroutable"] = PARAMS["unknown"] = max(1, args.seeds // 4)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    first = args.first_seed

    def report(label, ok, detail):
        print(f"{label}: {detail} : {'OK' if ok else 'REGRESSION'}", flush=True)

    def save(case, what, text, stderr=""):
        if args.out:
            name = f"{case['tier']}-{case['shape']}-{case['seed']}-{what}"
            (args.out / f"{name}.mlir").write_text(text)
            if stderr:
                (args.out / f"{name}.err").write_text(stderr)

    problems = check_model()
    for line in problems:
        print("WRONG MODEL:", line)
    report("model-validation", not problems, f"{len(problems)} wrong answer(s)")
    problems = check_verifier()
    for line in problems:
        print("WRONG VERIFIER:", line)
    report("verifier-validation", not problems, f"{len(problems)} problem(s)")

    cases, gave_up = [], 0
    for seed in range(first, first + PARAMS["routable"]):
        case = routable_case(seed, args.device)
        if case is None:
            gave_up += 1
        else:
            cases.append(case)
    shapes = Counter(c["shape"] for c in cases)
    batches = [
        (hops_on, cases[k : k + PARAMS["batch"]])
        for hops_on in (True, False)
        for k in range(0, len(cases), PARAMS["batch"])
    ]

    def route(job):
        hops_on, group = job
        tagged = [(c["seed"], c["design"].emit(c["seed"])) for c in group]
        t0 = time.monotonic()
        outs, errs, _ = route_batch(tagged, hops_on)
        spent = time.monotonic() - t0
        return outs, errs, route_batch(tagged, hops_on, debug_flags()), spent

    failed, illegal, nondet = [], [], 0
    known = Counter()
    totals = defaultdict(int)
    route_time = 0.0
    spaces = []
    for (hops_on, group), (outs, errs, again, spent) in zip(
        batches, run_cases(batches, route)
    ):
        route_time += spent
        mode = "" if hops_on else " hops-off"
        for c in group:
            seed, d = c["seed"], c["design"]
            rc, stderr = errs.get(seed, (0, ""))
            spaces.append(
                route_space(
                    d, rc, outs.get(seed, ""), stderr, again[2].get(seed, ""), hops_on
                )
            )
            if seed not in outs:
                rc, stderr = errs[seed]
                if rc > 0 and (label := known_bug(d, [first_error(stderr)])):
                    known[label] += 1
                    save(c, "known" + mode.strip(), d.emit(), stderr)
                    continue
                failed.append(
                    f"seed {seed}{mode} ({c['shape']}): rc {rc}: {first_error(stderr)[:300]}"
                )
                save(c, "failed" + mode.strip(), d.emit(), stderr)
                continue
            if outs[seed] != again[0].get(seed):
                nondet += 1
            problems, stats = verify(d, c["analysis"], outs[seed], hops_on)
            if problems and (label := known_bug(d, problems)):
                known[label] += 1
                save(c, "known" + mode.strip(), d.emit())
            elif problems:
                illegal.append(f"seed {seed}{mode} ({c['shape']}): {problems[0]}")
                save(c, "illegal" + mode.strip(), d.emit())
            if hops_on:
                for k, v in stats.items():
                    totals[k] += v
                totals["constructed"] += c["truth"]["witness_hops"]
                totals["designs"] += 1
    runs = 2 * len(cases)
    for line in failed[:10]:
        print("FAILED:", line)
    for line in illegal[:10]:
        print("ILLEGAL:", line)
    report(
        "completeness",
        runs
        and (runs - len(failed)) / runs >= PARAMS["min_routed_rate"]
        and len(cases) >= PARAMS["routable"] * 0.9,
        f"{runs - len(failed)}/{runs} routed ({len(cases)} designs, "
        f"{shapes['fixed-ir']} with fixed switchboxes, both hop modes, "
        f"{gave_up} generator gave up)",
    )
    report(
        "legality",
        not illegal,
        f"{len(illegal)} illegal of {runs - len(failed)} routed",
    )
    report("determinism", nondet == 0, f"{nondet} unstable")

    ucases = [
        unroutable_case(s, args.device)
        for s in range(first, first + PARAMS["unroutable"])
    ]

    def route_one(c):
        p = aie_opt(c["design"].emit(), c["hops_on"], extra=debug_flags())
        spaces.append(
            route_space(
                c["design"], p.returncode, p.stdout, p.stderr, p.stderr, c["hops_on"]
            )
        )
        return p

    wrong, exact, undecided = [], 0, 0
    for c, p in zip(ucases, run_cases(ucases, route_one)):
        want = c["truth"]["expect"]
        tag = f"seed {c['seed']} {c['shape']} ({c['design'].dev})"
        if want is None:
            undecided += 1
            if "no two of" in p.stderr:
                wrong.append(
                    f"{tag}: model finds no overfull tile, router: {first_error(p.stderr)[:300]}"
                )
            continue
        if p.returncode == 0:
            wrong.append(f"{tag}: routed, wants {want[:200]}")
        elif p.returncode < 0 or want not in p.stderr:
            wrong.append(
                f"{tag}: rc {p.returncode}: {first_error(p.stderr)[:300]}, wants {want[:300]}"
            )
        else:
            exact += c["shape"] in ("arbiters", "wormhole")
            continue
        save(c, "wrong", c["design"].emit(), p.stderr)
    for line in wrong[:10]:
        print("WRONG VERDICT:", line)
    decided = len(ucases) - undecided
    report(
        "unroutable",
        not wrong and decided,
        f"{decided - len(wrong)}/{decided} rejected as predicted ({exact} with the "
        f"exact message), {undecided} left to the model",
    )

    kcases = [
        unknown_case(s, args.device) for s in range(first, first + PARAMS["unknown"])
    ]
    outcomes = defaultdict(lambda: [0, 0])
    unknown_illegal = []
    for c, p in zip(kcases, run_cases(kcases, route_one)):
        outcomes[c["shape"]][p.returncode != 0] += 1
        tag = f"seed {c['seed']} ({c['shape']})"
        if p.returncode < 0:
            unknown_illegal.append(f"{tag}: crashed: {first_error(p.stderr)[:300]}")
            save(c, "crash", c["design"].emit(), p.stderr)
        elif p.returncode == 0:
            problems, _ = verify(c["design"], c["analysis"], p.stdout, c["hops_on"])
            if problems and (label := known_bug(c["design"], problems)):
                known[label] += 1
                save(c, "known", c["design"].emit())
            elif problems:
                unknown_illegal.append(f"{tag}: {problems[0]}")
                save(c, "illegal", c["design"].emit())
    for line in unknown_illegal[:10]:
        print("ILLEGAL:", line)
    print(
        "unknown: "
        + ", ".join(
            f"{k} {v[0]} routed {v[1]} failed" for k, v in sorted(outcomes.items())
        )
    )
    report(
        "unknown-legality",
        not unknown_illegal,
        f"{len(unknown_illegal)} illegal or crashed of {len(kcases)}",
    )

    for label, count in sorted(known.items()):
        print(f"known: {count} {label}")
    n = max(totals["designs"], 1)
    ratio = totals["hops"] / max(totals["constructed"], 1)
    report(
        "hops",
        ratio <= PARAMS["max_hop_ratio"],
        f"{totals['hops']} switchbox hops, {ratio:.3f}x the constructed routing, "
        f"{totals['hops'] / max(totals['bound'], 1):.3f}x the Manhattan bound "
        f"(max {PARAMS['max_hop_ratio']}x constructed)",
    )
    report(
        "arbiters",
        totals["amsels"] / n <= PARAMS["max_amsels"],
        f"{totals['amsels'] / n:.2f} amsels on {totals['arbiters'] / n:.2f} arbiters per "
        f"design, {totals['promoted']} hops circuit switched (max {PARAMS['max_amsels']} amsels)",
    )
    report(
        "priority",
        totals["low_priority"] / n <= PARAMS["max_low_priority"],
        f"{totals['low_priority'] / n:.3f} priority hops per design under a non-priority "
        f"msel, {totals['mixed_ctrl']} ctrl mastersets also carrying non-priority flows "
        f"(max {PARAMS['max_low_priority']})",
    )
    dims, pairs, hit, _ = space_coverage(spaces, args.device)
    floors = SPACE_FLOORS.get(args.device, {})
    low = [f"{k} {dims[k][0]} < {n}" for k, n in floors.items() if dims[k][0] < n]
    ph = sum(h for h, _ in pairs.values())
    pn = sum(n for _, n in pairs.values())
    report(
        "route-space",
        not low,
        f"{sum(h for h, _ in dims.values())}/{sum(n for _, n in dims.values())} "
        f"values over {len(dims)} dimensions, {ph}/{pn} pairs over "
        f"{'/'.join(PAIRWISE)}"
        + ("" if debug_flags() else ", no router internals (no debug build)")
        + (f"; below floor: {', '.join(low)}" if low else ""),
    )
    if args.space_out:
        args.space_out.write_text(
            json.dumps(
                [
                    dict(sp, dims={k: sorted(v) for k, v in sp["dims"].items()})
                    for sp in spaces
                ]
            )
        )

    per_design = route_time / max(runs, 1) * 1000
    report(
        "time",
        per_design <= PARAMS["max_ms_per_design"],
        f"{per_design:.1f} ms per design (max {PARAMS['max_ms_per_design']})",
    )
    print(f"total: {time.monotonic() - t_start:.1f}s")
    return 0


# Every property must report OK; any REGRESSION fails the test.
# CHECK: model-validation: {{.*}} : OK
# CHECK: verifier-validation: {{.*}} : OK
# CHECK: completeness: {{.*}} : OK
# CHECK: legality: {{.*}} : OK
# CHECK: determinism: {{.*}} : OK
# CHECK: unroutable: {{.*}} : OK
# CHECK: unknown-legality: {{.*}} : OK
# CHECK: hops: {{.*}} : OK
# CHECK: arbiters: {{.*}} : OK
# CHECK: priority: {{.*}} : OK
# CHECK: route-space: {{.*}} : OK
# CHECK: time: {{.*}} : OK
# CHECK-NOT: REGRESSION

if __name__ == "__main__":
    sys.exit(main())
