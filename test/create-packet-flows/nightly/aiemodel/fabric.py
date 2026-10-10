#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The fabric: bundles, ports, and the TargetModel view of a device."""

from aie.dialects.aie import AIEDevice, WireBundle, get_target_model
from aie.ir import Context

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
# The ops LoopLikeOpInterface covers that a core body can hold.
LOOP_OPS = ("scf.for", "scf.while", "scf.parallel", "scf.forall", "affine.for")


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
    """What the TargetModel says about a device, plus its crossbar."""

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
            self.aie1 = tm.get_target_arch() == 1
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
        """AIE1TargetModel and AIE2TargetModel::isLegalTileConnection."""
        (sb, si), (db, di) = sp, mp
        if si >= self.slaves[tile][sb] or di >= self.masters[tile][db]:
            return False
        kind = self.kinds[tile]
        if self.aie1:
            return sb != TRACE or db == SOUTH
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
