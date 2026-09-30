#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %python %s | FileCheck %s

"""An infeasible hold cycle with 32 alternative holders needs 65 searches."""

import router_properties as rp
from router_properties import DMA, EAST, MM2S, NORTH, S2MM, SOUTH, WEST


def make_design():
    d = rp.Design("npu1_2col")
    d.add_packet_flow(2, [(0, 1, DMA, 0)], [(0, 2, DMA, 0)])
    for ch, ids in (
        (0, [2]),
        (1, [0, 1]),
        (2, [0]),
        (3, [0]),
        (4, list(range(31))),
    ):
        rp.add_program(
            d, (0, 1), MM2S, ch, [rp.bd_block(256, pid) for pid in ids], False
        )
    for r in (2, 3):
        pa, ca, pb, cb = (d.lock((0, r), init) for init in (1, 0, 1, 0))
        rp.add_program(
            d,
            (0, r),
            S2MM,
            0,
            [rp.bd_block(64, None, pa if r == 3 else None, ca)],
            True,
        )
        rp.add_program(d, (0, r), S2MM, 1, [rp.bd_block(64, None, pb, cb)], True)
        d.cores[(0, r)] = [(2, ca, 1), (2, cb, 1), (1, pa, 1), (1, pb, 1)]
    rp.add_program(d, (0, 4), S2MM, 0, [rp.bd_block(256)], True)

    def connect(sb, si, db, di):
        return ("connect", (sb, si), (db, di))

    def amsel(name, arbiter, msel=0):
        return ("amsel", name, arbiter, msel)

    def master(bundle, channel, *names):
        return ("masterset", (bundle, channel), names, None, False)

    def rules(bundle, channel, *entries):
        return ("rules", (bundle, channel), entries, False)

    d.boxes[(0, 1)] = [
        *[amsel(f"reserved0_{m}", 0, m) for m in range(4)],
        amsel("y", 1),
        amsel("dj", 5),
        amsel("xi", 2),
        amsel("z", 3),
        amsel("di", 4),
        master(NORTH, 0, "y"),
        master(NORTH, 1, "dj"),
        master(NORTH, 2, "xi"),
        master(NORTH, 3, "z"),
        master(NORTH, 4, "di"),
        rules(DMA, 1, (31, 0, "y"), (31, 1, "dj")),
        rules(DMA, 2, (31, 0, "xi")),
        rules(DMA, 3, (31, 0, "z")),
        rules(DMA, 4, (0, 0, "di")),
    ]
    d.boxes[(0, 2)] = [
        amsel("y", 1),
        amsel("dj", 2),
        master(EAST, 0, "y"),
        master(DMA, 1, "dj"),
        rules(SOUTH, 0, (31, 0, "y")),
        rules(EAST, 0, (31, 1, "dj")),
        connect(SOUTH, 1, NORTH, 1),
        connect(SOUTH, 2, EAST, 2),
        connect(SOUTH, 3, NORTH, 0),
        connect(SOUTH, 4, EAST, 3),
    ]
    d.boxes[(1, 2)] = [
        amsel("xi", 0, 0),
        amsel("y", 0, 1),
        amsel("dj", 0, 2),
        master(NORTH, 0, "xi"),
        master(NORTH, 1, "y"),
        master(WEST, 0, "dj"),
        rules(WEST, 0, (31, 0, "y")),
        rules(NORTH, 1, (31, 1, "dj")),
        rules(WEST, 2, (31, 0, "xi")),
        rules(WEST, 3, (0, 0, "xi")),
    ]
    d.boxes[(1, 3)] = [
        connect(SOUTH, 0, WEST, 0),
        connect(SOUTH, 1, NORTH, 0),
        connect(WEST, 1, SOUTH, 1),
    ]
    d.boxes[(0, 3)] = [
        amsel("xi", 0),
        amsel("z", 1),
        master(DMA, 0, "xi"),
        master(DMA, 1, "z"),
        rules(EAST, 0, (0, 0, "xi")),
        rules(SOUTH, 0, (31, 0, "z")),
        connect(SOUTH, 1, EAST, 1),
    ]
    d.boxes[(1, 4)] = [connect(SOUTH, 0, WEST, 0)]
    d.boxes[(0, 4)] = [
        amsel("y", 0),
        master(DMA, 0, "y"),
        rules(EAST, 0, (31, 0, "y")),
    ]
    return d


def main():
    d = make_design()
    analysis = rp.Analysis(d)
    direct = [((0, 1), (DMA, 0), 1), ((0, 2), (SOUTH, 5), 0)]
    routes = [direct] + [s.hops for s in analysis.streams[1:]]
    assert analysis.hold_cycle(routes) is None
    result = rp.aie_opt(d.emit(), False)
    assert result.returncode == 0, result.stderr
    streams = rp.trace_routed_streams(rp.load_design(result.stdout))
    routed = [s.hops for s in streams if s.src == (0, 1, DMA, 0) and s.pid == 2]
    assert routed == [direct], routed
    print("hold-cycle-search: feasible direct route retained")


# CHECK: hold-cycle-search: feasible direct route retained

if __name__ == "__main__":
    main()
