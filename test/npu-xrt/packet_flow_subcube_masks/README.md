<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Relaxed packet-rule masks on hardware

A slave port has only `getNumSlaveSlots()` (four) packet-rule slots, so
`--aie-create-pathfinder-flows` does not give each packet id its own exact rule.
`computeSubcubeCover` minimises the rule set: it relaxes masks so one rule
covers several ids, splitting into more cubes only when a single cube would
swallow an id belonging to another destination.

Every other on-device packet test uses exact (`mask = 31`) rules, so that
minimisation had only ever been checked by FileCheck. This test runs it on
silicon.

## What the router emits here

Ids 0, 3, 4 go to memtile S2MM 0 and id 1 to memtile S2MM 1. At the memtile
slave port that produces a **two-cube cover**:

```mlir
aie.packet_rules(South : 4) {
  aie.rule(31, 1, %1)   // exact,   id 1       -> DMA:1
  aie.rule(27, 0, %0)   // relaxed, ids {0, 4} -> DMA:0
  aie.rule(31, 3, %0)   // exact,   id 3       -> DMA:0
}
```

`{0, 3, 4}` cannot be one cube: the smallest cube enclosing it is `{0..7}`,
which would swallow id 1. So the branch-and-bound minimum-cover step splits it
into the relaxed `(27, 0)` plus an exact rule for 3. The relaxed rule sits one
bit from id 1 — `1 & 27 == 1`, not `0` — so a single wrong mask bit misroutes.

The shim slave port gets its own relaxed rule, `aie.rule(24, 0, %0)`, covering
all four ids on the way up.

Nothing in `aie.mlir` is hand-routed. The switchboxes come from the router, and
that is the point.

## Checking

Each id carries a distinguishable payload, `1000 + 100*id + i`, so claiming one
id too many shows up as the *wrong* payload rather than as a missing transfer.
Ids 0, 3, 4 arrive at S2MM 0 in send order and fill `a0`, `a1`, `a2`; id 1 fills
`b0`. All four are read back and compared.

## What makes this falsifiable

Taking the routed IR and diverting one rule — `aie.rule(31, 3, %0)` to
`%1`, so id 3 goes to `DMA:1` — turns the pass into a failure on NPU Phoenix
(npu1), while the unmodified routed IR passes:

```
routed, unmodified:   PASS!

one rule diverted:    packet id 3 slot 0: got 1400 want 1300
                      packet id 3 slot 1: got 1401 want 1301
                      ...
```

`1400` is id 4's payload turning up where id 3's belongs: with id 3 diverted,
only ids 0 and 4 reach `DMA:0`, so they land in `a0` and `a1`. The check
notices, which is what makes a green run meaningful.

## No path can deadlock

The receive BDs acquire from a pre-filled `ready` lock rather than gating on
data, so a misrouted id cannot stall a channel. The readbacks gate on `go_a` /
`go_b`, which the runtime sequence sets after the shim has finished sending —
on the program, not on data — so a buffer that received nothing still reads back
as the `-1` sentinel it was initialised with. A misroute always shows up as
wrong data, never as a hang.

Ordering: only the last input BD issues a token. The shim drains its queue in
order, so waiting on that one means all four have left before the readbacks are
released.

Two details that are easy to get wrong: `aie.dma` (and the CDO backend) require
exactly one acquire and one release per BD, which is why the BDs are written
with `dma_start` and why `ready` / `unused` exist; and memtile BD ids are
partitioned by channel parity, even channels taking BD < 24 and odd channels
BD >= 24.
