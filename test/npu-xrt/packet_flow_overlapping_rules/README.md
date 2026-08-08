<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Overlapping packet rules misroute silently

Reproduces [Xilinx/mlir-aie#437](https://github.com/Xilinx/mlir-aie/issues/437)
on hardware.

A stream switch matches packet rules **in order and routes on the first hit**.
Two rules whose `(mask, value)` cubes intersect therefore send every shared id
to whichever rule comes first; the later rule is dead for those ids. The issue's
example, reproduced verbatim in [`aie.mlir`](./aie.mlir):

```mlir
aie.packet_rules(South : 4) {
  aie.rule(26, 10, %0)   // mask 0b11010, val 0b01010 -> masterset DMA:0
  aie.rule(24,  8, %1)   // mask 0b11000, val 0b01000 -> masterset DMA:1
}
```

Packet id 14 (`0b01110`) matches both — `14 & 26 == 10` and `14 & 24 == 8` — so
it is routed to `DMA:0` instead of `DMA:1`. Ids 10, 11 and 15 are shared too.

## Data path

Two packet streams leave shim DMA MM2S 0 and must fan apart at the memtile:

| stream | pkt id | payload   | intended destination |
| ------ | ------ | --------- | -------------------- |
| `x`    | 10     | `100..107`| memtile S2MM 0 → `buf_a` |
| `y`    | 14     | `200..207`| memtile S2MM 1 → `buf_b` |

Both memtile buffers are read back to DDR over circuit-switched flows.

Everything except the two rules above is verbatim what the router emits. To
regenerate it, write the design at the flow level and dump it:

```mlir
aie.packet_flow(10) { aie.packet_source<%t00, DMA : 0>  aie.packet_dest<%t01, DMA : 0> }
aie.packet_flow(14) { aie.packet_source<%t00, DMA : 0>  aie.packet_dest<%t01, DMA : 1> }
aie.flow(%t01, DMA : 0, %t00, DMA : 0)
aie.flow(%t01, DMA : 1, %t00, DMA : 1)
```

```bash
aie-opt --aie-create-pathfinder-flows design.mlir
```

The router produces `aie.rule(31, 14, %1)` / `aie.rule(31, 10, %0)` for the
memtile slave port — exact masks that cannot overlap. Relaxing them is the only
edit.

## What makes this falsifiable

The design is correct apart from the two masks. Substituting the exact-match
rules the router would have emitted makes it pass, measured on NPU Phoenix
(npu1):

```
$ sed -e 's/aie.rule(26, 10, %0)/aie.rule(31, 10, %0)/' \
      -e 's/aie.rule(24, 8, %1)/aie.rule(31, 14, %1)/' ...
buf_a (packet id 10, not checked): 100 101 102 103 104 105 106 107
buf_b (packet id 14): 200 201 202 203 204 205 206 207

PASS!
```

With the overlapping rules, on the same hardware:

```
buf_a (packet id 10, not checked): 200 201 202 203 204 205 206 207
buf_b (packet id 14): -1 -1 -1 -1 -1 -1 -1 -1

packet id 14 did not reach its destination
```

`buf_a` holding `200..207` is direct evidence of the misroute: that is `y`'s
payload arriving at `x`'s destination. `buf_b` keeps the `-1` sentinel it was
initialised with.

The host asserts on `buf_b` only. `buf_a` is printed but not checked: under the
bug it receives both streams, and whether the readback catches `y` or `x` is a
race.

## No path can deadlock

A misroute starves the intended destination, so the obvious version of this test
hangs rather than failing. The DMA structure avoids that, and is load-bearing:

- `buf_a`'s S2MM BD acquires from `ready` (pre-filled), so it accepts one packet
  or two without stalling, and releases `a_arrived`.
- `buf_a`'s MM2S BD acquires `a_arrived`, so the readback is ordered behind the
  data. Path A is fed under both correct and broken routing.
- `buf_b`'s S2MM BD also acquires from `ready`; under the bug it simply never
  fires.
- `buf_b`'s MM2S BD acquires `b_go`, which the **runtime sequence** sets after
  path A has round-tripped — gated on the program, not on data, so the readback
  completes even when nothing ever arrived.

Sending `y` before `x` makes the check deterministic: both share one wire, so
`y`'s memtile write retires first and `buf_b` has settled by the time path A
round-trips.

`aie.dma` requires exactly one acquire and one release per BD (as does the CDO
backend), which is why the BDs are written with `dma_start` and why `ready` /
`unused` exist. Memtile BD ids are partitioned by channel parity: even channels
take BD < 24, odd channels BD >= 24.

## Why the compiler did not catch it

Nothing rejected this configuration. `--aie-create-pathfinder-flows` never sees
it, because the switchboxes are already routed.

And `-aie-find-flows`, which traces packets backwards from destination to
source, evaluated every rule independently with no notion of priority. On this
design it reported the `DMA:1` route as a live flow:

```
$ aie-opt -aie-find-flows aie_arch.mlir
    aie.packet_flow(10) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 1>      <-- never delivered
    aie.packet_flow(10) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 0>
```

That second destination does not exist on hardware — the shadowed rule carries
nothing. This is precisely the report in #437.

## How it is caught

`PacketRulesOp::verify()` rejects a rule set in which two rules pointing at
different amsels match a common id, naming the lowest one:

```
$ aiecc ... aie_arch.mlir
aie_arch.mlir:69:9: error: 'aie.rule' op is shadowed for packet id 10: an earlier
rule in this aie.packet_rules matches it too, and the switch routes on the first
match
        aie.rule(24, 8, %1)
```

That is an op verifier, so it runs at parse time and after every pass: this
design cannot reach hardware by any route, which is why `run.lit` asserts the
diagnostic rather than running anything.

`-aie-find-flows` tracks reachable packet ids as an explicit set rather than a
`(mask, value)` cube and walks a port's rules in order, so a rule claims only
the ids no earlier rule took. `test/find-flows/shadowed_rule.mlir` covers that;
it uses a same-amsel overlap, the only kind the verifier permits.
