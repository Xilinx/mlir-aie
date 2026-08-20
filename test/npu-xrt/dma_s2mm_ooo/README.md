<!---//===- README.md ---------------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Out-of-order S2MM

A deterministic many-to-one merge: `N` senders stream into one S2MM channel, and
each packet lands in a fixed destination slot chosen by its packet header, not by
the order it arrives in.

## How it works

The design has four stages.

1. **Senders.** `N` compute cores (the bottom row, one per column) each hold an
   `arange` slice and send it as one or more packets. Each packet is stamped with
   an *out-of-order id* -- the destination slot -- using the dataflow
   `Bd(out_of_order_id=...)` field, so no runtime `writebd` is needed.

2. **Routing.** Each sender uses a distinct route `pkt_id`, and its `PacketFlow`
   sets `keep_pkt_header=True` so the out-of-order id survives to the receiver.
   (The route `pkt_id` and the out-of-order id are separate header fields.)

3. **Receiver.** A compute **core** or a **memtile** puts its S2MM channel in
   out-of-order mode. Each incoming packet is written to the receive BD whose
   pinned `bd_id` equals the packet's out-of-order id -- placement follows the
   id, not the arrival order or the slot's position in the BD list.

4. **Completion and drain.** Completion is gated on-chip by a counting lock (see
   [On-chip completion](#on-chip-completion)); an MM2S on the receiver then
   drains the merged buffer to the host via the egress shim tile, with no host
   round-trip and no completion token.

The test proves placement is id-directed: sender `s` targets slot `(s + shift) %
N`, so each merged buffer is the send order rotated by `shift`, and the verifier
runs every non-identity rotation (an in-order channel could match at most one by
luck). The receive BDs are pinned to non-sequential `bd_id`s so a match cannot be
explained by position.

## Options

| Flag | Values | Meaning |
|------|--------|---------|
| `--recv-tile` | `core`, `mem` | Receiver (merge) tile type. |
| `--channels`  | `1`, `2`      | Out-of-order channels on the receiver tile. |
| `-n`/`--sources` | `2..8`     | Merge width (number of senders). |
| `--packets`   | `m` (default 1) | Packets per source; fills `n*m` sub-buffers. |
| `--nonuniform`| flag          | Slot `j` gets `j+1` packets (per-slot count). Overrides `--packets`. |

**`--packets m`.** Each source sends `m` packets carrying `m` distinct
`tile_words`-sized sub-slices, and a receive-side BD iteration advances the write
offset so they land in `m` consecutive sub-buffers. Both sides use
`BdIteration(size=m, stride=tile_words)`; the sender replays its BD with
`repeat_count`.

**`--nonuniform`.** Slot `j` gets `j+1` packets, so the slots of one merge hold
different counts (`1, 2, ..., n`, total `n(n+1)/2`). The spec places no
restriction on how many packets each source sends; the receiver derives the total
from the per-BD iteration sizes, and slot offsets are the running prefix sums.

## Limits

| Config | Max `n` | Bound |
|--------|---------|-------|
| `mem`, 1 or 2 channels | 8 | full width |
| `core`, 1 channel      | 8 | full width |
| `core`, 2 channels     | 7 | 16-BD core-tile budget |

- **BD budget.** Each channel needs `n` receive BDs plus 1 egress BD, so
  `c*(n+1) <= 16` on a core tile. Two channels at `n=8` need 18 BDs; `n=7` fits.
- **Routing.** A stream-switch slave port holds at most 4 packet rules. Centering
  the receiver at column `N//2` splits the funnel (west routes east, east routes
  west), keeping every port under 4 rules through `n=8`. A column-0 receiver would
  overflow a port and cap the 2-channel merge at `n=6`.
- **Packet count.** The egress lock acquires the total packet count, and an AIE
  lock value is 6-bit, so the total must stay `<= 63` (an `n=8` 1-channel merge
  tops out at `m=4`). This is orthogonal to the routing limit above.

Two out-of-order channels may share one tile using disjoint pinned `bd_id`s (a
memtile odd channel requires `bd_id >= 24`).

## Receiver API

The receiver is expressed entirely with the first-class IRON API:

```python
DmaChannel(
    direction=DMAChannelDir.S2MM,
    channel=0,
    bds=recv_bds,          # one packet-enabled BD per slot
    out_of_order=True,     # arm the channel out-of-order (S2MM only)
    # no repeat_count: the packet count is derived from the receive BDs
)
```

`out_of_order=True`:

- arms the channel out-of-order at config time (while idle, before it is enabled);
- lowers the receive BDs with `use_next_bd=0`: the BD chain is ignored and each
  packet is placed by its header id, so the chain exists only to configure the BDs;
- honors each BD's pinned `Bd(bd_id=...)`, so placement follows the id, not the
  BD's position in the list;
- reads `iteration=BdIteration(size=m, stride=tile_words)` on each BD to spread a
  source's `m` packets across `m` consecutive sub-buffers;
- derives the packet count as the sum of the receive BDs' iteration sizes
  (`n*m`); a nonzero `repeat_count` would repeat that whole round, and this
  example uses a single round.

## Sender side

Each sender is a compute core. Its buffer is preloaded with the source slice, and
a trivial worker releases a lock to launch the send. The core's MM2S BD stamps the
out-of-order id via the dataflow `Bd(out_of_order_id=...)` field -- no
`aiex.npu.writebd`. With `m > 1`, the BD carries
`iteration=BdIteration(size=m, stride=tile_words)` and `repeat_count` replays it
`m` times, so each of the `m` packets carries a distinct sub-slice under the one
out-of-order id. The host drains each merged buffer with a high-level shim DMA
task (`shim_dma_single_bd_task`), so the whole design uses no `writebd`.

## On-chip completion

Each receive BD holds a **release-only** lock; the egress BD acquires the total
packet count of it:

```python
# completion only (this example):
recv_bd  = Bd(..., packet=(0, route), releases=[Release(ooo_cons, 1)])
egress   = Bd(..., acquires=[Acquire(ooo_cons, n * m)])
```

As packets land, the counter climbs to the total and the egress drains -- on-chip,
with no host round-trip or completion token. The receive BDs only release (they
never acquire each other's locks), so there is no inter-BD dependency to deadlock.
The total must stay `<= 63` (the 6-bit lock-value max).

For guaranteed backpressure when the buffer is reused, add a second credit lock
the consumer hands back: each receive BD acquires a free slot before writing, and
the egress releases the credits after draining.

```python
# + guaranteed backpressure (free init n*m):
recv_bd  = Bd(..., packet=(0, route),
              acquires=[Acquire(ooo_prod, 1)], releases=[Release(ooo_cons, 1)])
egress   = Bd(..., acquires=[Acquire(ooo_cons, n * m)],
              releases=[Release(ooo_prod, n * m)])
```

## Running

```
python dma_s2mm_ooo.py                                       # memtile, 1 channel, n=3
python dma_s2mm_ooo.py --recv-tile mem  --channels 2 -n 8    # 8-way merge, 2 channels
python dma_s2mm_ooo.py --recv-tile core --channels 2 -n 7    # core 2-channel ceiling
python dma_s2mm_ooo.py --recv-tile mem  --channels 1 -n 4 --packets 4   # m=4, n*m=16
python dma_s2mm_ooo.py --recv-tile mem  --channels 1 -n 4 --nonuniform  # slot j gets j+1
python dma_s2mm_ooo.py --emit-mlir                           # print the generated MLIR
```
