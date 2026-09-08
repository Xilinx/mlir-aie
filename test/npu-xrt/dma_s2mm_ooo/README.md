<!---//===- README.md ---------------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Out-of-order S2MM

A deterministic out-of-order merge (not to be confused with deterministic merge
*mode* in packet arbitration). `N` senders stream into one S2MM channel, and
each packet lands in a fixed destination slot chosen by its packet header rather
than by the order it arrives in.

## How it works

The design has four stages.

1. **Senders.** `N` compute cores (the bottom row, one per column) each hold an
   `arange` slice and send it as one or more packets. Each packet is stamped with
   an *out-of-order id*, the destination slot, through the `Bd(out_of_order_id=...)`
   field.

2. **Routing.** Each sender uses a distinct route `pkt_id`, and its `PacketFlow`
   sets `keep_pkt_header=True` because the `out_of_order_id` must survive to the
   receiver. The route `pkt_id` and the `out_of_order_id` are separate header
   fields.

3. **Receiver.** A compute **core** or a **memtile** puts its S2MM channel in
   `out_of_order` mode. The receiver writes each incoming packet to the receive BD
   whose `bd_id` equals the packet's out-of-order id. Placement follows the
   id, not the arrival order and not the slot's position in the BD list.

4. **Completion and drain.** A counting lock gates completion on-chip (see
   [On-chip completion](#on-chip-completion)). An MM2S on the receiver then drains
   the merged buffer to the host through the egress shim tile, with no host
   round-trip and no completion token.

The test proves placement is id-directed. Sender `s` targets slot `(s + shift) %
N`, which rotates each merged buffer by `shift`, and the verifier runs every
non-identity rotation. An in-order channel could match at most one rotation by
luck. The receive BDs use non-sequential `bd_id`s because a match must not be
explainable by position. With `--channels 2` each channel is fed distinct data
(each `(channel, round, source)` chunk is a unique `arange` slice), which also
lets the verifier catch a channel swap or a cross-channel misroute, not just a
wrong slot.

## Options

| Flag | Values | Meaning |
|------|--------|---------|
| `--recv-tile` | `core`, `mem` | Receiver (merge) tile type. |
| `--recv-config` | `static`, `runtime` | How the out-of-order receive channel is configured (see [Configuring the receiver](#configuring-the-receiver)). |
| `--channels`  | `1`, `2`      | Out-of-order channels on the receiver tile. |
| `-n`/`--sources` | `1..8`     | Merge width (`n=1` is a degenerate 1-way merge). |
| `--packets`   | `m` (default 1) | Packets per source; fills `n*m` sub-buffers. |
| `--nonuniform`| flag          | Slot `j` gets `j+1` packets (per-slot count). Leave `--packets` at 1. |
| `--repeat-count` | `k` (default 0) | Extra merge rounds; the receiver runs `k+1` rounds reusing the one buffer. |
| `--recv-backpressure` | flag | Single-producer (`n=1`) reuse via a receiver-side credit instead of the sender-side barrier. |

**`--packets m`.** Each source sends `m` packets carrying `m` distinct
`tile_words`-sized sub-slices. A receive-side BD iteration advances the write
offset because the sub-slices must land in `m` consecutive sub-buffers. Both sides
use `BdIteration(size=m, stride=tile_words)`, and the sender replays its BD with
`repeat_count`.

**`--nonuniform`.** Slot `j` gets `j+1` packets, which gives the slots of one merge
different counts (`1, 2, ..., n`, total `n(n+1)/2`). There is no restriction on how
many packets each source sends. The receiver derives the total from the per-BD
iteration sizes, and each slot begins right after the previous one (slot `j`'s
offset is the sum of the earlier slots' sizes).

**`--repeat-count k`.** Runs `k+1` merge rounds that reuse the one buffer, each a
genuine `n`-way out-of-order merge. See [Multiple rounds](#multiple-rounds).
It composes with `--channels`, `--packets`, and `--nonuniform`.

## Limits

| Config | Max `n` | Bound |
|--------|---------|-------|
| `mem`, 1 or 2 channels | 8 | full width |
| `core`, 1 channel      | 8 | full width |
| `core`, 2 channels     | 7 | 16-BD core tile budget |

- **BD budget.** Each channel needs `n` receive BDs plus 1 egress BD, which
  requires `c*(n+1) <= 16` on a core tile. Two channels at `n=8` need 18 BDs, and
  `n=7` fits.
- **Routing.** A stream-switch slave port holds at most 4 packet rules. Centering
  the receiver at column `N//2` splits the funnel and keeps every port under 4
  rules.
- **Packet count.** The total must stay `<= 63` because the egress lock acquires
  the total packet count and a lock value is 6-bit.
- **Multi-round (`--repeat-count k`).** Let `M` be the per-round packet count
  (`n*m`, or `sum(ms)` under `--nonuniform`). The 8-bit repeat field bounds the
  all-rounds total to `M*(k+1) <= 256`. The sender's per-round credit and the
  token-credit lock are each a single 6-bit lock, which bounds `channels * max(ms)
  <= 63` and `rounds (k+1) <= 63`. The sender walks all rounds from one BD, whose
  iteration wrap bounds `max(ms) * (k+1) <= 64`. On a core tile with 2 channels the
  `2n+3` receive/drain/token BDs bound it to `n <= 6`.

Two out-of-order channels may share one tile using disjoint pinned `bd_id`s (a
memtile odd channel requires `bd_id >= 24`).

## Receiver API

The receiver is expressed with the following API:

```python
DmaChannel(
    direction=DMAChannelDir.S2MM,
    channel=0,
    bds=recv_bds,          # one packet-enabled BD per slot
    out_of_order=True,     # arm the channel out-of-order (S2MM only)
    repeat_count=0         # one merge round; per-round packet count is derived
                           # from the receive BDs
)
```

`out_of_order=True`:

- arms the channel out-of-order at config time (while idle, before it is enabled)
- lowers the receive BDs with `use_next_bd=0`, which ignores the BD chain and
  places each packet by its header id (the chain exists only to configure the BDs)
- honors each BD's pinned `Bd(bd_id=...)`, which makes placement follow the id
  rather than the BD's position in the list
- reads `iteration=BdIteration(size=m, stride=tile_words)` on each BD to spread a
  source's `m` packets across `m` consecutive sub-buffers
- reads `repeat_count = k` as `k` extra merge rounds. The hardware field is the
  all-rounds packet count `M*(k+1) - 1`, which the channel derives from the
  receive BDs and `repeat_count` (see [Multiple rounds](#multiple-rounds))

## Configuring the receiver

`--recv-config` selects how the channel is armed; both paths produce the same
on-device behavior and share everything else (senders, routing, counting-lock
completion, egress drain, verifier).

- **`static`** (default): part of the receiver tile's static DMA program,
  `DmaChannel(..., out_of_order=True)` on a `TileDma`. Lowers to
  `aie.dma_start {out_of_order}`.
- **`runtime`**: armed from the host sequence with
  `aiex.dma_configure_task(receiver, S2MM, ch) {out_of_order}`
  followed by `dma_start_task`; only the drain MM2S stays static. Supports the
  full merge matrix (multi-packet, multi-channel, multi-round). Two runtime-only
  wrinkles: per-slot BD iteration is expressed via the outermost `sizes`/`strides`
  dimension rather than the static `BdIteration` attribute (which the
  runtime-sequence path rejects); and on a **core** tile the static drain/token
  BDs are pinned off the receive ids, because a core's tile-wide bd_ids let the
  static allocator -- blind to the runtime-pinned receive ids -- otherwise reuse a
  receive slot and deadlock (a memtile restricts ids per channel and avoids this).

End-to-end out-of-order reception on a **shim** tile is out of scope here: the
receiver stays on-chip so its own MM2S drains the merged buffer to the host.

## Multiple rounds

`--repeat-count k` reuses the one buffer for `k+1` successive out-of-order merges.
The receive BDs' iteration wraps every round, and round `r+1` overwrites round
`r`'s sub-buffers. That overwrite is safe only after round `r` has drained.

A single out-of-order channel is a FIFO that places by header id. A receiver-side
lock therefore cannot enforce "one packet per slot per round". A shared credit
counts *any* `n*m` writes, which miscounts a fast source's round-`r+1` packet into
round `r`. A per-slot credit deadlocks because the channel head-of-line stalls on
the fast source's blocked round-`r+1` packet while a slow source's round-`r` packet
waits behind it. The verifier forbids a receive BD from acquiring a sibling receive
BD's lock for that reason.

The barrier is therefore **sender-side**. The receiver carries a dedicated MM2S
that broadcasts a one-word credit token once all channels have drained the round (a
`both` join across the `c` drains). Each sender's send gates on a `go` credit that a
worker seeds for round 0 and the token replenishes for every round after. No sender
starts round `r+1` until round `r` has drained on every channel. The `n` senders
still race within a round, which keeps each round a genuine `n`-way out-of-order
merge. The token needs no channel of its own because it packet-shares channel 0's
drain MM2S under a distinct `pkt_id` (chained after the drain), which fits
multi-round onto a core tile's two MM2S even at `--channels 2`.

### Single producer: receiver-side backpressure

The barrier above is *sender-side backpressure* -- the receiver tells the sources
when to proceed. For a single producer (`--recv-backpressure`, `n=1`) a cheaper
*receiver-side backpressure* suffices, with no token at all. Each receive BD
acquires a free-slot credit before writing, and the egress returns the round's
credits after draining. The stalled receive DMA then backpressures the one sender
through the stream:

```python
# --recv-backpressure (single producer, ooo_prod init M):
recv_bd  = Bd(..., packet=(0, route),
              acquires=[Acquire(ooo_prod, 1)], releases=[Release(ooo_cons, 1)])
egress   = Bd(..., acquires=[Acquire(ooo_cons, M)],
              releases=[Release(ooo_prod, M)])
```

This works only for one producer, because the round-agnostic credit cannot hold
per-round grouping across several sources -- a fast source would spend a whole
round's credits on its own packets. Because the one sender's launch credit is a
single 6-bit lock, this mode also requires `M*(k+1) <= 63`. The test drives it at a
high `--repeat-count` so the reused buffer would overrun without the credit, which
keeps the credit load-bearing rather than inert.

## Sender side

Each sender is a compute core. Its buffer is preloaded with the source slice, and a
trivial worker releases a lock to launch the send. The core's MM2S BD stamps the
out-of-order id through the `Bd(out_of_order_id=...)` field. With `m > 1`, the BD
carries `iteration=BdIteration(size=m, stride=tile_words)` and `repeat_count`
replays it `m` times, which gives each of the `m` packets a distinct sub-slice under
the one out-of-order id. The host drains each merged buffer with a high-level shim
DMA task (`shim_dma_single_bd_task`). `--nonuniform` instead varies `m` across BDs.
With `--channels 2` a sender emits a distinct sub-slice to each channel, and with
`--repeat-count` its send gates on a credit token per round (see
[Multiple rounds](#multiple-rounds)).

## On-chip completion

Each receive BD holds a **release-only** lock, and the egress BD acquires the total
packet count of it:

```python
recv_bd  = Bd(..., packet=(0, route), releases=[Release(ooo_cons, 1)])
egress   = Bd(..., acquires=[Acquire(ooo_cons, M)])   # M = total packet count
```

As packets land, the counter climbs to the total and the egress drains, on-chip,
with no host round-trip or completion token. The receive BDs only release and never
acquire each other's locks, which removes any inter-BD dependency that could
deadlock. The total must stay `<= 63` because a lock value is 6-bit.

## Running

```
python dma_s2mm_ooo.py                                       # memtile, 1 channel, n=3
python dma_s2mm_ooo.py --recv-tile mem  --channels 2 -n 8    # 8-way merge, 2 channels
python dma_s2mm_ooo.py --recv-tile core --channels 2 -n 7    # core 2-channel ceiling
python dma_s2mm_ooo.py --recv-tile mem  --channels 1 -n 4 --packets 4   # m=4, n*m=16
python dma_s2mm_ooo.py --recv-tile mem  --channels 1 -n 4 --nonuniform  # slot j gets j+1
python dma_s2mm_ooo.py --recv-tile mem  --channels 1 -n 4 --repeat-count 2  # 3 merge rounds
python dma_s2mm_ooo.py --recv-config runtime --recv-tile mem  --channels 1 -n 4  # runtime-configured receiver
python dma_s2mm_ooo.py --emit-mlir                           # print the generated MLIR
```
