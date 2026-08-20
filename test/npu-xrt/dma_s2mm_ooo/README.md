<!---//===- README.md ---------------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Out-of-order S2MM

This example demonstrates an **out-of-order (OoO) S2MM DMA channel**: a
deterministic many-to-one merge where `N` senders stream into one receive
channel and each packet lands in a fixed destination slot chosen by its packet
header, independent of the order the packets arrive in.

## What it does

`N` compute-core senders -- the full bottom compute row, one per column
(row 2, columns `0..N-1`) -- each self-generate one packet per out-of-order
channel (`tile_words` int32s -- sender `s` owns the arange slice
`[s*tw, s*tw+tw)`), stamping it with a first-class out-of-order id via a
dataflow BD (no runtime `writebd`). The receiver tile -- a compute **core**
(row 3) or a **memtile** (row 1) (selectable), **centered at column `N//2`** so
packets funnel in from both sides -- puts each of its `1` or `2`
S2MM channels into out-of-order mode, so each packet is placed into the
receive buffer-descriptor (BD) whose pinned `bd_id` equals the packet header's
out-of-order id. To make placement observable, sender `s` stamps the id of
slot `(s + shift) % N`, so each merged buffer is the send order rotated by
`shift`. An MM2S on the receiver then drains each merged buffer to the host
via the egress **shim tile** (row 0), self-gated **on-chip**: each receive BD
releases a completion counter that the egress BD acquires `N` of (see below),
so the drain starts only after all `N` packets have landed -- no host
round-trip and no completion token.

`N` (`-n`/`--sources`) sweeps `2..8`. Every config reaches the full-width `n=8`
except a core receiver with 2 channels, which tops out at `n=7`:

| `--recv-tile` | `--channels` | max `n` |
|---------------|--------------|---------|
| `mem`         | 1 or 2       | 8       |
| `core`        | 1            | 8       |
| `core`        | 2            | 7       |

The core/2-channel ceiling is the **16-BD core-tile budget**: each channel needs
`n` receive BDs plus 1 egress BD, so `c * (n + 1) <= 16` -- 2 channels at `n=8`
need `2 * 9 = 18 > 16`, while `n=7` fits exactly (`2 * 8 = 16`).

That the budget (not routing) is the binding limit is thanks to **centering the
receiver**. A stream-switch slave port holds at most **4 packet rules**, and
distinct-pkt_id flows pile one rule per pkt_id onto the shared ports nearest the
receiver. With the receiver at column `N//2` the funnel splits -- the west half
routes east, the east half routes west -- so each direction carries only ~`N/2`
senders' rules, keeping every port under 4 through `n=8`. A col-0 receiver
(one-sided funnel) would instead overflow a port and cap the 2-channel merge at
`n=6` (`aie.packet_rules ... exceed the 4-slot limit`).

Two out-of-order channels share one tile (the per-tile "one out-of-order channel"
limit is gone) using disjoint pinned `bd_id`s; a memtile odd channel requires
`bd_id >= 24`.

Each sender uses a **distinct route packet id** (all still routed to the one OoO
S2MM channel; placement is by the separate out-of-order id). One shared pkt_id
across `N` senders over-subscribes a compute tile's switchbox arbiter when the
receiver is a core; distinct ids route cleanly, so both receiver tiles use one
code path.

## The out-of-order receiver API

The receiver is expressed entirely with the first-class IRON API:

```python
DmaChannel(
    direction=DMAChannelDir.S2MM,
    channel=0,
    bds=recv_bds,          # one packet-enabled BD per slot
    out_of_order=True,     # arm the channel out-of-order (S2MM only)
    repeat_count=n,        # number of packets the channel receives
)
```

`out_of_order=True`:
- arms the channel out-of-order at config time (while it is idle, before it is
  enabled)
- lowers the receive BDs with `use_next_bd=0`; out-of-order mode ignores the
  BD chain and places each packet by its header id, so the chain only exists to
  get all `N` BDs configured
- honors each receive BD's explicit, non-sequential `Bd(bd_id=...)`: a packet is
  placed in the BD whose `bd_id` equals its header out-of-order id, so placement
  follows the pinned id, not the slot's position in `bds`

## Sender side

Each sender is a compute core: its buffer is pre-initialized with the sender's
slice, a trivial worker releases a lock to launch the send, and the core's MM2S
BD stamps the out-of-order id with the first-class dataflow field
`Bd(out_of_order_id=<target slot's bd_id>)` -- no runtime `aiex.npu.writebd`. The
ingress `PacketFlow`s set `keep_pkt_header=True` so the header (carrying the id)
reaches the S2MM, and each sender uses a distinct route `pkt_id` (the
out-of-order id is a separate header field from the routing id). The host only
drains each merged buffer to the output with a high-level shim DMA task
(`shim_dma_single_bd_task`) -- so the whole design uses no `writebd`.

## On-chip completion

Each receive BD carries a **release-only** lock (`releases=[Release(ooo_cons,
value=1)]`); the egress BD does `acquires=[Acquire(ooo_cons, value=n)]`. As the
`N` packets land, the counter climbs to `N` and the egress drains -- entirely
on-chip, with no host round-trip or completion token. The receive BDs only
release (they never acquire each other's locks), so there is no inter-BD
dependency to deadlock.

For guaranteed backpressure (no overwrite when the buffer is reused), add a
second credit lock the consumer hands back. Each receive BD also acquires a free
slot before writing, and the egress releases the credits after draining:

```python
# completion only (this example):
recv_bd  = Bd(..., packet=(0, route), releases=[Release(ooo_cons, 1)])
egress   = Bd(..., acquires=[Acquire(ooo_cons, n)])

# + guaranteed backpressure (free init n):
recv_bd  = Bd(..., packet=(0, route),
              acquires=[Acquire(ooo_prod, 1)], releases=[Release(ooo_cons, 1)])
egress   = Bd(..., acquires=[Acquire(ooo_cons, n)], releases=[Release(ooo_prod, n)])
```

## Running

```
python dma_s2mm_ooo.py                                      # memtile receiver, 1 channel, n=3
python dma_s2mm_ooo.py --recv-tile mem  --channels 2 -n 8   # 8-way merge, 2 channels
python dma_s2mm_ooo.py --recv-tile core --channels 2 -n 7   # core 2-channel ceiling
python dma_s2mm_ooo.py --emit-mlir                          # print the generated MLIR
```
