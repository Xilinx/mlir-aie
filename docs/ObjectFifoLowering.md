<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# ObjectFifo Lowering

An `aie.objectfifo` describes a data movement in one line: a rotating set of
objects, the tiles that produce and consume them, and the ordering between the
two. Turning that into hardware configuration means deciding many smaller
things — where the buffers live, which locks guard them, which DMA channels move
them, which stream route connects those channels, and what buffer descriptor
chain each channel runs.

A pipeline of passes makes those decisions, and each pass writes its answers
back into the IR. This page describes the operations that carry those answers
and the pass that produces each one.

It is written for two readers: someone working on the lowering, and someone who
wants to enter the pipeline partway through with hand-written IR.

## Why the state is in the IR

The lowering used to be one pass. It was all or nothing: a design that needed
one decision made differently had to abandon the whole thing and write buffers,
locks, DMA programs and flows by hand.

Now every decision lands in the IR as an operation or an attribute, so a design
can start at any point in the pipeline and every pass respects what it finds.
Being specific is optional but honored: write down a DMA channel and it is kept;
leave it out and one is chosen.

The same property keeps the passes honest. Each pass erases what it consumes, so
running the pipeline over its own output is a no-op. Anything a pass forgot to
record would show up as a leftover operation instead of a silently lost side
table. This is tested directly.

## The operations

| Operation | What it holds |
| --- | --- |
| `aie.objectfifo.pool` | A rotating set of buffers, the segments they may be accessed in, and the locks guarding those segments. |
| `aie.objectfifo.core_endpoint` | What a core needs to fill or drain the next object in a pool. |
| `aie.objectfifo.dma_endpoint` | The same for a DMA channel. |
| `aie.objectfifo.dangling_endpoint` | One end of a stream this compiler does not program: a shim the host runtime drives, a PLIO boundary, or a core's raw stream port. |
| `aie.objectfifo.flow` | A circuit- or packet-switched connection from one draining endpoint to the filling endpoints it feeds. |

All of them are transient. `--aie-objectfifo-split` introduces them and they are
gone by the end of the pipeline. What survives is ordinary AIE IR: `aie.buffer`,
`aie.lock`, `aie.flow`, `aie.mem` / `aie.memtile_dma` / `aie.shim_dma`, and
`aie.shim_dma_allocation`.

### Pools

```mlir
aie.objectfifo.pool @P(%tile) {
  depth    = 2 : i32,
  buffers  = [@P_buff_0, @P_buff_1],
  segments = [#aie.objectfifo_segment<offset = 0,  size = 16, produceLock = @p0, consumeLock = @c0>,
              #aie.objectfifo_segment<offset = 16, size = 20, produceLock = @p1, consumeLock = @c1>]
} : memref<36xi32>
```

The tile is where the buffers live and `depth` is how many of them rotate.

**Buffers and segments are different axes.** Buffers are the rotation axis:
`depth` objects taking turns. Segments partition a *single* object, and every
buffer carries every segment. Segments are listed in increasing offset order, do
not overlap, and together cover the element type exactly.

An ordinary pool has one segment spanning the whole object. Only a joined or
distributed pool has more.

On devices with binary locks a pool carries `locks` — one per buffer, following
the rotation axis — instead of per-segment locks, and has a single implicit
segment. A pool carries either `locks` or `segments`, never both.

### Endpoints

An endpoint is an actor: something that fills or drains a pool.

```mlir
aie.objectfifo.core_endpoint @c(%tile) fills  @P
aie.objectfifo.dma_endpoint  @d(%tile) drains @P {channelIndex = 0 : i32}
```

`fills` and `drains` say what the actor does to the pool's buffers. The
endpoint's own tile is where the actor is, which for shared memory differs from
the pool's tile.

**Every segment has exactly one filler and exactly one drainer.** A pool whose
segments each have one core filling and one DMA draining is an ordinary fifo
end. A pool with several fillers is a join; one with several drainers is a
distribute.

#### Selecting segments

`segments` picks which of the pool's segments an actor handles:

```mlir
aie.objectfifo.dma_endpoint @d(%tile) fills @P {segments = array<i32: 0, 1>}
```

Indices must be strictly increasing. Omitting the attribute means segment zero,
and is only valid on a single-segment pool. There is no "all segments"
shorthand: on a partitioned pool that would hide which slices an actor really
touches.

#### Per-segment transforms

A DMA endpoint may carry `dimensions` (a strided access pattern) and
`padDimensions`. Both are **positional**: entry N describes the segment named by
`segments[N]`. When present, either attribute must have exactly as many entries
as `segments`. An empty inner array means no transform for that segment.

```mlir
aie.objectfifo.dma_endpoint @d(%tile) drains @P {
  segments   = array<i32: 0, 1>,
  dimensions = #aie<bd_dim_layout_array_array[
    [<size = 4, stride = 4>, <size = 4, stride = 1>],   // segment 0
    []]>                                               // segment 1, linear
}
```

Padding only appears on a draining endpoint, since padding is added as data goes
out on the stream.

#### Dangling endpoints

An end with no pool behind it is a `dangling_endpoint`. It holds a tile, a
direction and a port — enough for a flow to reach it — and nothing else:

```mlir
aie.objectfifo.dangling_endpoint @in(%shim) MM2S DMA  {fifoName = "of_in"}
aie.objectfifo.dangling_endpoint @s(%tile)  MM2S Core {channelIndex = 0 : i32}
```

A shim whose transfers the runtime sequence issues has nothing to pool, because
the addresses only arrive at dispatch; it lowers to an `aie.shim_dma_allocation`
recording the tile, direction and channel. A shim that registers external
buffers *does* have a pool and is an ordinary `dma_endpoint` with a real BD
chain.

### Locks

A segment's `produceLock` is taken by whoever fills it and its `consumeLock` by
whoever drains it:

- filling actor: acquire `produceLock`, release `consumeLock`
- draining actor: acquire `consumeLock`, release `produceLock`

With semaphore locks the value counts how many objects are claimed. With binary
locks one lock per buffer serves both directions, and the distinction is carried
in the lock value.

### Flows

```mlir
aie.objectfifo.flow from @d1 to [@d2, @d3]
```

Several destinations are a broadcast: one source channel feeding a multicast
route.

A flow marked `packet` becomes an `aie.packet_flow` and shares the stream
instead of reserving a circuit; the two kinds coexist in one device. The choice
sits on the flow, not on an endpoint, because a packet route is one id agreed by
the source and every destination — per-endpoint marks could disagree.
`packet_id` pins that id, otherwise allocation picks the lowest one no other flow
uses.

```mlir
aie.objectfifo.flow from @d1 to [@d2] {packet, packet_id = 7 : i8}
```

At the frontend this is chosen per fifo, with the same two attributes on
`aie.objectfifo`, or `ObjectFifo(..., packet=True, packet_id=7)` in IRON.

## Shapes

### Through DMAs

```mlir
aie.objectfifo.pool @producer_pool(%t02) ...
aie.objectfifo.core_endpoint @prod_core(%t02) fills  @producer_pool
aie.objectfifo.dma_endpoint  @prod_dma(%t02)  drains @producer_pool

aie.objectfifo.pool @consumer_pool(%t25) ...
aie.objectfifo.dma_endpoint  @cons_dma(%t25)  fills  @consumer_pool
aie.objectfifo.core_endpoint @cons_core(%t25) drains @consumer_pool

aie.objectfifo.flow from @prod_dma to [@cons_dma]
```

Two pools, one per tile, joined by a stream.

### No DMAs at all

A fifo used within one core, or across neighboring cores that share memory, is
one pool with two core endpoints — no flow and no BD chains:

```mlir
aie.objectfifo.pool @smem(%t02) {depth = 4 : i32,
    segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
aie.objectfifo.core_endpoint @prod(%t02) fills  @smem
aie.objectfifo.core_endpoint @cons(%t03) drains @smem
```

### Passthrough

A one-to-one `aie.objectfifo.link` is a single pool with a DMA filling it and
another draining it:

```mlir
aie.objectfifo.pool @link_pool(%t21) ...
aie.objectfifo.dma_endpoint @link_in(%t21)  fills  @link_pool
aie.objectfifo.dma_endpoint @link_out(%t21) drains @link_pool
```

The two ends may move different amounts per transfer. That sets the stream
granularity; it does not partition the pool.

### Join

```mlir
aie.objectfifo.pool @out_pool(%t21) {
  depth = 2 : i32,
  segments = [#aie.objectfifo_segment<offset = 0,  size = 16>,
              #aie.objectfifo_segment<offset = 16, size = 32>]} : memref<48xi32>

aie.objectfifo.dma_endpoint @fill_0(%t21)    fills  @out_pool {segments = array<i32: 0>}
aie.objectfifo.dma_endpoint @fill_1(%t21)    fills  @out_pool {segments = array<i32: 1>}
aie.objectfifo.dma_endpoint @drain_all(%t21) drains @out_pool {segments = array<i32: 0, 1>}
```

Two DMAs fill different segments of the same pool, each ordered by its own
locks, and one DMA drains the whole object. That is what joining the data means.
Distribute is the same picture reversed: one endpoint fills the whole object and
N endpoints each drain one segment.

A link point needs a memory module to hold the shared object, and a device with
counting locks to guard each slice independently. A MemTile or a compute tile
satisfies both; a shim tile does not.

## The pipeline

`--aie-objectFifo-stateful-transform` runs these in order. Its options:
`skip-verify=true` drops step 2, and `packet-sw-objFifos=true` makes every flow
packet-switched.

| Pass | Emits | Consumes |
| --- | --- | --- |
| `--aie-objectfifo-split` | pools, endpoints, flows | `aie.objectfifo`, `aie.objectfifo.link` |
| `--aie-objectfifo-verify` | diagnostics only | — |
| `--aie-objectfifo-allocate` | `aie.buffer`, `aie.lock`, channel indices, `aie.flow` / `aie.packet_flow`, `aie.shim_dma_allocation` | `aie.objectfifo.flow` |
| `--aie-objectfifo-lower-dmas` | BD chains | `dma_endpoint`, `dangling_endpoint` |
| `--aie-objectfifo-lower-cores` | `use_lock` and buffer selection | `core_endpoint`, `acquire`, `release` |
| `--aie-objectfifo-erase-pools` | — | unreferenced pools |

Each endpoint op has exactly one consuming pass, so `lower-dmas` and
`lower-cores` are order-independent. Pools outlive the endpoints that named
them, so the record of which buffers and locks belong to which fifo survives
lowering; `erase-pools` drops it for consumers that do not want it.

### 1. `--aie-objectfifo-split`

Splits `aie.objectfifo` and `aie.objectfifo.link` into pools, endpoints and
flows, deciding which tiles hold objects and which ends need DMAs.

```mlir
// before
aie.objectfifo @of1 (%tile02, {%tile25}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
%e = aie.objectfifo.acquire @of1(Produce) : memref<16xi32>

// after
aie.objectfifo.pool @of1_pool(%tile_0_2) {depth = 2 : i32, fifoName = "of1",
    segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
aie.objectfifo.core_endpoint @of1_prod(%tile_0_2) fills @of1_pool
aie.objectfifo.dma_endpoint @of1_prod_dma(%tile_0_2) drains @of1_pool {fifoName = "of1"}
aie.objectfifo.pool @of1_cons_pool(%tile_2_5) {depth = 2 : i32, fifoName = "of1",
    segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
aie.objectfifo.core_endpoint @of1_cons(%tile_2_5) drains @of1_cons_pool
aie.objectfifo.dma_endpoint @of1_cons_dma(%tile_2_5) fills @of1_cons_pool {fifoName = "of1"}
aie.objectfifo.flow from @of1_prod_dma to [@of1_cons_dma]

%e = aie.objectfifo.acquire @of1_prod : memref<16xi32>
```

A core's `acquire` and `release` name the endpoint they work through and carry
no `Produce`/`Consume` port. The endpoint's role already says which end, so the
two cannot disagree.

This pass also picks pool depth: enough to cover the largest acquire a core on
that tile makes, plus one so the core can hold an object while the next arrives.
A tile no core touches gets the fifo's declared size.

### 2. `--aie-objectfifo-verify`

Checks completeness (listed below). It is a separate pass so incomplete designs
stay legal: a missing endpoint says nothing about whether a BD-level actor
exists elsewhere.

### 3. `--aie-objectfifo-allocate`

Gives pools their buffers and locks, and endpoints their channels, flows and
shim allocations.

```mlir
%of1_buff_0 = aie.buffer(%tile_0_2) {sym_name = "of1_buff_0"} : memref<16xi32>
%of1_buff_1 = aie.buffer(%tile_0_2) {sym_name = "of1_buff_1"} : memref<16xi32>
%of1_prod_lock_0 = aie.lock(%tile_0_2) {init = 2 : i32, sym_name = "of1_prod_lock_0"}
%of1_cons_lock_0 = aie.lock(%tile_0_2) {init = 0 : i32, sym_name = "of1_cons_lock_0"}

aie.objectfifo.pool @of1_pool(%tile_0_2) {buffers = [@of1_buff_0, @of1_buff_1],
    depth = 2 : i32, fifoName = "of1",
    segments = [#aie.objectfifo_segment<offset = 0, size = 16,
                 produceLock = @of1_prod_lock_0, consumeLock = @of1_cons_lock_0>]} : memref<16xi32>
aie.objectfifo.dma_endpoint @of1_prod_dma(%tile_0_2) drains @of1_pool {channelIndex = 0 : i32, fifoName = "of1"}
aie.flow(%tile_0_2, DMA : 0, %tile_2_5, DMA : 0)
```

Anything already written down is left alone: hand-placed buffers, locks already
on a segment, a pinned channel.

### 4. `--aie-objectfifo-lower-dmas`

Turns each `dma_endpoint` into the BD chain that walks its pool's buffers. One
rule, buffer-major and segment-minor:

```
for each buffer b in pool.buffers:
  for each segment s selected by the endpoint:
    acquire( drains ? s.consumeLock : s.produceLock )
    dma_bd(b, s.offset, s.size)
    release( drains ? s.produceLock : s.consumeLock )
```

```mlir
%mem_0_2 = aie.mem(%tile_0_2) {
  %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
^bb1:
  aie.use_lock(%of1_cons_lock_0, AcquireGreaterEqual, %c1_i32)
  aie.dma_bd(%of1_buff_0 : memref<16xi32> offset = 0 len = 16)
  aie.use_lock(%of1_prod_lock_0, Release, %c1_i32_0)
  aie.next_bd ^bb2
^bb2:   // the same, with %of1_buff_1
  ...
```

A join's draining endpoint selects every segment and so emits `depth × segments`
descriptors; each filling endpoint selects one and emits `depth`.

### 5. `--aie-objectfifo-lower-cores`

Turns `acquire` and `release` into `use_lock` plus a rotating buffer selection.

```mlir
%0 = arith.subi %c1_i32, %c0_i32 : i32   // acquire(N) is absolute:
%1 = arith.maxsi %0, %c0_i32_0 : i32     // delta = max(N - held, 0)
aie.use_lock(%of1_prod_lock_0, AcquireGreaterEqual, %1)
%4 = scf.index_switch %3 -> memref<16xi32>
case 0 { scf.yield %of1_buff_0 : memref<16xi32> }
case 1 { scf.yield %of1_buff_1 : memref<16xi32> }
```

The `index_switch` folds to a concrete buffer once the loops unroll.

A core sees one memref, so a core endpoint's segments must be a contiguous run
of the object. It is handed the buffer itself when that run is the whole object
and a `memref.subview` otherwise. An acquire emits one `AcquireGreaterEqual` per
selected segment, all with the same delta.

### 6. `--aie-objectfifo-erase-pools`

Drops pool metadata once nothing refers to it. Optional: knowing which buffers
and locks belong to which fifo is often worth keeping, and a pool still named by
something else — a re-arm binding in a runtime sequence, say — stays.

## Entering the pipeline partway

Every pass respects what it finds, which is what makes partial designs work.
Implementing one half of a fifo by hand at BD level and letting the other half
be generated is a supported shape, not a degenerate one:

- a pool that already names buffers and locks keeps them
- an endpoint that already names a channel keeps it
- a channel a hand-written `aie.dma_start` already programs is not programmed
  again — that is an error, not a silent overwrite
- a segment whose filler or drainer lives elsewhere simply has no endpoint here

Some things this enables:

**Substitute your own DMA program.** Write the `aie.mem` block yourself and leave
out the `dma_endpoint`. The pool, its locks and the other endpoints are still
generated, and your BD chain uses the same locks.

**Pin placement or channels.** Write the `aie.buffer` ops and name them in the
pool, or set `channelIndex` on an endpoint. Allocation fills in only what is
missing.

**Start from pools instead of `aie.objectfifo`.** Skip `split` and write pools,
endpoints and flows directly. This is the level at which shapes the frontend
cannot express — an unusual segment partition, a mixed core and DMA join —
become writable.

**Keep the frontend but change one decision.** Run the passes one at a time,
edit the IR in between, and carry on.

Because each pass erases what it consumes, adding a new `aie.objectfifo` to
already-lowered IR and re-running the whole pipeline lowers only the new one and
leaves everything else untouched.

## What is checked

### By the op verifiers

True of every well-formed module.

Pools:

- segments are in increasing offset order, and there is at least one
- `buffers` and `locks`, when present, have `depth` entries
- the lock kind matches the device: binary locks in `locks`, counting locks on segments
- more than one segment requires semaphore locks

Endpoints:

- segment indices are in range and strictly increasing, and there is at least one
- omitted `segments` is only valid on a single-segment pool
- the named pool exists
- a core endpoint's segments are contiguous
- `dimensions` and `padDimensions` have one entry per selected segment, with matching ranks, and stay within segment bounds
- `padDimensions` appears only on a draining endpoint, and a nonzero `padValue` needs one
- `iter_count` is only supported on a MemTile
- a dangling DMA or PLIO end is on a shim tile, and its bundle is DMA, PLIO or Core

Flows and core accesses:

- a flow has at least one destination, and its source and destinations are endpoints
- `packet_id` is only meaningful on a packet flow
- `acquire` and `release` sit inside a core, and `acquire` takes at least one object of the pool's element type

### By `--aie-objectfifo-verify`

- a pool's segments do not overlap and cover the element type exactly
- each segment has one filling and one draining endpoint. A filler may be absent
  when lock initializers mark the objects as starting full, and a pool over
  external buffers has a host-side actor with no op
- every DMA and dangling endpoint appears in exactly one flow, since an endpoint
  drives one channel
- no loop body releases more than it acquires

### By `--aie-objectfifo-lower-dmas`

- a `dma_endpoint` whose channel an existing `aie.dma_start` already programs is
  an error

## What the IR cannot say

Some combinations the old representation allowed are gone:

- **A fifo with several producers and no consumer.** Every segment needs exactly
  one filler and one drainer.
- **A core `acquire` whose port disagrees with the end it is on.** The
  endpoint's role is the only place that is written down.
- **A DMA endpoint with no buffers behind it.** An end this compiler does not
  program is a `dangling_endpoint`; a `dma_endpoint` always names a pool and
  always lowers to a BD chain.
- **A subview of an acquired object.** `acquire` returns the objects asked for.
