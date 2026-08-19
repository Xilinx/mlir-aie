<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# ObjectFifo Lowering

An `aie.objectfifo` enables data movement at object-size granularity in a
first-in-first-out manner, either between two tiles (1:1), as a broadcast (1:N),
or, using `aie.objectfifo.link`, as join and distribute patterns (N:M). The
ObjectFifo provides buffering, and users may acquire and hold multiple objects
at a time (e.g. sliding windows). The current implementation provides
buffering using a rotating set of `aie.buffer`s, synchronization between
producers and consumers using `aie.lock`s, routing using `aie.flow` or
`aie.packet_flow`, movement of data either using shared memory or DMA programs
(`aie.mem`, `aie.memtile_dma` and `aie.shim_dma`), and object access on cores.

This page documents how we arrive at a concrete lowering for a given high-level
ObjectFifo in multiple passes. Users requiring lower-level control may find the
intermediate levels useful.

## Frontend operations

| Operation | What it holds |
| --- | --- |
| `aie.objectfifo` | The whole data movement: element type, depth, producer tile and consumer tiles. |
| `aie.objectfifo.link` | Two or more fifos meeting on one tile, optionally with the offsets that make it a join or a distribute. |
| `aie.objectfifo.acquire` / `.release` | A core taking objects out of a fifo and giving them back. |
| `aie.objectfifo.allocate` | A delegate tile whose memory holds the objects. |
| `aie.objectfifo.register_external_buffers` | DDR buffers backing a shim end. |

## Operations after splitting

`--aie-objectfifo-split` replaces the frontend operations with these:

| Operation | What it holds |
| --- | --- |
| `aie.objectfifo.pool` | A set of buffers and the contract for accessing them: the segments they may be accessed in, and the locks to acquire and release. |
| `aie.objectfifo.core_endpoint` | What a core needs to fill or drain the next object in a pool. |
| `aie.objectfifo.dma_endpoint` | The same for a DMA channel. |
| `aie.objectfifo.dangling_endpoint` | One end of a stream this compiler does not program: a shim the host runtime drives, a PLIO boundary, or a core's raw stream port. |
| `aie.objectfifo.flow` | A circuit- or packet-switched connection from one draining endpoint to the filling endpoints it feeds. |

These are transient. By the end of the pipeline what remains is `aie.buffer`,
`aie.lock`, `aie.flow`, `aie.mem` / `aie.memtile_dma` / `aie.shim_dma`, and
`aie.shim_dma_allocation`.

### Pools

A pool is a set of buffers and the contract for accessing them.

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
buffer carries every segment. Segments are listed in increasing offset order,
do not overlap, and together cover the element type exactly.

An ordinary pool has one segment spanning the whole object. A joined or
distributed pool has one per participant.

On devices with binary locks a pool carries `locks`, one per buffer, and has a
single implicit segment.

### The access contract

A segment's `produceLock` is taken by whoever fills it and its `consumeLock` by
whoever drains it:

- filling actor: acquire `produceLock`, release `consumeLock`
- draining actor: acquire `consumeLock`, release `produceLock`

With semaphore locks the value counts how many objects are claimed. With binary
locks one lock per buffer serves both directions, and the direction is carried
in the lock value.

### Endpoints

An endpoint is an actor: something that fills or drains a pool.

```mlir
aie.objectfifo.core_endpoint @c(%tile) fills  @P
aie.objectfifo.dma_endpoint  @d(%tile) drains @P {channelIndex = 0 : i32}
```

`fills` and `drains` say what the actor does to the pool's buffers. The
endpoint's own tile is where the actor is, which for shared memory differs from
the pool's tile.

**Every segment has exactly one filler and exactly one drainer.** A pool with
several fillers is a join; one with several drainers is a distribute.

#### Selecting segments

`segments` picks which of the pool's segments an actor handles. Indices are
strictly increasing. Omitting the attribute selects segment zero, and is valid
on a single-segment pool.

```mlir
aie.objectfifo.dma_endpoint @d(%tile) fills @P {segments = array<i32: 0, 1>}
```

#### Per-segment transforms

A DMA endpoint may carry `dimensions` (a strided access pattern) and
`padDimensions`. Both are **positional**: entry N describes the segment named by
`segments[N]`, and each must have exactly as many entries as `segments`. An
empty inner array means no transform for that segment.

```mlir
aie.objectfifo.dma_endpoint @d(%tile) drains @P {
  segments   = array<i32: 0, 1>,
  dimensions = #aie<bd_dim_layout_array_array[
    [<size = 4, stride = 4>, <size = 4, stride = 1>],   // segment 0
    []]>                                               // segment 1, linear
}
```

Padding applies on a draining endpoint, where data goes out on the stream.

#### Dangling endpoints

An end with no pool behind it is a `dangling_endpoint`. It holds a tile, a
direction and a port, which is what a flow needs to reach it:

```mlir
aie.objectfifo.dangling_endpoint @in(%shim) MM2S DMA  {fifoName = "of_in"}
aie.objectfifo.dangling_endpoint @s(%tile)  MM2S Core {channelIndex = 0 : i32}
```

A shim whose transfers the runtime sequence issues has nothing to pool, since
the addresses arrive at dispatch; it lowers to an `aie.shim_dma_allocation`
recording the tile, direction and channel. A shim that registers external
buffers has a pool and is an ordinary `dma_endpoint` with a BD chain.

### Flows

```mlir
aie.objectfifo.flow from @d1 to [@d2, @d3]
```

Several destinations are a broadcast: one source channel feeding a multicast
route.

A flow marked `packet` becomes an `aie.packet_flow` and shares the stream;
circuit flows reserve theirs. Both kinds coexist in one device. `packet_id`
pins the id; otherwise allocation picks the lowest id no other flow uses.

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

### Shared memory

A fifo used within one core, or across neighboring cores that share memory, is
one pool with two core endpoints:

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

The two ends may move different amounts per transfer, which sets the stream
granularity. The pool keeps one segment.

### Join and distribute

```mlir
aie.objectfifo.pool @out_pool(%t21) {
  depth = 2 : i32,
  segments = [#aie.objectfifo_segment<offset = 0,  size = 16>,
              #aie.objectfifo_segment<offset = 16, size = 32>]} : memref<48xi32>

aie.objectfifo.dma_endpoint @fill_0(%t21)    fills  @out_pool {segments = array<i32: 0>}
aie.objectfifo.dma_endpoint @fill_1(%t21)    fills  @out_pool {segments = array<i32: 1>}
aie.objectfifo.dma_endpoint @drain_all(%t21) drains @out_pool {segments = array<i32: 0, 1>}
```

Two DMAs fill different segments of one pool, each ordered by its own locks, and
one DMA drains the whole object. Distribute is the same picture reversed: one
endpoint fills the whole object and N endpoints each drain one segment.

A link point needs a memory module for the shared object and a device with
counting locks: a MemTile or a compute tile.

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

Each pass replaces the operations it consumes, so the pipeline is idempotent:
lower, edit the result, and re-run. Adding a new `aie.objectfifo` to lowered IR
and re-running lowers the new one and leaves the rest alone.

Each endpoint has exactly one consuming pass, so `lower-dmas` and `lower-cores`
are order-independent.

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

A core's `acquire` and `release` name the endpoint they work through; the
endpoint's role gives the direction.

Pool depth covers the largest acquire a core on that tile makes, plus one so the
core can hold an object while the next arrives. A tile no core touches gets the
fifo's declared size.

Writing pools, endpoints and flows by hand and starting the pipeline at step 3
gives access to shapes beyond what the frontend expresses, such as a custom
segment partition.

### 2. `--aie-objectfifo-verify`

Checks the completeness rules listed below. Designs that supply an endpoint by
hand at BD level lower with `skip-verify=true`.

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

Attributes already present are kept, so writing `aie.buffer` ops and naming them
in the pool, attaching locks to a segment, or setting `channelIndex` on an
endpoint overrides that choice; this pass fills in the rest.

### 4. `--aie-objectfifo-lower-dmas`

Turns each `dma_endpoint` into the BD chain that walks its pool's buffers,
buffer-major and segment-minor:

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

A join's draining endpoint selects every segment and emits `depth × segments`
descriptors; each filling endpoint selects one and emits `depth`.

To supply your own BD chain, write the `aie.mem` block and omit the
`dma_endpoint`; the pool and its locks are still generated, and your chain uses
those locks. A channel an existing `aie.dma_start` programs is an error here.

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

A core sees one memref, so a core endpoint's segments are a contiguous run of
the object. It is handed the buffer itself when that run is the whole object,
and a `memref.subview` for a shorter run. An acquire emits one
`AcquireGreaterEqual` per selected segment, all with the same delta.

### 6. `--aie-objectfifo-erase-pools`

Drops pool metadata once nothing refers to it. A pool still named by something
else, such as a re-arm binding in a runtime sequence, stays.

## What is checked

### By the op verifiers

Pools:

- segments are in increasing offset order, and there is at least one
- `buffers` and `locks`, when present, have `depth` entries
- the lock kind matches the device: binary locks in `locks`, counting locks on segments
- more than one segment requires semaphore locks

Endpoints:

- the named pool exists
- segment indices are in range and strictly increasing, and there is at least one
- omitting `segments` is valid on a single-segment pool
- a core endpoint's segments are contiguous
- `dimensions` and `padDimensions` have one entry per selected segment, with matching ranks, within segment bounds
- `padDimensions` appears on a draining endpoint, and a nonzero `padValue` requires one
- `iter_count` is supported on a MemTile
- a dangling DMA or PLIO end is on a shim tile, and its bundle is DMA, PLIO or Core

Flows and core accesses:

- a flow has at least one destination, and its source and destinations are endpoints
- `packet_id` requires a packet flow
- `acquire` and `release` sit inside a core, and `acquire` takes at least one object of the pool's element type

### By `--aie-objectfifo-verify`

- a pool's segments do not overlap and cover the element type exactly
- each segment has one filling and one draining endpoint. A filler may be absent
  when lock initializers mark the objects as starting full, and a pool over
  external buffers has a host-side actor with no op
- every DMA and dangling endpoint appears in exactly one flow
- no loop body releases more than it acquires

### By `--aie-objectfifo-lower-dmas`

- a `dma_endpoint` whose channel an existing `aie.dma_start` already programs
