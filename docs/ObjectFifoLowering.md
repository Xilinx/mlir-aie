<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# ObjectFifo Lowering

An `aie.objectfifo` enables data movement at object-size granularity in a
first-in-first-out manner, either between two tiles, as a broadcast,
or, using `aie.objectfifo.link`, as join and distribute patterns. The
ObjectFifo provides buffering, and users may acquire and hold multiple objects
at a time (e.g., sliding windows). The current implementation provides
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
| `aie.objectfifo` | A data movement pipe between producer and consumer tiles, with the capacity to buffer `depth` objects of the given element type. |
| `aie.objectfifo.link` | A connection between two or more fifos meeting on one tile; multiple input or output fifos describe a join or distribute, respectively. |
| `aie.objectfifo.acquire` / `.release` | A core taking objects out of a fifo and giving them back. |
| `aie.objectfifo.allocate` | An (optional) manual allocation indicating on which tile the objects of the fifo should be held. |
| `aie.objectfifo.register_external_buffers` | DDR buffers backing a shim end. |

## Operations after splitting

`--aie-objectfifo-split` replaces the frontend operations with these:

| Operation | What it holds |
| --- | --- |
| `aie.objectfifo.pool` | A set of buffers and the rules for accessing them: how the buffers are sliced into segments, and which locks users must acquire/release to access the segments. |
| `aie.objectfifo.core_endpoint` | Information the `acquire` and `release` operations on a core need to fill or drain the next object in a pool. |
| `aie.objectfifo.dma_endpoint` | A DMA channel and a reference to a pool; lowers to a DMA program that fills or drains the buffer pool from/onto the given channel. |
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
do not overlap, and together cover the entire buffer.

An ordinary pool has one segment spanning the whole object. A joined or
distributed pool has one per participant.

On devices with binary locks (AIE1) a pool carries `locks`, one per buffer, and
has a single implicit segment. Join/distribute is not supported on AIE1.

### The access contract

On AIE2 (semaphore locks), users of a pool are expected to acquire locks as
follows before reading from or writing to buffers in the pool:

- filling actor: acquire `produceLock`, release `consumeLock`
- draining actor: acquire `consumeLock`, release `produceLock`

With semaphore locks the value counts how many objects are claimed.

AIE1 uses binary locks, which are supplied in the `locks` attribute. Producers
and consumers toggle the same lock with opposite values (0/1).

### Replication and iteration

`repeatCount` sits on the pool the data leaves from. It makes that pool's
draining end put each object on the stream that many times, and the filling end
covers a whole batch in one go: it takes and gives back `repeatCount` lock units
at a time, with the lock initializers scaled to match.

`iterCount` sits on each DMA endpoint and says how many passes its chain makes.
Both ends of a fifo carry `depth * repeat_count * iter_count` objects, so
`--aie-objectfifo-split` resolves the fifo's `iter_count` differently for each:

| endpoint | descriptors in its chain | passes it makes |
| --- | --- | --- |
| draining | `depth * repeat_count` | `iter_count` |
| filling | `depth` | `repeat_count * iter_count` |

A chain given an `iterCount` ends in `aie.end` and carries `iterCount - 1` on
its channel start queue. A chain without one loops back on itself and runs
until the design stops. Locks gate every descriptor either way, so a bounded
chain simply stops asking once it has made its passes.

### Endpoints

An endpoint is an actor: something that fills or drains a pool.

Endpoints lower to code (DMA program or core code) that accesses the buffers
in the referenced pool in sequence, following its expected synchronization
protocol (lock acquire/releases).

```mlir
aie.objectfifo.core_endpoint @c(%tile) fills  @P
aie.objectfifo.dma_endpoint  @d(%tile) drains @P {channelIndex = 0 : i32}
```

**Every segment has exactly one filler and exactly one drainer.** A pool with
several fillers is a join; one with several drainers is a distribute.

#### Selecting segments

`segments` picks which of the pool's segments an actor handles. Indices are
strictly increasing. Omitting the attribute selects segment zero, and is valid
on a single-segment pool.

```mlir
aie.objectfifo.dma_endpoint @d(%tile) fills @P {segments = array<i32: 0, 1>}
// Results in a DMA program that accesses the first two segments of buffers in @P.
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
    []]>                                                // segment 1, linear
}
```

Padding applies on a draining endpoint, where data goes out on the stream.

#### Dangling endpoints

The `dangling_endpoint` is an atypical endpoint that does not name a pool:
It is an escape hatch for endpoints that do not want to use the pool abstraction,
as a stand-in for the other end of a `objectfifo.flow`. It names only a DMA
channel on a tile, which a user can then program in other ways.

```mlir
aie.objectfifo.dangling_endpoint @in(%shim) MM2S DMA  {fifoName = "of_in"}
aie.objectfifo.dangling_endpoint @s(%tile)  MM2S Core {channelIndex = 0 : i32}
```

### Flows

Flows name two endpoints and lower to a circuit- or packet-switched route
between them.

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

## Examples

### Core-to-Core FIFO via DMAs

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

### Core-to-Core FIFO via Shared Memory

A fifo used within one core, or across neighboring cores that share memory, is
one pool with two core endpoints:

```mlir
aie.objectfifo.pool @smem(%t02) {depth = 4 : i32,
    segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
aie.objectfifo.core_endpoint @prod(%t02) fills  @smem
aie.objectfifo.core_endpoint @cons(%t03) drains @smem
```

### Buffered Pass-through via DMA

A one-to-one `aie.objectfifo.link` is a single pool with a DMA filling it and
another draining it:

```mlir
aie.objectfifo.pool @link_pool(%t21) ...
aie.objectfifo.dma_endpoint @link_in(%t21)  fills  @link_pool
aie.objectfifo.dma_endpoint @link_out(%t21) drains @link_pool
```

The two ends may move different amounts per transfer, which sets the stream
granularity. The pool keeps one segment.

### 2:1 Join via DMA

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

## The pipeline

`--aie-objectFifo-stateful-transform` runs these in order. Its options:
`skip-verify=true` drops step 2, and `packet-sw-objFifos=true` makes every flow
packet-switched.

| Pass | Emits | Consumes |
| --- | --- | --- |
| `--aie-objectfifo-split` | pools, endpoints, flows | `aie.objectfifo`, `aie.objectfifo.link` |
| `--aie-objectfifo-verify` | diagnostics only | — |
| `--aie-objectfifo-allocate` | `pool`s with buffer/lock attributes, `aie.buffer`, `aie.lock`, channel indices, `aie.flow` / `aie.packet_flow`, `aie.shim_dma_allocation` | `aie.objectfifo.flow`, `pool`s without buffer/lock attributes |
| `--aie-objectfifo-lower-dmas` | BD chains | `dma_endpoint`, `dangling_endpoint` |
| `--aie-objectfifo-lower-cores` | `use_lock` and buffer selection | `core_endpoint`, `acquire`, `release` |
| `--aie-objectfifo-erase-pools` | — | unreferenced pools |

Each pass replaces the operations it consumes, so the pipeline is idempotent:
lower, edit the result, and re-run. Adding a new `aie.objectfifo` to lowered IR
and re-running lowers the new one and leaves the rest alone.

### 1. `--aie-objectfifo-split`

Splits `aie.objectfifo` and `aie.objectfifo.link` into pools, endpoints and
flows, deciding which tiles hold objects and which ends need DMAs.

```mlir
// before
aie.objectfifo @of1 (%tile02, {%tile25}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
%e = aie.objectfifo.acquire @of1 (Produce, 1) : memref<16xi32>

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

%e = aie.objectfifo.acquire @of1_prod (1) : memref<16xi32>
```

A core's `acquire` and `release` name the endpoint they work through; the
endpoint's role gives the direction.

Pool depth covers the largest acquire a core on that tile makes, plus one so the
core can hold an object while the next arrives. A tile no core touches gets the
fifo's declared size.

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

aie.objectfifo.pool @of1_pool(%tile_0_2) {
    depth = 2 : i32, fifoName = "of1",
    buffers = [@of1_buff_0, @of1_buff_1],
    segments = [#aie.objectfifo_segment<offset = 0, size = 16,
                 produceLock = @of1_prod_lock_0, consumeLock = @of1_cons_lock_0>]} : memref<16xi32>
// Note the set buffer and lock attributes.
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

### 5. `--aie-objectfifo-lower-cores`

Turns `acquire` and `release` operations in AIE core code into `use_lock`
and a rotating buffer selection.

```mlir
%0 = arith.subi %c1_i32, %c0_i32 : i32   // acquire(N) is absolute:
%1 = arith.maxsi %0, %c0_i32_0 : i32     // delta = max(N - held, 0)
aie.use_lock(%of1_prod_lock_0, AcquireGreaterEqual, %1)
%4 = scf.index_switch %3 -> memref<16xi32>
case 0 { scf.yield %of1_buff_0 : memref<16xi32> }
case 1 { scf.yield %of1_buff_1 : memref<16xi32> }
```

The `index_switch` folds to a concrete buffer once the loops unroll.

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
