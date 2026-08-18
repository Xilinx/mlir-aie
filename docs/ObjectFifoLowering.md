<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# ObjectFifo Lowering

An `aie.objectfifo` is a high-level description of a data movement: a rotating
set of objects, the tiles that produce and consume them, and the locks that
order the two. Turning it into hardware configuration means deciding a great
many smaller things -- where the buffers live, which locks guard them, which DMA
channels move them, which stream route connects those channels, and what buffer
descriptor chain each channel runs.

Those decisions are made by a pipeline of passes, each of which writes its
answers back into the IR. This page describes the operations that carry those
answers and the pass that produces each one.

## Why the state is in the IR

The lowering used to be a single pass. It did a lot of useful things, but it was
all or nothing: a design that needed one of its decisions made differently had
to abandon the whole thing and write buffers, locks, DMA programs and flows by
hand.

Because each decision now lands in the IR as an attribute or an operation, a
design can enter the pipeline at any point with hand-written IR, and every pass
respects what it finds. Specificity is optional but respected: write down a DMA
channel and it is honored, leave it out and one is chosen.

The same property keeps the passes honest. Each pass erases what it consumes, so
running the pipeline over its own output is a no-op -- anything a pass forgot to
record would show up as a leftover operation rather than as a silently lost side
table. That is tested directly.

## The operations

| Operation | What it holds |
| --- | --- |
| `aie.objectfifo.pool` | A rotating set of buffers, the segments they may be accessed in, and the locks guarding those segments. |
| `aie.objectfifo.core_endpoint` | What a core needs to fill or drain the next object in a pool. |
| `aie.objectfifo.dma_endpoint` | The same for a DMA channel. |
| `aie.objectfifo.dangling_endpoint` | One end of a stream this compiler does not program: a shim the host runtime drives, a PLIO boundary, or a core's raw stream port. |
| `aie.objectfifo.flow` | A circuit- or packet-switched connection from one draining endpoint to the filling endpoints it feeds. |

All of them are transient. They are introduced by `--aie-objectfifo-split` and
are gone by the end of the pipeline; what survives is ordinary AIE IR --
`aie.buffer`, `aie.lock`, `aie.flow`, `aie.mem` / `aie.memtile_dma` /
`aie.shim_dma`, and `aie.shim_dma_allocation`.

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
Buffers and segments are orthogonal axes: buffers are the rotation axis, while
segments partition each individual object, and every buffer carries every
segment. Segments are listed in increasing offset order, do not overlap, and
together cover the element type exactly.

An ordinary pool has one segment spanning the whole object. Only a joined or
distributed pool has more.

On devices with binary locks a pool carries `locks`, one per buffer, instead of
per-segment locks, and has a single implicit segment.

### Endpoints

```mlir
aie.objectfifo.core_endpoint @c(%tile) fills  @P
aie.objectfifo.core_endpoint @c(%tile) drains @P {segments = array<i32: 0>}
aie.objectfifo.dma_endpoint  @d(%tile) drains @P {channelIndex = 0 : i32}
```

`fills` and `drains` state what the actor does to the pool's buffers, and
`segments` selects which of the pool's segments it handles. Omitting it selects
segment zero and is only valid for a single-segment pool. The endpoint's own
tile is where the actor is, which for shared memory differs from the pool's
tile.

Every segment has exactly one filler and exactly one drainer. A pool whose
segments each have one core filling and one DMA draining is an ordinary fifo
end; a pool with several fillers is a join, and one with several drainers is a
distribute.

An end with no pool behind it is an `aie.objectfifo.dangling_endpoint`. It holds
a tile, a direction and a port, which is enough for a flow to reach it, and
nothing else:

```mlir
aie.objectfifo.dangling_endpoint @in(%shim) MM2S DMA  {fifoName = "in"}
aie.objectfifo.dangling_endpoint @out(%tile) S2MM Core {channelIndex = 0 : i32}
```

### Locks

A segment's `produceLock` is acquired by whoever fills it and its `consumeLock`
by whoever drains it:

- filling actor: acquire `produceLock`, release `consumeLock`
- draining actor: acquire `consumeLock`, release `produceLock`

On devices with semaphore locks the lock value carries how many objects are
claimed. On devices with binary locks a single lock per buffer serves both
directions, and the distinction is carried in the lock value.

### Flows

```mlir
aie.objectfifo.flow from @d1 to [@d2, @d3]
```

Several destinations are a broadcast: one source channel feeding a multicast
route. A flow marked `packet` becomes an `aie.packet_flow` instead, sharing the
stream rather than reserving a circuit, and the two kinds coexist in one device.
The choice sits on the flow because a packet route is one id agreed by the
source and every destination; per-endpoint marks could disagree with each other.

## Worked examples

### An objectFifo through DMAs

```mlir
aie.objectfifo.pool @producer_pool(%t12) ...
aie.objectfifo.core_endpoint @prod_core(%t12) fills  @producer_pool
aie.objectfifo.dma_endpoint  @prod_dma(%t12)  drains @producer_pool

aie.objectfifo.pool @consumer_pool(%t33) ...
aie.objectfifo.dma_endpoint  @cons_dma(%t33)  fills  @consumer_pool
aie.objectfifo.core_endpoint @cons_core(%t33) drains @consumer_pool

aie.objectfifo.flow from @prod_dma to [@cons_dma]
```

### A MemTile passthrough

What `aie.objectfifo.link` describes is one pool with a DMA filling it and
another draining it:

```mlir
aie.objectfifo.pool @memtile_pool(%t21) ...
aie.objectfifo.dma_endpoint @memtile_cons(%t21) fills  @memtile_pool
aie.objectfifo.dma_endpoint @memtile_prod(%t21) drains @memtile_pool
```

### An N-way join

```mlir
aie.objectfifo.pool @out_pool(%t21) {
  depth = 2 : i32,
  segments = [#aie.objectfifo_segment<offset = 0,  size = 16>,
              #aie.objectfifo_segment<offset = 16, size = 32>]} : memref<48xi32>

aie.objectfifo.dma_endpoint @fill_0(%t21)    fills  @out_pool {segments = array<i32: 0>}
aie.objectfifo.dma_endpoint @fill_1(%t21)    fills  @out_pool {segments = array<i32: 1>}
aie.objectfifo.dma_endpoint @drain_all(%t21) drains @out_pool {segments = array<i32: 0, 1>}
```

Two DMAs fill different segments of the same pool, each segment ordered by its
own locks, and one DMA drains the whole object -- which is what joining the data
means. Distribute is the same picture with the arrows reversed: one endpoint
fills the whole object and N endpoints each drain one segment.

### A connection with no DMAs

An objectFifo used within one core, or across neighboring cores that share
memory, is a pool with two core endpoints:

```mlir
aie.objectfifo.pool @smem(%t12) {depth = 4 : i32,
    segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
aie.objectfifo.core_endpoint @prod(%t12) fills  @smem
aie.objectfifo.core_endpoint @cons(%t12) drains @smem
```

Because a core endpoint selects segments the same way a DMA endpoint does, join
and distribute on cores fall out of the same model. A core sees one memref, so
its segments must be a contiguous run of the object, and it is handed a
`memref.subview` over that run.

## The passes

### 1. `--aie-objectfifo-split`

Splits `aie.objectfifo` and `aie.objectfifo.link` into pools, endpoints and
flows, deciding which tiles hold objects and which ends need DMAs.

```mlir
// before
aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
%elem = aie.objectfifo.acquire @of1(Produce) : memref<16xi32>

// after
aie.objectfifo.pool @of1_pool(%tile_1_2) {depth = 2 : i32, fifoName = "of1",
    segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
aie.objectfifo.core_endpoint @of1_prod(%tile_1_2) fills @of1_pool
aie.objectfifo.dma_endpoint  @of1_prod_dma(%tile_1_2) drains @of1_pool {fifoName = "of1"}
aie.objectfifo.pool @of1_cons_pool(%tile_3_3) ...
aie.objectfifo.core_endpoint @of1_cons(%tile_3_3) drains @of1_cons_pool
aie.objectfifo.dma_endpoint  @of1_cons_dma(%tile_3_3) fills @of1_cons_pool {fifoName = "of1"}
aie.objectfifo.flow from @of1_prod_dma to [@of1_cons_dma]

%elem = aie.objectfifo.acquire @of1_prod : memref<16xi32>
```

A core's `acquire` and `release` name the endpoint they work through, so they
carry no `Produce`/`Consume` port of their own: the endpoint's role already says
which end, and the two cannot disagree.

### 2. `--aie-objectfifo-verify`

Checks the completeness rules -- every segment has one filler and one drainer,
segments tile the object exactly, every endpoint is reached by exactly one flow,
and no loop body releases more than it acquires.

The composite `--aie-objectFifo-stateful-transform` pipeline runs this check
after splitting. A partial design or compatibility flow may pass
`skip-verify=true` and lower the explicit pool/endpoint IR without the
completeness check.

### 3. `--aie-objectfifo-allocate`

Gives pools their buffers and locks, and endpoints their DMA channels,
`aie.flow` / `aie.packet_flow` and shim allocations.

```mlir
%of1_buff_0 = aie.buffer(%tile_1_2) {sym_name = "of1_buff_0"} : memref<16xi32>
%of1_buff_1 = aie.buffer(%tile_1_2) {sym_name = "of1_buff_1"} : memref<16xi32>
%of1_prod_lock_0 = aie.lock(%tile_1_2) {init = 2 : i32, sym_name = "of1_prod_lock_0"}
%of1_cons_lock_0 = aie.lock(%tile_1_2) {init = 0 : i32, sym_name = "of1_cons_lock_0"}

aie.objectfifo.pool @of1_pool(%tile_1_2) {buffers = [@of1_buff_0, @of1_buff_1], depth = 2 : i32,
    segments = [#aie.objectfifo_segment<offset = 0, size = 16,
                 produceLock = @of1_prod_lock_0, consumeLock = @of1_cons_lock_0>]} : memref<16xi32>
aie.objectfifo.dma_endpoint @of1_prod_dma(%tile_1_2) drains @of1_pool {channelIndex = 0 : i32}
aie.flow(%tile_1_2, DMA : 0, %tile_3_3, DMA : 0)
```

Anything already written down is left alone: hand-placed buffers, locks already
attached to a segment, and a channel the design pinned.

### 4. `--aie-objectfifo-lower-dmas`

Turns each `dma_endpoint` into the buffer descriptor chain that walks its pool's
buffers. The rule is the same for every endpoint, buffer-major and
segment-minor:

```
for each buffer b in pool.buffers:
  for each segment s selected by the endpoint:
    acquire( drains ? s.consumeLock : s.produceLock )
    dma_bd(b, s.offset, s.size)
    release( drains ? s.produceLock : s.consumeLock )
```

```mlir
%mem_1_2 = aie.mem(%tile_1_2) {
  %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
^bb1:
  aie.use_lock(%of1_cons_lock_0, AcquireGreaterEqual, %c1_i32)
  aie.dma_bd(%of1_buff_0 : memref<16xi32> offset = 0 len = 16)
  aie.use_lock(%of1_prod_lock_0, Release, %c1_i32_0)
  aie.next_bd ^bb2
^bb2:  // same, with %of1_buff_1
```

A join's draining endpoint selects every segment and so emits
`depth x segments` descriptors; each filling endpoint selects one and emits
`depth`.

### 5. `--aie-objectfifo-lower-cores`

Turns `acquire` and `release` on cores into `use_lock` plus a rotating buffer
selection.

```mlir
%0 = arith.subi %c1_i32, %c0_i32 : i32      // acquire(N) is absolute:
%1 = arith.maxsi %0, %c0_i32_0 : i32        // delta = max(N - held, 0)
aie.use_lock(%of1_prod_lock_0, AcquireGreaterEqual, %1)
%4 = scf.index_switch %3 -> memref<16xi32>
case 0 { scf.yield %of1_buff_0 : memref<16xi32> }
case 1 { scf.yield %of1_buff_1 : memref<16xi32> }
```

The `index_switch` folds to a concrete buffer once the loops unroll.

### 6. `--aie-objectfifo-erase-pools`

Drops the pool metadata once nothing refers to it any more. This is optional:
the record of which buffers and locks belong to which objectFifo is often worth
keeping, and a pool still named by something else -- a re-arm binding in a
runtime sequence, say -- is left in place.

## What the IR makes unsayable

Some combinations the old representation allowed are no longer expressible:

- An objectFifo with several producers and no consumer. Every segment must have
  exactly one filler and one drainer, and `--aie-objectfifo-verify` says so.
- A core `acquire` whose port disagrees with the end it is on. The endpoint's
  role is the only place that is written.
- A DMA endpoint with no buffers behind it. An end this compiler does not
  program is a `dangling_endpoint`, and a `dma_endpoint` always names a pool and
  always lowers to a buffer descriptor chain.
