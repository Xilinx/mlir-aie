//===- 00-model.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DESIGN SKETCH -- not a lit test. The model the other files in this directory
// use. Read this first.
//
//===----------------------------------------------------------------------===//
//
// THE OPS
//
//   aie.objectfifo.pool           a rotating set of buffers, the sub-segments
//                                 they may be accessed in, and the locks
//                                 guarding those segments
//   aie.objectfifo.core_endpoint  everything a core needs to fill or drain the
//                                 next object in a pool
//   aie.objectfifo.dma_endpoint   the same for a DMA channel
//   aie.objectfifo.flow           a stream connection between two DMA endpoints
//
// All four are transient. They are introduced by --aie-objectfifo-split and are
// gone by the end of the pipeline; what survives is ordinary AIE IR --
// aie.buffer, aie.lock, aie.flow, aie.mem / aie.memtile_dma / aie.shim_dma, and
// aie.shim_dma_allocation.
//
//===----------------------------------------------------------------------===//
//
// POOL
//
//   aie.objectfifo.pool @P(%tile) {
//     depth    = 2 : i32,
//     buffers  = [@P_buff_0, @P_buff_1],
//     segments = [<produceLock = @p0, consumeLock = @c0, offset = 0,  size = 16>,
//                 <produceLock = @p1, consumeLock = @c1, offset = 16, size = 20>]
//   } : memref<36xi32>
//
// The tile is where the buffers live. `depth` is how many buffers rotate.
//
// SEGMENTS AND BUFFERS ARE ORTHOGONAL AXES. buffers is the rotation axis;
// segments partition each individual object, and every buffer carries every
// segment. offset and size are always written. Segments are listed in
// increasing offset order, are non-overlapping, and together cover the element
// type exactly.
//
// An ordinary pool has one segment spanning the whole object. Only a joined or
// distributed pool has more.
//
// On binary-lock devices a pool carries `locks = [...]`, one per buffer, instead
// of `segments`, and has a single implicit segment. See 06-aie1.mlir.
//
//===----------------------------------------------------------------------===//
//
// ENDPOINTS
//
//   aie.objectfifo.core_endpoint @c(%tile) fills  @P
//   aie.objectfifo.core_endpoint @c(%tile) drains @P {segments = [0]}
//   aie.objectfifo.dma_endpoint  @d(%tile) drains @P {channel = MM2S 0}
//
// `fills` and `drains` state what the actor does to the pool's buffers. The
// endpoint's own tile is where the actor is, which for shared memory differs
// from the pool's tile.
//
// `segments` selects which of the pool's segments this actor handles. Omitted
// means all of them.
//
// EVERY SEGMENT HAS EXACTLY ONE FILLER AND EXACTLY ONE DRAINER. A pool whose
// segments each have one core filling and one DMA draining is an ordinary fifo
// end; a pool with several fillers is a join; one with several drainers is a
// distribute.
//
// A dma_endpoint on a shim tile has no pool: there are no buffers and no locks
// at the shim/DDR boundary. It carries a channel, and lowers to an
// aie.shim_dma_allocation recording the tile, direction and channel for the
// runtime sequence.
//
//===----------------------------------------------------------------------===//
//
// LOCKS
//
//   produceLock   acquired by whoever FILLS the segment
//   consumeLock   acquired by whoever DRAINS it
//
// and the handshake is:
//
//   filling actor:   acquire produceLock, release consumeLock
//   draining actor:  acquire consumeLock, release produceLock
//
// On AIE2 the semaphore value carries how many objects are claimed. On AIE1 a
// single binary lock per buffer is used for both directions and the distinction
// is carried in the lock value: 1 for a fill-release or a drain-acquire, 0
// otherwise.
//
//===----------------------------------------------------------------------===//
//
// FLOW
//
//   aie.objectfifo.flow from @d1 to [@d2]
//
// Several destinations are a broadcast: one source channel feeding a multicast
// route. --aie-objectfifo-allocate turns a flow into a channel on each endpoint
// plus an aie.flow, and consumes it.
//
// A flow marked `packet` becomes an aie.packet_flow instead, sharing the stream
// rather than reserving a circuit; the two kinds coexist in one device. The
// choice sits on the flow because a packet route is one id agreed by the source
// and every destination -- per-endpoint marks could disagree with each other.
// `packet_id` pins that id; otherwise allocation picks the lowest one no other
// flow uses. The source endpoint's `packet` attribute is the consequence, not
// the request: it is the header its buffer descriptors stamp, written by
// allocation.
//
//===----------------------------------------------------------------------===//
//
// BD EMISSION
//
// One rule for every DMA endpoint:
//
//     for each buffer b in pool.buffers:
//       for each segment s selected by the endpoint:
//         acquire( drains ? s.consumeLock : s.produceLock )
//         dma_bd(b, s.offset, s.size)
//         release( drains ? s.produceLock : s.consumeLock )
//
// buffer-major, segment-minor. A join's draining endpoint selects every segment
// and so emits depth x segments BDs; each filling endpoint selects one and emits
// depth. See 03-join.mlir.
//
//===----------------------------------------------------------------------===//
//
// HOW A CORE SEES SEGMENTS
//
// A DMA takes segments one at a time, each BD an independent transfer that the
// stream reassembles by concatenation. A core wants one memref, so:
//
//   acquire EVERY selected segment's lock, then hand over the memref spanning
//   their union
//
// which gives three shapes:
//
//   one segment spanning the object   -> the buffer itself
//   a run of segments short of it     -> a memref.subview over that run
//   every segment of a partitioned object -> the buffer itself, after N acquires
//
// A core endpoint's selection is fixed, so every acquire on it yields the same
// shape.
//
// Bookkeeping is one held counter and one buffer-index counter per endpoint. An
// acquire emits one AcquireGreaterEqual per selected segment, all with the same
// delta.
//
//===----------------------------------------------------------------------===//
//
// PIPELINE                        emits                      consumes
//
//   --aie-objectfifo-verify       (diagnostics only)
//   --aie-objectfifo-split        pools, endpoints, flows    objectfifo, link
//   --aie-objectfifo-allocate     buffers, locks, channels,  flow
//                                 aie.flow, shim allocations
//   --aie-objectfifo-lower-dmas   BD chains                  dma_endpoint
//   --aie-objectfifo-lower-cores  lock/buffer accesses       core_endpoint,
//                                                            acquire, release
//   --aie-objectfifo-erase-pools  (nothing)                  unreferenced pools
//   --aie-objectfifo-unroll       (existing pass)
//
// Pools outlive the endpoints that named them, so the annotation of which
// buffers and locks belong to which fifo survives lowering. --aie-objectfifo-
// erase-pools drops it for consumers that have no use for it. Since each
// endpoint op has exactly one consuming pass, lower-dmas and lower-cores are
// order-independent.
//
// Every pass is idempotent, because each erases what it consumes. Adding a new
// aie.objectfifo to already-lowered IR and re-running the pipeline lowers it
// without disturbing anything already there.
//
// Entering mid-pipeline is supported throughout. A pool that already names
// buffers and locks is left alone by allocate; an endpoint that already names a
// channel keeps it. Hand-written BD programs and hand-written pools compose with
// generated ones; see PARTIAL DESIGNS below.
//
//===----------------------------------------------------------------------===//
//
// PARTIAL DESIGNS
//
// A design may implement one half of a fifo by hand, at BD level, and have the
// other half generated. Writing the producer side as an objectFifo and the
// consumer side as an aie.mem block of one's own is a supported shape, not a
// degenerate one, and it is what the pass boundaries are arranged to permit:
//
//   - a pool that already names buffers and locks keeps them
//   - an endpoint that already names a channel keeps it
//   - a channel already programmed by a hand-written aie.dma_start is not
//     programmed again
//   - a segment whose filler or drainer is implemented elsewhere has no endpoint
//     for it here
//
// The last of these is why completeness is checked by a pass rather than by an
// op verifier: absence of an endpoint carries no information about whether the
// actor exists.
//
//===----------------------------------------------------------------------===//
//
// STRUCTURAL VERIFICATION
//
// Checked by the op verifiers, and therefore true of every well-formed module:
//
//  - a pool's segments are in increasing offset order
//  - every segment states offset and size
//  - a pool's locks match the device's lock kind: binary locks in `locks`,
//    counting locks on the segments
//  - an endpoint's segment indices are in range for its pool
//  - an endpoint's tile is the pool's tile or shares a memory module with it
//  - a pool with more than one segment requires semaphore locks
//  - a flow connects DMA endpoints; a dma_endpoint appears in at most one flow
//  - across a flow, the source's per-object size is an integer multiple of each
//    destination's
//  - a dma_endpoint without a pool is on a shim tile
//  - a core's acquire names a core_endpoint on its own tile
//
//===----------------------------------------------------------------------===//
//
// COMPLETENESS VERIFICATION
//
// Checked by --aie-objectfifo-verify, which a partial design may decline to run:
//
//  - a pool's segments do not overlap and cover the element type exactly
//  - each of a pool's segments has one filling endpoint and one draining
//    endpoint. A filler may be absent when the lock initializers mark the
//    objects as starting full
//  - every flow reaches a destination
//
// Checked by --aie-objectfifo-lower-dmas, where it is an error:
//
//  - a dma_endpoint whose channel is already programmed by an existing
//    aie.dma_start
//
//===----------------------------------------------------------------------===//
//
// IMPLEMENTATION NOTES
//
// 1. A core sees one memref, so a core endpoint's segments must be a single
//    run of the object. The subview it is handed carries the run's offset in
//    its layout, giving a type like memref<16xi32, strided<[1], offset: 16>>.
//
// 2. A multi-segment core acquire takes one lock per segment before the object
//    may be touched. The delta computation is unchanged; it is applied to each.
//
// 3. Pre-filled fifos desugar during split: the initial contents become
//    initializers on the aie.buffer ops, and "N objects start full" becomes the
//    pool's lock initializers.
//
//===----------------------------------------------------------------------===//
//
// OPEN WORK
//
//  - tests for join and distribute with core participants: over shared memory
//    with no DMA at all, over DMAs where the many side is a core, and with one
//    core taking several parts
//
//===----------------------------------------------------------------------===//
