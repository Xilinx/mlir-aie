<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# ObjectFifo lowering refactor — implementation plan

Target IR at each stage is documented in `docs/design/objfifo-stages/`. Read
`00-model.mlir` first; this file covers only the order of work.

## Gate applied to every commit

- `test/objectFifo-stateful-transform` stays green: 159 pass, 1 XFAIL.
- Existing tests are unchanged, or changed only mechanically (RUN lines, op
  spellings). A behavioural diff in an expected output means the commit is wrong.
- New behaviour ships with new tests in the same commit.
- Comments follow `docs/skills` / the code-commenting skill: no history, no
  before/after framing, no restating the adjacent line.

### Idempotency

Each pass erases what it consumes, so running the pipeline over its own output
changes nothing. Every commit adds a test asserting it for the passes it owns:

```
// RUN: aie-opt --aie-objectfifo-<pass> %s -o %t1.mlir
// RUN: aie-opt --aie-objectfifo-<pass> %t1.mlir -o %t2.mlir
// RUN: diff %t1.mlir %t2.mlir
```

and one covering the whole pipeline. A second test appends a fresh
`aie.objectfifo` to already-lowered IR and checks that re-running lowers it while
leaving everything already lowered byte-identical.

These are what keep the design honest: an op left behind by the pass that should
have consumed it shows up as a diff rather than as silent growth.

## How the two pipelines coexist

The monolith consumes `aie.objectfifo`; split destroys it. The two cannot be
chained, so the new passes are built as a second, complete pipeline rather than
by hollowing the monolith out from the inside. Commits 1–5 add one pass each,
tested on its own IR, while `--aie-objectFifo-stateful-transform` keeps running
unchanged. Commit 6 points that name at the new pipeline, ports the 160 existing
RUN lines, and deletes the monolith.

This means the gate below is trivially met for commits 1–5 and carries no
information until commit 6, which is where the two pipelines are compared
against the same expected output. The compensating discipline is that every pass
ships with tests covering the patterns it handles, written against the design
sketches rather than against whatever the monolith happens to emit.

## Source map

Everything below moves out of
`lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp` (~2900 lines).
`AIEObjectFifoUnroll.cpp` is untouched throughout.

---

## 1. `--aie-objectfifo-split`

New ops in `include/aie/Dialect/AIE/IR/AIEOps.td`: `ObjectFifoPoolOp`,
`ObjectFifoCoreEndpointOp`, `ObjectFifoDmaEndpointOp`, `ObjectFifoFlowOp`, plus a
segment attribute. New pass `AIEObjectFifoSplit.cpp`.

Moves: the split phase (`runOnOperation` §"Split objectFifos into a consumer end
and producer end"), `requiresDMAs`, `isSharedMemory`, `getOptionalLinkOp`, and
the buffer-ownership ladder in `createObjectFifoElements` that decides which fifo
of a link owns the elements.

`splitBecauseLink`, `objFifoLinks` and `linkTarget` disappear: pool identity
answers what all three asked.

Also here: the `@fifo` → shim symbol rewrite, since split is where the shim-side
endpoint gets its identity. The endpoint keeps the fifo's name in a `fifoName`
attribute so allocate can emit an `aie.shim_dma_allocation` under the name the
runtime sequence expects.

Segment coverage — segments tiling the element type exactly — is a completeness
rule, so it belongs in the verify pass, not the pool's own verifier. The op
verifier checks only ordering. `getDistributeTransferLengths` mixes element and
byte units for ND distributes (`nd_dma_distribute_AIE2.mlir`), which is what
makes the distinction matter in practice.

New tests: `test/objectFifo-split/` — pool and endpoint IR for the seven patterns
in the design sketches.

## 2. `--aie-objectfifo-verify`

Runs **after** split. Analyses are per-pool rather than per-link, and a design
entering below `aie.objectfifo` is still checked. Diagnostics keep pointing at
the original fifos via location info threaded through split.

Moves: `verifyObjectFifoAccesses`, `verifyObjectFifoOverRelease`,
`verifyObjectFifoTilesArePlaced`.

`verifyObjectFifoLinks` does not move as-is — split consumes links, so its
"objectfifo cannot be in more than one ObjectFifoLinkOp" becomes an
`ObjectFifoLinkOp` verifier, and its intent is re-expressed as the per-segment
filler/drainer check.

Structural rules become op verifiers; completeness rules live in this pass so a
partial design can skip it. Both lists are in `00-model.mlir`.

Existing negative tests (`aie_stream/bad_*.mlir`, `dma_channel_alloc/*_bad.mlir`,
`data_movement_patterns/broadcast_error_test.mlir`) must emit the same
diagnostics.

## 3. `--aie-objectfifo-allocate`

Moves: `createObjectFifoElements`, `createObjectFifoLocks`,
`calculateCurrentUsedMemory`, `findOrCreateTile` and the MemTile largest-first
ordering and spillover; `DMAChannelAnalysis`, `assignDMAChannelIndices`,
`reservePinnedChannels`, `getStartPacketID`; flow and packet-flow emission;
`createObjectFifoAllocationInfo`; `detectExternalBuffers` / `addExternalBuffer`.

Writes buffers, locks, segments and channels onto pools and endpoints; emits
`aie.flow` and `aie.shim_dma_allocation`; consumes `aie.objectfifo.flow`.

Pools and endpoints that already carry resources are left as written — this is
the commit that makes hand-placed buffers and pinned channels work.

`joinDistribFactor` resolves here into one lock pair per segment.

The rearm-binding phase needs locks and channels, so it lands here; it reads them
off pools instead of `rearmChannelsPerFifo` / `locksPerFifo`.

New tests: hand-written pools and endpoints surviving allocation untouched.

## 4. `--aie-objectfifo-lower-dmas`

Moves: `createAIETileDMA`, `createShimDMA`, `createMemTileDMA`, `createBdBlock`,
`createBd`, collapsed into one emitter driven by the rule in `00-model.mlir`
(buffer-major, segment-minor, per DMA endpoint).

Deleted rather than moved: `extraOffset`, `joinDistribFactor`,
`joinDistribLockIndex`, the `isJoin`/`isDistribute` branches and the
`getJoinTransferLengths` / `getDistributeTransferLengths` calls — segments carry
all of it.

Erases `dma_endpoint`s. Errors when an endpoint's channel already has an
`aie.dma_start`.

Highest-risk commit: `repeat_count`, `iter_count`, `padDimensions`, `padValue`,
`disable_synchronization` and the `aie_stream` port cases all currently branch
inside these three functions and must survive the collapse. The
`repeat_count/`, `init_values/` and `aie_stream/` test directories are the ones
to watch.

## 5. `--aie-objectfifo-lower-cores`

Moves: `LoweringContext`, `LowerObjectFifoAcquire`, `LowerObjectFifoRelease`,
`LowerObjectFifoSubviewAccess`, `emitAdvanceBufferIndexCounter`,
`buildRotatingSwitch`, `getSemaphoreLockToUse`, `emitBinaryUseLocks`,
`emitCoreCounters`, `annotateUnrollHints`, `promoteBookkeepingSlots`.

The unroll hint and the alloca-to-SSA promotion belong to this pass: the hint
preserves fifo-level information the lowering discards, and the allocas exist
only to reach SSA form.

New behaviour, with tests:

- an acquire on an endpoint selecting several segments takes one lock per
  segment, all with the same delta, before the object is handed over;
- an endpoint whose segments are a strict slice of its pool's buffers receives a
  `memref.subview` at the segment offset;
- join and distribute with core participants: over shared memory with no DMA at
  all, over DMAs where the many side is a core, and with one core taking several
  parts.

Erases `core_endpoint`s, and pools once their last endpoint is gone.

## 6. Drop `--aie-objectFifo-stateful-transform`

The name becomes a pipeline over the five new passes, forwarding
`dynamic-objFifos` and `packet-sw-objFifos` to the members that read them, and
the monolith is deleted. The 160 existing tests move onto the new pipeline here;
their expected output is what proves the two agree.

The bulk erase and `computeTopologicalSorting` at the end of the monolith do not
move. Sorting exists only so that one sweep can erase a use-def chain — the
`ObjectFifoSubviewAccessOp` reading its `ObjectFifoAcquireOp` — users before
definitions. With each pass erasing what it consumes through rewrite patterns,
there is nothing left over to sweep, and commit 7 removes the chain outright. The
idempotency tests are what prove it.

## 7. Drop `aie.objectfifo.subview.access`

`aie.objectfifo.acquire` returns one memref result per acquired object.
`!aie.objectfifosubview`, `ObjectFifoSubviewAccessOp`,
`LowerObjectFifoSubviewAccess`, the "subview operand must be the direct result of
an aie.objectfifo.acquire" verifier and
`test/objectFifo-stateful-transform/subview_escape_via_iter_args.mlir` all go.

Mechanical but wide: most objectFifo tests use `subview.access`. Land it last so
it never blocks a behavioural commit.

---

## Ordering notes

- 2 depends only on 1. 3 depends on 1. 4 and 5 depend on 3 and are independent of
  each other, matching the pass ordering.
- 4 and 5 each erase only the endpoint kind they consume, so neither constrains
  the other's position in the pipeline.
- 7 is independent of 2–6 and can be pulled forward if it stops being convenient
  to carry.
