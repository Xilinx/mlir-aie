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

`--aie-objectFifo-stateful-transform` is registered as a pass pipeline from
commit 1 so existing RUN lines keep working while the passes are peeled off. Its
options (`dynamic-objFifos`, `packet-sw-objFifos`) forward to the members.

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

The stateful transform keeps every remaining phase but is re-anchored — it walks
pools and endpoints, and `ObjectFifoState`'s maps key on `ObjectFifoPoolOp`
instead of `ObjectFifoCreateOp`. It still creates buffers, locks, channels, DMAs
and core code exactly as now.

Also here: the `@fifo` → shim symbol rewrite, since split is where the shim-side
endpoint gets its identity.

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

Nothing is left in it. `tools/aiecc` runs the pipeline; the pipeline
registration is either kept as a convenience alias or removed and test RUN lines
updated.

Residual phases still to place before this commit can land: the dead-op cleanup
and `computeTopologicalSorting`, which currently run at the end of the monolith.

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
