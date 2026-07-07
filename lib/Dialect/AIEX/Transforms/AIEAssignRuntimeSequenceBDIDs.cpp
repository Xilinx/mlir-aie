//===- AIEAssignRuntimeSequenceBDIDs.cpp ------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Assigns buffer-descriptor (BD) IDs to the DMA tasks configured in a runtime
// sequence. Control-flow-aware: tasks inside scf.for / scf.if are allocated
// against a real per-tile pool, with arm-exclusivity (scf.if arms are swept
// independently and merged) and same-iteration reuse (scf.for bodies recurse).
//
// This pass runs AFTER --aie-unroll-runtime-sequence-loops, so every
// constant-trip loop has already been unrolled to straight-line IR. A rolled
// ping-pong (a handle freed a later iteration than configured) therefore only
// survives when its trip count is a runtime value; that form needs a
// runtime-selected BD id and is rejected here for the dynamic EmitC path.
// Genuinely unallocatable forms (ambiguous handles, unbounded in-loop leaks,
// peak liveness exceeding the tile pool) are rejected with a clear diagnostic.
// The liveness analysis that classifies all of these lives in
// AIERuntimeBdLiveness.{h,cpp}.
//
//===----------------------------------------------------------------------===//

#include "AIERuntimeBdLiveness.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEASSIGNRUNTIMESEQUENCEBDIDS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct AIEAssignRuntimeSequenceBDIDsPass
    : xilinx::AIEX::impl::AIEAssignRuntimeSequenceBDIDsBase<
          AIEAssignRuntimeSequenceBDIDsPass> {

  llvm::DenseMap<AIE::TileOp, BdIdGenerator> gens;

  BdIdGenerator &getGeneratorForTile(AIE::TileOp tile) {
    auto it = gens.find(tile);
    if (it == gens.end()) {
      const AIETargetModel &targetModel =
          tile->getParentOfType<AIE::DeviceOp>().getTargetModel();
      it = gens.insert({tile, BdIdGenerator(tile.getCol(), tile.getRow(),
                                            targetModel)})
               .first;
    }
    return it->second;
  }

  // Maps each completion-sync op to the configure(s) it completes, built once
  // per sequence from the shared forward handle-trace in AIERuntimeBdLiveness.
  // Replaces a separate backward tracer: a free/await completes a configure iff
  // the forward trace from that configure reaches this sync (the two are duals).
  llvm::DenseMap<Operation *, SmallVector<DMAConfigureTaskOp>> syncToConfigures;

  //===--------------------------------------------------------------------===//
  // Validation: reject control-flow forms whose static BD-ID write-back is not
  // supported yet (they need a runtime-selected BD id).
  //===--------------------------------------------------------------------===//

  // Reject every form the static allocator cannot handle, BEFORE any IDs are
  // assigned, so allocation only ever runs on supported IR. This is the
  // complete gatekeeper: anything not rejected here must allocate without
  // crashing.
  LogicalResult validate(AIE::RuntimeSequenceOp seq) {
    // 1. Per-configure liveness classification. Same-iteration tasks (freed in
    // the iteration they are configured) and scf.if value-joins are allocated
    // statically; only genuinely unallocatable forms are rejected here.
    WalkResult wr = seq.walk([&](DMAConfigureTaskOp configure) -> WalkResult {
      TaskLiveRange range = resolveTaskLiveRange(configure);
      if (range.ambiguous) {
        configure.emitOpError(
            "task handle cannot be statically resolved to a single completion "
            "(it has multiple carries/syncs, or is carried unchanged across a "
            "loop back-edge); static BD-ID allocation is not supported for "
            "this "
            "form");
        return WalkResult::interrupt();
      }
      if (range.leaked && range.enclosingLoop) {
        configure.emitOpError(
            "buffer descriptor configured in a loop is never completed (no "
            "aiex.dma_await_task / aiex.dma_free_task reachable); its BD would "
            "not be reusable across iterations");
        return WalkResult::interrupt();
      }
      // A handle freed a later iteration than it was configured (a rolled
      // ping-pong) needs a distinct physical BD per outstanding iteration. For
      // a constant trip count that was already resolved by unrolling before
      // this pass ran, so a surviving back-edge crossing means the loop is
      // runtime-bound: its BD id must be selected at runtime, which is the
      // dynamic EmitC path, not static allocation.
      if (range.backEdgesCrossed > 0) {
        configure.emitOpError(
            "buffer descriptor is held across a runtime-bound loop back-edge "
            "(a rolled ping-pong with a non-constant trip count); its BD id "
            "must be selected at runtime. Use a compile-time-constant trip "
            "count so the loop can be unrolled, or lower this sequence with "
            "the dynamic EmitC path");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (wr.wasInterrupted())
      return failure();

    // 2. Every free/await must complete at least one configure op. The shared
    // forward trace maps each sync to the configures it completes; a sync absent
    // from that map (e.g. a free of an scf.for result seeded by a non-task
    // constant) would crash the recycle path and is rejected here.
    wr = seq.walk([&](Operation *op) -> WalkResult {
      Value task;
      if (auto f = dyn_cast<DMAFreeTaskOp>(op))
        task = f.getTask();
      else if (auto a = dyn_cast<DMAAwaitTaskOp>(op))
        task = a.getTask();
      else
        return WalkResult::advance();
      if (!syncToConfigures.count(op)) {
        // Emit "lower this first" notes for unlowered task ops.
        (void)emitUnresolvedTask(op, task);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (wr.wasInterrupted())
      return failure();

    // 3. A configure must not be buried inside a region op the allocator does
    // not sweep (e.g. scf.while); its BDs would be silently left unassigned.
    // (Pool-overflow is caught during allocation itself, when nextBdId runs
    // dry.)
    wr = seq.walk([&](DMAConfigureTaskOp configure) -> WalkResult {
      Operation *p = configure->getParentOp();
      while (p && p != seq.getOperation()) {
        if (!isa<scf::ForOp, scf::IfOp>(p)) {
          configure.emitOpError("is nested in an unsupported control-flow op '")
              << p->getName()
              << "'; BD-ID allocation only handles scf.for and scf.if";
          return WalkResult::interrupt();
        }
        p = p->getParentOp();
      }
      return WalkResult::advance();
    });
    return failure(wr.wasInterrupted());
  }

  //===--------------------------------------------------------------------===//
  // Allocation
  //===--------------------------------------------------------------------===//

  LogicalResult allocateConfigure(DMAConfigureTaskOp op) {
    AIE::TileOp tile = op.getTileOp();
    BdIdGenerator &gen = getGeneratorForTile(tile);

    // First, honor all the user-specified BD IDs.
    WalkResult result = op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd_op) {
      if (bd_op.getBdId().has_value()) {
        if (gen.bdIdAlreadyAssigned(bd_op.getBdId().value())) {
          op.emitOpError("Specified buffer descriptor ID ")
              << bd_op.getBdId().value()
              << " is already in use. Emit an aiex.dma_free_task operation to "
                 "reuse BDs.";
          return WalkResult::interrupt();
        }
        gen.assignBdId(bd_op.getBdId().value());
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();

    // Now allocate BD IDs for all unspecified BDs.
    result = op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd_op) {
      if (bd_op.getBdId().has_value())
        return WalkResult::advance();
      // channelIndex only affects allocation on MemTiles, where the AIE2 model
      // partitions BDs by channel parity (isBdChannelAccessible). Runtime
      // sequences configure BDs on shim (and compute) tiles only, which are
      // channel-agnostic (always accessible), so passing 0 is correct here.
      std::optional<int32_t> next_id = gen.nextBdId(/*channelIndex=*/0);
      if (!next_id) {
        const AIETargetModel &tm =
            tile->getParentOfType<AIE::DeviceOp>().getTargetModel();
        op.emitOpError()
            << "Too many simultaneously active buffer descriptors on tile("
            << tile.getCol() << "," << tile.getRow() << "), which supports up to "
            << tm.getNumBDs(tile.getCol(), tile.getRow())
            << ". Emit an aiex.dma_free_task / aiex.dma_await_task to reuse BDs.";
        return WalkResult::interrupt();
      }
      bd_op.setBdId(*next_id);
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();

    return success();
  }

  // Configures already completed by an aiex.dma_await_task. Awaiting a task
  // returns its BD ids to the pool (like a free), but a subsequent
  // aiex.dma_free_task of the same task is the common "wait, then release"
  // idiom, not a double free -- so freeing an awaited task's already-returned
  // ids is tolerated.
  llvm::SmallPtrSet<Operation *, 8> awaitedConfigures;

  // Return the ids of `task_op`'s chain to the pool.
  //   isAwait          -- this sync is an await (records the configure so a
  //                       later free of it is treated as a redundant release).
  //   tolerateAlreadyFree -- set when this configure is one of several an
  //                       scf.if-join free resolves to: only the arm that ran
  //                       holds the ids, so the others legitimately appear
  //                       already-freed.
  // Otherwise an already-freed id is a real double free (or a free of a task
  // that was never started) and is an error.
  LogicalResult recycle(DMAConfigureTaskOp task_op, Operation *freeOp,
                        bool isAwait, bool tolerateAlreadyFree) {
    BdIdGenerator &gen = getGeneratorForTile(task_op.getTileOp());
    bool redundantAfterAwait = awaitedConfigures.contains(task_op);
    WalkResult result = task_op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd) {
      if (!bd.getBdId().has_value()) {
        bd.emitOpError("Free called on BD chain with unassigned IDs.");
        return WalkResult::interrupt();
      }
      if (gen.bdIdAlreadyAssigned(bd.getBdId().value())) {
        gen.freeBdId(bd.getBdId().value());
      } else if (!tolerateAlreadyFree && !redundantAfterAwait) {
        freeOp->emitOpError("frees buffer descriptor ID ")
            << bd.getBdId().value()
            << ", which is not currently in use; it was already completed by "
               "an "
               "earlier aiex.dma_free_task or aiex.dma_await_task";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();
    if (isAwait)
      awaitedConfigures.insert(task_op);
    return success();
  }

  // Emit the "does not reference a valid configure_task" diagnostic, pointing
  // at an unlowered task op (bd_chain / configure_task_for) when that is the
  // cause.
  LogicalResult emitUnresolvedTask(Operation *op, Value task) {
    auto err =
        op->emitOpError("does not reference a valid configure_task operation.");
    if (Operation *def = task.getDefiningOp()) {
      if (isa<DMAStartBdChainOp>(def))
        err.attachNote(def->getLoc()) << "Lower this operation first using the "
                                         "--aie-materialize-bd-chains pass.";
      if (isa<DMAConfigureTaskForOp>(def))
        err.attachNote(def->getLoc())
            << "Lower this operation first using the "
               "--aie-substitute-shim-dma-allocations pass.";
    }
    return err;
  }

  // Recycle every configure this sync op completes (a single configure, or the
  // per-arm configures behind an scf.if result). When it completes more than one
  // configure, the free joins mutually-exclusive scf.if arms and only the arm
  // that ran holds the ids, so an already-freed id on the others is expected
  // rather than a double-free. validate() has already checked the sync is in the
  // map, so a miss here means an unlowered task op.
  LogicalResult recycleTargets(Value task, Operation *op, bool isAwait) {
    auto it = syncToConfigures.find(op);
    if (it == syncToConfigures.end())
      return emitUnresolvedTask(op, task);
    ArrayRef<DMAConfigureTaskOp> targets = it->second;
    bool tolerateAlreadyFree = targets.size() > 1;
    for (DMAConfigureTaskOp t : targets)
      if (failed(recycle(t, op, isAwait, tolerateAlreadyFree)))
        return failure();
    return success();
  }

  LogicalResult onFree(DMAFreeTaskOp op) {
    if (failed(recycleTargets(op.getTask(), op, /*isAwait=*/false)))
      return failure();
    op.erase();
    return success();
  }

  LogicalResult onAwait(DMAAwaitTaskOp op) {
    // Awaiting a task guarantees completion, so its BD IDs can be reused after.
    return recycleTargets(op.getTask(), op, /*isAwait=*/true);
  }

  // Snapshot/restore of the per-tile allocation state, for sweeping scf.if arms
  // independently (mutually exclusive arms do not interfere).
  using Snapshot = llvm::DenseMap<AIE::TileOp, BdIdGenerator::AssignedState>;
  Snapshot snapshot() {
    Snapshot s;
    for (auto &kv : gens)
      s[kv.first] = kv.second.saveAssigned();
    return s;
  }
  void restore(const Snapshot &s) {
    for (auto &kv : gens) {
      auto it = s.find(kv.first);
      kv.second.restoreAssigned(it == s.end() ? BdIdGenerator::AssignedState()
                                              : it->second);
    }
  }
  void unionInto(const Snapshot &other) {
    for (auto &kv : other)
      getGeneratorForTile(kv.first).mergeAssigned(kv.second);
  }

  LogicalResult sweepRegion(Region &region) {
    for (Block &block : region) {
      // Collect frees to erase after iterating (onFree erases the op).
      SmallVector<Operation *> ops;
      for (Operation &op : block)
        ops.push_back(&op);
      for (Operation *op : ops)
        if (failed(sweepOp(op)))
          return failure();
    }
    return success();
  }

  LogicalResult sweepOp(Operation *op) {
    if (auto cfg = dyn_cast<DMAConfigureTaskOp>(op))
      return allocateConfigure(cfg);
    if (auto freeOp = dyn_cast<DMAFreeTaskOp>(op))
      return onFree(freeOp);
    if (auto await = dyn_cast<DMAAwaitTaskOp>(op))
      return onAwait(await);
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Constant-trip loops were unrolled before this pass, so any loop here is
      // runtime-bound. Its tasks must be same-iteration (allocate and free
      // within the body) -- validate() has already rejected handles carried
      // across the back-edge. Sweep the body once: ids taken by a task are
      // returned by its free within the same iteration, so the allocation is
      // correct for every iteration.
      return sweepRegion(forOp.getRegion());
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // Arms are mutually exclusive: sweep each from the same starting state
      // and union the resulting allocations (a BD still held when an arm
      // finishes stays reserved regardless of which arm runs).
      Snapshot before = snapshot();
      if (failed(sweepRegion(ifOp.getThenRegion())))
        return failure();
      Snapshot afterThen = snapshot();
      restore(before);
      if (!ifOp.getElseRegion().empty()) {
        if (failed(sweepRegion(ifOp.getElseRegion())))
          return failure();
      }
      unionInto(afterThen);
      return success();
    }
    // Other ops (incl. those with regions not expected here) are inert for BD
    // allocation.
    return success();
  }

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();

    WalkResult wr = device.walk([&](AIE::RuntimeSequenceOp seq) -> WalkResult {
      // Build the sync->configures map once from the shared forward trace; both
      // validation and the recycle path read it.
      syncToConfigures = mapSyncsToConfigures(seq);
      if (failed(validate(seq)))
        return WalkResult::interrupt();
      gens.clear();
      if (failed(sweepRegion(seq.getBody())))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (wr.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEAssignRuntimeSequenceBDIDsPass() {
  return std::make_unique<AIEAssignRuntimeSequenceBDIDsPass>();
}
