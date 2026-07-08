//===- AIEAssignRuntimeSequenceBDIDs.cpp ------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Assigns buffer-descriptor (BD) IDs to the DMA tasks configured in a runtime
// sequence.
//
// This pass runs on straight-line IR: --aie-unroll-runtime-sequence-loops has
// unrolled every constant-trip scf.for, and canonicalization has folded every
// constant-predicate scf.if, before this pass runs. So in the static path no
// scf op survives to reach the allocator -- a rolled ping-pong over a constant
// loop is just N straight-line configures whose ids ordinary liveness reuse
// recycles. Any scf.for/scf.if still present is therefore runtime-valued, which
// the static path cannot lower (the runtime sequence becomes a flat, branchless
// NPU instruction stream); such forms are rejected here for the dynamic EmitC
// path (Phase 2).
//
// Before allocating, the pass also rejects task-completion-token (TCT)
// imbalances -- an await with no matching issue_token push on its channel would
// deadlock the host. On the straight-line IR the allocator sees, this is a
// simple per-channel token count (see verifyTokenBalance).
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <array>
#include <map>

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

  // Reject control flow the static path cannot lower. Constant-trip scf.for is
  // unrolled and constant-predicate scf.if is folded before this pass, so any
  // scf op here is runtime-valued and belongs to the dynamic EmitC path.
  LogicalResult rejectRuntimeControlFlow(AIE::RuntimeSequenceOp seq) {
    WalkResult wr = seq.walk([&](Operation *op) -> WalkResult {
      if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op)) {
        op->emitOpError(
            "Runtime-valued control flow in a runtime sequence is not "
            "supported by this static BD-ID allocation pass. Either pass "
            "only constant-valued predicates to scf.for and scf.if so "
            "`aie-unroll-runtime-sequence-loops` can unroll/fold them, or "
            "use the dynamic EmitC path instead.");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return failure(wr.wasInterrupted());
  }

  // Reject task-completion-token (TCT) imbalances that deadlock the host.
  //
  // Each issue_token dma_start_task pushes one token onto a per-(tile,
  // direction, channel) FIFO; each dma_await_task on that channel pops one. A
  // sync that runs when the FIFO is empty blocks the runtime sequence forever.
  // Over-production is always safe (leftover tokens are harmless) -- awaiting
  // the last of N pushes on a channel and letting the FIFO cover the rest is a
  // legal, common idiom -- so only under-production (an await with no matching
  // push) is an error.
  //
  // rejectRuntimeControlFlow has already run, so this walks straight-line IR: a
  // single program-order pass counting available tokens per channel is exact,
  // with no control-flow reasoning. (Non-issue_token awaits are left to the
  // per-op check in aie-dma-tasks-to-npu, which owns that diagnostic.)
  LogicalResult verifyTokenBalance(AIE::RuntimeSequenceOp seq) {
    using ChannelKey = std::array<int, 4>; // {col, row, direction, channel}
    auto keyOf = [](DMAConfigureTaskOp cfg) -> ChannelKey {
      AIE::TileOp tile = cfg.getTileOp();
      return {tile.getCol(), tile.getRow(),
              static_cast<int>(cfg.getDirection()),
              static_cast<int>(cfg.getChannel())};
    };

    std::map<ChannelKey, int> avail;
    WalkResult wr = seq.walk([&](Operation *op) -> WalkResult {
      if (auto start = dyn_cast<DMAStartTaskOp>(op)) {
        DMAConfigureTaskOp cfg = start.getTaskOp();
        if (cfg && cfg.getIssueToken())
          avail[keyOf(cfg)]++;
      } else if (auto await = dyn_cast<DMAAwaitTaskOp>(op)) {
        DMAConfigureTaskOp cfg = await.getTaskOp();
        // A non-issue_token await is diagnosed later by aie-dma-tasks-to-npu;
        // an unresolved task is diagnosed by the recycle path. Skip both here.
        if (!cfg || !cfg.getIssueToken())
          return WalkResult::advance();
        int &tokens = avail[keyOf(cfg)];
        if (tokens < 1) {
          await.emitOpError(
              "awaits a task-completion token on a channel where no "
              "outstanding token is guaranteed to have been produced; the "
              "runtime sequence would block here forever. Ensure a prior "
              "issue_token dma_start_task on the same tile, direction and "
              "channel reaches this await");
          return WalkResult::interrupt();
        }
        tokens--;
      }
      return WalkResult::advance();
    });
    return failure(wr.wasInterrupted());
  }

  LogicalResult validate(AIE::RuntimeSequenceOp seq) {
    // Reject runtime control flow first, so the token-balance pass below runs
    // on straight-line IR and needs no control-flow reasoning.
    if (failed(rejectRuntimeControlFlow(seq)))
      return failure();
    if (failed(verifyTokenBalance(seq)))
      return failure();
    return success();
  }

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
    result =
        op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd_op) {
          if (bd_op.getBdId().has_value())
            return WalkResult::advance();
          // channelIndex only affects allocation on MemTiles, where the AIE2
          // model partitions BDs by channel parity (isBdChannelAccessible).
          // Runtime sequences configure BDs on shim (and compute) tiles only,
          // which are channel-agnostic (always accessible), so passing 0 is
          // correct here.
          std::optional<int32_t> next_id = gen.nextBdId(/*channelIndex=*/0);
          if (!next_id) {
            const AIETargetModel &tm =
                tile->getParentOfType<AIE::DeviceOp>().getTargetModel();
            op.emitOpError()
                << "Too many simultaneously active buffer descriptors on tile ("
                << tile.getCol() << "," << tile.getRow()
                << "), which supports up to "
                << tm.getNumBDs(tile.getCol(), tile.getRow())
                << ". Emit an aiex.dma_free_task / aiex.dma_await_task to "
                   "reuse BDs.";
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

  // Return the ids of the configure's chain to the pool. `isAwait` records the
  // configure so a later free of it is treated as a redundant release rather
  // than a double free. Otherwise an already-freed id is a real double free (or
  // a free of a task that was never started) and is an error.
  LogicalResult recycle(DMAConfigureTaskOp task_op, Operation *freeOp,
                        bool isAwait) {
    BdIdGenerator &gen = getGeneratorForTile(task_op.getTileOp());
    bool redundantAfterAwait = awaitedConfigures.contains(task_op);
    WalkResult result = task_op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd) {
      if (!bd.getBdId().has_value()) {
        bd.emitOpError("Free called on BD chain with unassigned IDs.");
        return WalkResult::interrupt();
      }
      if (gen.bdIdAlreadyAssigned(bd.getBdId().value())) {
        gen.freeBdId(bd.getBdId().value());
      } else if (!redundantAfterAwait) {
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

  // Resolve a free/await to its configure. In straight-line IR the task value
  // is defined directly by the configure op; anything else is an unlowered task
  // op.
  LogicalResult recycleTask(Value task, Operation *op, bool isAwait) {
    auto cfg = dyn_cast_or_null<DMAConfigureTaskOp>(task.getDefiningOp());
    if (!cfg) {
      auto err = op->emitOpError(
          "does not reference a valid configure_task operation.");
      if (Operation *def = task.getDefiningOp()) {
        if (isa<DMAStartBdChainOp>(def))
          err.attachNote(def->getLoc())
              << "Lower this operation first using the "
                 "--aie-materialize-bd-chains pass.";
        if (isa<DMAConfigureTaskForOp>(def))
          err.attachNote(def->getLoc())
              << "Lower this operation first using the "
                 "--aie-substitute-shim-dma-allocations pass.";
      }
      return err;
    }
    return recycle(cfg, op, isAwait);
  }

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();

    WalkResult wr = device.walk([&](AIE::RuntimeSequenceOp seq) -> WalkResult {
      if (failed(validate(seq)))
        return WalkResult::interrupt();
      gens.clear();
      awaitedConfigures.clear();

      // Straight-line walk. Collect frees to erase after (recycling reads the
      // configure the free points at, so erase only once the walk is done).
      // This allocation strategy works only for straight-line IR without
      // branching or conditionals; the verifier of this pass ensures this is
      // the case on the input IR.
      SmallVector<DMAFreeTaskOp> frees;
      WalkResult r = seq.walk([&](Operation *op) -> WalkResult {
        if (auto cfg = dyn_cast<DMAConfigureTaskOp>(op)) {
          if (failed(allocateConfigure(cfg)))
            return WalkResult::interrupt();
        } else if (auto await = dyn_cast<DMAAwaitTaskOp>(op)) {
          if (failed(recycleTask(await.getTask(), await, /*isAwait=*/true)))
            return WalkResult::interrupt();
        } else if (auto freeOp = dyn_cast<DMAFreeTaskOp>(op)) {
          if (failed(recycleTask(freeOp.getTask(), freeOp, /*isAwait=*/false)))
            return WalkResult::interrupt();
          frees.push_back(freeOp);
        }
        return WalkResult::advance();
      });
      if (r.wasInterrupted())
        return WalkResult::interrupt();
      for (DMAFreeTaskOp freeOp : frees)
        freeOp.erase();
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
