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
// A task whose BD must rotate through several physical IDs across loop
// iterations (the ping-pong "free the previous iteration" pattern, where the
// handle crosses a loop back-edge) cannot be expressed with a single static
// `bd_id` attribute -- it needs a runtime-selected BD id. That lowering is not
// yet available, so such tasks are rejected with a clear diagnostic rather than
// miscompiled. The liveness analysis that classifies these lives in
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
#include "llvm/ADT/TypeSwitch.h"

#include <map>
#include <set>

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

  std::map<AIE::TileOp, BdIdGenerator> gens;

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

  //===--------------------------------------------------------------------===//
  // Validation: reject control-flow forms whose static BD-ID write-back is not
  // supported yet (they need a runtime-selected BD id).
  //===--------------------------------------------------------------------===//

  // Reject every form the static allocator cannot handle, BEFORE any IDs are
  // assigned, so allocation only ever runs on supported IR. This is the
  // complete gatekeeper: anything not rejected here must allocate without
  // crashing.
  LogicalResult validate(AIE::RuntimeSequenceOp seq) {
    // 1. Per-configure liveness classification.
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
      if (range.backEdgesCrossed > 0) {
        configure.emitOpError(
            "buffer descriptor is held across a loop back-edge (e.g. a rolled "
            "ping-pong that frees a previous iteration); this needs a "
            "runtime-selected BD id, which is not yet supported. Unroll the "
            "loop, or free the task within the same iteration");
        return WalkResult::interrupt();
      }
      if (range.crossedIfJoin) {
        configure.emitOpError(
            "task handle escapes its control-flow region via an scf.if result; "
            "static BD-ID allocation across a value join is not yet supported");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (wr.wasInterrupted())
      return failure();

    // 2. Every free/await must resolve to a configure op. A free/await of an
    // scf.for/if result or block argument (not a dma_configure_task) would
    // crash the recycle path; reject it cleanly here instead.
    wr = seq.walk([&](Operation *op) -> WalkResult {
      Value task;
      if (auto f = dyn_cast<DMAFreeTaskOp>(op))
        task = f.getTask();
      else if (auto a = dyn_cast<DMAAwaitTaskOp>(op))
        task = a.getTask();
      else
        return WalkResult::advance();
      if (!isa_and_nonnull<DMAConfigureTaskOp>(task.getDefiningOp())) {
        // Preserve the helpful "lower this first" notes for unlowered task ops.
        (void)emitUnresolvedTask(op, task);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (wr.wasInterrupted())
      return failure();

    // 3. A configure must not be buried inside a region op the allocator does
    // not sweep (e.g. scf.while); its BDs would be silently left unassigned.
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
        op.emitOpError() << "Allocator exhausted available buffer descriptor "
                            "IDs.";
        return WalkResult::interrupt();
      }
      bd_op.setBdId(*next_id);
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();

    return success();
  }

  LogicalResult recycle(DMAConfigureTaskOp task_op) {
    BdIdGenerator &gen = getGeneratorForTile(task_op.getTileOp());
    WalkResult result = task_op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd) {
      if (!bd.getBdId().has_value()) {
        bd.emitOpError("Free called on BD chain with unassigned IDs.");
        return WalkResult::interrupt();
      }
      // Ignore double-frees.
      if (gen.bdIdAlreadyAssigned(bd.getBdId().value()))
        gen.freeBdId(bd.getBdId().value());
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
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

  LogicalResult onFree(DMAFreeTaskOp op) {
    DMAConfigureTaskOp task_op = op.getTaskOp();
    if (!task_op)
      return emitUnresolvedTask(op, op.getTask());
    if (failed(recycle(task_op)))
      return failure();
    op.erase();
    return success();
  }

  LogicalResult onAwait(DMAAwaitTaskOp op) {
    // Awaiting a task guarantees completion, so its BD IDs can be reused after.
    DMAConfigureTaskOp task_op = op.getTaskOp();
    if (!task_op)
      return emitUnresolvedTask(op, op.getTask());
    return recycle(task_op);
  }

  // Snapshot/restore of the per-tile allocation state, for sweeping scf.if arms
  // independently (mutually exclusive arms do not interfere).
  using Snapshot = std::map<AIE::TileOp, std::set<uint32_t>>;
  Snapshot snapshot() {
    Snapshot s;
    for (auto &kv : gens)
      s[kv.first] = kv.second.alreadyAssigned;
    return s;
  }
  void restore(const Snapshot &s) {
    for (auto &kv : gens) {
      auto it = s.find(kv.first);
      kv.second.alreadyAssigned =
          it == s.end() ? std::set<uint32_t>() : it->second;
    }
  }
  void unionInto(const Snapshot &other) {
    for (auto &kv : other) {
      BdIdGenerator &gen = getGeneratorForTile(kv.first);
      gen.alreadyAssigned.insert(kv.second.begin(), kv.second.end());
    }
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
    if (auto forOp = dyn_cast<scf::ForOp>(op))
      // Same-iteration reuse: the body allocates and frees within one
      // iteration (back-edge-crossing handles are rejected by validate()), so
      // sweeping the body once leaves the pool as it was before the loop.
      return sweepRegion(forOp.getRegion());
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
