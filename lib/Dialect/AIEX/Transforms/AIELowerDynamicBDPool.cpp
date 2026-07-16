//===- AIELowerDynamicBDPool.cpp --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The dynamic counterpart to AIEAssignRuntimeSequenceBDIDs. That pass runs on
// straight-line IR and rejects runtime-bound scf.for; this pass handles the
// runtime-bound case by keeping the loop rolled and drawing buffer-descriptor
// IDs from a per-tile runtime free-list pool.
//
// Each dma_configure_task gets an aiex.dma_bd_pool_pop whose SSA bd_id is
// routed into the task (its bd_id_val operand, consumed by the dynamic BD
// lowering). Each dma_free_task, and the recycling half of a dma_await_task,
// becomes an aiex.dma_bd_pool_push of the matching id.
//
// The matching id must be available where the push is emitted. A task value
// (Index) may cross an scf.for back edge (the ping-pong "%prev" carry) and exit
// as a loop result, so the popped i32 id is carried in lockstep: for every
// iter_arg that carries a task, a parallel i32 iter_arg carrying its id is
// added. Then a push of any task value looks up the paired id at the same
// scope.
//
// v1 restricts to single-BD tasks: a multi-BD chain under a runtime loop would
// need per-BD runtime next_bd cross-references, which is a follow-up.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIELOWERDYNAMICBDPOOL
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct AIELowerDynamicBDPoolPass
    : xilinx::AIEX::impl::AIELowerDynamicBDPoolBase<AIELowerDynamicBDPoolPass> {

  // Maps a task value (the Index result of a configure, or any Index the carry
  // propagates it into) to the i32 pool id available at that same program
  // point.
  llvm::DenseMap<Value, Value> pairedId;
  // The (col,row) of the tile each task's id belongs to, so a push names the
  // right pool even when the id is a loop result rather than a direct pop.
  llvm::DenseMap<Value, std::pair<int, int>> tileForTask;
  // A representative configure for each task value whose (tile,dir,channel) an
  // await can reference. For a loop-carried task this stays the init (iteration
  // 0) configure, which is loop-invariant and dominates a post-loop await --
  // the value the await originally named (a loop result) has no defining
  // configure to resolve, so awaits are redirected here.
  llvm::DenseMap<Value, DMAConfigureTaskOp> originConfigure;

  // Count the BD ops in a configure's body (across all its blocks).
  static unsigned countBds(DMAConfigureTaskOp cfg) {
    unsigned n = 0;
    cfg.walk([&](AIE::DMABDOp) { ++n; });
    return n;
  }

  LogicalResult lowerConfigure(DMAConfigureTaskOp cfg) {
    if (cfg.getBdIdVal())
      return success(); // already lowered

    if (countBds(cfg) != 1)
      return cfg.emitOpError(
          "dynamic BD pool lowering supports single-BD tasks only; a multi-BD "
          "chain under a runtime-bound loop needs runtime next_bd chaining "
          "(not yet implemented)");

    AIE::TileOp tile = cfg.getTileOp();
    OpBuilder b(cfg);
    Value bdId = DMABdPoolPopOp::create(b, cfg.getLoc(), b.getI32Type(),
                                        tile.getCol(), tile.getRow())
                     .getBdId();
    cfg.getBdIdValMutable().assign(bdId);
    pairedId[cfg.getResult()] = bdId;
    tileForTask[cfg.getResult()] = {tile.getCol(), tile.getRow()};
    originConfigure[cfg.getResult()] = cfg;
    return success();
  }

  // Add a parallel i32 iter_arg for every task-carrying iter_arg of `forOp`, so
  // the popped id flows in lockstep with the task Index. Returns the new loop.
  // Updates `pairedId` for the new region iter_args and results.
  scf::ForOp carryIdsThroughLoop(scf::ForOp forOp) {
    SmallVector<unsigned> carried; // iter_arg positions carrying a paired task
    SmallVector<Value> newInits;
    for (unsigned k = 0; k < forOp.getInitArgs().size(); ++k) {
      Value init = forOp.getInitArgs()[k];
      auto it = pairedId.find(init);
      if (it == pairedId.end())
        continue;
      carried.push_back(k);
      newInits.push_back(it->second);
    }
    if (carried.empty())
      return forOp;

    unsigned nOrig = forOp.getInitArgs().size();
    IRRewriter rewriter(forOp);
    auto newYields = [&](OpBuilder &b, Location loc,
                         ArrayRef<BlockArgument> newBBArgs) {
      // The value yielded for each new id iter_arg is the paired id of the task
      // that carried position yields. For a rotating ping-pong the body yields
      // a fresh configure result, whose id is already in the map.
      auto *yield = forOp.getBody()->getTerminator();
      SmallVector<Value> yielded;
      for (unsigned k : carried) {
        Value bodyTask = yield->getOperand(k);
        Value id = pairedId.lookup(bodyTask);
        assert(id && "carried task yields an unpaired id");
        yielded.push_back(id);
      }
      return yielded;
    };

    FailureOr<LoopLikeOpInterface> newLoop = forOp.replaceWithAdditionalYields(
        rewriter, newInits,
        /*replaceInitOperandUsesInLoop=*/false, newYields);
    assert(succeeded(newLoop) && "scf.for additional-yields rewrite failed");
    auto newFor = cast<scf::ForOp>(newLoop->getOperation());

    // The body block was moved intact, so original region iter_args keep their
    // positions; the id iter_args are appended after the original nOrig. Pair
    // each carried task iter_arg / result with its new id iter_arg / result,
    // and propagate the tile from the corresponding init task.
    for (auto [i, k] : llvm::enumerate(carried)) {
      Value init = forOp.getInitArgs()[k];
      auto tile = tileForTask.lookup(init);
      DMAConfigureTaskOp origin = originConfigure.lookup(init);
      pairedId[newFor.getRegionIterArg(k)] = newFor.getRegionIterArg(nOrig + i);
      tileForTask[newFor.getRegionIterArg(k)] = tile;
      originConfigure[newFor.getRegionIterArg(k)] = origin;
      pairedId[newFor.getResult(k)] = newFor.getResult(nOrig + i);
      tileForTask[newFor.getResult(k)] = tile;
      originConfigure[newFor.getResult(k)] = origin;
    }
    return newFor;
  }

  LogicalResult lowerRelease(Value task, Operation *op,
                             llvm::DenseSet<Value> &released) {
    Value id = pairedId.lookup(task);
    if (!id)
      return op->emitOpError(
          "does not resolve to a task allocated from the runtime pool; cannot "
          "return its buffer descriptor ID");
    // await returns the id; a later free of the same task is the "wait then
    // release" idiom, not a double free -- push once.
    if (!released.insert(task).second)
      return success();
    auto tile = tileForTask.lookup(task);
    OpBuilder b(op);
    DMABdPoolPushOp::create(b, op->getLoc(), tile.first, tile.second, id);
    return success();
  }

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();

    WalkResult wr = device.walk([&](AIE::RuntimeSequenceOp seq) -> WalkResult {
      bool hasRuntimeCF = false;
      seq.walk([&](Operation *op) {
        if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op))
          hasRuntimeCF = true;
      });
      if (!hasRuntimeCF)
        return WalkResult::advance();

      pairedId.clear();
      tileForTask.clear();

      // 1. Every configure pops an id (program order: a pop dominates every use
      //    of its id).
      WalkResult r = seq.walk([&](DMAConfigureTaskOp cfg) -> WalkResult {
        return failed(lowerConfigure(cfg)) ? WalkResult::interrupt()
                                           : WalkResult::advance();
      });
      if (r.wasInterrupted())
        return WalkResult::interrupt();

      // 2. Carry the ids through loops so a push sees the id at its own scope.
      //    Innermost-first: a body's yielded ids must be paired before its
      //    enclosing loop consumes them.
      SmallVector<scf::ForOp> loops;
      seq.walk([&](scf::ForOp f) { loops.push_back(f); });
      for (scf::ForOp f : llvm::reverse(loops))
        carryIdsThroughLoop(f);

      // 3. free/await -> push (await keeps its sync, adds a push). An await on a
      //    loop result / region-arg cannot resolve its configure (a loop result
      //    has no defining configure), so redirect it to the loop-invariant
      //    origin configure, whose tile/dir/channel is identical every
      //    iteration.
      llvm::DenseSet<Value> released;
      SmallVector<Operation *> toErase;
      r = seq.walk([&](Operation *op) -> WalkResult {
        if (auto freeOp = dyn_cast<DMAFreeTaskOp>(op)) {
          if (failed(lowerRelease(freeOp.getTask(), freeOp, released)))
            return WalkResult::interrupt();
          toErase.push_back(freeOp);
        } else if (auto await = dyn_cast<DMAAwaitTaskOp>(op)) {
          if (failed(lowerRelease(await.getTask(), await, released)))
            return WalkResult::interrupt();
          if (!await.getTask().getDefiningOp<DMAConfigureTaskOp>()) {
            DMAConfigureTaskOp origin = originConfigure.lookup(await.getTask());
            if (!origin)
              return await.emitOpError(
                         "awaits a task whose configure cannot be resolved for "
                         "the dynamic pool path"),
                     WalkResult::interrupt();
            await.getTaskMutable().assign(origin.getResult());
          }
        }
        return WalkResult::advance();
      });
      if (r.wasInterrupted())
        return WalkResult::interrupt();
      for (Operation *op : toErase)
        op->erase();
      return WalkResult::advance();
    });
    if (wr.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIELowerDynamicBDPoolPass() {
  return std::make_unique<AIELowerDynamicBDPoolPass>();
}
