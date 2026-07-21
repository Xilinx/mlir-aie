//===- AIELowerDynamicBDPool.cpp --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The dynamic counterpart to AIEAssignRuntimeSequenceBDIDs: where that pass
// rejects a runtime-bound scf.for, this keeps the loop rolled and draws bd_ids
// from a per-tile runtime free-list pool. Each configure gets a dma_bd_pool_pop
// (its SSA id feeding bd_id_val); each free/await gets a dma_bd_pool_push.
//
// A popped id must be reachable at its push. Since a task value can cross a
// loop back edge and exit as a result, the id is carried in lockstep: every
// task iter_arg gets a parallel i32 id iter_arg, and a push looks up the paired
// id.
//
// v1 is single-BD only; multi-BD chains would need runtime next_bd (follow-up).
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
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
  // A representative (loop-invariant) configure per task, so a post-loop await
  // on a loop result -- which has no defining configure -- can be redirected to
  // one with the right (tile,dir,channel).
  llvm::DenseMap<Value, DMAConfigureTaskOp> originConfigure;

  // Every SSA value that carries a DMA task: configure results plus everything
  // the task flows into (for iter_args/results, if results). Computed by
  // forward closure from configures, so it is independent of the order in which
  // control flow is later rewritten.
  llvm::DenseSet<Value> taskValues;

  // A grown scf.for/scf.if whose appended i32 yield operands are still
  // placeholders. carried[i] is the original task position that the appended id
  // at nOrig+i shadows; the yields are wired once every position is paired.
  struct Fixup {
    Operation *op;
    SmallVector<unsigned> carried;
    unsigned nOrig;
    SmallVector<Operation *> placeholders; // dead consts to erase (scf.if only)
  };
  SmallVector<Fixup> fixups;

  // Count the BD ops in a configure's body (across all its blocks).
  static unsigned countBds(DMAConfigureTaskOp cfg) {
    unsigned n = 0;
    cfg.walk([&](AIE::DMABDOp) { ++n; });
    return n;
  }

  void markTask(Value v, SmallVectorImpl<Value> &worklist) {
    if (taskValues.insert(v).second)
      worklist.push_back(v);
  }

  // Forward-propagate task-ness from every configure result through scf.for
  // (init -> iter_arg, yield -> result) and scf.if (yield -> result). A value
  // that is never derived from a configure (e.g. an unrelated index yielded on
  // one branch) stays out of the set, which is what lets the divergent-yield
  // check below reject it.
  void computeTaskValues(AIE::RuntimeSequenceOp seq) {
    taskValues.clear();
    SmallVector<Value> worklist;
    seq.walk([&](DMAConfigureTaskOp c) { markTask(c.getResult(), worklist); });
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      for (OpOperand &use : v.getUses()) {
        Operation *user = use.getOwner();
        if (auto f = dyn_cast<scf::ForOp>(user)) {
          for (unsigned k = 0; k < f.getInitArgs().size(); ++k)
            if (f.getInitArgs()[k] == v)
              markTask(f.getRegionIterArg(k), worklist);
        } else if (isa<scf::YieldOp>(user)) {
          unsigned k = use.getOperandNumber();
          Operation *parent = user->getParentOp();
          if (auto pf = dyn_cast<scf::ForOp>(parent))
            markTask(pf.getResult(k), worklist);
          else if (auto pi = dyn_cast<scf::IfOp>(parent))
            markTask(pi.getResult(k), worklist);
        }
      }
    }
  }

  LogicalResult lowerConfigure(DMAConfigureTaskOp cfg) {
    if (cfg.getBdIdVal())
      return success(); // already lowered

    if (countBds(cfg) != 1)
      return cfg.emitOpError(
          "dynamic BD pool lowering supports single-BD tasks only; a multi-BD "
          "chain under a runtime-bound loop needs runtime next_bd chaining "
          "(not yet implemented)");

    // The pool owns all id allocation here, so a hand-pinned bd_id would
    // collide with a runtime pop. Allocation is all-pool or all-static (for a
    // straight-line sequence), never mixed.
    WalkResult pinned = cfg.walk([&](AIE::DMABDOp bd) {
      if (bd.getBdId().has_value()) {
        bd.emitOpError(
            "pins a buffer descriptor ID inside a runtime-bound "
            "sequence that draws IDs from the dynamic pool; a pinned "
            "ID would collide with a runtime-allocated one. Allocate "
            "every BD in this sequence from the pool (drop the "
            "bd_id), or make the sequence straight-line so the static "
            "allocator assigns all IDs.");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (pinned.wasInterrupted())
      return failure();

    AIE::TileOp tile = cfg.getTileOp();
    OpBuilder b(cfg);
    Value bdId = DMABdPoolPopOp::create(b, cfg.getLoc(), b.getI32Type(),
                                        tile.getCol(), tile.getRow())
                     .getBdId();
    cfg.getBdIdValMutable().assign(bdId);
    pairedId[cfg.getResult()] = bdId;
    return success();
  }

  // Grow an scf.for with a parallel i32 iter_arg for every task-carrying
  // iter_arg, so the popped id can flow in lockstep with the task Index. The
  // yields are left as placeholders (the new id block arg itself) and wired to
  // the real per-iteration id later, once every position is paired -- a body
  // may yield an id defined by a not-yet-grown inner op. Pairs the new
  // iter_args/results in `pairedId` and records a Fixup.
  scf::ForOp carryIdsThroughLoop(scf::ForOp forOp) {
    SmallVector<unsigned> carried; // iter_arg positions carrying a task
    SmallVector<Value> newInits;
    for (unsigned k = 0; k < forOp.getInitArgs().size(); ++k) {
      Value init = forOp.getInitArgs()[k];
      if (!taskValues.contains(init))
        continue;
      Value id = pairedId.lookup(init);
      assert(id && "for init carries a task but has no paired id");
      carried.push_back(k);
      newInits.push_back(id);
    }
    if (carried.empty())
      return forOp;

    unsigned nOrig = forOp.getInitArgs().size();
    IRRewriter rewriter(forOp);
    auto newYields = [&](OpBuilder &b, Location loc,
                         ArrayRef<BlockArgument> newBBArgs) {
      // Placeholder: yield each new id iter_arg straight back. Same type,
      // dominates the yield, and is a no-op for an identity (task-reuse) carry.
      // wireFixups overwrites it with the real per-iteration id.
      return SmallVector<Value>(newBBArgs.begin(), newBBArgs.end());
    };

    FailureOr<LoopLikeOpInterface> newLoop = forOp.replaceWithAdditionalYields(
        rewriter, newInits,
        /*replaceInitOperandUsesInLoop=*/false, newYields);
    assert(succeeded(newLoop) && "scf.for additional-yields rewrite failed");
    auto newFor = cast<scf::ForOp>(newLoop->getOperation());

    for (auto [i, k] : llvm::enumerate(carried)) {
      // The rewrite gave the loop new block args/results; migrate task-ness
      // onto them so a later-grown consumer of this loop's result still sees a
      // task.
      taskValues.insert(newFor.getRegionIterArg(k));
      taskValues.insert(newFor.getResult(k));
      pairedId[newFor.getRegionIterArg(k)] = newFor.getRegionIterArg(nOrig + i);
      pairedId[newFor.getResult(k)] = newFor.getResult(nOrig + i);
    }
    fixups.push_back({newFor, carried, nOrig, {}});
    return newFor;
  }

  // Grow an scf.if with a parallel i32 result per task it yields out (its
  // branch-local id), so a task freed after the if finds its id. Yields start
  // as placeholders, wired later. A result carries a task iff BOTH branches
  // yield one; a divergent yield (only one branch) is rejected -- no id to push
  // on the other path.
  LogicalResult carryIdsThroughIf(scf::IfOp ifOp) {
    if (ifOp.getNumResults() == 0 || ifOp.elseBlock() == nullptr)
      return success();

    auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());

    SmallVector<unsigned> carried; // result positions carrying a task
    for (unsigned k = 0; k < ifOp.getNumResults(); ++k) {
      bool thenT = taskValues.contains(thenYield.getOperand(k));
      bool elseT = taskValues.contains(elseYield.getOperand(k));
      if (thenT && elseT)
        carried.push_back(k);
      else if (thenT || elseT)
        return ifOp.emitOpError(
            "yields a pool-allocated task on only one branch of an scf.if; "
            "both "
            "branches must yield the task (and its buffer descriptor id) so it "
            "can be freed after the if");
    }
    if (carried.empty())
      return success();

    unsigned nOrig = ifOp.getNumResults();
    OpBuilder b(ifOp);
    SmallVector<Type> resultTypes(ifOp.getResultTypes());
    for (unsigned k : carried)
      (void)k, resultTypes.push_back(b.getI32Type());

    auto newIf =
        scf::IfOp::create(b, ifOp.getLoc(), resultTypes, ifOp.getCondition(),
                          /*withElseRegion=*/true);
    newIf.getThenRegion().takeBody(ifOp.getThenRegion());
    newIf.getElseRegion().takeBody(ifOp.getElseRegion());

    // Append a placeholder i32 to each branch's yield; wireFixups replaces it
    // with the branch's real id (which may be defined by a not-yet-grown inner
    // op). The placeholder is a dead constant erased during wiring.
    SmallVector<Operation *> placeholders;
    auto appendPlaceholders = [&](scf::YieldOp y) {
      OpBuilder yb(y);
      SmallVector<Value> ids;
      for (unsigned k : carried) {
        (void)k;
        auto c =
            arith::ConstantOp::create(yb, y.getLoc(), yb.getI32IntegerAttr(0));
        placeholders.push_back(c);
        ids.push_back(c);
      }
      y->insertOperands(y->getNumOperands(), ids);
    };
    appendPlaceholders(cast<scf::YieldOp>(newIf.thenBlock()->getTerminator()));
    appendPlaceholders(cast<scf::YieldOp>(newIf.elseBlock()->getTerminator()));

    for (auto [i, k] : llvm::enumerate(carried)) {
      // Migrate task-ness onto the rebuilt result so a later-grown consumer of
      // this if's result still sees a task.
      taskValues.insert(newIf.getResult(k));
      pairedId[newIf.getResult(k)] = newIf.getResult(nOrig + i);
    }
    for (unsigned k = 0; k < nOrig; ++k)
      ifOp.getResult(k).replaceAllUsesWith(newIf.getResult(k));
    ifOp.erase();
    fixups.push_back({newIf, carried, nOrig, placeholders});
    return success();
  }

  // Sweep 2: with every position paired, set each grown op's appended yield
  // operand(s) to the real id yielded at that position. Runs after all growth,
  // so an id defined by an inner op (grown later than its enclosing op) is
  // resolvable, and an identity carry resolves to its own iter_arg's id.
  void wireFixups() {
    for (Fixup &fx : fixups) {
      if (auto forOp = dyn_cast<scf::ForOp>(fx.op)) {
        auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        for (auto [i, k] : llvm::enumerate(fx.carried)) {
          Value id = pairedId.lookup(yield.getOperand(k));
          assert(id && "carried task yields an unpaired id");
          yield.setOperand(fx.nOrig + i, id);
        }
      } else {
        auto ifOp = cast<scf::IfOp>(fx.op);
        for (Block *blk : {ifOp.thenBlock(), ifOp.elseBlock()}) {
          auto yield = cast<scf::YieldOp>(blk->getTerminator());
          for (auto [i, k] : llvm::enumerate(fx.carried)) {
            Value id = pairedId.lookup(yield.getOperand(k));
            assert(id && "carried task yields an unpaired id");
            yield.setOperand(fx.nOrig + i, id);
          }
        }
      }
    }
    for (Fixup &fx : fixups)
      for (Operation *p : fx.placeholders)
        p->erase();
  }

  // Sweep 3: over the final IR, give every task value its tile (for the push)
  // and a representative origin configure (for an await that names a loop/if
  // result with no defining configure). Order-independent forward closure:
  //   - scf.for: the init's tile/origin flow to the iter_arg AND the result, so
  //     a result takes the loop-invariant init configure (which dominates a
  //     post-loop await), not the per-iteration body configure.
  //   - scf.if: the then-branch yield's tile/origin flow to the result.
  // A push only reads the tile, so both branches must agree; a genuinely
  // cross-tile scf.if is rejected.
  LogicalResult computeMetadata(AIE::RuntimeSequenceOp seq) {
    tileForTask.clear();
    originConfigure.clear();
    SmallVector<Value> worklist;
    auto setMeta = [&](Value v, std::pair<int, int> tile,
                       DMAConfigureTaskOp origin) {
      if (tileForTask.count(v))
        return;
      tileForTask[v] = tile;
      originConfigure[v] = origin;
      worklist.push_back(v);
    };
    seq.walk([&](DMAConfigureTaskOp c) {
      AIE::TileOp t = c.getTileOp();
      setMeta(c.getResult(), {t.getCol(), t.getRow()}, c);
    });
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      auto tile = tileForTask[v];
      DMAConfigureTaskOp origin = originConfigure[v];
      for (OpOperand &use : v.getUses()) {
        Operation *user = use.getOwner();
        if (auto f = dyn_cast<scf::ForOp>(user)) {
          for (unsigned k = 0; k < f.getInitArgs().size(); ++k)
            if (f.getInitArgs()[k] == v) {
              setMeta(f.getRegionIterArg(k), tile, origin);
              setMeta(f.getResult(k), tile, origin);
            }
        } else if (isa<scf::YieldOp>(user)) {
          auto ifOp = dyn_cast<scf::IfOp>(user->getParentOp());
          if (ifOp && user->getBlock() == ifOp.thenBlock())
            setMeta(ifOp.getResult(use.getOperandNumber()), tile, origin);
        }
      }
    }
    // A carried scf.if result is one BD id and (if awaited post-if) one sync
    // channel, so both branches must agree on the full physical channel
    // (tile,dir,channel) -- otherwise the single push/sync would be ambiguous.
    WalkResult wr = seq.walk([&](scf::IfOp ifOp) -> WalkResult {
      if (ifOp.getNumResults() == 0 || ifOp.elseBlock() == nullptr)
        return WalkResult::advance();
      auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
      auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
      for (unsigned k = 0; k < ifOp.getNumResults(); ++k) {
        DMAConfigureTaskOp to = originConfigure.lookup(thenYield.getOperand(k));
        DMAConfigureTaskOp eo = originConfigure.lookup(elseYield.getOperand(k));
        if (to && eo && syncSig(to) != syncSig(eo)) {
          ifOp.emitOpError("yields tasks on different physical channels "
                           "(tile/direction/channel) from its two branches at "
                           "the same result; a pooled buffer descriptor id and "
                           "its completion sync belong to one channel");
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return wr.wasInterrupted() ? failure() : success();
  }

  // The physical channel a configure targets: (col, row, direction, channel).
  static std::tuple<int, int, int, int> syncSig(DMAConfigureTaskOp cfg) {
    AIE::TileOp t = cfg.getTileOp();
    return {t.getCol(), t.getRow(), (int)cfg.getDirection(), cfg.getChannel()};
  }

  LogicalResult lowerRelease(Value task, Operation *op) {
    Value id = pairedId.lookup(task);
    if (!id)
      return op->emitOpError(
          "does not resolve to a task allocated from the runtime pool; cannot "
          "return its buffer descriptor ID");
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

      // Idempotency: a sequence already carrying pool pops has been lowered by
      // a prior run of this pass. Its awaits may have dropped their task
      // operand (sync_* attrs) and its scf ops already carry i32 ids, so
      // re-running the carry logic would misfire. Skip it.
      bool alreadyLowered = false;
      seq.walk([&](DMABdPoolPopOp) { alreadyLowered = true; });
      if (alreadyLowered)
        return WalkResult::advance();

      // scf.while has no bounded trip count, so a BD id popped in its body has
      // no place to be pushed back deterministically; only scf.for and scf.if
      // are supported.
      WalkResult whileCheck = seq.walk([&](scf::WhileOp w) {
        w.emitOpError("scf.while in a runtime sequence is not supported by the "
                      "dynamic BD pool path; only scf.for and scf.if are "
                      "supported");
        return WalkResult::interrupt();
      });
      if (whileCheck.wasInterrupted())
        return WalkResult::interrupt();

      pairedId.clear();
      tileForTask.clear();
      originConfigure.clear();
      fixups.clear();

      // 1. Every configure pops an id (program order: a pop dominates every use
      //    of its id).
      WalkResult r = seq.walk([&](DMAConfigureTaskOp cfg) -> WalkResult {
        return failed(lowerConfigure(cfg)) ? WalkResult::interrupt()
                                           : WalkResult::advance();
      });
      if (r.wasInterrupted())
        return WalkResult::interrupt();

      // 2. Thread the id alongside the task everywhere the task flows. The task
      //    Index already threads (it is valid IR); shadowing every position it
      //    occupies with a parallel i32 makes the id in-scope at every push.
      //    a. Task closure: which SSA values carry a task (order-independent).
      //    b. Grow structure outermost-first (pre-order): an enclosing op's new
      //       iter_arg/result exists before an inner op yields through it. A
      //       divergent scf.if is rejected here.
      //    c. Wire the placeholder yields now that every position is paired --
      //       handles both siblings (producer grown before consumer) and
      //       parent-defines/child-consumes (inner id resolvable after growth).
      computeTaskValues(seq);
      SmallVector<Operation *> cf;
      seq.walk<WalkOrder::PreOrder>([&](Operation *o) {
        if (isa<scf::ForOp, scf::IfOp>(o))
          cf.push_back(o);
      });
      for (Operation *o : cf) {
        if (auto f = dyn_cast<scf::ForOp>(o)) {
          carryIdsThroughLoop(f);
        } else if (failed(carryIdsThroughIf(cast<scf::IfOp>(o)))) {
          return WalkResult::interrupt();
        }
      }
      wireFixups();

      // 3. Metadata over the final IR: each task's tile (for the push) and a
      //    dominating origin configure (for an await on a loop/if result).
      if (failed(computeMetadata(seq)))
        return WalkResult::interrupt();

      // 4. Return ids to the pool. A free returns the id (push). An await is a
      //    TCT sync (npu_sync), NOT a release -- it pushes ONLY when the id it
      //    carries is never returned by a free, so an awaited-but-never-freed
      //    task still frees its BD exactly once. This split keeps the runtime
      //    free-list balanced when a BD is reused across iterations or awaited
      //    on one SSA value and freed on another (the same id via the carry).
      //
      //    freedByCarry is the set of task values whose carried id reaches a
      //    free: the backward closure from each free's operand through the
      //    carry (an scf result comes from its branch/body yields and, for a
      //    loop, its init; a loop iter_arg from its init and back-edge yield).
      //    An await whose task is in this set is on a path a later free already
      //    balances.
      llvm::DenseSet<Value> freedByCarry;
      SmallVector<Value> fworklist;
      auto markFreed = [&](Value v) {
        if (v && taskValues.contains(v) && freedByCarry.insert(v).second)
          fworklist.push_back(v);
      };
      seq.walk([&](DMAFreeTaskOp freeOp) { markFreed(freeOp.getTask()); });
      while (!fworklist.empty()) {
        Value v = fworklist.pop_back_val();
        if (auto res = dyn_cast<OpResult>(v)) {
          Operation *def = res.getOwner();
          unsigned k = res.getResultNumber();
          if (auto f = dyn_cast<scf::ForOp>(def)) {
            markFreed(f.getInitArgs()[k]);
            markFreed(f.getBody()->getTerminator()->getOperand(k));
          } else if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
            markFreed(ifOp.thenBlock()->getTerminator()->getOperand(k));
            markFreed(ifOp.elseBlock()->getTerminator()->getOperand(k));
          }
        } else if (auto ba = dyn_cast<BlockArgument>(v)) {
          if (auto f = dyn_cast<scf::ForOp>(ba.getOwner()->getParentOp())) {
            unsigned k = ba.getArgNumber() - f.getNumInductionVars();
            markFreed(f.getInitArgs()[k]);
            markFreed(f.getBody()->getTerminator()->getOperand(k));
          }
        }
      }

      DominanceInfo domInfo(seq);
      SmallVector<Operation *> toErase;
      r = seq.walk([&](Operation *op) -> WalkResult {
        if (auto freeOp = dyn_cast<DMAFreeTaskOp>(op)) {
          if (failed(lowerRelease(freeOp.getTask(), freeOp)))
            return WalkResult::interrupt();
          toErase.push_back(freeOp);
        } else if (auto await = dyn_cast<DMAAwaitTaskOp>(op)) {
          // Push only if no free balances this id; the await keeps its sync.
          if (!freedByCarry.contains(await.getTask()) &&
              failed(lowerRelease(await.getTask(), await)))
            return WalkResult::interrupt();
          if (!await.getTask().getDefiningOp<DMAConfigureTaskOp>()) {
            DMAConfigureTaskOp origin = originConfigure.lookup(await.getTask());
            if (!origin)
              return await.emitOpError(
                         "awaits a task whose configure cannot be resolved for "
                         "the dynamic pool path"),
                     WalkResult::interrupt();
            // The sync reads only the configure's static tile/dir/channel. If a
            // configure dominates the await (e.g. a loop's pre-loop init),
            // redirect the operand to it -- the common case, no attrs needed.
            // Otherwise (the awaited task was configured in an scf.if branch,
            // which cannot dominate a sibling/post-if await) stamp the channel
            // as attributes; computeMetadata verified both branches agree on
            // it.
            if (domInfo.dominates(origin.getResult(), await)) {
              await.getTaskMutable().assign(origin.getResult());
            } else {
              if (!origin.getIssueToken())
                return origin.emitOpError("awaited task does not issue a "
                                          "completion token; add "
                                          "issue_token = true"),
                       WalkResult::interrupt();
              AIE::TileOp t = origin.getTileOp();
              await.setSyncCol(t.getCol());
              await.setSyncRow(t.getRow());
              await.setSyncDirection(origin.getDirection());
              await.setSyncChannel(origin.getChannel());
              // Drop the operand: the sync only needs the channel, and keeping
              // a use of the branch-carried task would block the configure from
              // lowering (it requires use_empty). BD reuse stays serialized by
              // queue backpressure, so the dropped ordering edge is not needed.
              await.getTaskMutable().clear();
            }
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
