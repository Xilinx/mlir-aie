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
// Before allocating, the pass also checks the per-channel hardware resources a
// sequence can exhaust (see verifyChannelUsage): task-completion-token (TCT)
// imbalance -- an await with no matching issue_token push on its channel would
// deadlock the host -- and DMA task-queue overflow, where more transfers are
// pushed onto a channel than its queue holds. On the straight-line IR the
// allocator sees, both are single-pass per-channel counts.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Dialect/AIEX/Utils/DmaQueueModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <array>
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
  using Base = xilinx::AIEX::impl::AIEAssignRuntimeSequenceBDIDsBase<
      AIEAssignRuntimeSequenceBDIDsPass>;
  AIEAssignRuntimeSequenceBDIDsPass() = default;
  AIEAssignRuntimeSequenceBDIDsPass(
      const AIEAssignRuntimeSequenceBDIDsOptions &options)
      : Base(options) {}

  llvm::DenseMap<AIE::TileOp, BdIdGenerator> gens;

  // Mark every BD id a static DMA already took on `tile`, so this allocator
  // doesn't hand the same id to a runtime-sequence task.
  static void seedFromStaticBds(AIE::DeviceOp device, AIE::TileOp tile,
                                BdIdGenerator &gen) {
    for (AIE::DmaBody program : device.getOps<AIE::DmaBody>())
      if (program.getTileID() == tile.getTileID())
        for (uint32_t id : AIE::getAssignedBdIds(program))
          if (!gen.bdIdAlreadyAssigned(id))
            gen.assignBdId(id);
  }

  BdIdGenerator &getGeneratorForTile(AIE::TileOp tile) {
    auto it = gens.find(tile);
    if (it == gens.end()) {
      AIE::DeviceOp device = tile->getParentOfType<AIE::DeviceOp>();
      it = gens.insert({tile, BdIdGenerator(tile.getCol(), tile.getRow(),
                                            device.getTargetModel())})
               .first;
      seedFromStaticBds(device, tile, it->second);
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

  // Track the two per-channel resources a sequence can exhaust. Their failure
  // modes are opposites: over-production is harmless for the TCT FIFO (leftover
  // tokens cost nothing) but not for the task queue, where a push onto a full
  // queue is dropped and its transfer never runs.
  //
  // Queue counting differs from token counting in three ways: every push takes
  // a slot, not just issue_token ones; a non-token push is still retired
  // implicitly, since in-order execution means awaiting a token drains
  // everything queued ahead of it too; and repeat_count does not multiply
  // slots. rejectRuntimeControlFlow has already run, so one program-order pass
  // over straight-line IR is exact.
  LogicalResult verifyChannelUsage(AIE::RuntimeSequenceOp seq) {
    using ChannelKey = DmaQueueModel::ChannelKey;
    auto keyOf = [](DMAConfigureTaskOp cfg) -> ChannelKey {
      AIE::TileOp tile = cfg.getTileOp();
      return {tile.getCol(), tile.getRow(),
              static_cast<int>(cfg.getDirection()),
              static_cast<int>(cfg.getChannel())};
    };

    const AIETargetModel &tm =
        seq->getParentOfType<AIE::DeviceOp>().getTargetModel();

    std::map<ChannelKey, int> avail;
    DmaQueueModel queue;

    WalkResult wr = seq.walk([&](Operation *op) -> WalkResult {
      if (auto start = dyn_cast<DMAStartTaskOp>(op)) {
        DMAConfigureTaskOp cfg = start.getTaskOp();
        if (!cfg)
          return WalkResult::advance();
        ChannelKey key = keyOf(cfg);
        AIE::TileOp tile = cfg.getTileOp();
        uint32_t depth = tm.getDmaTaskQueueDepth(
            tile.getCol(), tile.getRow(), cfg.getChannel(), cfg.getDirection());
        if (queue.wouldOverflow(key, depth))
          guardQueueOverflow(queue, start, tm, key, depth, enforceQueueDepth);
        queue.push(key, cfg.getIssueToken());
        if (cfg.getIssueToken())
          avail[key]++;
      } else if (auto await = dyn_cast<DMAAwaitTaskOp>(op)) {
        DMAConfigureTaskOp cfg = await.getTaskOp();
        // A non-issue_token await is diagnosed later by aie-dma-tasks-to-npu;
        // an unresolved task is diagnosed by the recycle path. Skip both here.
        if (!cfg || !cfg.getIssueToken())
          return WalkResult::advance();
        ChannelKey key = keyOf(cfg);
        int &tokens = avail[key];
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
        queue.awaitToken(key);
      } else if (auto sync = dyn_cast<NpuSyncOp>(op)) {
        // Deliberately the queue only, never the `avail` token balance above.
        // That balance drives a hard error, so it stays with the ops whose
        // token flags it can read off the IR; the queue only warns or polls,
        // so it can afford to credit a raw sync it cannot fully account for.
        awaitSync(queue, sync);
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
    if (failed(verifyChannelUsage(seq)))
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
              << " is already in use. Release the earlier task first: "
                 "aiex.dma_await_task waits for hardware completion before its "
                 "BDs are reusable. aiex.dma_free_task also releases them but "
                 "does NOT wait, so it is only safe when some other "
                 "synchronization already guarantees that task has finished "
                 "(see programming_guide/section-2/section-2d/DMATasks.md).";
          return WalkResult::interrupt();
        }
        checkReallocation(tile, bd_op.getBdId().value(), bd_op);
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
                << ". Emit an aiex.dma_await_task to free BDs for reuse; it "
                   "waits for hardware completion, so the recycled ids are no "
                   "longer in flight. aiex.dma_free_task also recycles ids but "
                   "does NOT wait for completion -- using it before the task "
                   "has finished is a race -- so reach for it only when some "
                   "other synchronization already guarantees completion (see "
                   "programming_guide/section-2/section-2d/DMATasks.md).";
            return WalkResult::interrupt();
          }
          checkReallocation(tile, *next_id, bd_op);
          bd_op.setBdId(next_id);
          return WalkResult::advance();
        });
    if (result.wasInterrupted())
      return failure();

    return success();
  }

  // Tasks started on each channel, in program order, that are not yet known to
  // have completed.
  std::map<DmaQueueModel::ChannelKey, SmallVector<DMAConfigureTaskOp, 8>>
      startedOnChannel;
  // Configures whose completion an await has established.
  llvm::SmallPtrSet<Operation *, 16> knownComplete;
  // BD ids released by aiex.dma_free_task while the task could still have been
  // in flight, keyed by tile, with the free that released them.
  std::map<std::pair<int, int>, std::map<uint32_t, Operation *>> freedInFlight;

  static DmaQueueModel::ChannelKey channelOf(DMAConfigureTaskOp cfg) {
    AIE::TileOp tile = cfg.getTileOp();
    return {tile.getCol(), tile.getRow(), static_cast<int>(cfg.getDirection()),
            static_cast<int>(cfg.getChannel())};
  }

  // A channel runs its tasks in order, so awaiting one establishes that it and
  // everything started ahead of it on that channel has finished. This is the
  // reasoning DMATasks.md blesses for "free X after awaiting Y", made explicit.
  void noteAwaited(DMAConfigureTaskOp cfg) {
    auto &started = startedOnChannel[channelOf(cfg)];
    auto *it = llvm::find(started, cfg);
    if (it == started.end()) {
      knownComplete.insert(cfg);
      return;
    }
    for (auto *p = started.begin(); p <= it; ++p)
      knownComplete.insert(*p);
    started.erase(started.begin(), std::next(it));
  }

  // Record ids released without any completion guarantee. nextBdId scans upward
  // from 0, so a just-freed low id is the first one handed out again -- the
  // worst case for aliasing a BD that is still running.
  void noteFreedInFlight(DMAConfigureTaskOp cfg, Operation *freeOp) {
    if (knownComplete.contains(cfg))
      return;
    AIE::TileOp tile = cfg.getTileOp();
    auto &ids = freedInFlight[{tile.getCol(), tile.getRow()}];
    cfg.walk([&](AIE::DMABDOp bd) {
      if (bd.getBdId().has_value())
        ids[bd.getBdId().value()] = freeOp;
    });
  }

  // Warn where the hazard actually bites: reusing the id, not releasing it.
  void checkReallocation(AIE::TileOp tile, uint32_t id, AIE::DMABDOp bd) {
    if (!warnUnsafeBdReuse)
      return;
    auto tileIt = freedInFlight.find({tile.getCol(), tile.getRow()});
    if (tileIt == freedInFlight.end())
      return;
    auto idIt = tileIt->second.find(id);
    if (idIt == tileIt->second.end())
      return;
    Operation *freeOp = idIt->second;
    tileIt->second.erase(idIt);
    auto diag = bd->emitWarning()
        << "reuses buffer descriptor ID " << id << " on tile ("
        << tile.getCol() << "," << tile.getRow()
        << ") after it was released by an aiex.dma_free_task that had no "
           "completion guarantee, so the DMA it belonged to may still be "
           "running and this reprograms it underneath. Await the earlier task, "
           "or await a later one on the same tile, direction and channel -- a "
           "channel completes its tasks in order, so that covers everything "
           "queued before it";
    diag.attachNote(freeOp->getLoc()) << "released here";
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
    else
      noteFreedInFlight(task_op, freeOp);
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
      // Skip sequences already handled by the dynamic free-list pool path
      // (aie-lower-dynamic-bd-pool): they draw BD ids at runtime via
      // dma_bd_pool_pop and keep their scf.for rolled, which the static
      // straight-line allocator neither needs to touch nor can validate.
      bool dynamicPool = false;
      seq.walk([&](DMABdPoolPopOp) { dynamicPool = true; });
      if (dynamicPool)
        return WalkResult::advance();

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
        } else if (auto start = dyn_cast<DMAStartTaskOp>(op)) {
          if (DMAConfigureTaskOp cfg = start.getTaskOp())
            startedOnChannel[channelOf(cfg)].push_back(cfg);
        } else if (auto await = dyn_cast<DMAAwaitTaskOp>(op)) {
          if (DMAConfigureTaskOp cfg = await.getTaskOp())
            noteAwaited(cfg);
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

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEAssignRuntimeSequenceBDIDsPass(
    const AIEAssignRuntimeSequenceBDIDsOptions &options) {
  return std::make_unique<AIEAssignRuntimeSequenceBDIDsPass>(options);
}
