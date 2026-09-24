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
// A start whose constant repeat count is past the hardware field is first
// issued as several starts of the same task (see splitLongRepeats), so every
// start below is exactly one queue push.
//
// Before allocating, the pass also checks the per-channel hardware resources a
// sequence can exhaust (see verifyChannelUsage): task-completion-token (TCT)
// imbalance -- an await with no matching issue_token push on its channel would
// deadlock the host -- and DMA task-queue overflow, where more transfers are
// pushed onto a channel than its queue holds. On the straight-line IR the
// allocator sees, both are single-pass per-channel counts.
//
// When a tile runs out of ids, the pass takes them back from a started task
// that was never released (see reclaimFor) rather than failing: from one some
// status poll already proves finished, or else by inserting a poll that does.
// Only allocations that would fail change, so a sequence that fits compiles to
// the same instructions either way.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Dialect/AIEX/Utils/DmaQueueModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

  static std::optional<DmaQueueModel::ChannelKey>
  otherTokenChannel(Operation *op) {
    if (auto push = dyn_cast<NpuPushQueueOp>(op)) {
      if (push.getIssueToken())
        return DmaQueueModel::ChannelKey{static_cast<int>(push.getColumn()),
                                         static_cast<int>(push.getRow()),
                                         static_cast<int>(push.getDirection()),
                                         static_cast<int>(push.getChannel())};
    } else if (auto copy = dyn_cast<NpuDmaMemcpyNdOp>(op)) {
      auto allocation = AIE::ShimDMAAllocationOp::getForSymbol(
          op->getParentOfType<AIE::DeviceOp>(),
          copy.getMetadata().getRootReference());
      if (allocation &&
          (copy.getIssueToken() ||
           allocation.getChannelDir() == AIE::DMAChannelDir::S2MM)) {
        if (AIE::TileOp tile = allocation.getTileOp())
          return DmaQueueModel::ChannelKey{
              tile.getCol(), tile.getRow(),
              static_cast<int>(allocation.getChannelDir()),
              static_cast<int>(allocation.getChannelIndex())};
      }
    }
    return std::nullopt;
  }

  static std::optional<DmaQueueModel::ChannelKey>
  otherAwaitChannel(Operation *op) {
    if (auto sync = dyn_cast<NpuSyncOp>(op))
      return syncChannelKey(sync);
    if (auto wait = dyn_cast<NpuDmaWaitOp>(op)) {
      auto allocation = AIE::ShimDMAAllocationOp::getForSymbol(
          op->getParentOfType<AIE::DeviceOp>(), wait.getSymbol());
      if (allocation)
        if (AIE::TileOp tile = allocation.getTileOp())
          return DmaQueueModel::ChannelKey{
              tile.getCol(), tile.getRow(),
              static_cast<int>(allocation.getChannelDir()),
              static_cast<int>(allocation.getChannelIndex())};
    }
    return std::nullopt;
  }

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
  // everything queued ahead of it too; and a repeat_count the hardware field
  // holds does not multiply slots (splitLongRepeats has already made one start
  // per push of a larger one). rejectRuntimeControlFlow has already run, so one
  // program-order pass over straight-line IR is exact.
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
        bool issuesToken = start.getPushIssueToken(cfg);
        queue.push(key, issuesToken);
        if (issuesToken)
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
      } else if (auto key = otherAwaitChannel(op)) {
        // Raw waits may also consume tokens from outside this sequence. Do not
        // diagnose those here, but never reuse a token already consumed by one.
        avail[*key] = std::max(0, avail[*key] - 1);
        queue.awaitToken(*key);
      } else if (auto key = otherTokenChannel(op)) {
        avail[*key]++;
        queue.push(*key, true);
      }
      return WalkResult::advance();
    });
    return failure(wr.wasInterrupted());
  }

  // Issue each start whose constant repeat count does not fit the queue push's
  // field as several starts of the same task: full-size ones first, then the
  // remainder, which is the original op. Leading starts withhold the token, so
  // an await on the task still returns only after the last pass. Splitting
  // here, not at lowering, makes each push its own start, so the queue-depth
  // guard below can poll between them. A runtime-valued count is left alone;
  // aie-dma-to-npu guards it.
  void splitLongRepeats(AIE::RuntimeSequenceOp seq) {
    uint32_t maxRepeat = seq->getParentOfType<AIE::DeviceOp>()
                             .getTargetModel()
                             .getMaxRepeatCount();
    // A target without a repeat field (maxRepeat 0) cannot be split into
    // anything; its push verifier reports the count instead.
    if (maxRepeat == 0)
      return;
    SmallVector<DMAStartTaskOp> starts;
    seq.walk([&](DMAStartTaskOp start) { starts.push_back(start); });
    for (DMAStartTaskOp start : starts) {
      DMAConfigureTaskOp cfg = start.getTaskOp();
      if (!cfg)
        continue;
      std::optional<int64_t> rc =
          getConstantIntValue(start.getPushRepeatCount(cfg));
      if (!rc || *rc <= maxRepeat)
        continue;
      OpBuilder b(start);
      int64_t runs = *rc + 1;
      for (; runs > maxRepeat + 1; runs -= maxRepeat + 1)
        DMAStartTaskOp::create(b, start.getLoc(), start.getTask(),
                               b.getI32IntegerAttr(maxRepeat),
                               /*no_token=*/b.getUnitAttr());
      start.setRepeatCountAttr(b.getI32IntegerAttr(runs - 1));
    }
  }

  LogicalResult validate(AIE::RuntimeSequenceOp seq) {
    // Reject runtime control flow first, so the token-balance pass below runs
    // on straight-line IR and needs no control-flow reasoning.
    if (failed(rejectRuntimeControlFlow(seq)))
      return failure();
    splitLongRepeats(seq);
    if (failed(verifyChannelUsage(seq)))
      return failure();
    return success();
  }

  LogicalResult allocateConfigure(DMAConfigureTaskOp op) {
    AIE::TileOp tile = op.getTileOp();
    BdIdGenerator &gen = getGeneratorForTile(tile);

    const AIETargetModel &targetModel =
        tile->getParentOfType<AIE::DeviceOp>().getTargetModel();

    // First, honor all the user-specified BD IDs.
    WalkResult result = op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd_op) {
      if (bd_op.getBdId().has_value()) {
        if (!targetModel.isBdChannelAccessible(tile.getCol(), tile.getRow(),
                                               bd_op.getBdId().value(),
                                               op.getChannel())) {
          bd_op.emitOpError("Buffer descriptor ID ")
              << bd_op.getBdId().value() << " cannot be submitted on channel "
              << op.getChannel() << " of tile (" << tile.getCol() << ","
              << tile.getRow()
              << "), which partitions its buffer descriptors by channel "
                 "parity: an even channel reaches only the low half of the "
                 "ids and an odd channel only the high half.";
          return WalkResult::interrupt();
        }
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
    result = op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd_op) {
      if (bd_op.getBdId().has_value())
        return WalkResult::advance();
      // channelIndex matters on a MemTile, where the AIE2 model partitions
      // BDs by channel parity (isBdChannelAccessible: an even channel can
      // only submit ids below 24, an odd channel only 24 and above).
      std::optional<int32_t> next_id = gen.nextBdId(op.getChannel());
      while (!next_id && reclaimBds && succeeded(reclaimFor(op)))
        next_id = gen.nextBdId(op.getChannel());
      if (!next_id) {
        auto diag =
            op.emitOpError()
            << "Too many simultaneously active buffer descriptors on tile ("
            << tile.getCol() << "," << tile.getRow()
            << "), which supports up to "
            << targetModel.getNumBDsForChannel(tile.getCol(), tile.getRow(),
                                               op.getChannel())
            << ". Emit an aiex.dma_await_task to free BDs for reuse; it "
               "waits for hardware completion, so the recycled ids are no "
               "longer in flight. aiex.dma_free_task also recycles ids but "
               "does NOT wait for completion -- using it before the task "
               "has finished is a race -- so reach for it only when some "
               "other synchronization already guarantees completion (see "
               "programming_guide/section-2/section-2d/DMATasks.md).";
        if (reclaimBds)
          diag << " The compiler could not take any back either: every "
                  "task holding one is not started yet, is started again "
                  "later, or runs on a channel whose status it cannot "
                  "poll.";
        return WalkResult::interrupt();
      }
      checkReallocation(tile, *next_id, bd_op);
      bd_op.setBdId(next_id);
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();

    liveByTile[tile].push_back(op);
    return success();
  }

  // One push onto a channel: the task it started (null for a raw push or
  // memcpy transfer) and whether that push issues a token. A task started more
  // than once may issue a token on only some of its pushes.
  struct StartedTask {
    DMAConfigureTaskOp task;
    bool issuesToken;
    // Program order of the push, for picking the oldest task to reclaim.
    uint64_t order;
    // A status poll proved it finished. Its token, if any, is still waiting
    // for an await, so it stays here until one consumes it.
    bool finished = false;
  };
  // Pushes on each channel, in program order, whose tokens no await has
  // consumed yet.
  std::map<DmaQueueModel::ChannelKey, SmallVector<StartedTask, 8>>
      startedOnChannel;
  uint64_t pushCount = 0;

  void notePush(const DmaQueueModel::ChannelKey &key, DMAConfigureTaskOp task,
                bool issuesToken) {
    startedOnChannel[key].push_back({task, issuesToken, pushCount++});
  }

  static bool isStarted(ArrayRef<StartedTask> started,
                        DMAConfigureTaskOp task) {
    return llvm::any_of(started,
                        [&](const StartedTask &s) { return s.task == task; });
  }
  static bool isRunning(ArrayRef<StartedTask> started,
                        DMAConfigureTaskOp task) {
    return llvm::any_of(started, [&](const StartedTask &s) {
      return s.task == task && !s.finished;
    });
  }
  // Configures whose completion an await has established.
  llvm::SmallPtrSet<Operation *, 16> knownComplete;
  // Configures whose completion a status poll has established. Kept apart from
  // knownComplete so that the ids an await releases do not move: only
  // reclaimFor reads this.
  llvm::SmallPtrSet<Operation *, 16> pollProven;
  // Configures whose ids reclaimFor took back.
  llvm::SmallPtrSet<Operation *, 16> reclaimed;
  // Configures whose ids went back to the pool by any route.
  llvm::SmallPtrSet<Operation *, 16> releasedTasks;
  // Configures allocated on each tile, in program order.
  llvm::DenseMap<AIE::TileOp, SmallVector<DMAConfigureTaskOp, 16>> liveByTile;
  // BD ids released by aiex.dma_free_task while the task could still have been
  // in flight, keyed by tile, with the task and the free that released them.
  struct ReleasedTask {
    DMAConfigureTaskOp configure;
    Operation *freeOp;
  };
  std::map<std::pair<int, int>, std::map<uint32_t, ReleasedTask>> freedInFlight;

  static DmaQueueModel::ChannelKey channelOf(DMAConfigureTaskOp cfg) {
    AIE::TileOp tile = cfg.getTileOp();
    return {tile.getCol(), tile.getRow(), static_cast<int>(cfg.getDirection()),
            static_cast<int>(cfg.getChannel())};
  }

  // An await consumes the oldest outstanding token on the channel, regardless
  // of the configure named by its SSA operand. Only that token-issuing task and
  // the tasks queued ahead of it are known to have finished.
  void noteAwaited(const DmaQueueModel::ChannelKey &key) {
    auto &started = startedOnChannel[key];
    auto *it = llvm::find_if(
        started, [](const StartedTask &s) { return s.issuesToken; });
    if (it == started.end())
      return;
    SmallVector<StartedTask, 8> retired(started.begin(), std::next(it));
    started.erase(started.begin(), std::next(it));
    for (const StartedTask &s : retired)
      if (s.task && !isStarted(started, s.task))
        knownComplete.insert(s.task);
  }

  // A status poll proved at most `unfinished` pushes on `key` are still queued
  // or running. The channel runs its queue in order, so those are the newest.
  void noteDrained(const DmaQueueModel::ChannelKey &key, size_t unfinished) {
    auto &started = startedOnChannel[key];
    if (started.size() <= unfinished)
      return;
    auto done = MutableArrayRef<StartedTask>(started).drop_back(unfinished);
    for (StartedTask &s : done)
      s.finished = true;
    for (StartedTask &s : done)
      if (s.task && !isRunning(started, s.task))
        pollProven.insert(s.task);
  }

  // Credit a maskpoll on a channel's status register, whether the queue-depth
  // guard, reclaimFor or the design emitted it. A poll bounding the queue size
  // by b leaves at most b + 1 unfinished (see getDmaTaskQueueSizeMask); one
  // that also clears the running and stall bits leaves none.
  void notePoll(NpuMaskPollOp poll) {
    std::optional<uint32_t> address = poll.getAbsoluteAddress();
    std::optional<uint32_t> mask = getConstantIntOperand(poll.getMask());
    std::optional<uint32_t> value = getConstantIntOperand(poll.getValue());
    const AIETargetModel &tm =
        poll->getParentOfType<AIE::DeviceOp>().getTargetModel();
    uint32_t field = tm.getDmaTaskQueueSizeMask();
    uint32_t idle = tm.getDmaChannelIdleMask();
    if (!address || !mask || !value || !field)
      return;
    for (auto &[key, started] : startedOnChannel) {
      if (tm.getDmaStatusAddress(key[0], key[1], key[3],
                                 static_cast<AIE::DMAChannelDir>(key[2])) !=
          address)
        continue;
      if (idle && (*mask & idle) == idle && (*value & idle) == 0)
        return noteDrained(key, 0);
      unsigned shift = llvm::countr_zero(field);
      uint32_t bound = 0;
      for (uint32_t size = 0; size <= field >> shift; ++size)
        if (((size << shift) & *mask & field) == (*value & *mask & field))
          bound = size;
      return noteDrained(key, bound + 1);
    }
  }

  // Record ids released without any completion guarantee. nextBdId scans upward
  // from 0, so a just-freed low id is the first one handed out again -- the
  // worst case for aliasing a BD that is still running.
  void noteFreedInFlight(DMAConfigureTaskOp cfg, Operation *freeOp) {
    if (!isStarted(startedOnChannel[channelOf(cfg)], cfg))
      return;
    AIE::TileOp tile = cfg.getTileOp();
    auto &ids = freedInFlight[{tile.getCol(), tile.getRow()}];
    cfg.walk([&](AIE::DMABDOp bd) {
      if (bd.getBdId().has_value())
        ids[bd.getBdId().value()] = {cfg, freeOp};
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
    ReleasedTask released = idIt->second;
    tileIt->second.erase(idIt);
    if (knownComplete.contains(released.configure) ||
        pollProven.contains(released.configure))
      return;
    auto diag =
        bd->emitWarning()
        << "reuses buffer descriptor ID " << id << " on tile (" << tile.getCol()
        << "," << tile.getRow()
        << ") after it was released by an aiex.dma_free_task that had no "
           "completion guarantee, so the DMA it belonged to may still be "
           "running and this reprograms it underneath. Consume the outstanding "
           "tokens on the same tile, direction and channel through a task "
           "queued at or after this transfer -- each await consumes the oldest "
           "token, regardless of the task it names";
    diag.attachNote(released.freeOp->getLoc()) << "released here";
  }

  // Configures already completed by an aiex.dma_await_task. Awaiting a task
  // returns its BD ids to the pool (like a free), but a subsequent
  // aiex.dma_free_task of the same task is the common "wait, then release"
  // idiom, not a double free -- so freeing an awaited task's already-returned
  // ids is tolerated.
  llvm::SmallPtrSet<Operation *, 8> awaitedConfigures;
  // Awaited configures whose IDs remain live until their FIFO completion.
  llvm::SmallPtrSet<Operation *, 8> pendingAwaitReleases;

  // Return the ids of the configure's chain to the pool. `isAwait` records the
  // configure so a later free of it is treated as a redundant release rather
  // than a double free. Otherwise an already-freed id is a real double free (or
  // a free of a task that was never started) and is an error.
  LogicalResult recycle(DMAConfigureTaskOp task_op, Operation *freeOp,
                        bool isAwait) {
    pendingAwaitReleases.erase(task_op);
    // Those IDs may now belong to another configure. A redundant release must
    // not inspect or change the generator's current ownership.
    if (awaitedConfigures.contains(task_op) || reclaimed.contains(task_op))
      return success();
    BdIdGenerator &gen = getGeneratorForTile(task_op.getTileOp());
    WalkResult result = task_op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd) {
      if (!bd.getBdId().has_value()) {
        bd.emitOpError("Free called on BD chain with unassigned IDs.");
        return WalkResult::interrupt();
      }
      if (gen.bdIdAlreadyAssigned(bd.getBdId().value())) {
        gen.freeBdId(bd.getBdId().value());
      } else {
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
    releasedTasks.insert(task_op);
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
    if (!isAwait)
      return recycle(cfg, op, /*isAwait=*/false);
    pendingAwaitReleases.insert(cfg);
    return recycleCompletedTasks(op);
  }

  LogicalResult recycleCompletedTasks(Operation *op) {
    SmallVector<Operation *, 8> pending(pendingAwaitReleases.begin(),
                                        pendingAwaitReleases.end());
    for (Operation *pendingOp : pending) {
      auto pendingCfg = cast<DMAConfigureTaskOp>(pendingOp);
      // An earlier await may have named this task while consuming an older
      // token. Revisit its release when a later await proves it complete.
      if (!knownComplete.contains(pendingCfg))
        continue;
      // Retain ownership across any remaining starts of the same configure.
      if (llvm::any_of(pendingCfg.getResult().getUsers(), [&](Operation *user) {
            return isa<DMAStartTaskOp>(user) &&
                   user->getBlock() == op->getBlock() &&
                   op->isBeforeInBlock(user);
          }))
        continue;
      if (failed(recycle(pendingCfg, op, /*isAwait=*/true)))
        return failure();
    }
    return success();
  }

  static bool isStartedAfter(DMAConfigureTaskOp task, Operation *op) {
    return llvm::any_of(task.getResult().getUsers(), [&](Operation *user) {
      return isa<DMAStartTaskOp>(user) && user->getBlock() == op->getBlock() &&
             op->isBeforeInBlock(user);
    });
  }

  // Take back the ids of a started task that is not started again after `op`
  // and holds an id `op`'s channel can use (on a mem tile, its parity's half).
  // One a poll already proved finished is taken as is. Otherwise a poll before
  // `op` proves one finished: the oldest with j >= 1 pushes queued behind it,
  // for which Task_Queue_Size <= j - 1 is proof (as a masked equality, the
  // largest 2^k - 1 <= j - 1), so it waits on no push not yet issued; failing
  // that, the oldest, until its channel is idle. The poll never returns if that
  // task's inputs come from a later push through a core, which the compiler
  // cannot see; a design with such a dependence releases its own BDs.
  LogicalResult reclaimFor(DMAConfigureTaskOp op) {
    AIE::TileOp tile = op.getTileOp();
    const AIETargetModel &tm =
        tile->getParentOfType<AIE::DeviceOp>().getTargetModel();
    auto isCandidate = [&](DMAConfigureTaskOp task) {
      if (releasedTasks.contains(task) || isStartedAfter(task, op))
        return false;
      bool usable = false;
      task.walk([&](AIE::DMABDOp bd) {
        usable |= bd.getBdId() &&
                  tm.isBdChannelAccessible(tile.getCol(), tile.getRow(),
                                           *bd.getBdId(), op.getChannel());
      });
      return usable;
    };
    auto take = [&](DMAConfigureTaskOp task) {
      BdIdGenerator &gen = getGeneratorForTile(tile);
      task.walk([&](AIE::DMABDOp bd) { gen.freeBdId(*bd.getBdId()); });
      reclaimed.insert(task);
      releasedTasks.insert(task);
      pendingAwaitReleases.erase(task);
    };

    ArrayRef<DMAConfigureTaskOp> live = liveByTile[tile];
    for (DMAConfigureTaskOp task : live)
      if ((knownComplete.contains(task) || pollProven.contains(task)) &&
          isCandidate(task)) {
        take(task);
        return success();
      }

    uint32_t field = tm.getDmaTaskQueueSizeMask();
    uint32_t idle = tm.getDmaChannelIdleMask();
    DMAConfigureTaskOp victim;
    size_t victimQueuedBehind = 0;
    uint64_t victimOrder = 0;
    for (DMAConfigureTaskOp task : live) {
      if (!isCandidate(task))
        continue;
      DmaQueueModel::ChannelKey key = channelOf(task);
      if (!tm.getDmaStatusAddress(key[0], key[1], key[3],
                                  static_cast<AIE::DMAChannelDir>(key[2])))
        continue;
      ArrayRef<StartedTask> started = startedOnChannel[key];
      auto last =
          llvm::find_if(llvm::reverse(started),
                        [&](const StartedTask &s) { return s.task == task; });
      // Never started, or retired by an await and caught above.
      if (last == started.rend() || last->finished)
        continue;
      size_t queuedBehind = std::distance(started.rbegin(), last);
      if (queuedBehind == 0 ? !idle : !field)
        continue;
      // Rule 1 before rule 2, then oldest.
      if (victim && std::make_pair(queuedBehind == 0, last->order) >=
                        std::make_pair(victimQueuedBehind == 0, victimOrder))
        continue;
      victim = task;
      victimQueuedBehind = queuedBehind;
      victimOrder = last->order;
    }
    if (!victim)
      return failure();

    DmaQueueModel::ChannelKey key = channelOf(victim);
    std::optional<uint32_t> status = tm.getDmaStatusAddress(
        key[0], key[1], key[3], static_cast<AIE::DMAChannelDir>(key[2]));
    if (!status)
      return failure();
    uint32_t mask = idle;
    size_t unfinished = 0;
    if (victimQueuedBehind > 0) {
      unsigned shift = llvm::countr_zero(field);
      uint32_t bound =
          std::min<uint32_t>(llvm::bit_floor(victimQueuedBehind),
                             llvm::bit_floor((field >> shift) + 1) / 2) -
          1;
      mask = field & ~(bound << shift);
      unfinished = bound + 1;
    }
    OpBuilder b(op);
    auto cst = [&](uint32_t v) {
      return arith::ConstantOp::create(b, op.getLoc(), b.getI32Type(),
                                       b.getI32IntegerAttr(v))
          .getResult();
    };
    Value maskValue = cst(mask);
    Value compareValue = cst(0);
    Value statusValue = cst(*status);
    NpuMaskPollOp::create(b, op.getLoc(), statusValue, compareValue, maskValue,
                          /*buffer=*/nullptr, /*column=*/nullptr,
                          /*row=*/nullptr);
    noteDrained(key, unfinished);
    take(victim);
    return success();
  }

  // All of this is scoped to one runtime sequence. `gens` restarts BD id
  // allocation per sequence, so hazard state left over from an earlier one
  // describes ids that no longer name the same tasks. freedInFlight is the
  // one that bites rather than merely misleads: it holds aiex.dma_free_task
  // pointers, and those ops are erased once the sequence that owns them is
  // done, so carrying an entry forward leaves a dangling note location.
  void resetPerSequenceState() {
    gens.clear();
    awaitedConfigures.clear();
    pendingAwaitReleases.clear();
    startedOnChannel.clear();
    pushCount = 0;
    knownComplete.clear();
    pollProven.clear();
    reclaimed.clear();
    releasedTasks.clear();
    liveByTile.clear();
    freedInFlight.clear();
  }

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();

    WalkResult wr = device.walk([&](AIE::RuntimeSequenceOp seq) -> WalkResult {
      // Skip sequences already handled by the dynamic free-list pool path
      // (aie-lower-dynamic-bd-pool): they draw BD ids at runtime via
      // dma_bd_pool_pop and keep their scf.for rolled, which the static
      // straight-line allocator neither needs to touch nor can validate.
      bool dynamicPool = false;
      bool hasTasks = false;
      seq.walk([&](Operation *op) {
        dynamicPool |= isa<DMABdPoolPopOp>(op);
        hasTasks |=
            isa<DMAConfigureTaskOp, DMAConfigureTaskForOp, DMAStartBdChainOp,
                DMAStartTaskOp, DMAAwaitTaskOp, DMAFreeTaskOp>(op);
      });
      // Already-lowered instruction-only control flow needs no BD allocation.
      if (dynamicPool || !hasTasks)
        return WalkResult::advance();

      if (failed(validate(seq)))
        return WalkResult::interrupt();
      resetPerSequenceState();

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
          if (DMAConfigureTaskOp cfg = start.getTaskOp()) {
            knownComplete.erase(cfg);
            pollProven.erase(cfg);
            notePush(channelOf(cfg), cfg, start.getPushIssueToken(cfg));
          }
        } else if (auto poll = dyn_cast<NpuMaskPollOp>(op)) {
          notePoll(poll);
        } else if (auto await = dyn_cast<DMAAwaitTaskOp>(op)) {
          if (DMAConfigureTaskOp cfg = await.getTaskOp())
            if (cfg.getIssueToken())
              noteAwaited(channelOf(cfg));
          if (failed(recycleTask(await.getTask(), await, /*isAwait=*/true)))
            return WalkResult::interrupt();
        } else if (auto key = otherAwaitChannel(op)) {
          noteAwaited(*key);
          if (failed(recycleCompletedTasks(op)))
            return WalkResult::interrupt();
        } else if (auto freeOp = dyn_cast<DMAFreeTaskOp>(op)) {
          if (failed(recycleTask(freeOp.getTask(), freeOp, /*isAwait=*/false)))
            return WalkResult::interrupt();
          frees.push_back(freeOp);
        } else if (auto key = otherTokenChannel(op)) {
          notePush(*key, DMAConfigureTaskOp{}, true);
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
