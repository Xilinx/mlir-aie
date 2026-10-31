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
// aie-split-long-repeats has already made every start exactly one queue push.
//
// Before allocating, the pass also checks the per-channel hardware resources a
// sequence can exhaust (see verifyChannelUsage): task-completion-token (TCT)
// imbalance -- an await with no matching issue_token push on its channel would
// deadlock the host -- and DMA task-queue overflow, where more transfers are
// pushed onto a channel than its queue holds. On the straight-line IR the
// allocator sees, both are single-pass per-channel counts. What an await here
// proves relies on checkTokensConsumed (DmaQueueModel.h).
//
// With reclaim-bds, when a tile runs out of ids, the pass takes them back from
// a started task that was never released (see reclaimFor) rather than failing:
// from one a status poll already proves finished, or else by inserting a poll
// that does. Only allocations that would fail change, so a sequence that fits
// compiles to the same instructions either way. It is opt-in because the poll
// never returns if the task's completion depends on a later push.
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
#include "llvm/Support/MathExtras.h"

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
    std::optional<DmaQueueModel::ChannelKey> key = rawPushChannel(op);
    auto push = dyn_cast<NpuPushQueueOp>(op);
    auto copy = dyn_cast<NpuDmaMemcpyNdOp>(op);
    bool issuesToken =
        push
            ? push.getIssueToken()
            : copy && (copy.getIssueToken() ||
                       (key && (*key)[2] ==
                                   static_cast<int>(AIE::DMAChannelDir::S2MM)));
    return issuesToken ? key : std::nullopt;
  }

  // The channel a raw push, an aiex.npu.push_queue or aiex.npu.dma_memcpy_nd,
  // queues on, whether or not it issues a token.
  static std::optional<DmaQueueModel::ChannelKey>
  rawPushChannel(Operation *op) {
    if (auto push = dyn_cast<NpuPushQueueOp>(op))
      return DmaQueueModel::ChannelKey{static_cast<int>(push.getColumn()),
                                       static_cast<int>(push.getRow()),
                                       static_cast<int>(push.getDirection()),
                                       static_cast<int>(push.getChannel())};
    if (auto copy = dyn_cast<NpuDmaMemcpyNdOp>(op)) {
      auto allocation = AIE::ShimDMAAllocationOp::getForSymbol(
          op->getParentOfType<AIE::DeviceOp>(),
          copy.getMetadata().getRootReference());
      if (allocation)
        if (AIE::TileOp tile = allocation.getTileOp())
          return DmaQueueModel::keyOf(tile, allocation.getChannelDir(),
                                      allocation.getChannelIndex());
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
          return DmaQueueModel::keyOf(tile, allocation.getChannelDir(),
                                      allocation.getChannelIndex());
    }
    return std::nullopt;
  }

  // The BD ids each device's static DMAs take, by tile. Generators restart
  // at every PDI load, so this is read once per tile per load.
  //
  // Known limitation: after a load_pdi of another device, this and
  // loopingStartsOf read that sibling device's DMA programs, which races with
  // the sibling's own nested pipeline (aie-dma-to-npu adds globals to its
  // block) once devices run in parallel. The MLIR aiecc builds against runs
  // them one at a time today. Splitting the nest does not help, since the pass
  // manager merges adjacent device nests; a parallel build needs these
  // summaries computed at module scope before the nest.
  llvm::DenseMap<Operation *,
                 llvm::DenseMap<AIE::TileID, SmallVector<uint32_t>>>
      staticBdIds;

  // Mark every BD id a static DMA already took on `tile`, so this allocator
  // doesn't hand the same id to a runtime-sequence task.
  void seedFromStaticBds(AIE::DeviceOp device, AIE::TileOp tile,
                         BdIdGenerator &gen) {
    auto [it, inserted] = staticBdIds.try_emplace(device);
    if (inserted)
      for (AIE::DmaBody program : device.getOps<AIE::DmaBody>())
        llvm::append_range(it->second[program.getTileID()],
                           AIE::getAssignedBdIds(program));
    for (uint32_t id : it->second.lookup(tile.getTileID()))
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
      if (loadedDevice && loadedDevice != device)
        seedFromStaticBds(loadedDevice, tile, it->second);
    }
    return it->second;
  }

  // Each device's static aie.dma_starts whose BD chain never ends, by channel.
  llvm::DenseMap<Operation *,
                 std::map<DmaQueueModel::ChannelKey, AIE::DMAStartOp>>
      loopingStarts;
  // Channels an aiex.dma_channel_reset drained since the latest load.
  std::set<DmaQueueModel::ChannelKey> resetChannels;

  std::map<DmaQueueModel::ChannelKey, AIE::DMAStartOp> &
  loopingStartsOf(AIE::DeviceOp device) {
    auto [it, inserted] = loopingStarts.try_emplace(device);
    auto &starts = it->second;
    if (!inserted)
      return starts;
    for (AIE::DmaBody program : device.getOps<AIE::DmaBody>()) {
      AIE::TileID tile = program.getTileID();
      program->walk([&](AIE::DMAStartOp start) {
        if (start.getEndpoint())
          return;
        llvm::SmallPtrSet<Block *, 8> visited;
        Block *bd = start.getDest();
        while (bd && visited.insert(bd).second)
          bd = bd->getNumSuccessors() == 1 ? bd->getSuccessor(0) : nullptr;
        if (bd)
          starts.try_emplace({tile.col, tile.row,
                              static_cast<int>(start.getChannelDir()),
                              start.getChannelIndex()},
                             start);
      });
    }
    return starts;
  }

  // A task pushed behind a static program that never ends never runs. Only
  // a named load tells which programs run; the sequence's own device does
  // until the first load.
  LogicalResult checkNotBehindLoop(DMAStartTaskOp start,
                                   DMAConfigureTaskOp cfg) {
    if (opaqueLoad)
      return success();
    DmaQueueModel::ChannelKey key = channelOf(cfg);
    if (resetChannels.count(key))
      return success();
    auto &starts =
        loopingStartsOf(loadedDevice ? loadedDevice : getOperation());
    auto it = starts.find(key);
    if (it == starts.end())
      return success();
    InFlightDiagnostic diag =
        start.emitOpError("starts a task on a channel whose static DMA "
                          "program loops forever, so the task never runs; "
                          "end that program's BD chain, or reset the channel "
                          "with aiex.dma_channel_reset first");
    diag.attachNote(it->second.getLoc()) << "the program starts here";
    return diag;
  }

  // The device the latest aiex.npu.load_pdi configured. A sequence that
  // aie-materialize-runtime-sequences inlined runs on that device's static
  // DMAs, which stay in it rather than in the device that holds the sequence.
  AIE::DeviceOp loadedDevice;
  // The latest aiex.npu.load_pdi, when it names no device in this module.
  NpuLoadPdiOp opaqueLoad;
  // Configures from before the latest aiex.npu.load_pdi.
  llvm::SmallPtrSet<Operation *, 16> unloaded;

  // A PDI load reconfigures the array, so the tasks configured so far no
  // longer hold ids and allocation restarts around the new static BDs.
  void noteLoadPdi(NpuLoadPdiOp load) {
    loadedDevice = {};
    if (FlatSymbolRefAttr ref = load.getDeviceRefAttr())
      if (Operation *parent = getOperation()->getParentOp())
        loadedDevice =
            SymbolTable::lookupNearestSymbolFrom<AIE::DeviceOp>(parent, ref);
    opaqueLoad = loadedDevice ? NpuLoadPdiOp() : load;
    resetChannels.clear();
    for (auto &[tile, tasks] : liveByTile)
      for (DMAConfigureTaskOp task : tasks)
        unloaded.insert(task);
    gens.clear();
    liveByTile.clear();
    pendingAwaitReleases.clear();
    freedInFlight.clear();
    pushesByAllocation.clear();
    creditedPushes.clear();
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
  // modes differ: over-production costs the TCT FIFO no capacity (see
  // checkTokensConsumed for what it costs later), but a push onto a full queue
  // is dropped and its transfer never runs.
  //
  // Queue counting differs from token counting in three ways: every push takes
  // a slot, not just issue_token ones; a non-token push is still retired
  // implicitly, since in-order execution means awaiting a token drains
  // everything queued ahead of it too; and a repeat_count does not multiply
  // slots. rejectRuntimeControlFlow has already run, so one program-order pass
  // over straight-line IR is exact.
  LogicalResult verifyChannelUsage(AIE::RuntimeSequenceOp seq) {
    using ChannelKey = DmaQueueModel::ChannelKey;

    const AIETargetModel &tm =
        seq->getParentOfType<AIE::DeviceOp>().getTargetModel();

    std::map<ChannelKey, int> avail;
    DmaQueueModel queue;

    WalkResult wr = seq.walk([&](Operation *op) -> WalkResult {
      if (auto start = dyn_cast<DMAStartTaskOp>(op)) {
        DMAConfigureTaskOp cfg = start.getTaskOp();
        if (!cfg)
          return WalkResult::advance();
        ChannelKey key = channelOf(cfg);
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
        ChannelKey key = channelOf(cfg);
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
        // A raw wait may consume a token from a push this count cannot see,
        // such as a raw register write. Do not diagnose those here, but never
        // reuse a token already consumed by one.
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

  LogicalResult validate(AIE::RuntimeSequenceOp seq) {
    WalkResult unplaced = seq.walk([](DMAConfigureTaskOp cfg) {
      if (cfg.getTile().getDefiningOp<AIE::TileOp>())
        return WalkResult::advance();
      cfg.emitOpError("Cannot assign buffer descriptor ids to a DMA task "
                      "whose tile is not placed; run placement first.");
      return WalkResult::interrupt();
    });
    if (unplaced.wasInterrupted())
      return failure();
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

    // A start whose repeat count is a scratchpad parameter pushes the head
    // BD with an update that clears the low two bits of its id.
    AIE::DMABDOp alignedHead =
        llvm::any_of(op->getUsers(),
                     [](Operation *user) {
                       auto start = dyn_cast<DMAStartTaskOp>(user);
                       return start && start.hasRepeatParameter();
                     })
            ? op.getFirstBd()
            : nullptr;

    // Now allocate BD IDs for all unspecified BDs.
    result = op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd_op) {
      if (bd_op.getBdId().has_value())
        return WalkResult::advance();
      if (opaqueLoad) {
        bd_op.emitOpError(
                 "needs a buffer descriptor ID, but the aiex.npu.load_pdi "
                 "before it does not name a device in this module, so the "
                 "compiler cannot tell which ids the loaded PDI's static "
                 "DMAs use. Name the loaded device with device_ref = "
                 "@<device>, or give this aie.dma_bd a bd_id that the PDI "
                 "leaves free.")
                .attachNote(opaqueLoad.getLoc())
            << "the PDI is loaded here";
        return WalkResult::interrupt();
      }
      // channelIndex matters on a MemTile, where the AIE2 model partitions
      // BDs by channel parity (isBdChannelAccessible: an even channel can
      // only submit ids below 24, an odd channel only 24 and above).
      uint32_t alignment = bd_op == alignedHead ? 4 : 1;
      std::optional<int32_t> next_id = gen.nextBdId(op.getChannel(), alignment);
      while (!next_id && reclaimBds && succeeded(reclaimFor(op)))
        next_id = gen.nextBdId(op.getChannel(), alignment);
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
        else
          diag << " Or let the compiler poll for finished tasks and take "
                  "their ids (reclaim-bds, aiecc --reclaim-runtime-bds), "
                  "provided no task depends on a push issued after it.";
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
  };
  // Pushes on a channel, in program order, whose tokens no await has consumed
  // yet. A status poll proves a prefix of them finished, the first `finished`;
  // their tokens, if any, still wait for an await, so they stay here until one
  // consumes them.
  struct ChannelPushes {
    SmallVector<StartedTask, 8> tasks;
    size_t finished = 0;
  };
  std::map<DmaQueueModel::ChannelKey, ChannelPushes> startedOnChannel;
  uint64_t pushCount = 0;
  // Per configure, its pushes in startedOnChannel and how many of them are not
  // yet finished. A configure pushes onto its own channel alone.
  llvm::DenseMap<Operation *, unsigned> startCount;
  llvm::DenseMap<Operation *, unsigned> runningCount;

  void notePush(const DmaQueueModel::ChannelKey &key, DMAConfigureTaskOp task,
                bool issuesToken, DMAStartTaskOp start = {}) {
    uint64_t order = pushCount++;
    startedOnChannel[key].tasks.push_back({task, issuesToken, order});
    if (task) {
      ++startCount[task];
      ++runningCount[task];
    }
    AIE::ShimDMAAllocationOp allocation = task ? allocationOf(task) : nullptr;
    if (!allocation)
      return;
    auto &pushes = pushesByAllocation[allocation];
    std::optional<Volume> through =
        pushes.empty() ? Volume() : pushes.back().through;
    std::optional<Volume> moved =
        start ? pushVolume(start, task) : std::nullopt;
    if (through && moved)
      for (auto [term, bytes] : *moved)
        (*through)[term] += bytes;
    else
      through = std::nullopt;
    pushes.push_back({order, through});
  }
  // Configures whose completion an await has established.
  llvm::SmallPtrSet<Operation *, 16> knownComplete;
  // Configures a status poll, or the consumption an output's await proves
  // (#aie.consumed_by), shows finished. Kept apart from knownComplete so that
  // the ids an await releases do not move.
  llvm::SmallPtrSet<Operation *, 16> provenFinished;
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
    return DmaQueueModel::keyOf(cfg.getTileOp(), cfg.getDirection(),
                                cfg.getChannel());
  }

  // An await consumes the oldest outstanding token on the channel, regardless
  // of the configure named by its SSA operand. Only that token-issuing task and
  // the tasks queued ahead of it are known to have finished.
  void noteAwaited(const DmaQueueModel::ChannelKey &key) {
    ChannelPushes &channel = startedOnChannel[key];
    auto &started = channel.tasks;
    auto *it = llvm::find_if(
        started, [](const StartedTask &s) { return s.issuesToken; });
    if (it == started.end())
      return;
    size_t retired = std::distance(started.begin(), it) + 1;
    for (size_t i = 0; i < retired; ++i)
      if (DMAConfigureTaskOp task = started[i].task) {
        --startCount[task];
        if (i >= channel.finished)
          --runningCount[task];
      }
    for (size_t i = 0; i < retired; ++i)
      if (DMAConfigureTaskOp task = started[i].task)
        if (startCount.lookup(task) == 0)
          knownComplete.insert(task);
    // Counts are cumulative, so the newest push an allocation retired here
    // credits all that its older ones would.
    SmallVector<std::pair<AIE::ShimDMAAllocationOp, uint64_t>> newest;
    for (const StartedTask &push :
         llvm::reverse(llvm::make_range(started.begin(), std::next(it))))
      if (AIE::ShimDMAAllocationOp output =
              push.task ? allocationOf(push.task) : nullptr)
        if (llvm::none_of(newest,
                          [&](auto &seen) { return seen.first == output; }))
          newest.push_back({output, push.order});
    started.erase(started.begin(), std::next(it));
    channel.finished -= std::min(channel.finished, retired);
    for (auto [output, order] : newest)
      creditInputs(output, order);
  }

  // A status poll, or consumption an output proves, shows at most
  // `unfinished` pushes on `key` still queued or running. The channel runs its
  // queue in order, so those are the newest.
  void noteDrained(const DmaQueueModel::ChannelKey &key, size_t unfinished) {
    ChannelPushes &channel = startedOnChannel[key];
    if (channel.tasks.size() <= unfinished)
      return;
    for (size_t done = channel.tasks.size() - unfinished;
         channel.finished < done; ++channel.finished)
      if (DMAConfigureTaskOp task = channel.tasks[channel.finished].task)
        if (--runningCount[task] == 0)
          provenFinished.insert(task);
  }

  // Bytes as a linear form in the sequence's scratchpad parameters: the
  // constant at key -1, each parameter's coefficient at its state-table index.
  using Volume = std::map<int64_t, int64_t>;

  // Whether a * scaleA <= b * scaleB whatever values the parameters take.
  // Every coefficient is a length and every parameter is at least 0, so it is
  // enough to compare them one by one.
  static bool atMost(const Volume &a, int64_t scaleA, const Volume &b,
                     int64_t scaleB) {
    for (auto [key, coefficient] : a) {
      auto it = b.find(key);
      int64_t lhs, rhs = 0;
      if (llvm::MulOverflow(coefficient, scaleA, lhs) ||
          (it != b.end() && llvm::MulOverflow(it->second, scaleB, rhs)) ||
          lhs > rhs)
        return false;
    }
    return true;
  }

  // Whether the host library holds the scratchpad parameter `seq` reads at
  // state-table index `index` at 0 or above. Commands that never run together
  // share slots, so where `seq` does not list what it reads, every parameter
  // on the slot must be.
  bool nonNegative(AIE::RuntimeSequenceOp seq, int64_t index) {
    auto [it, inserted] =
        nonNegativeByIndex.try_emplace({seq.getOperation(), index}, false);
    if (!inserted)
      return it->second;
    auto module = getOperation()->getParentOfType<ModuleOp>();
    if (!module)
      return false;
    ArrayAttr reads = seq.getScratchpadParametersAttr();
    bool found = false, held = true;
    for (auto parameter : module.getOps<ScratchpadParameterOp>()) {
      if (parameter.getStateTableIdx() != index ||
          (reads &&
           !llvm::is_contained(reads.getAsValueRange<FlatSymbolRefAttr>(),
                               parameter.getSymName())))
        continue;
      std::optional<int32_t> min = parameter.getMinValue();
      found = true;
      held &= min && *min >= 0;
    }
    it->second = found && held;
    return it->second;
  }
  llvm::DenseMap<std::pair<Operation *, int64_t>, bool> nonNegativeByIndex;

  // The bytes one push of `cfg` by `start` moves, or nullopt where its chain
  // loops, iterates, continues outside its body, or takes its length from a
  // parameter not held non-negative.
  std::optional<Volume> pushVolume(DMAStartTaskOp start,
                                   DMAConfigureTaskOp cfg) {
    std::optional<int64_t> repeat =
        start.hasRepeatParameter()
            ? std::nullopt
            : getConstantIntValue(start.getPushRepeatCount(cfg));
    Region &body = cfg.getBody();
    if (!repeat || *repeat < 0 || body.empty())
      return std::nullopt;
    Volume bytes;
    llvm::SmallPtrSet<Block *, 8> seen;
    for (Block *block = &body.front(); block;) {
      if (!seen.insert(block).second)
        return std::nullopt;
      for (AIE::DMABDOp bd : block->getOps<AIE::DMABDOp>()) {
        std::optional<int32_t> len = bd.getConstantLen();
        if (!len || bd.getIteration() || bd.getNextBdId() ||
            bd.getLengthParameterAttr())
          return std::nullopt;
        int64_t width = bd.getBufferElementTypeWidthInBytes();
        bytes[-1] += static_cast<uint32_t>(*len) * width;
        if (std::optional<uint8_t> index = bd.getLengthStateTableIdx()) {
          std::optional<int32_t> unit = bd.getLengthUnit();
          if (!unit ||
              !nonNegative(start->getParentOfType<AIE::RuntimeSequenceOp>(),
                           *index))
            return std::nullopt;
          bytes[*index] += static_cast<int64_t>(*unit) * width;
        }
      }
      auto next = dyn_cast<AIE::NextBDOp>(block->getTerminator());
      block = next ? next.getDest() : nullptr;
    }
    for (auto &[key, coefficient] : bytes)
      if (llvm::MulOverflow(coefficient, *repeat + 1, coefficient))
        return std::nullopt;
    return bytes;
  }

  // The device's shim allocations, by name; a task names one of its own
  // device's.
  llvm::DenseMap<StringAttr, AIE::ShimDMAAllocationOp> allocationsByName;
  // Each output's proofs, with the input that holds each.
  llvm::DenseMap<
      Operation *,
      SmallVector<std::pair<AIE::ShimDMAAllocationOp, AIE::ConsumedByAttr>>>
      proofsByOutput;

  // The shim allocation `cfg` was configured for, when substitution kept it.
  AIE::ShimDMAAllocationOp allocationOf(DMAConfigureTaskOp cfg) {
    FlatSymbolRefAttr alloc = cfg.getAllocAttr();
    return alloc ? allocationsByName.lookup(alloc.getAttr()) : nullptr;
  }

  // A shim allocation's pushes since the latest load, in program order, with
  // the bytes it has moved through each: nullopt from the first whose bytes
  // are unknown.
  struct AllocationPush {
    uint64_t order;
    std::optional<Volume> through;
  };
  llvm::DenseMap<Operation *, SmallVector<AllocationPush>> pushesByAllocation;
  // Per (input, output), how many of the input's pushes that output has
  // already proven taken: both counts only grow, so a credit resumes there.
  llvm::DenseMap<std::pair<Operation *, Operation *>, size_t> creditedPushes;
  llvm::DenseMap<std::pair<Operation *, Operation *>, bool> balancedPairs;

  // Whether every stretch of every runtime sequence of this device, from its
  // start or a load to the next load or its end, ends with all of `input`'s
  // bytes covered by `output`'s under `proof`. The counts restart at each
  // stretch, but a load of the PDI already loaded leaves the array as it was,
  // so a stretch starts where the previous one, perhaps another dispatch's,
  // left it, and its counts hold only when none leaves input bytes behind.
  // They count only the two allocations' own tasks, so any other push on
  // either channel, in any sequence, voids the pair.
  bool leavesNothing(AIE::ShimDMAAllocationOp input,
                     AIE::ShimDMAAllocationOp output,
                     AIE::ConsumedByAttr proof) {
    auto entry = balancedPairs.try_emplace({input, output}, true);
    auto it = entry.first;
    if (!entry.second)
      return it->second;
    DmaQueueModel::ChannelKey inKey = DmaQueueModel::keyOf(
        input.getTileOp(), input.getChannelDir(), input.getChannelIndex());
    DmaQueueModel::ChannelKey outKey = DmaQueueModel::keyOf(
        output.getTileOp(), output.getChannelDir(), output.getChannelIndex());
    for (auto seq : getOperation().getOps<AIE::RuntimeSequenceOp>()) {
      Volume in, out;
      seq.walk([&](Operation *op) {
        if (isa<NpuLoadPdiOp>(op)) {
          if (!atMost(in, proof.getOutputBytes(), out, proof.getInputBytes())) {
            it->second = false;
            return WalkResult::interrupt();
          }
          in.clear();
          out.clear();
          return WalkResult::advance();
        }
        auto start = dyn_cast<DMAStartTaskOp>(op);
        DMAConfigureTaskOp cfg = start ? start.getTaskOp() : nullptr;
        AIE::ShimDMAAllocationOp allocation = cfg ? allocationOf(cfg) : nullptr;
        std::optional<DmaQueueModel::ChannelKey> channel =
            cfg ? std::optional(channelOf(cfg)) : rawPushChannel(op);
        bool foreign =
            channel && ((*channel == inKey && allocation != input) ||
                        (*channel == outKey && allocation != output));
        if (foreign) {
          it->second = false;
          return WalkResult::interrupt();
        }
        if (!allocation || (allocation != input && allocation != output))
          return WalkResult::advance();
        std::optional<Volume> moved = pushVolume(start, cfg);
        if (!moved) {
          it->second = false;
          return WalkResult::interrupt();
        }
        for (auto [key, bytes] : *moved)
          (allocation == input ? in : out)[key] += bytes;
        return WalkResult::advance();
      });
      if (!it->second ||
          !atMost(in, proof.getOutputBytes(), out, proof.getInputBytes())) {
        it->second = false;
        break;
      }
    }
    return it->second;
  }

  // `output`'s pushes up to `order` have finished: the host has received the
  // R bytes it moved through that push. Each input whose consumed_by names it
  // has then had its first R * input_bytes / output_bytes bytes taken, so its
  // pushes within those bytes have finished.
  void creditInputs(AIE::ShimDMAAllocationOp output, uint64_t order) {
    auto outPushes = pushesByAllocation.find(output);
    auto proofs = proofsByOutput.find(output);
    if (outPushes == pushesByAllocation.end() || proofs == proofsByOutput.end())
      return;
    auto byOrder = [](const auto &push, uint64_t bound) {
      return push.order < bound;
    };
    const AllocationPush *done =
        llvm::lower_bound(outPushes->second, order, byOrder);
    if (done == outPushes->second.end() || done->order != order ||
        !done->through)
      return;
    for (auto [input, proof] : proofs->second) {
      auto inPushes = pushesByAllocation.find(input);
      DmaQueueModel::ChannelKey key = DmaQueueModel::keyOf(
          input.getTileOp(), input.getChannelDir(), input.getChannelIndex());
      if (inPushes == pushesByAllocation.end() ||
          !leavesNothing(input, output, proof))
        continue;
      ArrayRef<AllocationPush> pushes = inPushes->second;
      size_t &credited = creditedPushes[{input, output}];
      size_t covered = credited;
      for (; covered < pushes.size(); ++covered) {
        const std::optional<Volume> &through = pushes[covered].through;
        if (!through || !atMost(*through, proof.getOutputBytes(),
                                *done->through, proof.getInputBytes()))
          break;
      }
      if (covered == credited)
        continue;
      credited = covered;
      auto &tasks = startedOnChannel[key].tasks;
      const StartedTask *at =
          llvm::lower_bound(tasks, pushes[covered - 1].order, byOrder);
      if (at != tasks.end() && at->order == pushes[covered - 1].order)
        noteDrained(key, tasks.end() - at - 1);
    }
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
    for (const auto &[key, channel] : startedOnChannel) {
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
    if (!startCount.lookup(cfg))
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
        provenFinished.contains(released.configure))
      return;
    UnsafeReuse &reuse =
        unsafeReuses
            .try_emplace(bd.getLoc(),
                         UnsafeReuse{id, tile.getCol(), tile.getRow(),
                                     released.freeOp->getLoc()})
            .first->second;
    reuse.tiles.insert({tile.getCol(), tile.getRow()});
    reuse.count++;
  }

  // A full ELF inlines a sequence at every call, so one source transfer can
  // reuse an id once per call; it is reported once, with how often.
  struct UnsafeReuse {
    uint32_t id;
    int col, row;
    Location freeLoc;
    std::set<std::pair<int, int>> tiles = {};
    int64_t count = 0;
  };
  llvm::MapVector<Location, UnsafeReuse> unsafeReuses;

  void emitUnsafeReuses() {
    for (auto &[loc, reuse] : unsafeReuses) {
      // Not bd->emitWarning(): its "see current operation" note verifies and
      // numbers the whole device for each warning, quadratic in the sequence.
      auto diag =
          mlir::emitWarning(loc)
          << "reuses buffer descriptor ID " << reuse.id << " on tile ("
          << reuse.col << "," << reuse.row
          << ") after it was released by an aiex.dma_free_task that had no "
             "completion guarantee, so the DMA it belonged to may still be "
             "running and this reprograms it underneath. Consume the "
             "outstanding tokens on the same tile, direction and channel "
             "through a task queued at or after this transfer -- each await "
             "consumes the oldest token, regardless of the task it names";
      diag.attachNote(reuse.freeLoc) << "released here";
      if (reuse.count > 1) {
        Diagnostic &note = diag.attachNote();
        note << "and " << reuse.count - 1
             << " more times from this transfer, on tile"
             << (reuse.tiles.size() > 1 ? "s " : " ");
        llvm::interleaveComma(reuse.tiles, note, [&](auto tile) {
          note << "(" << tile.first << "," << tile.second << ")";
        });
      }
    }
    unsafeReuses.clear();
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
    if (awaitedConfigures.contains(task_op) || reclaimed.contains(task_op) ||
        unloaded.contains(task_op))
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
      if ((knownComplete.contains(task) || provenFinished.contains(task)) &&
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
      const ChannelPushes &channel = startedOnChannel[key];
      ArrayRef<StartedTask> started = channel.tasks;
      auto last =
          llvm::find_if(llvm::reverse(started),
                        [&](const StartedTask &s) { return s.task == task; });
      size_t queuedBehind = std::distance(started.rbegin(), last);
      // Never started, or retired by an await and caught above.
      if (last == started.rend() ||
          started.size() - 1 - queuedBehind < channel.finished)
        continue;
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
    auto poll = NpuMaskPollOp::create(b, op.getLoc(), statusValue, compareValue,
                                      maskValue,
                                      /*buffer=*/nullptr, /*column=*/nullptr,
                                      /*row=*/nullptr);
    SmallVector<int32_t> ids;
    victim.walk([&](AIE::DMABDOp bd) { ids.push_back(*bd.getBdId()); });
    // Not poll.emitRemark(): its "see current operation" note verifies and
    // numbers the whole device for each poll, quadratic in the sequence.
    auto remark = mlir::emitRemark(poll.getLoc())
                  << "tile (" << tile.getCol() << "," << tile.getRow()
                  << ") has no free buffer descriptor id, so this poll waits "
                  << (victimQueuedBehind > 0
                          ? "until the queue behind an earlier task is short "
                            "enough to prove it finished"
                          : "until the channel of an earlier task is idle")
                  << " and takes back its ids ";
    llvm::interleaveComma(ids, remark);
    remark.attachNote(victim.getLoc()) << "the task it waits for";
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
    startCount.clear();
    runningCount.clear();
    knownComplete.clear();
    provenFinished.clear();
    reclaimed.clear();
    releasedTasks.clear();
    liveByTile.clear();
    freedInFlight.clear();
    loadedDevice = {};
    opaqueLoad = {};
    unloaded.clear();
    resetChannels.clear();
    pushesByAllocation.clear();
    creditedPushes.clear();
  }

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();
    staticBdIds.clear();
    loopingStarts.clear();
    nonNegativeByIndex.clear();
    balancedPairs.clear();
    allocationsByName.clear();
    proofsByOutput.clear();
    for (auto allocation : device.getOps<AIE::ShimDMAAllocationOp>())
      allocationsByName[allocation.getSymNameAttr()] = allocation;
    for (auto input : device.getOps<AIE::ShimDMAAllocationOp>())
      if (ArrayAttr proofs = input.getConsumedByAttr())
        for (auto proof : proofs.getAsRange<AIE::ConsumedByAttr>())
          if (AIE::ShimDMAAllocationOp output =
                  allocationsByName.lookup(proof.getOutput().getAttr()))
            proofsByOutput[output].push_back({input, proof});

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
            if (unloaded.contains(cfg)) {
              start.emitOpError(
                  "starts a task configured before an aiex.npu.load_pdi, "
                  "which reconfigured the array; its buffer descriptor ids "
                  "may now belong to the loaded device's static DMAs. "
                  "Configure the task after the load_pdi instead.");
              return WalkResult::interrupt();
            }
            // Only an explicit aiex.dma_free_task releases a task that is
            // started again (an await or a reclaim keeps the ids of one that
            // is), so this start would push ids that may already describe
            // another task's transfer.
            if (releasedTasks.contains(cfg)) {
              start.emitOpError(
                  "starts a task whose buffer descriptors an earlier "
                  "aiex.dma_free_task released; their ids may already be "
                  "reprogrammed for another task. Free the task after its "
                  "last start instead.");
              return WalkResult::interrupt();
            }
            if (failed(checkNotBehindLoop(start, cfg)))
              return WalkResult::interrupt();
            knownComplete.erase(cfg);
            provenFinished.erase(cfg);
            notePush(channelOf(cfg), cfg, start.getPushIssueToken(cfg), start);
          }
        } else if (auto load = dyn_cast<NpuLoadPdiOp>(op)) {
          noteLoadPdi(load);
        } else if (auto reset = dyn_cast<DmaChannelResetOp>(op)) {
          resetChannels.insert(DmaQueueModel::keyOf(
              reset.getTileOp(), reset.getDirection(), reset.getChannel()));
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
    emitUnsafeReuses();
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
