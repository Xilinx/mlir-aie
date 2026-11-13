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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Threading.h"

#include <array>
#include <deque>
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

  llvm::DenseMap<AIE::TileID, BdIdGenerator> gens;

  // Bytes as a linear form in the sequence's scratchpad parameters: the
  // constant at key -1, each parameter's coefficient at its state-table index.
  using Volume = std::map<int64_t, int64_t>;

  // Whether `op` writes array registers raw, at an address it does not fix or
  // over one of `keys`' task queues, so pushing a task no count sees.
  static bool writesQueue(Operation *op,
                          ArrayRef<DmaQueueModel::ChannelKey> keys) {
    std::optional<uint32_t> first;
    uint64_t bytes = 4;
    if (auto write = dyn_cast<NpuWrite32Op>(op)) {
      if (write.getBuffer())
        return false;
      first = write.getAbsoluteAddress();
    } else if (auto write = dyn_cast<NpuMaskWrite32Op>(op)) {
      if (write.getBuffer())
        return false;
      first = write.getAbsoluteAddress();
    } else if (auto write = dyn_cast<NpuBlockWriteOp>(op)) {
      if (write.getBuffer())
        return false;
      first = write.getAbsoluteAddress();
      auto data = dyn_cast<MemRefType>(write.getData().getType());
      if (!data || !data.hasStaticShape())
        return true;
      bytes = data.getNumElements() * data.getElementTypeBitWidth() / 8;
    } else {
      return false;
    }
    if (!first)
      return true;
    const AIETargetModel &tm =
        op->getParentOfType<AIE::DeviceOp>().getTargetModel();
    return llvm::any_of(keys, [&](const DmaQueueModel::ChannelKey &key) {
      uint64_t queue = tm.getDmaQueueAddress(
          key[0], key[1], key[3], static_cast<AIE::DMAChannelDir>(key[2]));
      return queue >= *first && queue < *first + bytes;
    });
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

  // Keyed by (col, row), not the aie.tile op's own identity: two tile ops
  // for one coordinate (accepted elsewhere, see runtime_vs_static_multi_
  // tile.mlir) must share one generator, not bypass each other's claims.
  BdIdGenerator &getGeneratorForTile(AIE::TileOp tile) {
    AIE::TileID id = tile.getTileID();
    auto it = gens.find(id);
    if (it == gens.end()) {
      AIE::DeviceOp device = tile->getParentOfType<AIE::DeviceOp>();
      it = gens.insert({id, BdIdGenerator(tile.getCol(), tile.getRow(),
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
    holdsByChannel.clear();
    holdsOn.clear();
    copyOwned.clear();
    pendingAwaitReleases.clear();
    freedInFlight.clear();
    pushesByAllocation.clear();
    creditedPushes.clear();
    awaitedThrough.clear();
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
      } else if (QueueEffect raw = rawQueueEffect(op, *symbols);
                 raw.kind == QueueEffect::Kind::Await) {
        // A raw wait may consume a token from a push this count cannot see,
        // such as a raw register write. Do not diagnose those here, but never
        // reuse a token already consumed by one.
        avail[raw.key] = std::max(0, avail[raw.key] - 1);
        queue.awaitToken(raw.key);
      } else if (raw.kind == QueueEffect::Kind::Push) {
        avail[raw.key] += raw.issuesToken;
        queue.push(raw.key, raw.issuesToken);
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
        if (const HeldBy *held =
                copyHolding(tile.getTileID(), bd_op.getBdId().value())) {
          InFlightDiagnostic diag =
              op.emitOpError("Specified buffer descriptor ID ")
              << bd_op.getBdId().value()
              << " is the one an aiex.npu.dma_memcpy_nd on this tile "
                 "programs, and that copy may still be in flight here. Give "
                 "the task another ID, or ";
          if (held->issuesToken)
            diag << "wait for the copy first (aiex.npu.sync or "
                    "aiex.npu.dma_wait on its channel, consuming its token).";
          else
            diag << "retire the copy first: it issues no token, so only an "
                    "await of a later tokened task on its channel, a status "
                    "poll, or issue_token = true on it and a wait for it "
                    "shows it done.";
          diag.attachNote(held->copy->getLoc()) << "the copy is here";
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
        checkReallocation(tile, bd_op.getBdId().value(), bd_op.getLoc());
        gen.assignBdId(bd_op.getBdId().value());
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();

    AIE::DMABDOp head = op.getFirstBd();
    uint32_t headAlignment = DMAStartTaskOp::headBdIdAlignment(op.getResult());

    // The ids a copy reprograms while this task may run are not its to take,
    // including any a reclaim frees on the way.
    SmallVector<int32_t> blocked;
    auto block = [&] {
      auto it = forbidden.find(op);
      if (it == forbidden.end())
        return;
      for (auto [id, copy] : it->second)
        if (!gen.bdIdAlreadyAssigned(id)) {
          gen.assignBdId(id);
          blocked.push_back(id);
        }
    };
    block();
    unsigned tookBack = 0;

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
      uint32_t alignment = bd_op == head ? headAlignment : 1;
      std::optional<int32_t> next_id = gen.nextBdId(op.getChannel(), alignment);
      while (!next_id && reclaimBds && succeeded(reclaimFor(op))) {
        ++tookBack;
        block();
        next_id = gen.nextBdId(op.getChannel(), alignment);
      }
      if (!next_id) {
        auto diag =
            op.emitOpError()
            << "Too many simultaneously active buffer descriptors on tile ("
            << tile.getCol() << "," << tile.getRow()
            << "), which supports up to "
            << targetModel.getNumBDsForChannel(tile.getCol(), tile.getRow(),
                                               op.getChannel())
            << ".";
        SmallVector<std::tuple<int32_t, Operation *, bool>> copied;
        if (auto it = forbidden.find(op); it != forbidden.end())
          for (auto [id, copy] : it->second)
            if (llvm::is_contained(blocked, id))
              copied.push_back({id, copy, true});
        for (auto [owner, id] : copyOwned)
          if (owner == tile.getTileID() &&
              targetModel.isBdChannelAccessible(tile.getCol(), tile.getRow(),
                                                id, op.getChannel()))
            copied.push_back(
                {id, copyHolding(tile.getTileID(), id)->copy, false});
        llvm::sort(copied);
        if (!copied.empty()) {
          diag << " Of those, aiex.npu.dma_memcpy_nd ops program "
               << (copied.size() > 1 ? "ids " : "id ");
          llvm::interleaveComma(
              copied, diag, [&](auto &entry) { diag << std::get<0>(entry); });
          diag << " while this task may run (see the notes).";
        }
        // reclaimFor only ever considers a liveByTile task, never a copy, so
        // its advice -- and the task-shaped reasons it failed -- apply only
        // where some other task actually lives on this tile to reclaim from.
        bool otherLiveTask = llvm::any_of(
            liveByTile[tile.getTileID()],
            [&](DMAConfigureTaskOp t) { return !releasedTasks.contains(t); });
        if (otherLiveTask) {
          diag << " Emit an aiex.dma_await_task to free BDs for reuse; it "
                  "waits for hardware completion, so the recycled ids are no "
                  "longer in flight. aiex.dma_free_task also recycles ids but "
                  "does NOT wait for completion -- using it before the task "
                  "has finished is a race -- so reach for it only when some "
                  "other synchronization already guarantees completion (see "
                  "programming_guide/section-2/section-2d/DMATasks.md).";
          if (reclaimBds && tookBack)
            diag << " The ids the compiler took back did not cover it, and "
                    "it could not take more: every other task holding one is "
                    "not started yet, is started again later, waits on a "
                    "lock this or a later task releases, or runs on a "
                    "channel whose status it cannot poll.";
          else if (reclaimBds)
            diag << " The compiler could not take any back either: every "
                    "task holding one is not started yet, is started again "
                    "later, waits on a lock this or a later task releases, "
                    "or runs on a channel whose status it cannot poll.";
          else
            diag << " Or let the compiler poll for finished tasks and take "
                    "their ids (reclaim-bds, aiecc --reclaim-runtime-bds), "
                    "provided no task depends on a push issued after it.";
        } else if (!copied.empty()) {
          diag << " Every id in use here is an aiex.npu.dma_memcpy_nd copy "
                  "(reclaim-bds does not consider those); retire the ones "
                  "the notes name, or give this task's buffer descriptors "
                  "other ids.";
        }
        for (auto [id, copyOp, isForbidden] : copied) {
          auto copy = cast<NpuDmaMemcpyNdOp>(copyOp);
          Diagnostic &note = diag.attachNote(copy.getLoc());
          note << "this copy " << (isForbidden ? "reprograms" : "holds")
               << " buffer descriptor ID " << id
               << (isForbidden ? " while the task may run; "
                               : " until it is retired: ");
          if (rawQueueEffect(copy, *symbols).issuesToken)
            note << "wait for it (aiex.npu.sync or aiex.npu.dma_wait on its "
                    "channel, consuming its token).";
          else
            note << "it issues no token, so only an await of a later tokened "
                    "task on its channel, a status poll, or issue_token = "
                    "true on it and a wait for it shows it done.";
        }
        return WalkResult::interrupt();
      }
      checkReallocation(tile, *next_id, bd_op.getLoc());
      bd_op.setBdId(next_id);
      assignedHere.push_back(bd_op);
      return WalkResult::advance();
    });
    for (int32_t id : blocked)
      gen.freeBdId(id);
    if (result.wasInterrupted())
      return failure();

    liveByTile[tile.getTileID()].push_back(op);
    return success();
  }

  // One push onto a channel: the task it started (null for a raw push or
  // memcpy transfer) and whether that push issues a token. A task started more
  // than once may issue a token on only some of its pushes.
  struct StartedTask {
    DMAConfigureTaskOp task;
    Operation *origin;
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
  llvm::DenseMap<Operation *, uint64_t> lastPushOrder;

  void notePush(const DmaQueueModel::ChannelKey &key, DMAConfigureTaskOp task,
                bool issuesToken, Operation *origin,
                AIE::ShimDMAAllocationOp allocation,
                std::optional<Volume> moved) {
    uint64_t order = pushCount++;
    startedOnChannel[key].tasks.push_back({task, origin, issuesToken, order});
    if (task) {
      ++startCount[task];
      ++runningCount[task];
      lastPushOrder[task] = order;
    }
    if (!allocation)
      return;
    auto &pushes = pushesByAllocation[allocation];
    std::optional<Volume> through =
        pushes.empty() ? Volume() : pushes.back().through;
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
  // Each top-level op's place in the sequence, and per lock the last place a
  // task releasing it is configured or started (see reclaimFor).
  llvm::DenseMap<Operation *, unsigned> positionOf;
  llvm::DenseMap<Value, unsigned> lastReleaseOf;

  void indexLockReleases(AIE::RuntimeSequenceOp seq) {
    unsigned position = 0;
    for (Operation &op : seq.getBody().getOps()) {
      positionOf[&op] = position++;
      DMAConfigureTaskOp cfg = dyn_cast<DMAConfigureTaskOp>(op);
      if (auto start = dyn_cast<DMAStartTaskOp>(op))
        cfg = start.getTaskOp();
      if (!cfg)
        continue;
      cfg.walk([&](AIE::UseLockOp lock) {
        if (lock.release())
          lastReleaseOf[lock.getLock()] = positionOf[&op];
      });
    }
  }

  // Configures allocated on each tile, in program order.
  llvm::DenseMap<AIE::TileID, SmallVector<DMAConfigureTaskOp, 16>> liveByTile;

  // An aiex.npu.dma_memcpy_nd's BD id is held on its tile from its push until
  // the queue model counts that push finished. Holds are kept per channel in
  // push order, so they retire as a prefix; per tile and id, the copies
  // holding it by push order, and whether they took it in the generator.
  struct CopyHold {
    uint64_t order;
    AIE::TileID tile;
    int32_t id;
  };
  std::map<DmaQueueModel::ChannelKey, std::deque<CopyHold>> holdsByChannel;
  using TileBdId = std::pair<AIE::TileID, int32_t>;
  struct HeldBy {
    NpuDmaMemcpyNdOp copy;
    bool issuesToken;
  };
  llvm::DenseMap<TileBdId, std::map<uint64_t, HeldBy>> holdsOn;
  llvm::DenseSet<TileBdId> copyOwned;
  // Per configure, the ids it may not take and the copy that reprograms each
  // while it may run.
  llvm::DenseMap<Operation *, SmallVector<std::pair<int32_t, Operation *>>>
      forbidden;
  bool reallocate = false;
  // What one allocation of the sequence did that the next undoes.
  SmallVector<AIE::DMABDOp> assignedHere;
  SmallVector<Operation *> insertedHere;

  // The oldest copy still holding `id` on `tile`, if copies hold it.
  const HeldBy *copyHolding(AIE::TileID tile, int32_t id) const {
    if (!copyOwned.contains({tile, id}))
      return nullptr;
    return &holdsOn.find({tile, id})->second.begin()->second;
  }

  // Return a task's id to the pool; a copy waiting for it takes it at once.
  void releaseTaskBdId(AIE::TileID tile, BdIdGenerator &gen, int32_t id) {
    gen.freeBdId(id);
    if (!holdsOn.contains({tile, id}))
      return;
    gen.assignBdId(id);
    copyOwned.insert({tile, id});
  }

  static DMAStartTaskOp startAfter(DMAConfigureTaskOp task, Operation *op) {
    DMAStartTaskOp first;
    for (Operation *user : task.getResult().getUsers())
      if (auto start = dyn_cast<DMAStartTaskOp>(user);
          start && start->getBlock() == op->getBlock() &&
          op->isBeforeInBlock(start) &&
          (!first || start->isBeforeInBlock(first)))
        first = start;
    return first;
  }

  // The copy reprograms its BD at once, so a task on that id must neither be
  // in flight nor be started again after it.
  LogicalResult holdCopy(NpuDmaMemcpyNdOp copy,
                         const DmaQueueModel::ChannelKey &key, uint64_t order,
                         bool issuesToken) {
    AIE::TileOp tile = allocationOf(copy).getTileOp();
    AIE::TileID tileId = tile.getTileID();
    auto id = static_cast<int32_t>(copy.getId());
    if (auto held = holdsOn.find({tileId, id});
        held != holdsOn.end() && warnUnsafeBdReuse) {
      const HeldBy &earlier = held->second.begin()->second;
      auto [entry, inserted] = copiesOverCopies.try_emplace(
          copy.getLoc(),
          CopyOverCopy{id, tile.getCol(), tile.getRow(), earlier.copy->getLoc(),
                       earlier.issuesToken});
      entry->second.tiles.insert({tile.getCol(), tile.getRow()});
      entry->second.count++;
    }
    BdIdGenerator &gen = getGeneratorForTile(tile);
    holdsByChannel[key].push_back({order, tileId, id});
    holdsOn[{tileId, id}][order] = {copy, issuesToken};
    if (!gen.bdIdAlreadyAssigned(id)) {
      checkReallocation(tile, id, copy.getLoc());
      gen.assignBdId(id);
      copyOwned.insert({tileId, id});
      return success();
    }
    if (copyOwned.contains({tileId, id}))
      return success();
    for (DMAConfigureTaskOp task : liveByTile[tileId]) {
      if (releasedTasks.contains(task) || unloaded.contains(task))
        continue;
      AIE::DMABDOp held;
      task.walk([&](AIE::DMABDOp bd) {
        if (bd.getBdId() == id)
          held = bd;
      });
      if (!held)
        continue;
      bool running = runningCount.lookup(task);
      // A task not yet started even once has no prior push to its name: its
      // only start landing after the copy is its first, not a restart.
      bool everStarted = lastPushOrder.contains(task);
      DMAStartTaskOp restart = startAfter(task, copy);
      if (!running && !restart)
        return success();
      if (!llvm::is_contained(assignedHere, held)) {
        InFlightDiagnostic diag =
            task.emitOpError("Specified buffer descriptor ID ") << id;
        if (running)
          diag << " is the one an aiex.npu.dma_memcpy_nd on this tile "
                  "programs while this task may be in flight. Give the task "
                  "another ID, or await it before the copy.";
        else if (everStarted)
          diag << " is the one an aiex.npu.dma_memcpy_nd on this tile "
                  "programs before this task is started again, so the "
                  "restart would run the copy's descriptor. Give the task "
                  "another ID, or configure a new task after the copy.";
        else
          diag << " is the one an aiex.npu.dma_memcpy_nd on this tile "
                  "programs after this task's own configure wrote it, so "
                  "starting the task would launch the copy's descriptor "
                  "instead of its own. Give the task another ID, or start it "
                  "before the copy.";
        diag.attachNote(copy.getLoc()) << "the copy is here";
        if (!running)
          diag.attachNote(restart.getLoc())
              << (everStarted ? "the task is started again here"
                              : "the task starts here, after the copy");
        return failure();
      }
      assert(llvm::none_of(forbidden[task],
                           [&](auto &entry) { return entry.first == id; }) &&
             "a forbidden id was handed out again");
      forbidden[task].push_back({id, copy});
      reallocate = true;
      return success();
    }
    // The id was already taken in the generator, owned by neither a copy
    // nor a live task: only a static DMA program claims an id that way.
    copy.emitOpError("Specified buffer descriptor ID ")
        << id
        << " is already in use. Release the earlier task first: "
           "aiex.dma_await_task waits for hardware completion before its "
           "BDs are reusable. aiex.dma_free_task also releases them but "
           "does NOT wait, so it is only safe when some other "
           "synchronization already guarantees that task has finished "
           "(see programming_guide/section-2/section-2d/DMATasks.md).";
    return failure();
  }

  void releaseRetiredCopies(const DmaQueueModel::ChannelKey &key) {
    auto holds = holdsByChannel.find(key);
    if (holds == holdsByChannel.end())
      return;
    const ChannelPushes &channel = startedOnChannel[key];
    std::optional<uint64_t> oldest;
    if (channel.finished < channel.tasks.size())
      oldest = channel.tasks[channel.finished].order;
    while (!holds->second.empty() &&
           (!oldest || holds->second.front().order < *oldest)) {
      CopyHold hold = holds->second.front();
      holds->second.pop_front();
      auto on = holdsOn.find({hold.tile, hold.id});
      on->second.erase(hold.order);
      if (!on->second.empty())
        continue;
      holdsOn.erase(on);
      if (copyOwned.erase({hold.tile, hold.id}))
        gens.find(hold.tile)->second.freeBdId(hold.id);
    }
  }

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
              push.task ? allocationOf(push.task)
                        : allocationOf(dyn_cast<NpuDmaMemcpyNdOp>(push.origin)))
        if (llvm::none_of(newest,
                          [&](auto &seen) { return seen.first == output; }))
          newest.push_back({output, push.order});
    started.erase(started.begin(), std::next(it));
    channel.finished -= std::min(channel.finished, retired);
    for (auto [output, order] : newest)
      creditInputs(output, order);
    releaseRetiredCopies(key);
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
    releaseRetiredCopies(key);
  }

  // Whether a * scaleA <= b * scaleB whatever values the parameters take.
  // Every parameter is at least 0, so it is enough to compare the terms one
  // by one, a term missing from either side counting as 0.
  static bool atMost(const Volume &a, int64_t scaleA, const Volume &b,
                     int64_t scaleB) {
    std::set<int64_t> keys;
    for (const auto &term :
         llvm::concat<const std::pair<const int64_t, int64_t>>(a, b))
      keys.insert(term.first);
    for (int64_t key : keys) {
      auto x = a.find(key), y = b.find(key);
      int64_t lhs = 0, rhs = 0;
      if ((x != a.end() && llvm::MulOverflow(x->second, scaleA, lhs)) ||
          (y != b.end() && llvm::MulOverflow(y->second, scaleB, rhs)) ||
          lhs > rhs)
        return false;
    }
    return true;
  }

  // Whether `bytes` is a multiple of `granule` whatever values the parameters
  // take: every term is.
  static bool wholeGranules(const Volume &bytes, int64_t granule) {
    return !granule || llvm::all_of(bytes, [&](const auto &term) {
      return term.second % granule == 0;
    });
  }

  // The output bytes `proof` lets count: `received` rounded down to a
  // multiple of its granule, at least `received` - (granule - 1).
  static Volume countedOutput(const Volume &received,
                              AIE::ConsumedByAttr proof) {
    int64_t granule = proof.getGranuleBytes();
    Volume counted = received;
    if (!wholeGranules(received, granule))
      counted[-1] -= granule - 1;
    return counted;
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
        std::optional<int64_t> len = bd.getStaticLenInBytes();
        if (!len || bd.getIteration() || bd.getNextBdId() ||
            bd.getLengthParameterAttr())
          return std::nullopt;
        int64_t width = bd.getBufferElementTypeWidthInBytes();
        bytes[-1] += *len;
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

  // The bytes one push of `copy` moves, or nullopt unless all its sizes are
  // constants.
  static std::optional<Volume> copyVolume(NpuDmaMemcpyNdOp copy) {
    std::optional<int64_t> bytes = copy.getStaticBytesStreamed();
    if (!bytes)
      return std::nullopt;
    Volume volume;
    volume[-1] = *bytes;
    return volume;
  }

  // The device's symbols; a task names one of its own device's allocations.
  // runOnOperation sets it before visiting any op.
  std::optional<SymbolTable> symbols;
  // Each output's proofs, with the input that holds each.
  llvm::DenseMap<
      Operation *,
      SmallVector<std::pair<AIE::ShimDMAAllocationOp, AIE::ConsumedByAttr>>>
      proofsByOutput;

  // The shim allocation `cfg` was configured for, when substitution kept it.
  AIE::ShimDMAAllocationOp allocationOf(DMAConfigureTaskOp cfg) {
    FlatSymbolRefAttr alloc = cfg.getAllocAttr();
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return alloc ? symbols->lookup<AIE::ShimDMAAllocationOp>(alloc.getAttr())
                 : nullptr;
  }

  // The shim allocation `copy` moves through.
  AIE::ShimDMAAllocationOp allocationOf(NpuDmaMemcpyNdOp copy) {
    if (!copy)
      return nullptr;
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return symbols->lookup<AIE::ShimDMAAllocationOp>(
        copy.getMetadata().getRootReference());
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
  // Per output, the bytes moved through its newest push an await retired since
  // the latest load (nullopt where unknown), and how many pushes the sequence
  // had issued by that await.
  struct Awaited {
    std::optional<Volume> through;
    uint64_t pushesBefore;
  };
  llvm::DenseMap<Operation *, Awaited> awaitedThrough;

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
    if (entry.second)
      entry.first->second = llvm::all_of(
          getOperation().getOps<AIE::RuntimeSequenceOp>(),
          [&](AIE::RuntimeSequenceOp seq) {
            return stretchesUnbalanced(seq, nullptr, proofDevice(proof), input,
                                       output, {input}, {output},
                                       proof) == Unbalanced::No;
          });
    return entry.first->second;
  }

  // The device whose array `proof` describes: the one materialization copied
  // it from, else this one; none where that device is not in the module.
  AIE::DeviceOp proofDevice(AIE::ConsumedByAttr proof) {
    FlatSymbolRefAttr device = proof.getDevice();
    Operation *module = getOperation()->getParentOp();
    if (!device)
      return getOperation();
    return module ? SymbolTable::lookupNearestSymbolFrom<AIE::DeviceOp>(module,
                                                                        device)
                  : AIE::DeviceOp();
  }

  // The device `load` configures; none where it names none in the module.
  AIE::DeviceOp loadedBy(NpuLoadPdiOp load) {
    FlatSymbolRefAttr ref = load.getDeviceRefAttr();
    Operation *module = getOperation()->getParentOp();
    return ref && module ? SymbolTable::lookupNearestSymbolFrom<AIE::DeviceOp>(
                               module, ref)
                         : AIE::DeviceOp();
  }

  // Why a stretch of `proof`'s device may leave `input`'s bytes behind.
  enum class Unbalanced {
    No,
    // A push of either moves bytes no count follows: a parameter not held
    // non-negative, a chain, or a loop of a dynamic-pool sequence.
    Unsized,
    // Other traffic shares their channels.
    Traffic,
    // The input moves more than the output proves taken.
    Excess,
    // The output may end partway through an object of the join its proof
    // passes, whose later segments it has then taken less of.
    PartGranule,
  };

  // Whether each stretch of `seq` after `from` (all of them when null) with
  // `target` loaded ends with the bytes `proof`'s input channel moved covered
  // by those its output channel moved. Pushes through `inputs` and `outputs`
  // count as the proof's, and with `byChannel` any push on a circuit channel
  // (no packet header on it or on `input`/`output`), which carries only that
  // route's words; any other on those channels is traffic the proof does not
  // describe. Stretches under another PDI end at a real load of `target`.
  // `at` receives where the verdict stops, null for the sequence end, and the
  // stretch's counts there.
  struct Verdict {
    Operation *where = nullptr;
    Volume in, out;
  };
  Unbalanced stretchesUnbalanced(
      AIE::RuntimeSequenceOp seq, Operation *from, AIE::DeviceOp target,
      AIE::ShimDMAAllocationOp input, AIE::ShimDMAAllocationOp output,
      ArrayRef<AIE::ShimDMAAllocationOp> inputs,
      ArrayRef<AIE::ShimDMAAllocationOp> outputs, AIE::ConsumedByAttr proof,
      bool byChannel = false, Verdict *at = nullptr) {
    DmaQueueModel::ChannelKey inKey = DmaQueueModel::keyOf(input);
    DmaQueueModel::ChannelKey outKey = DmaQueueModel::keyOf(output);
    auto fromLoad = dyn_cast_or_null<NpuLoadPdiOp>(from);
    AIE::DeviceOp loaded = fromLoad ? loadedBy(fromLoad) : getOperation();
    bool counting = !from;
    Volume in, out;
    Unbalanced why = Unbalanced::No;
    seq.walk([&](Operation *op) {
      if (op == from) {
        counting = true;
        return WalkResult::advance();
      }
      if (!counting)
        return WalkResult::advance();
      if (auto load = dyn_cast<NpuLoadPdiOp>(op)) {
        if (target && loaded == target &&
            !wholeGranules(out, proof.getGranuleBytes())) {
          why = Unbalanced::PartGranule;
          if (at)
            *at = {op, in, out};
          return WalkResult::interrupt();
        }
        if (target && loaded == target &&
            !atMost(in, proof.getOutputBytes(), out, proof.getInputBytes())) {
          why = Unbalanced::Excess;
          if (at)
            *at = {op, in, out};
          return WalkResult::interrupt();
        }
        in.clear();
        out.clear();
        loaded = loadedBy(load);
        return WalkResult::advance();
      }
      if (!target || loaded != target)
        return WalkResult::advance();
      auto start = dyn_cast<DMAStartTaskOp>(op);
      DMAConfigureTaskOp cfg = start ? start.getTaskOp() : nullptr;
      auto copy = dyn_cast<NpuDmaMemcpyNdOp>(op);
      AIE::ShimDMAAllocationOp allocation =
          cfg ? allocationOf(cfg) : allocationOf(copy);
      std::optional<DmaQueueModel::ChannelKey> channel;
      if (cfg)
        channel = channelOf(cfg);
      else if (QueueEffect raw = rawQueueEffect(op, *symbols);
               raw.kind == QueueEffect::Kind::Push)
        channel = raw.key;
      bool circuit = byChannel && allocation && !allocation.getPacket();
      bool isInput =
          llvm::is_contained(inputs, allocation) ||
          (circuit && channel && *channel == inKey && !input.getPacket());
      bool isOutput =
          llvm::is_contained(outputs, allocation) ||
          (circuit && channel && *channel == outKey && !output.getPacket());
      if ((channel && ((*channel == inKey && !isInput) ||
                       (*channel == outKey && !isOutput))) ||
          writesQueue(op, {inKey, outKey})) {
        why = Unbalanced::Traffic;
        if (at)
          *at = {op, in, out};
        return WalkResult::interrupt();
      }
      if (!allocation || (!isInput && !isOutput))
        return WalkResult::advance();
      // A dynamic-pool sequence keeps its loops, whose trip counts these
      // counts cannot follow.
      std::optional<Volume> moved = op->getParentOp() != seq ? std::nullopt
                                    : cfg ? pushVolume(start, cfg)
                                          : copyVolume(copy);
      if (!moved) {
        why = Unbalanced::Unsized;
        if (at)
          *at = {op, in, out};
        return WalkResult::interrupt();
      }
      for (auto [key, bytes] : *moved)
        (isInput ? in : out)[key] += bytes;
      return WalkResult::advance();
    });
    if (why == Unbalanced::No && target && loaded == target &&
        !wholeGranules(out, proof.getGranuleBytes()))
      why = Unbalanced::PartGranule;
    else if (why == Unbalanced::No && target && loaded == target &&
             !atMost(in, proof.getOutputBytes(), out, proof.getInputBytes()))
      why = Unbalanced::Excess;
    if (at && why != Unbalanced::No && !at->where)
      *at = {nullptr, in, out};
    return why;
  }

  // Ending on another device's PDI hands that array on as leavesNothing
  // describes, so the last stretch is held to the loaded device's own proofs,
  // read off it as seedFromStaticBds reads its BDs: pushes on a proof's
  // channels count through the copies materialization stamped with it, and
  // any other there is traffic the proof does not describe.
  LogicalResult checkForeignTail(AIE::RuntimeSequenceOp seq) {
    NpuLoadPdiOp last;
    for (auto load : seq.getOps<NpuLoadPdiOp>())
      last = load;
    AIE::DeviceOp loaded = last ? loadedBy(last) : AIE::DeviceOp();
    if (!loaded || loaded == getOperation())
      return success();
    SmallVector<std::tuple<AIE::ShimDMAAllocationOp, AIE::ShimDMAAllocationOp,
                           Unbalanced>>
        unbalanced;
    for (auto input : loaded.getOps<AIE::ShimDMAAllocationOp>()) {
      ArrayAttr proofs = input.getConsumedByAttr();
      if (!proofs)
        continue;
      for (auto proof : proofs.getAsRange<AIE::ConsumedByAttr>()) {
        auto output = AIE::ShimDMAAllocationOp::getForSymbol(
            loaded, proof.getOutput().getValue());
        if (!output || proof.getDevice())
          continue;
        SmallVector<AIE::ShimDMAAllocationOp> inputs, outputs;
        for (auto copy : getOperation().getOps<AIE::ShimDMAAllocationOp>()) {
          ArrayAttr copied = copy.getConsumedByAttr();
          if (!copied ||
              DmaQueueModel::keyOf(copy) != DmaQueueModel::keyOf(input))
            continue;
          for (auto cp : copied.getAsRange<AIE::ConsumedByAttr>())
            if (proofDevice(cp) == loaded)
              // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
              if (auto out = symbols->lookup<AIE::ShimDMAAllocationOp>(
                      cp.getOutput().getAttr());
                  out &&
                  DmaQueueModel::keyOf(out) == DmaQueueModel::keyOf(output)) {
                inputs.push_back(copy);
                outputs.push_back(out);
              }
        }
        Unbalanced why = stretchesUnbalanced(seq, last, loaded, input, output,
                                             inputs, outputs, proof,
                                             /*byChannel=*/true);
        if (why != Unbalanced::No)
          unbalanced.push_back({input, output, why});
      }
    }
    // Every unbalanced pair, by its channels rather than declaration order.
    llvm::stable_sort(unbalanced, [&](const auto &a, const auto &b) {
      return std::pair(DmaQueueModel::keyOf(std::get<0>(a)),
                       DmaQueueModel::keyOf(std::get<1>(a))) <
             std::pair(DmaQueueModel::keyOf(std::get<0>(b)),
                       DmaQueueModel::keyOf(std::get<1>(b)));
    });
    for (auto [input, output, why] : unbalanced) {
      auto diag = last.emitOpError()
                  << "leaves " << last.getDeviceRefAttr()
                  << " loaded, and its next dispatch, whose proof that @"
                  << input.getSymName() << " is consumed by @"
                  << output.getSymName()
                  << " counts on a clean start, finds what this "
                     "sequence's last stretch left: it ";
      if (why == Unbalanced::Excess)
        diag << "moves more through that input than the output proves "
                "taken";
      else if (why == Unbalanced::PartGranule)
        diag << "may end with the output partway through an object of the "
                "join the proof passes";
      else if (why == Unbalanced::Traffic)
        diag << "has other traffic on their channels";
      else
        diag << "moves bytes on their channels whose count is not known, "
                "so cannot be proven balanced";
      diag << ". Balance them before the sequence ends, or load this "
              "device's PDI again.";
    }
    return failure(!unbalanced.empty());
  }

  // `output`'s pushes up to `order` have finished; see recordConsumption in
  // aie-objectfifo-verify for why that finishes its inputs' pushes too.
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
    if (done == outPushes->second.end() || done->order != order)
      return;
    awaitedThrough[output] = {done->through, pushCount};
    if (!done->through)
      return;
    AIE::DeviceOp configured = opaqueLoad     ? AIE::DeviceOp()
                               : loadedDevice ? loadedDevice
                                              : getOperation();
    for (auto [input, proof] : proofs->second) {
      if (!configured || proofDevice(proof) != configured)
        continue;
      auto inPushes = pushesByAllocation.find(input);
      DmaQueueModel::ChannelKey key = DmaQueueModel::keyOf(input);
      if (inPushes == pushesByAllocation.end() ||
          !leavesNothing(input, output, proof))
        continue;
      ArrayRef<AllocationPush> pushes = inPushes->second;
      size_t &credited = creditedPushes[{input, output}];
      size_t covered = credited;
      Volume counted = countedOutput(*done->through, proof);
      for (; covered < pushes.size(); ++covered) {
        const std::optional<Volume> &through = pushes[covered].through;
        if (!through || !atMost(*through, proof.getOutputBytes(), counted,
                                proof.getInputBytes()))
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

  // Prints `bytes` scaled by `scale` as "c + k x @param ...".
  void printVolume(llvm::raw_ostream &os, const Volume &bytes, int64_t scale,
                   AIE::RuntimeSequenceOp seq) {
    auto constant = bytes.find(-1);
    os << (constant == bytes.end() ? 0 : constant->second * scale);
    for (const auto &term : bytes) {
      if (term.first < 0 || !term.second)
        continue;
      os << " + " << term.second * scale << " x ";
      printParameter(os, term.first, seq);
    }
  }

  void printParameter(llvm::raw_ostream &os, int64_t index,
                      AIE::RuntimeSequenceOp seq) {
    auto module = getOperation()->getParentOfType<ModuleOp>();
    ArrayAttr reads = seq.getScratchpadParametersAttr();
    if (module)
      for (auto parameter : module.getOps<ScratchpadParameterOp>())
        if (parameter.getStateTableIdx() == index &&
            (!reads ||
             llvm::is_contained(reads.getAsValueRange<FlatSymbolRefAttr>(),
                                parameter.getSymName()))) {
          os << "@" << parameter.getSymName();
          return;
        }
    os << "parameter " << index;
  }

  // Why no proof shows `task`'s transfer finished, for the unsafe-reuse
  // warning: one clause per proof its allocation holds, and the ops they
  // point at as notes.
  void explainUnproven(DMAConfigureTaskOp task, llvm::raw_ostream &os,
                       SmallVectorImpl<std::pair<Location, std::string>> &at) {
    AIE::ShimDMAAllocationOp input = allocationOf(task);
    ArrayAttr proofs = input ? input.getConsumedByAttr() : ArrayAttr();
    if (!input) {
      os << "it names no shim allocation, so no output's progress can prove "
            "it consumed";
      return;
    }
    if (!proofs || proofs.empty()) {
      os << "aie-objectfifo-verify found no output whose progress proves @"
         << input.getSymName() << " consumed";
      return;
    }
    auto seq = task->getParentOfType<AIE::RuntimeSequenceOp>();
    AIE::DeviceOp configured = opaqueLoad     ? AIE::DeviceOp()
                               : loadedDevice ? loadedDevice
                                              : getOperation();
    size_t excess = 0;
    SmallVector<std::pair<int64_t, AIE::RuntimeSequenceOp>> exceeding;
    llvm::interleave(
        proofs.getAsRange<AIE::ConsumedByAttr>(), os,
        [&](AIE::ConsumedByAttr proof) {
          auto output = symbols->lookup<AIE::ShimDMAAllocationOp>(
              proof.getOutput().getAttr());
          os << "through " << proof.getOutput() << ", ";
          if (!output) {
            os << "that output is not in this device";
            return;
          }
          if (!configured || proofDevice(proof) != configured) {
            os << "the proof describes the array of "
               << (proof.getDevice() ? proof.getDevice()
                                     : FlatSymbolRefAttr::get(
                                           getOperation().getSymNameAttr()))
               << ", which is not the one loaded";
            return;
          }
          if (!leavesNothing(input, output, proof)) {
            for (auto other : getOperation().getOps<AIE::RuntimeSequenceOp>()) {
              Verdict verdict;
              Unbalanced why = stretchesUnbalanced(
                  other, nullptr, proofDevice(proof), input, output, {input},
                  {output}, proof, /*byChannel=*/false, &verdict);
              Operation *where = verdict.where;
              if (why == Unbalanced::No)
                continue;
              std::string note;
              if (why == Unbalanced::Traffic) {
                os << "another transfer shares their channels, which the "
                      "counts do not describe";
                note = "the other transfer";
              } else if (why == Unbalanced::PartGranule) {
                os << "a stretch of @" << other.getSymName()
                   << " may end with the output partway through an object of "
                      "the join the proof passes ("
                   << proof.getGranuleBytes()
                   << " output bytes), whose later segments it has then taken "
                      "less of";
                note = "the stretch ends here";
              } else if (why == Unbalanced::Unsized) {
                os << "a transfer on them moves a byte count the compiler "
                      "cannot follow (a BD chain, a loop, or a length "
                      "parameter without min_value >= 0)";
                note = "the transfer";
              } else {
                os << "a stretch of @" << other.getSymName()
                   << " can end having moved more through the input than the "
                      "output proves taken, so the next dispatch's counts do "
                      "not start clean (input bytes times "
                   << proof.getOutputBytes() << ", ";
                printVolume(os, verdict.in, proof.getOutputBytes(), other);
                os << ", against output bytes times " << proof.getInputBytes()
                   << ", ";
                printVolume(os, verdict.out, proof.getInputBytes(), other);
                os << ")";
                ++excess;
                for (const auto &term : verdict.in) {
                  int64_t index = term.first;
                  auto taken = verdict.out.find(index);
                  if (index >= 0 &&
                      term.second * proof.getOutputBytes() >
                          (taken == verdict.out.end() ? 0 : taken->second) *
                              proof.getInputBytes() &&
                      llvm::none_of(exceeding, [&](const auto &seen) {
                        return seen.first == index;
                      }))
                    exceeding.push_back({index, other});
                }
                note = "the stretch ends here";
              }
              std::pair<Location, std::string> pointer = {
                  where ? where->getLoc() : other.getLoc(), note};
              if (!llvm::is_contained(at, pointer))
                at.push_back(pointer);
              return;
            }
          }
          auto pushes = pushesByAllocation.find(input);
          auto order = lastPushOrder.find(task);
          const AllocationPush *push =
              pushes == pushesByAllocation.end() || order == lastPushOrder.end()
                  ? nullptr
                  : llvm::find_if(pushes->second, [&](const AllocationPush &p) {
                      return p.order == order->second;
                    });
          if (!push || push == pushes->second.end() || !push->through) {
            os << "the bytes the input has moved up to this transfer are not "
                  "known";
            return;
          }
          auto awaited = awaitedThrough.find(output);
          if (awaited == awaitedThrough.end()) {
            os << "no await on it has retired a transfer yet";
            return;
          }
          if (awaited->second.pushesBefore <= push->order) {
            os << "no await on it has retired a transfer since this one was "
                  "pushed";
            return;
          }
          const std::optional<Volume> &taken = awaited->second.through;
          if (!taken) {
            os << "the bytes its awaited transfers moved are not known";
            return;
          }
          Volume counted = countedOutput(*taken, proof);
          if (atMost(*push->through, proof.getOutputBytes(), counted,
                     proof.getInputBytes())) {
            os << "its awaited transfers cover this transfer's bytes, but "
                  "not those of every push of this task";
            return;
          }
          bool scaled = proof.getInputBytes() != proof.getOutputBytes();
          os << "the bytes its awaited transfers moved";
          if (!wholeGranules(*taken, proof.getGranuleBytes()))
            os << ", less the part of an object of the join the proof passes ("
               << proof.getGranuleBytes() << " bytes) they may end in,";
          if (scaled)
            os << " times " << proof.getInputBytes();
          os << ", ";
          printVolume(os, counted, proof.getInputBytes(), seq);
          os << ", fall short of the bytes the input moved up to this "
                "transfer";
          if (scaled)
            os << " times " << proof.getOutputBytes();
          os << ", ";
          printVolume(os, *push->through, proof.getOutputBytes(), seq);
          auto isParameter = [](auto term) { return term.first >= 0; };
          if (llvm::any_of(counted, isParameter) ||
              llvm::any_of(*push->through, isParameter))
            os << ", for some parameter values";
        },
        "; ");
    if (proofs.size() > 1 && excess == proofs.size()) {
      os << ". No one output's proof covers the input: it moves more than "
            "each proves alone";
      if (!exceeding.empty()) {
        os << ", by how much depending on the runtime parameter";
        if (exceeding.size() > 1)
          os << "s";
        os << " ";
        llvm::interleaveComma(exceeding, os, [&](const auto &parameter) {
          printParameter(os, parameter.first, parameter.second);
        });
      }
    }
  }

  // Warn where the hazard actually bites: reusing the id, not releasing it.
  void checkReallocation(AIE::TileOp tile, uint32_t id, Location at) {
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
    auto [entry, inserted] = unsafeReuses.try_emplace(
        at, UnsafeReuse{id, tile.getCol(), tile.getRow(), {}});
    UnsafeReuse &reuse = entry->second;
    if (inserted) {
      llvm::raw_string_ostream os(reuse.why);
      explainUnproven(released.configure, os, reuse.notes);
    }
    if (!llvm::is_contained(reuse.freeLocs, released.freeOp->getLoc()))
      reuse.freeLocs.push_back(released.freeOp->getLoc());
    reuse.tiles.insert({tile.getCol(), tile.getRow()});
    reuse.count++;
  }

  // A full ELF inlines a sequence at every call, so one source transfer can
  // reuse an id once per call; it is reported once, with how often. Several
  // distinct frees can still merge here -- a generated control-packet copy
  // and a user task both reusing the id this allocation pass's own location
  // names -- so every free actually seen is kept, not just the first.
  struct UnsafeReuse {
    uint32_t id;
    int col, row;
    SmallVector<Location> freeLocs;
    std::set<std::pair<int, int>> tiles = {};
    int64_t count = 0;
    std::string why = {};
    SmallVector<std::pair<Location, std::string>> notes = {};
  };
  llvm::MapVector<Location, UnsafeReuse> unsafeReuses;
  // Copies that reprogram a BD id an earlier copy on the tile may still have
  // in flight, reported once per copy like unsafeReuses.
  struct CopyOverCopy {
    int32_t id;
    int col, row;
    Location earlier;
    bool earlierIssuesToken;
    std::set<std::pair<int, int>> tiles = {};
    int64_t count = 0;
  };
  llvm::MapVector<Location, CopyOverCopy> copiesOverCopies;

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
             "consumes the oldest token, regardless of the task it names. "
             "The compiler cannot prove that transfer finished: "
          << reuse.why;
      for (Location freeLoc : reuse.freeLocs)
        diag.attachNote(freeLoc) << "released here";
      for (auto &[noteLoc, text] : reuse.notes)
        diag.attachNote(noteLoc) << text;
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
    for (auto &[loc, reuse] : copiesOverCopies) {
      auto diag = mlir::emitWarning(loc)
                  << "reuses buffer descriptor ID " << reuse.id << " on tile ("
                  << reuse.col << "," << reuse.row
                  << ") while an earlier aiex.npu.dma_memcpy_nd of that ID may "
                     "still be in flight, so this reprograms it underneath. "
                     "Give this copy another ID, or retire the earlier copy "
                     "first: ";
      if (reuse.earlierIssuesToken)
        diag << "wait for it (aiex.npu.sync or aiex.npu.dma_wait on its "
                "channel, consuming its token).";
      else
        diag << "it issues no token, so only an await of a later tokened "
                "task on its channel, a status poll, or issue_token = true on "
                "it and a wait for it shows it done.";
      diag.attachNote(reuse.earlier) << "the earlier copy is here";
      if (reuse.count > 1) {
        Diagnostic &note = diag.attachNote();
        note << "and " << reuse.count - 1
             << " more times from this copy, on tile"
             << (reuse.tiles.size() > 1 ? "s " : " ");
        llvm::interleaveComma(reuse.tiles, note, [&](auto tile) {
          note << "(" << tile.first << "," << tile.second << ")";
        });
      }
    }
    copiesOverCopies.clear();
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
      if (gen.bdIdAlreadyAssigned(bd.getBdId().value()) &&
          !copyOwned.contains(
              {task_op.getTileOp().getTileID(), bd.getBdId().value()})) {
        releaseTaskBdId(task_op.getTileOp().getTileID(), gen,
                        bd.getBdId().value());
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
  // that, the oldest, until its channel is idle. A task that acquires a lock
  // `op` or a later push releases is never polled for: it cannot finish
  // before that push, so the poll would never return. The same dependence
  // through a core the compiler cannot see; a design with one releases its own
  // BDs.
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
      task.walk([&](AIE::DMABDOp bd) {
        releaseTaskBdId(tile.getTileID(), gen, *bd.getBdId());
      });
      reclaimed.insert(task);
      releasedTasks.insert(task);
      pendingAwaitReleases.erase(task);
    };

    ArrayRef<DMAConfigureTaskOp> live = liveByTile[tile.getTileID()];
    for (DMAConfigureTaskOp task : live)
      if ((knownComplete.contains(task) || provenFinished.contains(task)) &&
          isCandidate(task)) {
        take(task);
        return success();
      }

    unsigned at = positionOf.lookup(op);
    auto waitsOnLater = [&](DMAConfigureTaskOp task) {
      bool waits = false;
      task.walk([&](AIE::UseLockOp lock) {
        auto last = lastReleaseOf.find(lock.getLock());
        waits |= (lock.acquire() || lock.acquireGE()) &&
                 last != lastReleaseOf.end() && last->second >= at;
      });
      return waits;
    };

    uint32_t field = tm.getDmaTaskQueueSizeMask();
    uint32_t idle = tm.getDmaChannelIdleMask();
    DMAConfigureTaskOp victim;
    size_t victimQueuedBehind = 0;
    uint64_t victimOrder = 0;
    for (DMAConfigureTaskOp task : live) {
      if (!isCandidate(task) || waitsOnLater(task))
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
      auto constant = arith::ConstantOp::create(b, op.getLoc(), b.getI32Type(),
                                                b.getI32IntegerAttr(v));
      insertedHere.push_back(constant);
      return constant.getResult();
    };
    Value maskValue = cst(mask);
    Value compareValue = cst(0);
    Value statusValue = cst(*status);
    auto poll = NpuMaskPollOp::create(b, op.getLoc(), statusValue, compareValue,
                                      maskValue,
                                      /*buffer=*/nullptr, /*column=*/nullptr,
                                      /*row=*/nullptr);
    insertedHere.push_back(poll);
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
    awaitedThrough.clear();
    lastPushOrder.clear();
    positionOf.clear();
    lastReleaseOf.clear();
    holdsByChannel.clear();
    holdsOn.clear();
    copyOwned.clear();
    reallocate = false;
    assignedHere.clear();
    insertedHere.clear();
  }

  void warnLeaks(AIE::RuntimeSequenceOp sequence) {
    if (!warnUnreleasedTransfers)
      return;
    for (const auto &[key, channel] : startedOnChannel) {
      SmallVector<Operation *> outstanding;
      for (const StartedTask &push :
           llvm::drop_begin(channel.tasks, channel.finished))
        if (!push.task || (!releasedTasks.contains(push.task) &&
                           !unloaded.contains(push.task)))
          outstanding.push_back(push.origin);
      if (outstanding.empty())
        continue;
      auto diag = mlir::emitWarning(sequence.getLoc())
                  << "may exit with " << outstanding.size()
                  << " DMA transfer(s) neither awaited nor freed on tile ("
                  << key[0] << "," << key[1] << ") "
                  << AIE::stringifyDMAChannelDir(
                         static_cast<AIE::DMAChannelDir>(key[2]))
                  << " channel " << key[3]
                  << "; runtime sequences called by aiex.run must finish using "
                     "their channels before returning. Await completion, or "
                     "free tasks whose completion is guaranteed indirectly";
      diag.attachNote(outstanding.front()->getLoc())
          << "first outstanding transfer starts here";
    }
  }

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();
    staticBdIds.clear();
    loopingStarts.clear();
    nonNegativeByIndex.clear();
    balancedPairs.clear();
    proofsByOutput.clear();
    symbols.emplace(device);
    for (auto input : device.getOps<AIE::ShimDMAAllocationOp>())
      if (ArrayAttr proofs = input.getConsumedByAttr())
        for (auto proof : proofs.getAsRange<AIE::ConsumedByAttr>())
          if (auto output = symbols->lookup<AIE::ShimDMAAllocationOp>(
                  proof.getOutput().getAttr()))
            proofsByOutput[output].push_back({input, proof});

    WalkResult wr = device.walk([&](AIE::RuntimeSequenceOp seq) -> WalkResult {
      if (failed(checkForeignTail(seq)))
        return WalkResult::interrupt();
      // Skip sequences already handled by the dynamic free-list pool path
      // (aie-lower-dynamic-bd-pool): they draw BD ids at runtime via
      // dma_bd_pool_pop and keep their scf.for rolled, which the static
      // straight-line allocator neither needs to touch nor can validate.
      bool dynamicPool = false;
      bool hasTasks = false;
      bool hasControlFlow = false;
      seq.walk([&](Operation *op) {
        dynamicPool |= isa<DMABdPoolPopOp>(op);
        hasControlFlow |= isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op);
        hasTasks |=
            isa<DMAConfigureTaskOp, DMAConfigureTaskForOp, DMAStartBdChainOp,
                DMAStartTaskOp, DMAAwaitTaskOp, DMAFreeTaskOp>(op);
      });
      if (dynamicPool || (!hasTasks && hasControlFlow))
        return WalkResult::advance();

      if (hasTasks && failed(validate(seq)))
        return WalkResult::interrupt();

      // Straight-line walk. Collect frees to erase after (recycling reads the
      // configure the free points at, so erase only once the walk is done).
      // This allocation strategy works only for straight-line IR without
      // branching or conditionals; the verifier of this pass ensures this is
      // the case on the input IR.
      SmallVector<DMAFreeTaskOp> frees;
      auto visit = [&](Operation *op) -> WalkResult {
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
            notePush(channelOf(cfg), cfg, start.getPushIssueToken(cfg), start,
                     allocationOf(cfg), pushVolume(start, cfg));
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
        } else if (auto freeOp = dyn_cast<DMAFreeTaskOp>(op)) {
          if (failed(recycleTask(freeOp.getTask(), freeOp, /*isAwait=*/false)))
            return WalkResult::interrupt();
          frees.push_back(freeOp);
        } else if (QueueEffect raw = rawQueueEffect(op, *symbols);
                   raw.kind == QueueEffect::Kind::Await) {
          noteAwaited(raw.key);
          if (failed(recycleCompletedTasks(op)))
            return WalkResult::interrupt();
        } else if (raw.kind == QueueEffect::Kind::Push) {
          auto copy = dyn_cast<NpuDmaMemcpyNdOp>(op);
          AIE::ShimDMAAllocationOp allocation = allocationOf(copy);
          notePush(raw.key, DMAConfigureTaskOp{}, raw.issuesToken, op,
                   allocation, copy ? copyVolume(copy) : std::nullopt);
          if (copy) {
            if (failed(holdCopy(copy, raw.key, pushCount - 1, raw.issuesToken)))
              return WalkResult::interrupt();
            if (reallocate)
              return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      };
      // A copy that meets a task in flight on its id forbids the task that id
      // and the sequence is allocated again; each round adds one such pair.
      // A round's diagnostics are held until it stands. Other threads' pass
      // instances share the context, so only this thread's are held.
      forbidden.clear();
      auto reusesBefore = unsafeReuses;
      auto copiesBefore = copiesOverCopies;
      WalkResult r = WalkResult::advance();
      std::vector<Diagnostic> held;
      for (;;) {
        resetPerSequenceState();
        indexLockReleases(seq);
        frees.clear();
        held.clear();
        {
          uint64_t thread = llvm::get_threadid();
          ScopedDiagnosticHandler hold(&getContext(), [&](Diagnostic &diag) {
            if (llvm::get_threadid() != thread)
              return failure();
            held.push_back(std::move(diag));
            return success();
          });
          r = seq.walk(visit);
        }
        if (!reallocate)
          break;
        for (AIE::DMABDOp bd : assignedHere)
          bd.removeBdIdAttr();
        for (Operation *inserted : llvm::reverse(insertedHere))
          inserted->erase();
        unsafeReuses = reusesBefore;
        copiesOverCopies = copiesBefore;
      }
      for (Diagnostic &diag : held)
        getContext().getDiagEngine().emit(std::move(diag));
      if (r.wasInterrupted())
        return WalkResult::interrupt();
      warnLeaks(seq);
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
