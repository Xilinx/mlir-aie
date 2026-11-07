//===- AIECtrlPacketToDma.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "CtrlPacketRuns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include <deque>
#include <map>
#include <optional>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIECTRLPACKETTODMA
#define GEN_PASS_DEF_AIECTRLPACKETINFERTILES
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

#define DEBUG_TYPE "aie-ctrl-packet-to-dma"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

static ShimDMAAllocationOp getCtrlPktShimAlloc(DeviceOp device, int col) {
  AIEDialect::IsCtrlPktOverlayAttrHelper overlay(device.getContext());
  for (auto alloc : device.getOps<AIE::ShimDMAAllocationOp>())
    if (overlay.isAttrPresent(alloc) &&
        alloc.getChannelDir() == AIE::DMAChannelDir::MM2S &&
        alloc.getTileOp().colIndex() == col)
      return alloc;
  return {};
}

// The packets that start a delivery run: the first of each run of groups
// (see groupRuns), and each control packet the user wrote right after a group,
// so a batch never mixes packets the user wrote with generated ones.
static llvm::DenseSet<Operation *> runStarts(Block &block) {
  llvm::DenseSet<Operation *> starts;
  for (const GroupRun &run :
       groupRuns(block, {/*byKindAndDevice=*/true, /*atBarriers=*/false}))
    starts.insert(run.front().packet);
  Operation *last = nullptr;
  for (Operation &op : block) {
    if (isa<NpuControlPacketOp>(op) &&
        isa_and_present<ControlPacketGroupOp>(last))
      starts.insert(&op);
    if (!isa<ControlPacketBarrierOp>(op))
      last = &op;
  }
  return starts;
}

struct AIECtrlPacketInferTilesPass
    : xilinx::AIEX::impl::AIECtrlPacketInferTilesBase<
          AIECtrlPacketInferTilesPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();
    OpBuilder devBuilder = OpBuilder::atBlockBegin(device.getBody());

    auto sequenceOps = device.getOps<AIE::RuntimeSequenceOp>();
    for (auto f : sequenceOps) {
      for (auto ctrlPktOp : controlPacketsOf(f.getBody().front())) {
        auto tOp = TileOp::getOrCreate(devBuilder, device,
                                       (int)ctrlPktOp.getColumnFromAddr(),
                                       (int)ctrlPktOp.getRowFromAddr());
        // Assign controller id
        auto tileIDMap = targetModel.getTileToControllerIdMap(true);
        if (tOp->hasAttr("controller_id"))
          continue;
        auto pktInfoAttr = AIE::PacketInfoAttr::get(
            tOp->getContext(), /*pkt_type*/ 0,
            /*pkt_id*/ tileIDMap[{tOp.colIndex(), tOp.rowIndex()}]);
        tOp->setAttr("controller_id", pktInfoAttr);
      }
    }
  }
};

// The descriptors the device's static DMAs leave on column `col`'s shim tile.
static uint32_t staticFreeBds(DeviceOp device, int col) {
  uint32_t used = 0;
  for (AIE::DmaBody program : device.getOps<AIE::DmaBody>())
    if (program.getTileID() == TileID{col, 0})
      used += AIE::getAssignedBdIds(program).size();
  uint32_t total = device.getTargetModel().getNumBDs(col, 0);
  return used < total ? total - used : 0;
}

// The shim tile column and descriptor count of a runtime task `op` configures,
// if it is on a shim tile.
static std::optional<std::pair<int, uint32_t>> shimTask(DeviceOp device,
                                                        Operation *op) {
  auto bds = [](Region &body) {
    uint32_t n = 0;
    body.walk([&](AIE::DMABDOp) { ++n; });
    return n;
  };
  auto chainLength = [&](AIE::BDChainOp chain) {
    return chain ? bds(chain.getBody()) : 0u;
  };
  auto shimCol = [](AIE::TileOp tile) -> std::optional<int> {
    if (tile && tile.isShimTile())
      return tile.getCol();
    return std::nullopt;
  };
  auto allocCol = [&](SymbolRefAttr alloc) -> std::optional<int> {
    auto shim = ShimDMAAllocationOp::getForSymbol(
        device, alloc.getRootReference().getValue());
    return shim ? shimCol(shim.getTileOp()) : std::nullopt;
  };
  std::optional<int> col;
  uint32_t n = 0;
  if (auto cfg = dyn_cast<DMAConfigureTaskOp>(op)) {
    col = shimCol(cfg.tryGetTileOp());
    n = bds(cfg.getBody());
  } else if (auto cfg = dyn_cast<DMAConfigureTaskForOp>(op)) {
    col = allocCol(cfg.getAlloc());
    n = bds(cfg.getBody());
  } else if (auto start = dyn_cast<DMAStartBdChainOp>(op)) {
    col = shimCol(start.getTileOp());
    n = chainLength(start.getBDChainOp());
  } else if (auto start = dyn_cast<DMAStartBdChainForOp>(op)) {
    col = allocCol(start.getAllocAttr());
    n = chainLength(device.lookupSymbol<AIE::BDChainOp>(start.getSymbol()));
  }
  if (!col)
    return std::nullopt;
  return std::make_pair(*col, n);
}

struct AIECtrlPacketToDmaPass
    : xilinx::AIEX::impl::AIECtrlPacketToDmaBase<AIECtrlPacketToDmaPass> {
  using Base::Base;

  // A transfer of control packets to one tile.
  struct BatchInfo {
    TileID tileId;
    int64_t startOffset;
    int64_t totalSize;
    std::string shimDmaAllocName;
    int shimChan;
    Operation *first;
    int run;   // batches of one run have no other op between them
    int phase; // the barriers before it in its run
    // From an aiex.control_packet_group, so a chain may carry it; control
    // packets the user wrote keep a transfer and a sync each.
    bool generated;
    // The multicast the packets go on.
    std::optional<int32_t> multicast;
  };

  // The descriptors each shim tile's static DMAs leave.
  llvm::DenseMap<int, uint32_t> staticFree;
  // The descriptors each shim tile's runtime tasks hold at the op the second
  // pass has reached: those of tasks configured since the last load_pdi and
  // not yet awaited or freed. aie-assign-runtime-sequence-bd-ids, which
  // allocates them later, keeps an awaited task's until it knows the task done.
  llvm::DenseMap<int, uint32_t> heldBds;
  llvm::DenseMap<Value, std::pair<int, uint32_t>> liveTasks;

  void noteHeld(Operation &op) {
    op.walk([&](Operation *inner) {
      Value done;
      if (auto await = dyn_cast<DMAAwaitTaskOp>(inner))
        done = await.getTask();
      else if (auto free = dyn_cast<DMAFreeTaskOp>(inner))
        done = free.getTask();
      if (isa<NpuLoadPdiOp>(inner)) {
        heldBds.clear();
        liveTasks.clear();
      } else if (done) {
        auto it = liveTasks.find(done);
        if (it != liveTasks.end()) {
          heldBds[it->second.first] -= it->second.second;
          liveTasks.erase(it);
        }
      } else if (auto task = shimTask(getOperation(), inner)) {
        heldBds[task->first] += task->second;
        liveTasks[inner->getResult(0)] = *task;
      }
    });
  }

  // The descriptors a chain may hold on column `col`'s shim tile: those its
  // static DMAs and runtime tasks leave, at least one.
  uint32_t chainBdLimit(DeviceOp device, int col) {
    auto [it, fresh] = staticFree.try_emplace(col);
    if (fresh)
      it->second = staticFreeBds(device, col);
    uint32_t held = heldBds.lookup(col);
    return held < it->second ? it->second - held : 1;
  }

  // One task over the chain's batches from `buffer`, each one buffer
  // descriptor, linked in order.
  static DMAConfigureTaskForOp emitTask(OpBuilder &builder, Location loc,
                                        Value buffer,
                                        ArrayRef<const BatchInfo *> chain) {
    auto task = DMAConfigureTaskForOp::create(
        builder, loc, builder.getIndexType(),
        SymbolRefAttr::get(builder.getContext(),
                           chain.front()->shimDmaAllocName),
        /*issue_token=*/true, /*repeat_count=*/0,
        /*repeat_count_val=*/Value());
    AIEXDialect::CtrlPktTransferAttrHelper(builder.getContext())
        .setAttr(task, builder.getUnitAttr());
    OpBuilder::InsertionGuard guard(builder);
    Region &body = task.getBody();
    for (size_t i = 0; i < chain.size(); ++i)
      builder.createBlock(&body, body.end());
    for (auto [block, batch] : llvm::zip(body, chain)) {
      builder.setInsertionPointToEnd(&block);
      AIE::DMABDOp::create(builder, loc, buffer,
                           static_cast<int>(batch->startOffset),
                           static_cast<int>(batch->totalSize));
      if (block.getNextNode())
        AIE::NextBDOp::create(builder, loc, block.getNextNode());
      else
        AIE::EndOp::create(builder, loc);
    }
    return task;
  }

  // parallel-shims (see the pass description).
  void emitParallelRun(OpBuilder &builder, Location loc, Value buffer,
                       DeviceOp device, ArrayRef<BatchInfo> batches) {
    const AIE::AIETargetModel &targetModel = device.getTargetModel();
    int phases = 0;
    for (const BatchInfo &batch : batches)
      phases = std::max(phases, batch.phase + 1);
    for (int phase = 0; phase < phases; ++phase) {
      // Each allocation's chains, and how many may be in flight at once.
      llvm::MapVector<StringRef, SmallVector<SmallVector<const BatchInfo *>>>
          tasks;
      llvm::StringMap<size_t> window;
      for (const BatchInfo &batch : batches) {
        if (batch.phase != phase)
          continue;
        int col = batch.tileId.col;
        uint32_t numBds = chainBdLimit(device, col);
        auto &list = tasks[batch.shimDmaAllocName];
        if (list.empty()) {
          uint32_t depth = targetModel.getDmaTaskQueueDepth(
              col, 0, batch.shimChan, DMAChannelDir::MM2S);
          window[batch.shimDmaAllocName] =
              chainBds ? 1 : (depth ? std::min(depth, numBds) : numBds);
        }
        if (list.empty() || !chainBds || list.back().size() >= numBds)
          list.emplace_back();
        list.back().push_back(&batch);
      }
      size_t rounds = 0;
      for (auto &[name, list] : tasks)
        rounds = std::max(rounds, list.size());
      SmallVector<std::deque<DMAConfigureTaskForOp>> outstanding(tasks.size());
      for (size_t r = 0; r < rounds; ++r) {
        for (auto [i, entry] : llvm::enumerate(tasks)) {
          auto &[name, list] = entry;
          if (r >= list.size())
            continue;
          if (outstanding[i].size() >= window[name]) {
            DMAAwaitTaskOp::create(builder, loc, outstanding[i].front());
            outstanding[i].pop_front();
          }
          DMAConfigureTaskForOp task = emitTask(builder, loc, buffer, list[r]);
          DMAStartTaskOp::create(builder, loc, task);
          outstanding[i].push_back(task);
        }
      }
      for (auto &queue : outstanding)
        for (DMAConfigureTaskForOp task : queue)
          DMAAwaitTaskOp::create(builder, loc, task);
    }
  }

  // A remark for each column whose chains in the run starting at `begin` the
  // descriptors runtime tasks hold make shorter.
  void remarkHeldBds(DeviceOp device, ArrayRef<BatchInfo> batches,
                     const BatchInfo *begin) {
    std::map<int, size_t> longest;
    std::map<std::pair<int, int>, size_t> perPhase;
    size_t length = 0;
    for (const BatchInfo *it = begin;
         it != batches.end() && it->run == begin->run; ++it) {
      int col = it->tileId.col;
      if (parallelShims) {
        length = ++perPhase[{col, it->phase}];
      } else {
        bool joins = it != begin && std::prev(it)->phase == it->phase &&
                     std::prev(it)->shimDmaAllocName == it->shimDmaAllocName;
        length = joins ? length + 1 : 1;
      }
      longest[col] = std::max(longest[col], length);
    }
    for (auto [col, n] : longest) {
      uint32_t held = heldBds.lookup(col);
      uint32_t limit = chainBdLimit(device, col);
      if (held && n > limit)
        begin->first->emitRemark("control packet chains on shim tile (")
            << col << ", 0) hold at most " << limit
            << " buffer descriptors: runtime transfers there not yet "
               "awaited or freed hold "
            << held;
    }
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();
    auto *ctx = device->getContext();
    auto loc = device->getLoc();

    if (targetModel.getTargetArch() == AIEArch::AIE1)
      return; // Disable this pass for AIE1; AIE1 support NYI.

    staticFree.clear();
    SmallVector<Operation *> erased;
    auto sequenceOps = device.getOps<AIE::RuntimeSequenceOp>();
    for (auto f : sequenceOps) {

      Block &entry = f.getBody().front();
      heldBds.clear();
      liveTasks.clear();
      llvm::DenseSet<Operation *> starts = runStarts(entry);
      llvm::DenseSet<Operation *> generated;
      for (auto group : entry.getOps<ControlPacketGroupOp>())
        for (Operation &packet : group.getBody().front())
          generated.insert(&packet);
      inlineControlPacketGroups(entry);
      auto controlPacketOps = f.getOps<AIEX::NpuControlPacketOp>();
      if (controlPacketOps.empty())
        continue;

      OpBuilder builder(f);

      IRMapping mapping;

      auto newSeq = AIE::RuntimeSequenceOp::create(
          builder, loc, f.getSymNameAttr(), BoolAttr{}, f.getTraceBufferAttr(),
          f.getTraceSlicesAttr(), f.getScratchpadParametersAttr());
      newSeq.getBody().push_back(new Block);

      // Copy the arguments from the old sequence to the new one.
      for (auto arg : f.getBody().getArguments()) {
        // Add the argument to the new sequence.
        auto newArg = newSeq.getBody().addArgument(arg.getType(), arg.getLoc());
        // Replace all uses of the old argument with the new one.
        arg.replaceAllUsesWith(newArg);
        // Add the mapping for the argument.
        mapping.map(arg, newArg);
      }

      // Using dynamic shape for ctrl pkt stream.
      auto ctrlPktMemrefType = MemRefType::get(
          ShapedType::kDynamic, IntegerType::get(ctx, 32), nullptr, nullptr);
      auto newBlockArg = newSeq.getBody().addArgument(ctrlPktMemrefType, loc);

      builder.setInsertionPointToStart(&newSeq.getBody().front());

      // Collect all npu.control_packet ops, grouped by location in 'batches'
      std::vector<BatchInfo> batches;

      int64_t ddrOffset = 0;

      // First pass: collect and batch control packet operations
      bool new_batch = true, inRun = false;
      int run = 0, phase = 0;
      for (Operation &o : entry) {
        auto ctrlPktOp = dyn_cast<NpuControlPacketOp>(&o);

        // A barrier ends the current batch and phase
        if (isa<ControlPacketBarrierOp>(o)) {
          new_batch = true;
          ++phase;
          continue;
        }
        // Any other op, or the start of a run, ends the current batch and run
        if (!ctrlPktOp || starts.contains(&o)) {
          if (inRun)
            ++run;
          new_batch = true;
          inRun = false;
          phase = 0;
          if (!ctrlPktOp)
            continue;
        }
        inRun = true;
        int col = ctrlPktOp.getColumnFromAddr();
        int row = ctrlPktOp.getRowFromAddr();

        // A read's length describes its response, not an outgoing payload.
        int64_t ctrlPktSize = 2; // ctrl info word + packet header
        if (auto data = ctrlPktOp.getData())
          ctrlPktSize += data->size();

        // Read requests must end their DMA transfer. A batch is one stream
        // packet, routed by its first header, so its packets share their
        // destination.
        bool batchable = ctrlPktOp.getOpcode() == 0;
        std::optional<int32_t> multicast;
        if (auto id = ctrlPktOp.getMulticastId())
          multicast = *id;
        if (targetModel.getTargetArch() == AIEArch::AIE2p && batchable &&
            !new_batch && batches.back().tileId == TileID{col, row} &&
            batches.back().multicast == multicast) {
          // Add to existing batch
          batches.back().totalSize += ctrlPktSize;
        } else {
          // Start new batch
          ShimDMAAllocationOp alloc = getCtrlPktShimAlloc(device, col);
          if (!alloc) {
            ctrlPktOp.emitOpError("column ")
                << col
                << " has no control packet shim DMA allocation; run "
                   "-aie-generate-column-control-overlay with "
                   "configuration=true, then -aie-import-overlay first";
            return signalPassFailure();
          }
          batches.push_back({TileID{col, row}, ddrOffset, ctrlPktSize,
                             alloc.getSymName().str(),
                             static_cast<int>(alloc.getChannelIndex()), &o, run,
                             phase, generated.contains(&o), multicast});
        }
        new_batch = !batchable;
        ddrOffset += ctrlPktSize;
      }

      // Second pass: emit batched operations in original order
      auto batchIt = batches.begin();

      for (Operation &o : entry) {
        auto ctrlPktOp = dyn_cast<NpuControlPacketOp>(&o);
        if (!ctrlPktOp) {
          noteHeld(o);
          if (!isa<ControlPacketBarrierOp>(o))
            builder.clone(o, mapping);
          continue;
        }

        // There are no more control packet batches to emit
        if (batchIt == batches.end())
          continue;

        // Check if this is the first packet of a new batch, otherwise skip it.
        if (batchIt->first != &o)
          continue;

        if (chainBds && batchIt->generated &&
            (batchIt == batches.begin() ||
             std::prev(batchIt)->run != batchIt->run))
          remarkHeldBds(device, batches, &*batchIt);

        int col = ctrlPktOp.getColumnFromAddr();

        if (parallelShims && batchIt->generated) {
          auto runEnd = batchIt;
          while (runEnd != batches.end() && runEnd->run == batchIt->run)
            ++runEnd;
          emitParallelRun(builder, loc, newBlockArg, device,
                          ArrayRef<BatchInfo>(&*batchIt, runEnd - batchIt));
          batchIt = runEnd;
          continue;
        }

        if (chainBds && batchIt->generated) {
          uint32_t maxBds = chainBdLimit(device, col);
          auto chainEnd = std::next(batchIt);
          while (chainEnd != batches.end() &&
                 static_cast<uint32_t>(chainEnd - batchIt) < maxBds &&
                 chainEnd->run == batchIt->run &&
                 chainEnd->phase == batchIt->phase &&
                 chainEnd->shimDmaAllocName == batchIt->shimDmaAllocName)
            ++chainEnd;

          SmallVector<const BatchInfo *> chain;
          for (auto it = batchIt; it != chainEnd; ++it)
            chain.push_back(&*it);
          DMAConfigureTaskForOp task =
              emitTask(builder, loc, newBlockArg, chain);
          DMAStartTaskOp::create(builder, loc, task);
          DMAAwaitTaskOp::create(builder, loc, task);
          batchIt = chainEnd;
          continue;
        }

        // Emit the batched DMA operation for this (col, row) pair
        const std::vector<int64_t> staticOffsets = {0, 0, 0,
                                                    batchIt->startOffset};
        const std::vector<int64_t> staticSizes = {1, 1, 1, batchIt->totalSize};
        const std::vector<int64_t> staticStrides = {0, 0, 0, 1};

        SymbolRefAttr metadata =
            SymbolRefAttr::get(builder.getContext(), batchIt->shimDmaAllocName);
        NpuDmaMemcpyNdOp::create(builder, loc, newBlockArg, staticOffsets,
                                 staticSizes, staticStrides, metadata,
                                 /*id=*/0, /*issue_token=*/true);

        Value shimRow = AIEX::createConstantI32(builder, loc, 0);
        Value shimCol = AIEX::createConstantI32(builder, loc, col);
        Value dir = AIEX::createConstantI32(builder, loc, 1); // MM2S
        Value chan = AIEX::createConstantI32(builder, loc, batchIt->shimChan);
        Value col_num = AIEX::createConstantI32(builder, loc, 1);
        Value row_num = AIEX::createConstantI32(builder, loc, 1);
        AIEX::NpuSyncOp::create(builder, loc, shimCol, shimRow, dir, chan,
                                col_num, row_num);
        ++batchIt;
      }

      erased.push_back(f);
    }

    for (auto *e : erased)
      e->erase();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIEX::createAIECtrlPacketInferTilesPass() {
  return std::make_unique<AIECtrlPacketInferTilesPass>();
}
std::unique_ptr<OperationPass<DeviceOp>> AIEX::createAIECtrlPacketToDmaPass() {
  return std::make_unique<AIECtrlPacketToDmaPass>();
}
std::unique_ptr<OperationPass<DeviceOp>>
AIEX::createAIECtrlPacketToDmaPass(const AIECtrlPacketToDmaOptions &options) {
  return std::make_unique<AIECtrlPacketToDmaPass>(options);
}
