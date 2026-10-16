//===- AIECtrlPacketToDma.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEGenerateColumnControlOverlay.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include <deque>

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

struct AIECtrlPacketInferTilesPass
    : xilinx::AIEX::impl::AIECtrlPacketInferTilesBase<
          AIECtrlPacketInferTilesPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();
    OpBuilder devBuilder = OpBuilder::atBlockBegin(device.getBody());

    auto sequenceOps = device.getOps<AIE::RuntimeSequenceOp>();
    for (auto f : sequenceOps) {
      auto ctrlPktOps = f.getOps<AIEX::NpuControlPacketOp>();
      for (auto ctrlPktOp : ctrlPktOps) {
        auto tOp = TileOp::getOrCreate(devBuilder, device,
                                       (int)ctrlPktOp.getColumnFromAddr(),
                                       (int)ctrlPktOp.getRowFromAddr());
        // Assign controller id
        auto tileIDMap = getTileToControllerIdMap(true, targetModel);
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

struct AIECtrlPacketToDmaPass
    : xilinx::AIEX::impl::AIECtrlPacketToDmaBase<AIECtrlPacketToDmaPass> {
  using Base::Base;

  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();
    auto *ctx = device->getContext();
    auto loc = device->getLoc();

    if (targetModel.getTargetArch() == AIEArch::AIE1)
      return; // Disable this pass for AIE1; AIE1 support NYI.

    SmallVector<Operation *> erased;
    auto sequenceOps = device.getOps<AIE::RuntimeSequenceOp>();
    for (auto f : sequenceOps) {

      auto controlPacketOps = f.getOps<AIEX::NpuControlPacketOp>();
      if (controlPacketOps.empty())
        continue;

      OpBuilder builder(f);

      IRMapping mapping;

      auto newSeq = AIE::RuntimeSequenceOp::create(
          builder, loc, f.getSymNameAttr(), BoolAttr{}, f.getTraceBufferAttr(),
          f.getTraceSlicesAttr());
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
      struct BatchInfo {
        TileID tileId;
        int64_t startOffset;
        int64_t totalSize;
        std::string shimDmaAllocName;
        int shimChan;
        Operation *first;
        int run;          // batches of one run have no other op between them
        bool enablePhase; // sent after the rest of the run
                          // (ctrl_pkt_enable_phase)
        // The multicast group the packets go to (aiex.ctrl_pkt_multicast_id).
        std::optional<int64_t> multicast;
      };
      std::vector<BatchInfo> batches;

      int64_t ddrOffset = 0;
      Block &entry = f.getBody().front();

      AIEXDialect::CtrlPktEnablePhaseAttrHelper enablePhaseStamp(ctx);
      AIEXDialect::CtrlPktMulticastIdAttrHelper multicastStamp(ctx);

      // First pass: collect and batch control packet operations
      bool new_batch = true;
      int run = 0;
      for (Operation &o : entry) {
        auto ctrlPktOp = dyn_cast<NpuControlPacketOp>(&o);

        // A non-control_packet op ends the current batch and run
        if (!ctrlPktOp) {
          if (!new_batch)
            ++run;
          new_batch = true;
          continue;
        }
        int col = ctrlPktOp.getColumnFromAddr();
        int row = ctrlPktOp.getRowFromAddr();

        // Calculate control packet size
        int64_t ctrlPktSize = 0;
        auto data = ctrlPktOp.getData();
        auto length = ctrlPktOp.getLength();
        if (data)
          ctrlPktSize = data->size();
        else if (length)
          ctrlPktSize = *length;
        ctrlPktSize++; // Ctrl info word
        ctrlPktSize++; // Packet header

        // Check if we can batch with the previous packet
        // A batch is one stream packet, routed by its first header, so its
        // packets share their destination.
        bool enablePhase = enablePhaseStamp.isAttrPresent(ctrlPktOp);
        std::optional<int64_t> multicast;
        if (auto id = multicastStamp.getAttr(ctrlPktOp))
          multicast = id.getInt();
        if (targetModel.getTargetArch() == AIEArch::AIE2p && !new_batch &&
            batches.back().tileId == TileID{col, row} &&
            batches.back().enablePhase == enablePhase &&
            batches.back().multicast == multicast) {
          // Add to existing batch
          batches.back().totalSize += ctrlPktSize;
        } else {
          // Start new batch
          std::optional<int> shimChan = getCtrlPktShimChan(device, col);
          if (!shimChan) {
            ctrlPktOp.emitOpError("column ")
                << col
                << " has no control packet shim DMA allocation; run "
                   "-aie-generate-column-control-overlay with "
                   "route-shim-to-tile-ctrl=true first";
            return signalPassFailure();
          }
          batches.push_back({TileID{col, row}, ddrOffset, ctrlPktSize,
                             getCtrlPktAllocName(col, true, *shimChan),
                             *shimChan, &o, run, enablePhase, multicast});
          new_batch = false;
        }
        ddrOffset += ctrlPktSize;
      }

      // One task over `chain`'s batches, each one buffer descriptor, linked in
      // order.
      auto emitTask = [&](OpBuilder &builder,
                          ArrayRef<const BatchInfo *> chain) {
        auto task = DMAConfigureTaskForOp::create(
            builder, loc, builder.getIndexType(),
            SymbolRefAttr::get(ctx, chain.front()->shimDmaAllocName),
            /*issue_token=*/true, /*repeat_count=*/0,
            /*repeat_count_val=*/Value());
        OpBuilder::InsertionGuard guard(builder);
        Region &body = task.getBody();
        for (size_t i = 0; i < chain.size(); ++i)
          builder.createBlock(&body, body.end());
        for (auto [block, batch] : llvm::zip(body, chain)) {
          builder.setInsertionPointToEnd(&block);
          AIE::DMABDOp::create(builder, loc, newBlockArg,
                               static_cast<int>(batch->startOffset),
                               static_cast<int>(batch->totalSize));
          if (block.getNextNode())
            AIE::NextBDOp::create(builder, loc, block.getNextNode());
          else
            AIE::EndOp::create(builder, loc);
        }
        builder.setInsertionPointAfter(task);
        return task;
      };

      // A run's shim channels send at once, in phases each awaited in full: the
      // configuration, then the enable phase. In a phase each channel keeps its
      // batches' order and at most `window` tasks outstanding, one chain or as
      // many single tasks as its queue holds.
      auto emitParallelRun = [&](OpBuilder &builder, auto run) {
        for (bool enablePhase : {false, true}) {
          llvm::MapVector<StringRef,
                          SmallVector<SmallVector<const BatchInfo *>>>
              tasks;
          llvm::StringMap<size_t> window;
          for (const BatchInfo &batch : run) {
            if (batch.enablePhase != enablePhase)
              continue;
            int col = batch.tileId.col;
            uint32_t numBds = targetModel.getNumBDs(col, 0);
            auto &list = tasks[batch.shimDmaAllocName];
            if (list.empty() || !chainBds || list.back().size() >= numBds)
              list.emplace_back();
            list.back().push_back(&batch);
            uint32_t depth = targetModel.getDmaTaskQueueDepth(
                col, 0, batch.shimChan, DMAChannelDir::MM2S);
            window[batch.shimDmaAllocName] =
                chainBds ? 1 : (depth ? std::min(depth, numBds) : numBds);
          }
          size_t rounds = 0;
          for (auto &[name, list] : tasks)
            rounds = std::max(rounds, list.size());
          SmallVector<std::deque<DMAConfigureTaskForOp>> outstanding(
              tasks.size());
          for (size_t r = 0; r < rounds; ++r) {
            for (auto [i, entry] : llvm::enumerate(tasks)) {
              auto &[name, list] = entry;
              if (r >= list.size())
                continue;
              if (outstanding[i].size() >= window[name]) {
                DMAAwaitTaskOp::create(builder, loc, outstanding[i].front());
                outstanding[i].pop_front();
              }
              DMAConfigureTaskForOp task = emitTask(builder, list[r]);
              DMAStartTaskOp::create(builder, loc, task);
              outstanding[i].push_back(task);
            }
          }
          for (auto &queue : outstanding)
            for (DMAConfigureTaskForOp task : queue)
              DMAAwaitTaskOp::create(builder, loc, task);
        }
      };

      // Second pass: emit batched operations in original order
      auto batchIt = batches.begin();

      for (Operation &o : entry) {
        auto ctrlPktOp = dyn_cast<NpuControlPacketOp>(&o);
        if (!ctrlPktOp) {
          builder.clone(o, mapping);
          continue;
        }

        // There are no more control packet batches to emit
        if (batchIt == batches.end())
          continue;

        // Check if this is the first packet of a new batch, otherwise skip it.
        if (batchIt->first != &o)
          continue;

        int col = ctrlPktOp.getColumnFromAddr();

        if (parallelShims) {
          auto runEnd = batchIt;
          while (runEnd != batches.end() && runEnd->run == batchIt->run)
            ++runEnd;
          emitParallelRun(builder, llvm::make_range(batchIt, runEnd));
          batchIt = runEnd;
          continue;
        }

        if (chainBds) {
          // The batches that follow on the same channel in the same run,
          // up to the shim tile's buffer descriptor count.
          uint32_t maxBds = targetModel.getNumBDs(col, 0);
          auto chainEnd = std::next(batchIt);
          while (chainEnd != batches.end() &&
                 static_cast<uint32_t>(chainEnd - batchIt) < maxBds &&
                 chainEnd->run == batchIt->run &&
                 chainEnd->shimDmaAllocName == batchIt->shimDmaAllocName)
            ++chainEnd;

          SmallVector<const BatchInfo *> chain;
          for (auto it = batchIt; it != chainEnd; ++it)
            chain.push_back(&*it);
          DMAConfigureTaskForOp task = emitTask(builder, chain);
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
