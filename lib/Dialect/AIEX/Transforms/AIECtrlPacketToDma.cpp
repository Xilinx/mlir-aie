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

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

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

struct AIECtrlPacketInferTilesPass
    : xilinx::AIEX::impl::AIECtrlPacketInferTilesBase<
          AIECtrlPacketInferTilesPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();
    OpBuilder devBuilder = OpBuilder::atBlockBegin(device.getBody());

    auto sequenceOps = device.getOps<AIE::RuntimeSequenceOp>();
    for (auto f : sequenceOps) {
      for (auto ctrlPktOp : f.getOps<NpuControlPacketOp>()) {
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

struct AIECtrlPacketToDmaPass
    : xilinx::AIEX::impl::AIECtrlPacketToDmaBase<AIECtrlPacketToDmaPass> {
  using Base::Base;

  struct BatchInfo {
    TileID tileId;
    int64_t startOffset;
    int64_t totalSize;
    std::string shimDmaAllocName;
    Operation *first;
    unsigned delivery;
    unsigned interval;
    bool ordinary;
    bool independent;
    bool disjointMulticast;
    std::optional<int32_t> multicast;
  };

  static DMAConfigureTaskForOp emitTask(OpBuilder &builder, Location loc,
                                        Value buffer, const BatchInfo &batch) {
    auto task = DMAConfigureTaskForOp::create(
        builder, loc, builder.getIndexType(),
        SymbolRefAttr::get(builder.getContext(), batch.shimDmaAllocName),
        /*issue_token=*/true, /*repeat_count=*/0,
        /*repeat_count_val=*/Value());
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&task.getBody());
    AIE::DMABDOp::create(builder, loc, buffer,
                         static_cast<int>(batch.startOffset),
                         static_cast<int>(batch.totalSize));
    AIE::EndOp::create(builder, loc);
    return task;
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();
    auto *ctx = device->getContext();
    auto loc = device->getLoc();

    if (targetModel.getTargetArch() == AIEArch::AIE1)
      return; // Disable this pass for AIE1; AIE1 support NYI.

    auto sequenceOps = device.getOps<AIE::RuntimeSequenceOp>();
    for (auto f : sequenceOps) {
      if (f.getBody().empty())
        continue;
      Block &entry = f.getBody().front();
      auto controlPacketOps = f.getOps<AIEX::NpuControlPacketOp>();
      if (controlPacketOps.empty())
        continue;

      auto ctrlPktMemrefType = MemRefType::get(
          ShapedType::kDynamic, IntegerType::get(ctx, 32), nullptr, nullptr);
      auto packetData = entry.addArgument(ctrlPktMemrefType, loc);
      std::vector<BatchInfo> batches;

      int64_t ddrOffset = 0;

      NpuControlPacketOp previous;
      bool previousIntervalMulticast = false;
      for (auto [index, interval] : llvm::enumerate(packetIntervals(entry))) {
        llvm::DenseSet<uint32_t> multicastWords;
        bool disjointMulticast = interval.ordinary;
        ArrayAttr groups = controlPacketMulticasts(
            device->getParentOfType<ModuleOp>(), interval.device);
        for (NpuControlPacketOp packet : interval.packets)
          if (packet.getMulticastId() && packet.getData()) {
            ArrayAttr destinations;
            if (groups)
              for (Attribute attribute : groups) {
                auto group = cast<DictionaryAttr>(attribute);
                if (group.getAs<IntegerAttr>("column").getInt() ==
                        packet.getColumnFromAddr() &&
                    group.getAs<IntegerAttr>("id").getInt() ==
                        *packet.getMulticastId())
                  destinations = group.getAs<ArrayAttr>("tiles");
              }
            if (!destinations) {
              disjointMulticast = false;
              continue;
            }
            for (auto tile : destinations.getAsRange<AIE::TileAttr>()) {
              auto destination = tile.getTileID();
              uint32_t address =
                  (destination.col << targetModel.getColumnShift()) |
                  (destination.row << targetModel.getRowShift()) |
                  packet.getLocalAddress();
              for (size_t word = 0; word < packet.getData()->size(); ++word)
                multicastWords.insert(address + 4 * word);
            }
          }
        for (NpuControlPacketOp packet : interval.packets)
          if (!packet.getMulticastId() && packet.getData())
            for (size_t word = 0; word < packet.getData()->size(); ++word)
              disjointMulticast &=
                  !multicastWords.contains(packet.getAddress() + 4 * word);
        bool new_batch = !previous || previous.getOpcode() != 0 ||
                         batches.back().delivery != interval.delivery ||
                         previousIntervalMulticast;
        for (NpuControlPacketOp ctrlPktOp : interval.packets) {
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
            batches.back().totalSize += ctrlPktSize;
            batches.back().interval = index;
            batches.back().ordinary &= interval.ordinary;
            batches.back().independent &= interval.independent;
            batches.back().disjointMulticast &= disjointMulticast;
          } else {
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
                               alloc.getSymName().str(), ctrlPktOp,
                               interval.delivery, static_cast<unsigned>(index),
                               interval.ordinary, interval.independent,
                               disjointMulticast, multicast});
          }
          new_batch = !batchable;
          ddrOffset += ctrlPktSize;
          previous = ctrlPktOp;
        }
        previousIntervalMulticast =
            llvm::any_of(interval.packets, [](NpuControlPacketOp packet) {
              return packet.getMulticastId().has_value();
            });
      }

      auto batchIt = batches.begin();
      bool pendingMulticast = false;
      for (; batchIt != batches.end(); ++batchIt) {
        OpBuilder builder(batchIt->first);
        DMAConfigureTaskForOp task =
            emitTask(builder, batchIt->first->getLoc(), packetData, *batchIt);
        DMAStartTaskOp::create(builder, loc, task);
        auto next = std::next(batchIt);
        bool independent = next != batches.end() &&
                           ((batchIt->independent && next->independent) ||
                            (batchIt->ordinary && next->ordinary &&
                             next->interval == batchIt->interval));
        pendingMulticast |= batchIt->multicast.has_value();
        bool defer =
            next != batches.end() && next->delivery == batchIt->delivery &&
            (!pendingMulticast || next->interval == batchIt->interval) &&
            ((!batchIt->multicast && !next->multicast) ||
             (((batchIt->independent && next->independent) ||
               (batchIt->ordinary && next->ordinary)) &&
              next->interval == batchIt->interval &&
              ((batchIt->multicast && next->multicast) ||
               (!next->multicast && batchIt->disjointMulticast)))) &&
            ((chainBds && next->shimDmaAllocName == batchIt->shimDmaAllocName &&
              (independent || next->tileId == batchIt->tileId)) ||
             (parallelShims && independent));
        DMAAwaitTaskOp::create(builder, loc, task,
                               defer ? builder.getUnitAttr() : UnitAttr{});
        pendingMulticast &= defer;
      }
      for (NpuControlPacketOp packet :
           llvm::make_early_inc_range(controlPacketOps))
        packet.erase();
    }
    device.walk([](ControlPacketBarrierOp barrier) { barrier.erase(); });
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
