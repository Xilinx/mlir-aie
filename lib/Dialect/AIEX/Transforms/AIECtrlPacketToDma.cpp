//===- AIECtrlPacketToDma.cpp -----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEGenerateColumnControlOverlay.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "aie-ctrl-packet-to-dma"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

struct AIECtrlPacketInferTilesPass
    : AIECtrlPacketInferTilesBase<AIECtrlPacketInferTilesPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();
    OpBuilder devBuilder = OpBuilder::atBlockBegin(device.getBody());

    auto sequenceOps = device.getOps<AIEX::RuntimeSequenceOp>();
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

struct AIECtrlPacketToDmaPass : AIECtrlPacketToDmaBase<AIECtrlPacketToDmaPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();
    auto ctx = device->getContext();
    auto loc = device->getLoc();
    OpBuilder devBuilder = OpBuilder::atBlockBegin(device.getBody());

    if (targetModel.getTargetArch() == AIEArch::AIE1)
      return; // Disable this pass for AIE1; AIE1 support NYI.

    SmallVector<Operation *> erased;
    auto sequenceOps = device.getOps<AIEX::RuntimeSequenceOp>();
    for (auto f : sequenceOps) {

      auto controlPacketOps = f.getOps<AIEX::NpuControlPacketOp>();
      if (controlPacketOps.empty())
        continue;

      OpBuilder builder(f);

      auto newSeq =
          builder.create<AIEX::RuntimeSequenceOp>(loc, f.getSymNameAttr());
      newSeq.getBody().push_back(new Block);

      // Using dynamic shape for ctrl pkt stream.
      auto ctrlPktMemrefType = MemRefType::get(
          ShapedType::kDynamic, IntegerType::get(ctx, 32), nullptr, 0);
      auto newBlockArg = newSeq.getBody().addArgument(ctrlPktMemrefType, loc);
      builder.setInsertionPointToStart(&newSeq.getBody().front());

      int ddrOffset = 0;
      Block &entry = f.getBody().front();
      for (auto &o : entry) {
        llvm::TypeSwitch<Operation *>(&o).Case<NpuControlPacketOp>(
            [&](auto op) {
              // Destination tile info
              int col = op.getColumnFromAddr();
              int row = op.getRowFromAddr();
              AIE::TileOp destTileOp =
                  TileOp::getOrCreate(devBuilder, device, col, row);
              assert(destTileOp->hasAttr("controller_id"));
              auto controllerIdPkt =
                  destTileOp->getAttrOfType<AIE::PacketInfoAttr>(
                      "controller_id");

              // Control packet offset (to raw data at ddr) and size
              uint32_t ctrlPktSize = 0;
              auto data = op.getData();
              auto length = op.getLength();
              if (data)
                ctrlPktSize = data->size();
              if (!data && length)
                ctrlPktSize = *length;
              ctrlPktSize++; // Ctrl info word

              const std::vector<int64_t> staticOffsets = {0, 0, 0, ddrOffset};
              ddrOffset += ctrlPktSize;
              const std::vector<int64_t> staticSizes = {1, 1, 1,
                                                        (int64_t)ctrlPktSize};
              const std::vector<int64_t> staticStrides = {0, 0, 0, 1};

              // Shim dma alloc symbol name
              std::string shimDmaAllocName = "ctrlpkt";
              shimDmaAllocName += "_col" + std::to_string(col);
              shimDmaAllocName += "_mm2s";
              auto rowToShimChanMap =
                  getRowToShimChanMap(targetModel, WireBundle::DMA);
              int shimChan = rowToShimChanMap[destTileOp.rowIndex()];
              shimDmaAllocName += "_chan" + std::to_string(shimChan);

              StringRef metadata = builder.getStringAttr(shimDmaAllocName);
              builder.create<NpuDmaMemcpyNdOp>(
                  builder.getUnknownLoc(), 0, 0, newBlockArg,
                  SmallVector<Value>{}, SmallVector<Value>{},
                  SmallVector<Value>{}, ArrayRef(staticOffsets),
                  ArrayRef(staticSizes), ArrayRef(staticStrides),
                  controllerIdPkt, metadata, 0, true, 0, 0, 0, 0, 0, 0);

              auto shimRow = builder.getI32IntegerAttr(0);
              auto shimCol = builder.getI32IntegerAttr(col);
              auto dir = builder.getI32IntegerAttr(1); // MM2S
              auto chan = builder.getI32IntegerAttr(shimChan);
              auto col_num = builder.getI32IntegerAttr(1);
              auto row_num = builder.getI32IntegerAttr(1);
              builder.create<AIEX::NpuSyncOp>(loc, shimCol, shimRow, dir, chan,
                                              col_num, row_num);
            });
      }

      erased.push_back(f);
    }

    for (auto e : erased)
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
