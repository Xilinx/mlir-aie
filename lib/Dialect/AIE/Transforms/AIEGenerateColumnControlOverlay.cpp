//===- AIEGenerateColumnControlOverlay.cpp ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEGenerateColumnControlOverlay.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallSet.h"

#define DEBUG_TYPE "aie-generate-column-control-overlay"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

AIE::PacketFlowOp createPacketFlowOp(OpBuilder &builder, int &flowID,
                                     Value source,
                                     xilinx::AIE::WireBundle sourceBundle,
                                     uint32_t sourceChannel, Value dest,
                                     xilinx::AIE::WireBundle destBundle,
                                     uint32_t destChannel,
                                     mlir::BoolAttr keep_pkt_header = nullptr) {
  AIE::PacketFlowOp pktFlow = builder.create<AIE::PacketFlowOp>(
      builder.getUnknownLoc(), flowID++, keep_pkt_header);
  Region &r_pktFlow = pktFlow.getPorts();
  Block *b_pktFlow = builder.createBlock(&r_pktFlow);
  builder.setInsertionPointToStart(b_pktFlow);
  builder.create<AIE::PacketSourceOp>(builder.getUnknownLoc(), source,
                                      sourceBundle, sourceChannel);
  builder.create<AIE::PacketDestOp>(builder.getUnknownLoc(), dest, destBundle,
                                    destChannel);
  builder.create<AIE::EndOp>(builder.getUnknownLoc());
  return pktFlow;
}

struct AIEGenerateColumnControlOverlayPass
    : AIEGenerateColumnControlOverlayBase<AIEGenerateColumnControlOverlayPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AIEDialect>();
  }
  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();
    OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());

    // Collect existing TileOps
    DenseMap<AIE::TileID, AIE::TileOp> tiles;
    llvm::SmallSet<int, 1> occupiedCols;
    for (auto tile : device.getOps<AIE::TileOp>()) {
      int colIndex = tile.colIndex();
      int rowIndex = tile.rowIndex();
      tiles[{colIndex, rowIndex}] = tile;
      occupiedCols.insert(colIndex);
    }

    for (int col : occupiedCols) {
      AIE::TileOp shimTile = nullptr;
      builder.setInsertionPointToStart(device.getBody());
      if (tiles.count({col, 0}))
        shimTile = tiles[{col, 0}];
      else
        shimTile = builder.create<AIE::TileOp>(builder.getUnknownLoc(), col, 0);

      if (clRouteShimCTRLToTCT) {
        // Get all tile ops on column col
        SmallVector<AIE::TileOp> tilesOnCol;
        for (auto &[tId, tOp] : tiles) {
          if (tId.col != col)
            continue;
          tilesOnCol.push_back(tOp);
        }

        int unusedPacketIdFrom = getUnusedPacketIdFrom(device);
        SmallVector<int> availableShimChans = {
            0}; // Only using SHIM dest SOUTH 0 for TCT.
        // Create packet flows per col
        generatePacketFlowsForControl(
            builder, device, targetModel, shimTile, AIE::WireBundle::South,
            availableShimChans, tilesOnCol, AIE::WireBundle::Ctrl, 0,
            unusedPacketIdFrom, false);
      }
      if (clRouteShimDmaToTileCTRL) {
        // Get all tile ops on column col
        SmallVector<AIE::TileOp> tilesOnCol;
        for (auto &[tId, tOp] : tiles) {
          if (tId.col != col)
            continue;
          if (targetModel.isCoreTile(tId.col, tId.row) ||
              targetModel.isMemTile(tId.col, tId.row))
            tilesOnCol.push_back(tOp);
        }

        int unusedPacketIdFrom = getUnusedPacketIdFrom(device);
        int numShimChans = targetModel.getNumSourceShimMuxConnections(
            shimTile.getCol(), shimTile.getRow(), AIE::WireBundle::DMA);
        SmallVector<int> availableShimChans = getAvailableShimChans(
            device, numShimChans, shimTile, AIE::WireBundle::DMA, true);
        // Create packet flows per col
        generatePacketFlowsForControl(builder, device, targetModel, shimTile,
                                      AIE::WireBundle::DMA, availableShimChans,
                                      tilesOnCol, AIE::WireBundle::Ctrl, 0,
                                      unusedPacketIdFrom, true);
      }
    }
  }

  // Create packet flows per col which moves control packets to and from shim
  // dma
  void generatePacketFlowsForControl(OpBuilder builder, DeviceOp device,
                                     const AIETargetModel &targetModel,
                                     TileOp shimTile, WireBundle shimWireBundle,
                                     SmallVector<int> availableShimChans,
                                     SmallVector<AIE::TileOp> coreOrMemTiles,
                                     WireBundle coreOrMemWireBundle,
                                     int coreOrMemChanId,
                                     int unusedPacketIdFrom, bool isShimMM2S) {
    // Get a set of available shim dma channels
    int numShimDmaChans = targetModel.getNumSourceShimMuxConnections(
        shimTile.getCol(), shimTile.getRow(), shimWireBundle);
    // Create packet flows
    SmallVector<int> thresholdsToNextShimChannel;
    for (int i = 1; i < (int)availableShimChans.size() + 1; i++)
      thresholdsToNextShimChannel.push_back(coreOrMemTiles.size() /
                                            availableShimChans.size() * i);
    int ctrlPktFlowID = unusedPacketIdFrom;
    int shimChanIdx = 0;
    for (int i = 0; i < (int)coreOrMemTiles.size(); i++) {
      builder.setInsertionPointToEnd(device.getBody());
      if (isShimMM2S)
        (void)createPacketFlowOp(
            builder, ctrlPktFlowID, shimTile, shimWireBundle,
            availableShimChans[shimChanIdx], coreOrMemTiles[i],
            coreOrMemWireBundle, coreOrMemChanId);
      else
        (void)createPacketFlowOp(builder, ctrlPktFlowID, coreOrMemTiles[i],
                                 coreOrMemWireBundle, coreOrMemChanId, shimTile,
                                 shimWireBundle,
                                 availableShimChans[shimChanIdx]);
      if (i >= thresholdsToNextShimChannel[shimChanIdx]) {
        shimChanIdx++;
        ctrlPktFlowID = unusedPacketIdFrom;
      }
    }
  }

  int getUnusedPacketIdFrom(DeviceOp device) {
    int unusedPacketIdFrom = 0;
    device.walk([&](AIE::PacketFlowOp pOp) {
      unusedPacketIdFrom = std::max(unusedPacketIdFrom, pOp.IDInt());
    });
    return unusedPacketIdFrom + 1;
  }

  // Get a vector of shim channels not reserved by any circuit-switched aie.flow
  // op
  SmallVector<int> getAvailableShimChans(DeviceOp device, int numShimChans,
                                         TileOp shimTile,
                                         WireBundle shimWireBundle,
                                         bool isShimMM2S) {
    SmallVector<int> availableShimChans;
    for (int i = 0; i < numShimChans; i++) {
      bool isShimChanReservedByAIEFlowOp = false;
      device.walk([&](AIE::FlowOp fOp) {
        if (isShimMM2S && fOp.getSource() == shimTile &&
            fOp.getSourceBundle() == shimWireBundle &&
            fOp.getSourceChannel() == i) {
          isShimChanReservedByAIEFlowOp = true;
          return;
        } else if (!isShimMM2S && fOp.getDest() == shimTile &&
                   fOp.getDestBundle() == shimWireBundle &&
                   fOp.getDestChannel() == i) {
          isShimChanReservedByAIEFlowOp = true;
          return;
        }
      });
      if (!isShimChanReservedByAIEFlowOp)
        availableShimChans.push_back(i);
    }
    return availableShimChans;
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEGenerateColumnControlOverlayPass() {
  return std::make_unique<AIEGenerateColumnControlOverlayPass>();
}

void populateAIEColumnControlOverlay(DeviceOp &device) {}
