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
        // Create packet flows per col
        SmallVector<int> thresholdsToNextChannel;
        int numShimDmaMM2SChans = targetModel.getNumSourceShimMuxConnections(
            shimTile.getCol(), shimTile.getRow(), AIE::WireBundle::DMA);
        for (int i = 1; i < numShimDmaMM2SChans + 1; i++)
          thresholdsToNextChannel.push_back(tilesOnCol.size() /
                                            numShimDmaMM2SChans * i);
        int unusedIdFrom =
            0; // TODO: get unused packet header for control packets
        int ctrlPktFlowID = unusedIdFrom;
        int currShimChan = 0;
        for (int i = 0; i < (int)tilesOnCol.size(); i++) {
          builder.setInsertionPointToEnd(device.getBody());
          (void)createPacketFlowOp(builder, ctrlPktFlowID, shimTile,
                                   AIE::WireBundle::DMA, currShimChan,
                                   tilesOnCol[i], AIE::WireBundle::Ctrl, 0);
          if (i >= thresholdsToNextChannel[currShimChan]) {
            currShimChan++;
            ctrlPktFlowID = unusedIdFrom;
          }
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEGenerateColumnControlOverlayPass() {
  return std::make_unique<AIEGenerateColumnControlOverlayPass>();
}

void populateAIEColumnControlOverlay(DeviceOp &device) {}
