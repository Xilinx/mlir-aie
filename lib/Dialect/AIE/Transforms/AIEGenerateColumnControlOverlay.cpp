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

int getUnusedPacketIdFrom(DeviceOp device) {
  int unusedPacketIdFrom = 0;
  device.walk([&](AIE::PacketFlowOp pOp) {
    unusedPacketIdFrom = std::max(unusedPacketIdFrom, pOp.IDInt());
  });
  device.walk([&](AIE::TileOp tOp) {
    if (!tOp->hasAttr("controller_id"))
      return;
    auto controllerIdPkt =
        tOp->getAttrOfType<AIE::PacketInfoAttr>("controller_id");
    unusedPacketIdFrom =
        std::max(unusedPacketIdFrom, (int)controllerIdPkt.getPktId());
  });
  return unusedPacketIdFrom + 1;
}

struct AIEAssignTileCtrlIDsPass
    : AIEAssignTileCtrlIDsBase<AIEAssignTileCtrlIDsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AIEDialect>();
  }
  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();

    if (targetModel.getTargetArch() == AIEArch::AIE1)
      return; // Disable this pass for AIE1; AIE1 support NYI.

    // Collect TileOps
    llvm::MapVector<AIE::TileID, AIE::TileOp> tiles;
    llvm::SmallSet<int, 1> occupiedCols;
    for (auto tile : device.getOps<AIE::TileOp>()) {
      int colIndex = tile.colIndex();
      int rowIndex = tile.rowIndex();
      tiles[{colIndex, rowIndex}] = tile;
      occupiedCols.insert(colIndex);
    }

    // Assign controller ids.
    int designUnusedPacketIdFrom = getUnusedPacketIdFrom(device);
    int unusedPacketIdFrom = designUnusedPacketIdFrom;
    for (int col : occupiedCols) {
      if (clColumnWiseUniqueIDs)
        unusedPacketIdFrom = designUnusedPacketIdFrom;
      SmallVector<AIE::TileOp> tilesOnCol;
      for (auto &[tId, tOp] : tiles) {
        if (tId.col != col)
          continue;
        tilesOnCol.push_back(tOp);
      }

      for (auto tOp : tilesOnCol) {
        if (tOp->hasAttr("controller_id"))
          continue;
        auto pktInfoAttr =
            AIE::PacketInfoAttr::get(tOp->getContext(), /*pkt_type*/ 0,
                                     /*pkt_id*/ unusedPacketIdFrom++);
        tOp->setAttr("controller_id", pktInfoAttr);
      }
    }
  }
};

struct AIEGenerateColumnControlOverlayPass
    : AIEGenerateColumnControlOverlayBase<AIEGenerateColumnControlOverlayPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AIEDialect>();
  }
  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();
    OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());

    if (targetModel.getTargetArch() == AIEArch::AIE1)
      return; // Disable this pass for AIE1; AIE1 support NYI.

    // Collect existing TileOps
    llvm::MapVector<AIE::TileID, AIE::TileOp> tiles;
    llvm::SmallSet<int, 1> occupiedCols;
    for (auto tile : device.getOps<AIE::TileOp>()) {
      int colIndex = tile.colIndex();
      int rowIndex = tile.rowIndex();
      tiles[{colIndex, rowIndex}] = tile;
      occupiedCols.insert(colIndex);
    }

    int designUnusedPacketIdFrom = getUnusedPacketIdFrom(device);
    for (int col : occupiedCols) {
      builder.setInsertionPointToStart(device.getBody());
      AIE::TileOp shimTile = TileOp::getOrCreate(builder, device, col, 0);

      if (clRouteShimCTRLToTCT == "all-tiles" ||
          clRouteShimCTRLToTCT == "shim-only") {
        // Get all tile ops on column col
        SmallVector<AIE::TileOp> tilesOnCol;
        for (auto &[tId, tOp] : tiles) {
          if (tId.col != col)
            continue;
          if (clRouteShimCTRLToTCT == "shim-only" && !tOp.isShimNOCorPLTile())
            continue;
          tilesOnCol.push_back(tOp);
        }

        int unusedPacketIdFrom = designUnusedPacketIdFrom;
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

        int unusedPacketIdFrom = designUnusedPacketIdFrom;
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

  AIE::PacketFlowOp
  createPacketFlowOp(OpBuilder &builder, int &flowID, Value source,
                     xilinx::AIE::WireBundle sourceBundle,
                     uint32_t sourceChannel, Value dest,
                     xilinx::AIE::WireBundle destBundle, uint32_t destChannel,
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

  // Create packet flows per col which moves control packets to and from shim
  // dma
  void generatePacketFlowsForControl(OpBuilder builder, DeviceOp device,
                                     const AIETargetModel &targetModel,
                                     TileOp shimTile, WireBundle shimWireBundle,
                                     SmallVector<int> availableShimChans,
                                     SmallVector<AIE::TileOp> ctrlTiles,
                                     WireBundle ctrlWireBundle,
                                     int coreOrMemChanId,
                                     int unusedPacketIdFrom, bool isShimMM2S) {
    // Create packet flows
    SmallVector<int>
        thresholdsToNextShimChannel; // a list of thresholds on the number of
                                     // control ports that the ith shim channel
                                     // could connect to, before advancing to
                                     // the next shim channel in round robin
    for (int i = 1; i < (int)availableShimChans.size() + 1; i++)
      thresholdsToNextShimChannel.push_back(ctrlTiles.size() /
                                            availableShimChans.size() * i);
    int ctrlPktFlowID = unusedPacketIdFrom;
    int shimChanIdx = 0;
    for (int i = 0; i < (int)ctrlTiles.size(); i++) {
      if (ctrlTiles[i]->hasAttr("controller_id"))
        ctrlPktFlowID =
            (int)ctrlTiles[i]
                ->getAttrOfType<AIE::PacketInfoAttr>("controller_id")
                .getPktId();
      builder.setInsertionPointToEnd(device.getBody());
      auto keep_pkt_header = builder.getBoolAttr(true);
      if (isShimMM2S)
        (void)createPacketFlowOp(
            builder, ctrlPktFlowID, shimTile, shimWireBundle,
            availableShimChans[shimChanIdx], ctrlTiles[i], ctrlWireBundle,
            coreOrMemChanId, keep_pkt_header);
      else
        (void)createPacketFlowOp(
            builder, ctrlPktFlowID, ctrlTiles[i], ctrlWireBundle,
            coreOrMemChanId, shimTile, shimWireBundle,
            availableShimChans[shimChanIdx], keep_pkt_header);
      if (i >= thresholdsToNextShimChannel[shimChanIdx]) {
        shimChanIdx++;
        ctrlPktFlowID = unusedPacketIdFrom;
      }
    }
  }

  // Get a vector of shim channels not reserved by any circuit-switched aie.flow
  // op
  SmallVector<int> getAvailableShimChans(DeviceOp device, int numShimChans,
                                         TileOp shimTile,
                                         WireBundle shimWireBundle,
                                         bool isShimMM2S) {
    SmallVector<int> availableShimChans;
    DenseMap<int, AIE::FlowOp> flowOpUsers;

    for (auto user : shimTile.getResult().getUsers()) {
      auto fOp = dyn_cast<AIE::FlowOp>(user);
      if (!fOp)
        continue;
      if (isShimMM2S && fOp.getSource() == shimTile)
        flowOpUsers[fOp.getSourceChannel()] = fOp;
      else if (!isShimMM2S && fOp.getDest() == shimTile)
        flowOpUsers[fOp.getDestChannel()] = fOp;
    }
    for (int i = 0; i < numShimChans; i++) {
      if (!flowOpUsers.count(i))
        availableShimChans.push_back(i);
    }
    return availableShimChans;
  }

  // Get packet-flow op with the same source or destination
  AIE::PacketFlowOp getPktFlowWithSameSrcOrDst(DeviceOp device, TileOp srcTile,
                                               WireBundle srcBundle,
                                               int srcChan, TileOp destTile,
                                               WireBundle destBundle,
                                               int destChan) {
    AIE::PacketFlowOp result = nullptr;
    device.walk([&](AIE::PacketFlowOp fOp) {
      for (auto srcOp : fOp.getOps<AIE::PacketSourceOp>()) {
        if (srcOp.getTile() == srcTile && srcOp.getBundle() == srcBundle &&
            srcOp.channelIndex() == srcChan) {
          result = fOp;
          return;
        }
      }
      for (auto destOp : fOp.getOps<AIE::PacketDestOp>()) {
        if (destOp.getTile() == destTile && destOp.getBundle() == destBundle &&
            destOp.channelIndex() == destChan) {
          result = fOp;
          return;
        }
      }
    });
    return result;
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEAssignTileCtrlIDsPass() {
  return std::make_unique<AIEAssignTileCtrlIDsPass>();
}

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEGenerateColumnControlOverlayPass() {
  return std::make_unique<AIEGenerateColumnControlOverlayPass>();
}

void populateAIEColumnControlOverlay(DeviceOp &device) {}
