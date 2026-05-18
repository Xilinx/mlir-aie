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

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEGENERATECOLUMNCONTROLOVERLAY
#define GEN_PASS_DEF_AIEASSIGNTILECTRLIDS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

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

// Delegate to AIETargetModel::getTileToControllerIdMap.
DenseMap<AIE::TileID, int>
getTileToControllerIdMap(bool clColumnWiseUniqueIDs,
                         const AIETargetModel &targetModel) {
  return targetModel.getTileToControllerIdMap(clColumnWiseUniqueIDs);
}

// AIE arch-specific row id to shim dma mm2s channel mapping. All shim mm2s
// channels were assumed to be available for control packet flow routing (i.e.
// not reserved by any aie.flow circuit-switched routing).
DenseMap<int, int> getRowToShimChanMap(const AIETargetModel &targetModel,
                                       WireBundle bundle) {
  DenseMap<int, int> rowMap;
  SmallVector<int>
      thresholdsToNextShimChannel; // a list of thresholds on the number of
                                   // control ports that the ith shim channel
                                   // could connect to, before advancing to
                                   // the next shim channel in round robin
  TileID shimTile = {0, 0};
  while (!targetModel.isShimNOCTile(shimTile.col, shimTile.row)) {
    shimTile.col++;
    if (shimTile.col == targetModel.columns()) {
      shimTile.col = 0;
      shimTile.row++;
    }
    assert(!(shimTile.col == targetModel.columns() &&
             shimTile.row == targetModel.rows()));
  }

  int numShimChans = targetModel.getNumSourceShimMuxConnections(
      shimTile.col, shimTile.row, AIE::WireBundle::DMA);
  for (int i = 1; i < numShimChans + 1; i++)
    thresholdsToNextShimChannel.push_back(targetModel.rows() / numShimChans *
                                          i);

  if (bundle == WireBundle::DMA) { // Ctrl packets
    int shimChanIdx = 0;
    for (int r = 0; r < targetModel.rows(); r++) {
      if (r >= thresholdsToNextShimChannel[shimChanIdx])
        shimChanIdx++;
      rowMap[r] = shimChanIdx;
    }
  } else if (bundle == WireBundle::South) { // TCT
    for (int r = 0; r < targetModel.rows(); r++)
      rowMap[r] = 0;
  }

  return rowMap;
}

struct AIEAssignTileCtrlIDsPass
    : xilinx::AIE::impl::AIEAssignTileCtrlIDsBase<AIEAssignTileCtrlIDsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AIEDialect>();
  }
  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();

    if (targetModel.getTargetArch() == AIEArch::AIE1)
      return; // Disable this pass for AIE1; AIE1 support NYI.

    // Collect all TileOps in columns occupied by the design.
    llvm::MapVector<AIE::TileID, AIE::TileOp> tiles;
    llvm::SmallSet<int, 1> occupiedCols;
    for (auto tile : device.getOps<AIE::TileOp>()) {
      int colIndex = tile.colIndex();
      int rowIndex = tile.rowIndex();
      tiles[{colIndex, rowIndex}] = tile;
      occupiedCols.insert(colIndex);
    }

    auto tileIDMap =
        getTileToControllerIdMap(clColumnWiseUniqueIDs, targetModel);
    for (int col : occupiedCols) {
      SmallVector<AIE::TileOp> tilesOnCol;
      for (auto &[tId, tOp] : tiles) {
        if (tId.col != col)
          continue;
        tilesOnCol.push_back(tOp);
      }

      for (auto tOp : tilesOnCol) {
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

struct AIEGenerateColumnControlOverlayPass
    : xilinx::AIE::impl::AIEGenerateColumnControlOverlayBase<
          AIEGenerateColumnControlOverlayPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AIEDialect>();
    registry.insert<memref::MemRefDialect>();
  }
  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();
    OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());

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

    auto tileIDMap = getTileToControllerIdMap(true, targetModel);
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

        generatePacketFlowsForControl(
            builder, device, shimTile, AIE::WireBundle::South, tilesOnCol,
            AIE::WireBundle::TileControl, 0, tileIDMap, false);
      }
      if (clRouteShimDmaToTileCTRL) {
        // Get all tile ops on column col
        SmallVector<AIE::TileOp> tilesOnCol;
        for (auto &[tId, tOp] : tiles) {
          if (tId.col != col)
            continue;
          tilesOnCol.push_back(tOp);
        }

        generatePacketFlowsForControl(
            builder, device, shimTile, AIE::WireBundle::DMA, tilesOnCol,
            AIE::WireBundle::TileControl, 0, tileIDMap, true);
      }
    }
  }

  AIE::PacketFlowOp createPacketFlowOp(OpBuilder &builder, int &flowID,
                                       Value source,
                                       xilinx::AIE::WireBundle sourceBundle,
                                       uint32_t sourceChannel, Value dest,
                                       xilinx::AIE::WireBundle destBundle,
                                       uint32_t destChannel,
                                       mlir::BoolAttr keep_pkt_header = nullptr,
                                       mlir::BoolAttr ctrl_pkt_flow = nullptr) {
    OpBuilder::InsertionGuard guard(builder);

    AIE::PacketFlowOp pktFlow =
        AIE::PacketFlowOp::create(builder, builder.getUnknownLoc(), flowID++,
                                  keep_pkt_header, ctrl_pkt_flow);
    Region &r_pktFlow = pktFlow.getPorts();
    Block *b_pktFlow = builder.createBlock(&r_pktFlow);
    builder.setInsertionPointToStart(b_pktFlow);
    AIE::PacketSourceOp::create(builder, builder.getUnknownLoc(), source,
                                sourceBundle, sourceChannel);
    AIE::PacketDestOp::create(builder, builder.getUnknownLoc(), dest,
                              destBundle, destChannel);
    AIE::EndOp::create(builder, builder.getUnknownLoc());
    return pktFlow;
  }

  // Get a vector of shim channels not reserved by any circuit-switched aie.flow
  // op
  SmallVector<int> getAvailableShimChans(DeviceOp device, TileOp shimTile,
                                         WireBundle shimWireBundle,
                                         bool isShimMM2S) {
    SmallVector<int> availableShimChans;
    DenseMap<int, AIE::FlowOp> flowOpUsers;
    const auto &targetModel = device.getTargetModel();

    for (auto user : shimTile.getResult().getUsers()) {
      auto fOp = dyn_cast<AIE::FlowOp>(user);
      if (!fOp)
        continue;
      if (isShimMM2S && fOp.getSource() == shimTile &&
          fOp.getSourceBundle() == shimWireBundle)
        flowOpUsers[fOp.getSourceChannel()] = fOp;
      else if (!isShimMM2S && fOp.getDest() == shimTile &&
               fOp.getDestBundle() == shimWireBundle)
        flowOpUsers[fOp.getDestChannel()] = fOp;
    }
    int numShimChans = 0;
    if (isShimMM2S)
      numShimChans = targetModel.getNumSourceShimMuxConnections(
          shimTile.colIndex(), shimTile.rowIndex(), shimWireBundle);
    else
      numShimChans = targetModel.getNumDestShimMuxConnections(
          shimTile.colIndex(), shimTile.rowIndex(), shimWireBundle);
    for (int i = 0; i < numShimChans; i++) {
      if (!flowOpUsers.count(i))
        availableShimChans.push_back(i);
    }

    return availableShimChans;
  }

  // Create packet flows per col which moves control packets to and from shim
  // dma
  void generatePacketFlowsForControl(OpBuilder builder, DeviceOp device,
                                     TileOp shimTile, WireBundle shimWireBundle,
                                     SmallVector<AIE::TileOp> ctrlTiles,
                                     WireBundle ctrlWireBundle,
                                     int coreOrMemChanId,
                                     DenseMap<TileID, int> tileIDMap,
                                     bool isShimMM2S) {
    int ctrlPktFlowID = 0;
    auto rowToShimChanMap =
        getRowToShimChanMap(device.getTargetModel(), shimWireBundle);
    // Get all available shim channels, to verify that the one being used is
    // available
    auto availableShimChans =
        getAvailableShimChans(device, shimTile, shimWireBundle, isShimMM2S);

    builder.setInsertionPoint(device.getBody()->getTerminator());
    for (auto tOp : ctrlTiles) {
      if (tOp->hasAttr("controller_id"))
        ctrlPktFlowID =
            (int)tOp->getAttrOfType<AIE::PacketInfoAttr>("controller_id")
                .getPktId();
      else
        ctrlPktFlowID = tileIDMap[{tOp.colIndex(), tOp.rowIndex()}];
      // Check shim channel availability
      if (!llvm::is_contained(availableShimChans,
                              rowToShimChanMap[tOp.rowIndex()])) {
        device->emitOpError(
            "failed to generate column control overlay from shim dma to tile "
            "ctrl ports, because some shim mm2s dma channels were reserved "
            "from routing control packets.");
        return signalPassFailure();
      }

      auto keep_pkt_header = builder.getBoolAttr(true);
      auto ctrl_pkt_flow = builder.getBoolAttr(true);
      if (isShimMM2S)
        (void)createPacketFlowOp(
            builder, ctrlPktFlowID, shimTile, shimWireBundle,
            rowToShimChanMap[tOp.rowIndex()], tOp, ctrlWireBundle,
            coreOrMemChanId, keep_pkt_header, ctrl_pkt_flow);
      else
        (void)createPacketFlowOp(builder, ctrlPktFlowID, tOp, ctrlWireBundle,
                                 coreOrMemChanId, shimTile, shimWireBundle,
                                 rowToShimChanMap[tOp.rowIndex()],
                                 keep_pkt_header, ctrl_pkt_flow);

      // Generate shim dma alloc ops as handle for runtime sequence to pickup,
      // when issuing control packets
      if (shimWireBundle != WireBundle::DMA)
        continue;

      AIE::DMAChannelDir dir =
          isShimMM2S ? AIE::DMAChannelDir::MM2S : AIE::DMAChannelDir::S2MM;
      int chan = rowToShimChanMap[tOp.rowIndex()];
      int col = shimTile.colIndex();
      std::string dma_name = "ctrlpkt";
      dma_name += "_col" + std::to_string(col);   // col
      dma_name += isShimMM2S ? "_mm2s" : "_s2mm"; // dir
      dma_name += "_chan" + std::to_string(chan); // chan

      // check to see if ShimDMAAllocationOp already exists
      if (device.lookupSymbol(dma_name))
        continue;

      AIE::ShimDMAAllocationOp::create(
          builder, builder.getUnknownLoc(), StringRef(dma_name),
          shimTile.getResult(), dir, rowToShimChanMap[tOp.rowIndex()], false,
          nullptr);
    }
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
