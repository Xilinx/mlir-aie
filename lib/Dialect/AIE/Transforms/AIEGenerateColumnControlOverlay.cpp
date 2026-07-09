//===- AIEGenerateColumnControlOverlay.cpp ----------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEGenerateColumnControlOverlay.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SetVector.h"

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
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    auto boolTrue = builder.getBoolAttr(true);
    auto boolFalse = builder.getBoolAttr(false);

    // Gather source devices in module order. Skip a previously-generated
    // overlay device so the pass is idempotent on its own output.
    SmallVector<DeviceOp> sourceDevices;
    for (auto dev : module.getOps<DeviceOp>()) {
      if (dev.getSymName() == "ctrl_pkt_overlay")
        continue;
      sourceDevices.push_back(dev);
    }

    // Baseline-preserving fast path: unless a standalone `@ctrl_pkt_overlay`
    // device was explicitly requested (the load-pdi-to-ctrl-pkt / reconfigure
    // flow), apply the overlay in place to every device exactly as the legacy
    // per-device pass did -- no extra attributes and no standalone device.
    // This keeps both the default `route-shim-to-tct` flow and direct
    // `route-shim-to-tile-ctrl=true` invocations byte-identical.
    if (!clEmitStandaloneOverlay) {
      for (auto dev : sourceDevices)
        if (failed(applyOverlayToDevice(dev)))
          return signalPassFailure();
      return;
    }

    // Ctrl-pkt flow: apply the overlay in place to each participating device
    // and emit a separate standalone `@ctrl_pkt_overlay` device.

    // Devices that participate in the standalone overlay (i.e. have not opted
    // out via `needs_ctrl_pkt_overlay = false`). Opted-out devices are tagged
    // immediately with `has_ctrl_pkt_overlay = false`.
    SmallVector<DeviceOp> participating;
    for (auto dev : sourceDevices) {
      if (deviceOptedOut(dev)) {
        dev->setAttr("has_ctrl_pkt_overlay", boolFalse);
        continue;
      }
      participating.push_back(dev);
    }

    // Step 1: apply the overlay in-place to each participating device, unless
    // the user asked us to suppress the in-place application via
    // `standalone-device-only`. Either way, tag each participating source
    // device with a `has_ctrl_pkt_overlay` boolean reflecting what happened.
    //
    // Before applying the overlay to a device, ensure that device contains
    // the full union of tiles referenced across all participating devices,
    // so the overlay shape (routes, shim_dma_allocations) is identical in
    // every device. When importing a missing tile, copy its attributes from
    // a participating device's prototype so downstream passes that compare
    // attribute dictionaries (e.g. AIEMaterializeRuntimeSequences) match.
    llvm::SmallSetVector<AIE::TileID, 8> unionTiles;
    llvm::DenseMap<AIE::TileID, AIE::TileOp> prototypeTile;
    for (auto dev : participating) {
      for (auto tOp : dev.getOps<AIE::TileOp>()) {
        AIE::TileID id{tOp.colIndex(), tOp.rowIndex()};
        unionTiles.insert(id);
        if (!prototypeTile.contains(id))
          prototypeTile[id] = tOp;
      }
    }

    for (auto dev : participating) {
      if (clStandaloneDeviceOnly) {
        dev->setAttr("has_ctrl_pkt_overlay", boolFalse);
        continue;
      }
      // Add any missing tiles from the union so the overlay shape is shared.
      llvm::SmallSet<AIE::TileID, 8> existing;
      for (auto tOp : dev.getOps<AIE::TileOp>())
        existing.insert({tOp.colIndex(), tOp.rowIndex()});
      OpBuilder b = OpBuilder::atBlockBegin(dev.getBody());
      for (auto id : unionTiles) {
        if (existing.contains(id))
          continue;
        auto proto = prototypeTile[id];
        auto cloned = cast<AIE::TileOp>(b.clone(*proto.getOperation()));
        (void)cloned;
      }
      if (failed(applyOverlayToDevice(dev)))
        return signalPassFailure();
      dev->setAttr("has_ctrl_pkt_overlay", boolTrue);
    }

    // Step 2: always emit a separate `@ctrl_pkt_overlay` device that contains
    // only the overlay (and the union of tiles it references). This gives
    // downstream consumers a single, isolated device to compile when they
    // need to ship a reconfigure-only PDI, regardless of whether the overlay
    // was also applied in place.
    if (participating.empty())
      return;

    // Verify all participating devices share the same target.
    auto refDevice = participating.front().getDevice();
    for (auto dev : llvm::drop_begin(participating)) {
      if (dev.getDevice() != refDevice) {
        dev->emitOpError(
            "cannot generate a single standalone ctrl_pkt_overlay device: "
            "participating devices have mismatched target architectures.");
        return signalPassFailure();
      }
    }

    // Refuse to overwrite an existing symbol with the reserved name.
    if (module.lookupSymbol("ctrl_pkt_overlay")) {
      module.emitOpError(
          "a symbol named `ctrl_pkt_overlay` already exists in the module; "
          "cannot create a standalone ctrl_pkt_overlay device.");
      return signalPassFailure();
    }

    // Create the new device at the end of the module.
    builder.setInsertionPointToEnd(module.getBody());
    Location loc = participating.front().getLoc();
    auto overlayDevice = AIE::DeviceOp::create(
        builder, loc, refDevice, builder.getStringAttr("ctrl_pkt_overlay"));
    overlayDevice.getRegion().emplaceBlock();
    Block *body = &overlayDevice.getRegion().front();
    builder.setInsertionPointToEnd(body);
    AIE::EndOp::create(builder, loc);

    // Clone the union of TileIDs referenced across participating devices.
    llvm::SmallSet<AIE::TileID, 8> seenTiles;
    builder.setInsertionPointToStart(body);
    for (auto dev : participating) {
      for (auto tOp : dev.getOps<AIE::TileOp>()) {
        AIE::TileID id{tOp.colIndex(), tOp.rowIndex()};
        if (!seenTiles.insert(id).second)
          continue;
        AIE::TileOp::create(builder, tOp.getLoc(), id.col, id.row);
      }
    }

    if (failed(applyOverlayToDevice(overlayDevice)))
      return signalPassFailure();

    overlayDevice->setAttr("has_ctrl_pkt_overlay", boolTrue);
  }

  // Apply the column-control overlay to `device` in place. Returns failure on
  // a routing conflict.
  LogicalResult applyOverlayToDevice(DeviceOp device) {
    const auto &targetModel = device.getTargetModel();
    OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());

    if (targetModel.getTargetArch() == AIEArch::AIE1)
      return success(); // Disable this pass for AIE1; AIE1 support NYI.

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

        if (failed(generatePacketFlowsForControl(
                builder, device, shimTile, AIE::WireBundle::South, tilesOnCol,
                AIE::WireBundle::TileControl, 0, tileIDMap, false)))
          return failure();
      }
      if (clRouteShimDmaToTileCTRL) {
        // Ensure tiles exist for the full range from shim (row 0) to the
        // highest existing tile in the column. Intermediate tiles (e.g. mem
        // tiles) are needed for control packet routing and will also need
        // their switchboxes configured via control packets.
        int maxRow = 0;
        for (auto &[tId, tOp] : tiles) {
          if (tId.col == col)
            maxRow = std::max(maxRow, tId.row);
        }
        SmallVector<AIE::TileOp> tilesOnCol;
        for (int row = 0; row <= maxRow; row++) {
          auto tOp = TileOp::getOrCreate(builder, device, col, row);
          tilesOnCol.push_back(tOp);
        }

        if (failed(generatePacketFlowsForControl(
                builder, device, shimTile, AIE::WireBundle::DMA, tilesOnCol,
                AIE::WireBundle::TileControl, 0, tileIDMap, true)))
          return failure();
      }
    }
    return success();
  }

  // Return true when the user has explicitly disabled overlay generation for
  // this device via `needs_ctrl_pkt_overlay = false`.
  static bool deviceOptedOut(DeviceOp device) {
    auto attr = device->getAttrOfType<BoolAttr>("needs_ctrl_pkt_overlay");
    return attr && !attr.getValue();
  }

  AIE::PacketFlowOp createPacketFlowOp(OpBuilder &builder, Location loc,
                                       int &flowID, Value source,
                                       xilinx::AIE::WireBundle sourceBundle,
                                       uint32_t sourceChannel, Value dest,
                                       xilinx::AIE::WireBundle destBundle,
                                       uint32_t destChannel,
                                       mlir::BoolAttr keep_pkt_header = nullptr,
                                       mlir::BoolAttr ctrl_pkt_flow = nullptr) {
    OpBuilder::InsertionGuard guard(builder);

    AIE::PacketFlowOp pktFlow = AIE::PacketFlowOp::create(
        builder, loc, flowID++, keep_pkt_header, ctrl_pkt_flow);
    Region &r_pktFlow = pktFlow.getPorts();
    Block *b_pktFlow = builder.createBlock(&r_pktFlow);
    builder.setInsertionPointToStart(b_pktFlow);
    AIE::PacketSourceOp::create(builder, loc, source, sourceBundle,
                                sourceChannel);
    AIE::PacketDestOp::create(builder, loc, dest, destBundle, destChannel);
    AIE::EndOp::create(builder, loc);
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
  LogicalResult generatePacketFlowsForControl(
      OpBuilder builder, DeviceOp device, TileOp shimTile,
      WireBundle shimWireBundle, SmallVector<AIE::TileOp> ctrlTiles,
      WireBundle ctrlWireBundle, int coreOrMemChanId,
      DenseMap<TileID, int> tileIDMap, bool isShimMM2S) {
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
        return failure();
      }

      auto keep_pkt_header = builder.getBoolAttr(true);
      auto ctrl_pkt_flow = builder.getBoolAttr(true);
      if (isShimMM2S)
        (void)createPacketFlowOp(
            builder, tOp.getLoc(), ctrlPktFlowID, shimTile, shimWireBundle,
            rowToShimChanMap[tOp.rowIndex()], tOp, ctrlWireBundle,
            coreOrMemChanId, keep_pkt_header, ctrl_pkt_flow);
      else
        (void)createPacketFlowOp(
            builder, tOp.getLoc(), ctrlPktFlowID, tOp, ctrlWireBundle,
            coreOrMemChanId, shimTile, shimWireBundle,
            rowToShimChanMap[tOp.rowIndex()], keep_pkt_header, ctrl_pkt_flow);

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
          builder, tOp.getLoc(), StringRef(dma_name), shimTile.getResult(), dir,
          rowToShimChanMap[tOp.rowIndex()], false, nullptr);
    }
    return success();
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

std::unique_ptr<OperationPass<mlir::ModuleOp>>
AIE::createAIEGenerateColumnControlOverlayPass() {
  return std::make_unique<AIEGenerateColumnControlOverlayPass>();
}

void populateAIEColumnControlOverlay(DeviceOp &device) {}
