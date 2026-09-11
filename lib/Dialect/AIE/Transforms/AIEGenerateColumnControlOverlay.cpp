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

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"

#include <map>
#include <optional>
#include <tuple>

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEGENERATECOLUMNCONTROLOVERLAY
#define GEN_PASS_DEF_AIEASSIGNTILECTRLIDS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-generate-column-control-overlay"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

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
    assert(shimTile.col != targetModel.columns() ||
           shimTile.row != targetModel.rows());
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

    // Gather source devices in module order. Skip a previously-generated
    // overlay device so the pass is idempotent on its own output.
    SmallVector<DeviceOp> sourceDevices;
    for (auto dev : module.getOps<DeviceOp>()) {
      if (dev.getSymName() == "ctrl_pkt_overlay")
        continue;
      sourceDevices.push_back(dev);
    }

    // Devices that receive the overlay: those that have not opted out via
    // `needs_ctrl_pkt_overlay = false`.
    SmallVector<DeviceOp> participating;
    for (auto dev : sourceDevices) {
      if (deviceOptedOut(dev)) {
        if (clEmitStandaloneOverlay) {
          dev->setAttr("has_ctrl_pkt_overlay", builder.getBoolAttr(false));
        }
        continue;
      }
      participating.push_back(dev);
    }

    // A standalone `@ctrl_pkt_overlay` device references a single overlay
    // shape, so every participating device must expose the same set of tiles
    // for that shape to be identical across them.
    if (clEmitStandaloneOverlay)
      shareTilesAcrossDevices(participating);

    // Apply the overlay in-place to participating devices.
    for (auto dev : participating) {
      if (failed(applyOverlayToDevice(dev)))
        return signalPassFailure();
      if (clEmitStandaloneOverlay)
        dev->setAttr("has_ctrl_pkt_overlay", builder.getBoolAttr(true));
    }

    // Emit standalone `@ctrl_pkt_overlay` device.
    if (clEmitStandaloneOverlay) {
      if (failed(createOverlayDevice(module, builder, participating))) {
        return signalPassFailure();
      }
    }
  }

  // Collect the union of tiles across `devices`, recording one prototype
  // TileOp per tile so its attributes can be copied when the tile is cloned.
  static void
  collectTileUnion(ArrayRef<DeviceOp> devices,
                   llvm::SmallSetVector<AIE::TileID, 8> &unionTiles,
                   llvm::DenseMap<AIE::TileID, AIE::TileOp> &prototypeTile) {
    for (auto dev : devices)
      for (auto tOp : dev.getOps<AIE::TileOp>()) {
        AIE::TileID id{tOp.colIndex(), tOp.rowIndex()};
        unionTiles.insert(id);
        if (!prototypeTile.contains(id))
          prototypeTile[id] = tOp;
      }
  }

  // Clone every tile in `unionTiles` not already present in `device`, copying
  // the prototype's attributes so downstream passes that compare attribute
  // dictionaries (e.g. AIEMaterializeRuntimeSequences) match.
  static void cloneMissingTiles(
      DeviceOp device, const llvm::SmallSetVector<AIE::TileID, 8> &unionTiles,
      const llvm::DenseMap<AIE::TileID, AIE::TileOp> &prototypeTile) {
    llvm::SmallSet<AIE::TileID, 8> existing;
    for (auto tOp : device.getOps<AIE::TileOp>())
      existing.insert({tOp.colIndex(), tOp.rowIndex()});
    OpBuilder b = OpBuilder::atBlockBegin(device.getBody());
    for (auto id : unionTiles) {
      if (existing.contains(id))
        continue;
      b.clone(*prototypeTile.lookup(id).getOperation());
    }
  }

  // Give every device in `devices` the union of their tiles, so an overlay
  // routed onto any of them has the same shape (routes, shim_dma_allocations).
  static void shareTilesAcrossDevices(ArrayRef<DeviceOp> devices) {
    llvm::SmallSetVector<AIE::TileID, 8> unionTiles;
    llvm::DenseMap<AIE::TileID, AIE::TileOp> prototypeTile;
    collectTileUnion(devices, unionTiles, prototypeTile);
    for (auto dev : devices)
      cloneMissingTiles(dev, unionTiles, prototypeTile);
  }

  // Emit a standalone `@ctrl_pkt_overlay` device holding only the overlay and
  // the union of tiles it references. Downstream consumers compile it on its
  // own to ship a reconfigure-only PDI.
  LogicalResult createOverlayDevice(ModuleOp module, OpBuilder &builder,
                                    ArrayRef<DeviceOp> participating) {
    if (participating.empty())
      return success();

    // All participating devices must share the same target.
    DeviceOp firstDev = participating.front();
    auto refDevice = firstDev.getDevice();
    for (auto dev : llvm::drop_begin(participating)) {
      if (dev.getDevice() != refDevice) {
        return dev->emitOpError(
            "cannot generate a single standalone ctrl_pkt_overlay device: "
            "participating devices have mismatched target architectures.");
      }
    }

    if (module.lookupSymbol("ctrl_pkt_overlay")) {
      return module.emitOpError(
          "a symbol named `ctrl_pkt_overlay` already exists in the module; "
          "cannot create a standalone ctrl_pkt_overlay device.");
    }

    builder.setInsertionPointToEnd(module.getBody());
    Location loc = firstDev.getLoc();
    auto overlayDevice = AIE::DeviceOp::create(
        builder, loc, refDevice, builder.getStringAttr("ctrl_pkt_overlay"));
    overlayDevice.getRegion().emplaceBlock();
    builder.setInsertionPointToEnd(&overlayDevice.getRegion().front());
    AIE::EndOp::create(builder, loc);

    // Populate the overlay device with the union of tiles referenced across
    // participating devices, then route the overlay onto it.
    llvm::SmallSetVector<AIE::TileID, 8> unionTiles;
    llvm::DenseMap<AIE::TileID, AIE::TileOp> prototypeTile;
    collectTileUnion(participating, unionTiles, prototypeTile);
    cloneMissingTiles(overlayDevice, unionTiles, prototypeTile);

    if (failed(applyOverlayToDevice(overlayDevice)))
      return failure();

    overlayDevice->setAttr("has_ctrl_pkt_overlay", builder.getBoolAttr(true));
    return success();
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
    for (auto tile : device.getOps<AIE::TileOp>())
      tiles[{tile.colIndex(), tile.rowIndex()}] = tile;
    if (tiles.empty())
      return success();

    int minOccupiedCol = tiles.front().first.col;
    int maxOccupiedCol = minOccupiedCol;
    int maxOccupiedRow = 0;
    llvm::SmallSet<int, 4> declaredCols;
    for (auto &[tId, tOp] : tiles) {
      minOccupiedCol = std::min(minOccupiedCol, tId.col);
      maxOccupiedCol = std::max(maxOccupiedCol, tId.col);
      maxOccupiedRow = std::max(maxOccupiedRow, tId.row);
      declaredCols.insert(tId.col);
    }

    // Both widenings below are scoped to the control-packet configuration path
    // (`route-shim-to-tile-ctrl`); do not add tile declarations if the control
    // overlay was not requested, as to not congest routing needlessly.
    SmallVector<int> colsToCover;
    if (clRouteShimDmaToTileCTRL) {
      // Cover the full column range between the leftmost and rightmost occupied
      // column, not just the occupied columns. This is required so that a shim
      // DMA allocation is later emitted for the intermediate columns that a
      // flow will route through.
      for (int col = minOccupiedCol; col <= maxOccupiedCol; col++)
        colsToCover.push_back(col);
    } else {
      for (auto &[tId, tOp] : tiles)
        if (!llvm::is_contained(colsToCover, tId.col))
          colsToCover.push_back(tId.col);
    }

    auto tileIDMap = getTileToControllerIdMap(true, targetModel);
    for (int col : colsToCover) {
      builder.setInsertionPointToStart(device.getBody());
      AIE::TileOp shimTile = TileOp::getOrCreate(builder, device, col, 0);
      if (clRouteShimDmaToTileCTRL)
        tiles[{col, 0}] = shimTile;

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
        // A column the design declared no tile in is only in range because
        // flows route through it, and such a flow can traverse it at any row up
        // to the highest row in use. getRowToShimChanMap splits the rows into
        // one contiguous range per shim channel, so covering only the shim row
        // here would allocate just one of the channels its packets get
        // addressed to. Test against the columns the design itself declared,
        // not against `tiles`, which now also holds the shim materialized just
        // above.
        if (!declaredCols.contains(col))
          maxRow = maxOccupiedRow;
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

  // A one-source, one-destination packet flow reduced to a comparable key:
  // the packet ID plus both endpoints as (col, row, bundle, channel). This is
  // the shape every control flow below has, and keying on coordinates rather
  // than on the defining op is what makes it match DeviceOp::verify's notion
  // of a duplicate.
  using CtrlFlowKey = std::tuple<int, int, int, int, int, int, int, int, int>;

  // Nullopt when either endpoint's coordinates are not known yet -- an
  // aie.logical_tile that --aie-place-tiles has not placed. Such a flow cannot
  // be told apart from the one about to be created, and the verifier skips it
  // for the same reason. A pinned aie.logical_tile does have coordinates, so
  // this goes through TileLike rather than TileOp to catch it.
  static std::optional<CtrlFlowKey>
  tryGetCtrlFlowKey(int id, Value srcTileValue, WireBundle srcBundle,
                    int srcChan, Value destTileValue, WireBundle destBundle,
                    int destChan) {
    auto srcTile = dyn_cast_or_null<TileLike>(srcTileValue.getDefiningOp());
    auto destTile = dyn_cast_or_null<TileLike>(destTileValue.getDefiningOp());
    if (!srcTile || !destTile)
      return std::nullopt;
    std::optional<int> srcCol = srcTile.tryGetCol();
    std::optional<int> srcRow = srcTile.tryGetRow();
    std::optional<int> destCol = destTile.tryGetCol();
    std::optional<int> destRow = destTile.tryGetRow();
    if (!srcCol || !srcRow || !destCol || !destRow)
      return std::nullopt;
    return CtrlFlowKey{
        id,      *srcCol,  *srcRow,  static_cast<int>(srcBundle),
        srcChan, *destCol, *destRow, static_cast<int>(destBundle),
        destChan};
  }

  // The packet flows `device` already declares in the shape this pass emits,
  // keyed so one can be recognised before it is created a second time. A flow
  // with more than one source or destination is a different shape and is left
  // out.
  static std::map<CtrlFlowKey, AIE::PacketFlowOp>
  collectExistingCtrlFlows(DeviceOp device) {
    std::map<CtrlFlowKey, AIE::PacketFlowOp> flows;
    for (auto flow : device.getOps<AIE::PacketFlowOp>()) {
      auto sources = flow.getOps<AIE::PacketSourceOp>();
      auto dests = flow.getOps<AIE::PacketDestOp>();
      if (!llvm::hasSingleElement(sources) || !llvm::hasSingleElement(dests))
        continue;
      AIE::PacketSourceOp src = *sources.begin();
      AIE::PacketDestOp dest = *dests.begin();
      std::optional<CtrlFlowKey> key = tryGetCtrlFlowKey(
          flow.IDInt(), src.getTile(), src.getBundle(), src.channelIndex(),
          dest.getTile(), dest.getBundle(), dest.channelIndex());
      if (key)
        flows.try_emplace(*key, flow);
    }
    return flows;
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

    for (auto *user : shimTile.getResult().getUsers()) {
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
      WireBundle shimWireBundle, const SmallVector<AIE::TileOp> &ctrlTiles,
      WireBundle ctrlWireBundle, int coreOrMemChanId,
      DenseMap<TileID, int> tileIDMap, bool isShimMM2S) {
    int ctrlPktFlowID = 0;
    auto rowToShimChanMap =
        getRowToShimChanMap(device.getTargetModel(), shimWireBundle);
    // Get all available shim channels, to verify that the one being used is
    // available
    auto availableShimChans =
        getAvailableShimChans(device, shimTile, shimWireBundle, isShimMM2S);
    // The overlay a device already carries -- from a user-written control flow,
    // or from an earlier run of this pass, which aiecc does whenever the input
    // was already overlaid -- must not be laid down a second time. A flow
    // declared twice is rejected by DeviceOp::verify.
    std::map<CtrlFlowKey, AIE::PacketFlowOp> existingCtrlFlows =
        collectExistingCtrlFlows(device);

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
      int shimChan = rowToShimChanMap[tOp.rowIndex()];
      std::optional<CtrlFlowKey> key =
          isShimMM2S ? tryGetCtrlFlowKey(ctrlPktFlowID, shimTile,
                                         shimWireBundle, shimChan, tOp,
                                         ctrlWireBundle, coreOrMemChanId)
                     : tryGetCtrlFlowKey(ctrlPktFlowID, tOp, ctrlWireBundle,
                                         coreOrMemChanId, shimTile,
                                         shimWireBundle, shimChan);
      // Only the flow itself is affected by this; the shim DMA allocation
      // below is emitted either way, guarded by its own name lookup.
      auto it = key ? existingCtrlFlows.find(*key) : existingCtrlFlows.end();
      if (it != existingCtrlFlows.end()) {
        // The device already declares this control flow -- written by hand, or
        // laid down by an earlier run of this pass over the same input, which
        // aiecc does whenever its input was already overlaid. Declaring it a
        // second time is the duplicate DeviceOp::verify rejects, so adopt the
        // overlay's attributes onto the flow that is already there instead.
        // They are what tells --aie-create-pathfinder-flows this is a control
        // flow; that pass takes the last writer per destination port, so
        // before this pass deduplicated, the copy it appended is what set
        // them. Dropping them here would silently reroute the switchbox.
        it->second.setKeepPktHeader(keep_pkt_header.getValue());
        it->second.setPriorityRoute(ctrl_pkt_flow.getValue());
      } else {
        AIE::PacketFlowOp created =
            isShimMM2S
                ? createPacketFlowOp(builder, tOp.getLoc(), ctrlPktFlowID,
                                     shimTile, shimWireBundle, shimChan, tOp,
                                     ctrlWireBundle, coreOrMemChanId,
                                     keep_pkt_header, ctrl_pkt_flow)
                : createPacketFlowOp(builder, tOp.getLoc(), ctrlPktFlowID, tOp,
                                     ctrlWireBundle, coreOrMemChanId, shimTile,
                                     shimWireBundle, shimChan, keep_pkt_header,
                                     ctrl_pkt_flow);
        // Both endpoints are placed tiles here, so the key is always present;
        // guarded rather than asserted so an unplaced one just means no
        // deduplication, never a dropped flow.
        if (key)
          existingCtrlFlows.try_emplace(*key, created);
      }

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

      // No elem_type: a control-overlay channel carries no objectFIFO.
      AIE::ShimDMAAllocationOp::create(
          builder, tOp.getLoc(), StringRef(dma_name), shimTile.getResult(), dir,
          rowToShimChanMap[tOp.rowIndex()], false, nullptr, nullptr);
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
