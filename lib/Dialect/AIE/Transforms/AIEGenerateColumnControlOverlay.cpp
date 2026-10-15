//===- AIEGenerateColumnControlOverlay.cpp ----------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEGenerateColumnControlOverlay.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"

#include <map>
#include <optional>
#include <set>
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

std::string getCtrlPktAllocName(int col, bool isShimMM2S, int chan) {
  std::string name = "ctrlpkt";
  name += "_col" + std::to_string(col);
  name += isShimMM2S ? "_mm2s" : "_s2mm";
  name += "_chan" + std::to_string(chan);
  return name;
}

ShimDMAAllocationOp getCtrlPktShimAlloc(DeviceOp device, int col) {
  AIEDialect::IsCtrlPktOverlayAttrHelper overlay(device.getContext());
  for (auto alloc : device.getOps<AIE::ShimDMAAllocationOp>())
    if (overlay.isAttrPresent(alloc) &&
        alloc.getChannelDir() == AIE::DMAChannelDir::MM2S &&
        alloc.getTileOp().colIndex() == col)
      return alloc;
  return {};
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
          dev.setHasCtrlPktOverlay(false);
        }
        continue;
      }
      // A device that already carries a control overlay must not receive a
      // second one.
      if (deviceHasControlOverlay(dev)) {
        if (clEmitStandaloneOverlay) {
          dev.setHasCtrlPktOverlay(true);
        }
        continue;
      }
      participating.push_back(dev);
    }

    // The overlay's shape must not depend on which device it is routed in, so
    // the channels are read from every device that carries it, before any of
    // them changes.
    SmallVector<DeviceOp> overlaid;
    for (auto dev : sourceDevices)
      if (!deviceOptedOut(dev))
        overlaid.push_back(dev);
    collectShimMM2SUses(overlaid);

    // A standalone `@ctrl_pkt_overlay` device references a single overlay
    // shape, so every participating device must expose the same set of tiles
    // for that shape to be identical across them.
    // The same holds for the tiles whose task-complete tokens need a route.
    llvm::SmallSet<AIE::TileID, 4> sharedTokenTiles;
    if (clEmitStandaloneOverlay) {
      shareTilesAcrossDevices(participating);
      sharedTokenTiles = collectTokenIssuingTiles(participating);
    }

    // Apply the overlay in-place to participating devices.
    for (auto dev : participating) {
      llvm::SmallSet<AIE::TileID, 4> tokenTiles =
          clEmitStandaloneOverlay ? sharedTokenTiles
                                  : collectTokenIssuingTiles(dev);
      if (failed(applyOverlayToDevice(dev, tokenTiles)))
        return signalPassFailure();
      if (clEmitStandaloneOverlay)
        dev.setHasCtrlPktOverlay(true);
    }

    // Emit standalone `@ctrl_pkt_overlay` device.
    if (clEmitStandaloneOverlay) {
      if (failed(createOverlayDevice(module, builder, participating,
                                     sharedTokenTiles))) {
        return signalPassFailure();
      }
    }
  }

  // Per column, the shim MM2S channels the designs move data through, and the
  // channels control requests already declared in them use.
  llvm::DenseMap<int, llvm::SmallSet<int, 2>> dataChans, declaredCtrlChans;
  // getCtrlChan's cache.
  llvm::DenseMap<int, int> ctrlChans;

  void collectShimMM2SUses(ArrayRef<DeviceOp> devices) {
    dataChans.clear();
    declaredCtrlChans.clear();
    ctrlChans.clear();
    AIEDialect::IsCtrlPktOverlayAttrHelper overlay(&getContext());
    auto shimCol = [](Value tile) -> std::optional<int> {
      auto tOp = dyn_cast_or_null<TileOp>(tile.getDefiningOp());
      if (!tOp || !tOp.isShimNOCorPLTile())
        return std::nullopt;
      return tOp.colIndex();
    };
    for (DeviceOp device : devices) {
      for (auto alloc : device.getOps<AIE::ShimDMAAllocationOp>()) {
        if (alloc.getChannelDir() != AIE::DMAChannelDir::MM2S)
          continue;
        int col = alloc.getTileOp().colIndex();
        int chan = static_cast<int>(alloc.getChannelIndex());
        // A design may declare the allocation by its conventional name.
        if (alloc.getSymName() == getCtrlPktAllocName(col, true, chan))
          overlay.setAttr(alloc, UnitAttr::get(alloc.getContext()));
        if (overlay.isAttrPresent(alloc))
          declaredCtrlChans[col].insert(chan);
        else
          dataChans[col].insert(chan);
      }
      for (auto flow : device.getOps<AIE::FlowOp>())
        if (flow.getSourceBundle() == WireBundle::DMA)
          if (auto col = shimCol(flow.getSource()))
            dataChans[*col].insert(flow.getSourceChannel());
      for (auto flow : device.getOps<AIE::PacketFlowOp>()) {
        bool control = llvm::all_of(
            flow.getOps<AIE::PacketDestOp>(), [](AIE::PacketDestOp dest) {
              return dest.getBundle() == WireBundle::TileControl;
            });
        for (auto src : flow.getOps<AIE::PacketSourceOp>())
          if (src.getBundle() == WireBundle::DMA)
            if (auto col = shimCol(src.getTile()))
              (control ? declaredCtrlChans : dataChans)[*col].insert(
                  src.channelIndex());
      }
      for (auto mux : device.getOps<AIE::ShimMuxOp>())
        for (auto connect :
             mux.getConnections().front().getOps<AIE::ConnectOp>())
          if (connect.getSourceBundle() == WireBundle::DMA)
            dataChans[mux.getTileOp().colIndex()].insert(connect.sourceIndex());
    }
  }

  // -1 when every channel of the column moves data.
  int getCtrlChan(const AIETargetModel &targetModel, int col) {
    auto [it, inserted] = ctrlChans.try_emplace(col, -1);
    if (!inserted)
      return it->second;
    auto declared = declaredCtrlChans.find(col);
    if (declared != declaredCtrlChans.end() && !declared->second.empty()) {
      it->second = *llvm::min_element(declared->second);
      return it->second;
    }
    int numChans = targetModel.getNumSourceShimMuxConnections(
        col, 0, AIE::WireBundle::DMA);
    for (int chan = 0; chan < numChans; chan++)
      if (!dataChans.lookup(col).contains(chan)) {
        it->second = chan;
        break;
      }
    return it->second;
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
  LogicalResult
  createOverlayDevice(ModuleOp module, OpBuilder &builder,
                      ArrayRef<DeviceOp> participating,
                      const llvm::SmallSet<AIE::TileID, 4> &tokenTiles) {
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

    if (failed(applyOverlayToDevice(overlayDevice, tokenTiles)))
      return failure();

    overlayDevice.setHasCtrlPktOverlay(true);
    return success();
  }

  // Apply the column-control overlay to `device` in place, routing the
  // TileControl port of every tile in `tokenTiles` back to its shim. Returns
  // failure on a routing conflict.
  LogicalResult
  applyOverlayToDevice(DeviceOp device,
                       const llvm::SmallSet<AIE::TileID, 4> &tokenTiles) {
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
    for (auto &[tId, tOp] : tiles) {
      minOccupiedCol = std::min(minOccupiedCol, tId.col);
      maxOccupiedCol = std::max(maxOccupiedCol, tId.col);
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
      // Plus the columns aiecc asks for (OverlayCoverage).
      for (int col : clCoverColumns)
        if (col >= 0 && col < targetModel.columns() &&
            !llvm::is_contained(colsToCover, col))
          colsToCover.push_back(col);
      llvm::sort(colsToCover);
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
          if (clRouteShimCTRLToTCT == "shim-only" && !tOp.isShimNOCorPLTile() &&
              !tokenTiles.contains(tId))
            continue;
          // A routed response only covers the tile it comes from.
          if (hasRoutedResponse(device, tOp, tileIDMap))
            continue;
          tilesOnCol.push_back(tOp);
        }

        if (!tilesOnCol.empty() &&
            failed(generatePacketFlowsForControl(
                builder, device, shimTile, AIE::WireBundle::South, 0,
                tilesOnCol, AIE::WireBundle::TileControl, 0, tileIDMap, false)))
          return failure();
      }
      if (clRouteShimDmaToTileCTRL) {
        int shimChan = getCtrlChan(targetModel, col);
        if (shimChan < 0)
          return device->emitOpError(
                     "failed to generate column control overlay: every shim "
                     "mm2s dma channel of column ")
                 << col << " moves data, so none is left for its control "
                 << "packets";
        // Cover every row, since the router may cross a covered column at any
        // row and one channel carries the whole column.
        SmallVector<AIE::TileOp> tilesOnCol;
        for (int row = 0; row < targetModel.rows(); row++) {
          auto tOp = TileOp::getOrCreate(builder, device, col, row);
          tilesOnCol.push_back(tOp);
        }

        if (failed(generatePacketFlowsForControl(
                builder, device, shimTile, AIE::WireBundle::DMA, shimChan,
                tilesOnCol, AIE::WireBundle::TileControl, 0, tileIDMap, true)))
          return failure();
      }
    }
    return success();
  }

  // Tiles a runtime sequence asks a task-complete token of. The token leaves
  // the tile through its TileControl port, so without a route from there to
  // the shim an await on it never returns.
  static llvm::SmallSet<AIE::TileID, 4>
  collectTokenIssuingTiles(ArrayRef<DeviceOp> devices) {
    llvm::SmallSet<AIE::TileID, 4> tiles;
    auto addTile = [&](Value tileValue) {
      if (!tileValue)
        return;
      if (auto tile = dyn_cast_or_null<TileOp>(tileValue.getDefiningOp()))
        tiles.insert({tile.colIndex(), tile.rowIndex()});
    };
    for (DeviceOp device : devices)
      device.walk([&](Operation *op) {
        if (auto task = dyn_cast<AIEX::DMAConfigureTaskOp>(op)) {
          if (task.getIssueToken())
            addTile(task.getTile());
        } else if (auto task = dyn_cast<AIEX::DMAConfigureTaskForOp>(op)) {
          if (!task.getIssueToken())
            return;
          if (auto endpoint = dyn_cast_or_null<RouteEndpointOp>(
                  SymbolTable::lookupSymbolIn(device, task.getAlloc())))
            addTile(endpoint.getTile());
        } else if (auto push = dyn_cast<AIEX::NpuPushQueueOp>(op)) {
          if (push.getIssueToken())
            tiles.insert({static_cast<int>(push.getColumn()),
                          static_cast<int>(push.getRow())});
        }
      });
    return tiles;
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

  // The packet ID of control packets to and from `tile`.
  static int getControllerID(TileOp tile, DenseMap<TileID, int> &tileIDMap) {
    if (auto id = tile->getAttrOfType<AIE::PacketInfoAttr>("controller_id"))
      return id.getPktId();
    return tileIDMap[{tile.colIndex(), tile.rowIndex()}];
  }

  // True when packets with ID `pktID` entering the switchbox of `from` on any
  // of `startPorts` are routed to `toPort` of `to`'s switchbox, through a
  // connect or master marked is_ctrl_pkt_overlay. The ID is traced through the
  // routed switchboxes rather than inferred from the final rule.
  static bool hasRoutedPacket(DeviceOp device, int pktID, TileID from,
                              ArrayRef<Port> startPorts, TileID to,
                              Port toPort) {
    AIEDialect::IsCtrlPktOverlayAttrHelper overlay(device.getContext());
    DenseMap<TileID, AIE::SwitchboxOp> switchboxes;
    for (auto switchbox : device.getOps<AIE::SwitchboxOp>()) {
      auto tile = dyn_cast<TileLike>(switchbox.getTile().getDefiningOp());
      if (!tile)
        continue;
      std::optional<int> c = tile.tryGetCol(), r = tile.tryGetRow();
      if (c && r)
        switchboxes[{*c, *r}] = switchbox;
    }

    SmallVector<std::pair<TileID, Port>> worklist;
    for (Port port : startPorts)
      worklist.push_back({from, port});
    std::set<std::tuple<int, int, int, int>> visited;
    while (!worklist.empty()) {
      auto [tileID, port] = worklist.pop_back_val();
      if (!visited
               .insert({tileID.col, tileID.row, static_cast<int>(port.bundle),
                        port.channel})
               .second)
        continue;
      auto switchbox = switchboxes.lookup(tileID);
      if (!switchbox)
        continue;
      Block &connections = switchbox.getConnections().front();

      SmallVector<std::pair<Port, Operation *>> outputs;
      for (auto connect : connections.getOps<AIE::ConnectOp>())
        if (connect.sourcePort() == port)
          outputs.push_back({connect.destPort(), connect});
      // The first rule matching the packet ID picks the master. Pathfinder
      // marks a slave's packet_rules only from the first group it routes, so
      // the marker on the rules is not reliable.
      for (auto rules : connections.getOps<AIE::PacketRulesOp>()) {
        if (rules.sourcePort() != port)
          continue;
        for (auto rule : rules.getRules().front().getOps<AIE::PacketRuleOp>()) {
          if ((pktID & rule.maskInt()) != rule.valueInt())
            continue;
          for (auto master : connections.getOps<AIE::MasterSetOp>())
            if (llvm::is_contained(master.getAmsels(), rule.getAmsel()))
              outputs.push_back({master.destPort(), master});
          break;
        }
      }

      for (auto [dest, op] : outputs) {
        if (tileID == to && dest == toPort) {
          if (overlay.isAttrPresent(op))
            return true;
          continue;
        }
        auto next =
            getConnectingWireEndpoint(device.getTargetModel(), tileID, dest);
        if (!next)
          continue;
        worklist.push_back(*next);
      }
    }
    return false;
  }

  // A routed response must actually connect TileControl:0 of `src` to South:0
  // on its column's shim. A priority request ending at TileControl is not a
  // response, even though pathfinder marks it is_ctrl_pkt_overlay.
  static bool hasRoutedResponse(DeviceOp device, TileOp src,
                                DenseMap<TileID, int> &tileIDMap) {
    int col = src.colIndex();
    return hasRoutedPacket(device, getControllerID(src, tileIDMap),
                           {col, src.rowIndex()},
                           {Port{WireBundle::TileControl, 0}}, {col, 0},
                           Port{WireBundle::South, 0});
  }

  // A routed request must connect DMA:`shimChan` of `shimTile` to
  // TileControl:0 of `dest`. Pathfinder moves the shim DMA behind the shim
  // mux, so the request enters the shim's switchbox on the South port the mux
  // connects that DMA channel to.
  static bool hasRoutedRequest(DeviceOp device, TileOp shimTile, int shimChan,
                               TileOp dest, DenseMap<TileID, int> &tileIDMap) {
    SmallVector<Port> startPorts = {Port{WireBundle::DMA, shimChan}};
    for (auto shimMux : device.getOps<AIE::ShimMuxOp>()) {
      if (shimMux.getTile() != shimTile.getResult())
        continue;
      for (auto connect :
           shimMux.getConnections().front().getOps<AIE::ConnectOp>())
        if (connect.sourcePort() == Port{WireBundle::DMA, shimChan} &&
            connect.destPort().bundle == WireBundle::North)
          startPorts.push_back(
              Port{WireBundle::South, connect.destPort().channel});
    }
    return hasRoutedPacket(device, getControllerID(dest, tileIDMap),
                           {shimTile.colIndex(), shimTile.rowIndex()},
                           startPorts, {dest.colIndex(), dest.rowIndex()},
                           Port{WireBundle::TileControl, 0});
  }

  // True when `tile` is used only by its routing, and its switchbox only
  // passes streams between neighbouring switchboxes: none starts or ends at
  // the tile itself.
  static bool isRoutedThroughOnly(TileOp tile) {
    auto isNeighbour = [](Port port) {
      return port.bundle == WireBundle::North ||
             port.bundle == WireBundle::South ||
             port.bundle == WireBundle::East || port.bundle == WireBundle::West;
    };
    bool hasSwitchbox = false;
    for (Operation *user : tile->getUsers()) {
      if (isa<AIE::WireOp>(user))
        continue;
      auto switchbox = dyn_cast<AIE::SwitchboxOp>(user);
      if (!switchbox)
        return false;
      hasSwitchbox = true;
      for (Operation &op : switchbox.getConnections().front()) {
        if (auto connect = dyn_cast<AIE::ConnectOp>(op)) {
          if (!isNeighbour(connect.sourcePort()) ||
              !isNeighbour(connect.destPort()))
            return false;
        } else if (auto rules = dyn_cast<AIE::PacketRulesOp>(op)) {
          if (!isNeighbour(rules.sourcePort()))
            return false;
        } else if (auto master = dyn_cast<AIE::MasterSetOp>(op)) {
          if (!isNeighbour(master.destPort()))
            return false;
        }
      }
    }
    return hasSwitchbox;
  }

  // Every occupied column needs its own shim response route. An overlay
  // routed on only one column must not suppress the remaining columns. With
  // route-shim-to-tct=all-tiles, every tile also needs its own response route,
  // except a tile that is only routed through. Pathfinder declares such a tile
  // when a flow crosses it, and a first run of this pass never covered it.
  bool hasRoutedResponses(DeviceOp device, DenseMap<TileID, int> &tileIDMap) {
    llvm::SmallSet<int, 4> cols;
    llvm::DenseMap<int, TileOp> shims;
    for (auto tile : device.getOps<AIE::TileOp>()) {
      cols.insert(tile.colIndex());
      if (tile.rowIndex() == 0)
        shims[tile.colIndex()] = tile;
    }
    if (cols.empty())
      return false;
    if (!llvm::all_of(cols, [&](int col) {
          auto it = shims.find(col);
          return it != shims.end() &&
                 hasRoutedResponse(device, it->second, tileIDMap);
        }))
      return false;
    if (clRouteShimCTRLToTCT != "all-tiles")
      return true;
    return llvm::all_of(device.getOps<AIE::TileOp>(), [&](TileOp tile) {
      return isRoutedThroughOnly(tile) ||
             hasRoutedResponse(device, tile, tileIDMap);
    });
  }

  // With route-shim-to-tile-ctrl, every tile applyOverlayToDevice sends
  // requests to needs a routed request from its column's shim DMA channel,
  // and that channel needs its shim DMA allocation. Tiles only routed through
  // do not widen the columns covered, as a first run never covered them.
  bool hasRoutedRequests(DeviceOp device, DenseMap<TileID, int> &tileIDMap) {
    llvm::DenseMap<TileID, TileOp> tiles;
    std::optional<int> minCol, maxCol;
    for (auto tile : device.getOps<AIE::TileOp>()) {
      int col = tile.colIndex(), row = tile.rowIndex();
      tiles[{col, row}] = tile;
      if (isRoutedThroughOnly(tile))
        continue;
      minCol = std::min(minCol.value_or(col), col);
      maxCol = std::max(maxCol.value_or(col), col);
    }
    if (!minCol || !maxCol)
      return false;
    llvm::SmallSetVector<int, 8> cols;
    for (int col = *minCol; col <= *maxCol; col++)
      cols.insert(col);
    for (int col : clCoverColumns)
      if (col >= 0 && col < device.getTargetModel().columns())
        cols.insert(col);
    for (int col : cols) {
      TileOp shimTile = tiles.lookup({col, 0});
      ShimDMAAllocationOp alloc = getCtrlPktShimAlloc(device, col);
      if (!shimTile || !alloc)
        return false;
      int chan = static_cast<int>(alloc.getChannelIndex());
      for (int row = 0; row < device.getTargetModel().rows(); row++) {
        TileOp tile = tiles.lookup({col, row});
        if (!tile || !hasRoutedRequest(device, shimTile, chan, tile, tileIDMap))
          return false;
      }
    }
    return true;
  }

  // True when `device` already carries every route this pass would add.
  // Responses alone do not complete an overlay that also routes requests.
  // Responses are only skipped when route-shim-to-tct=disable leaves requests
  // as the sole routes requested; with neither requested, they still decide.
  bool deviceHasControlOverlay(DeviceOp device) {
    auto tileIDMap = getTileToControllerIdMap(true, device.getTargetModel());
    if ((clRouteShimCTRLToTCT != "disable" || !clRouteShimDmaToTileCTRL) &&
        !hasRoutedResponses(device, tileIDMap))
      return false;
    return !clRouteShimDmaToTileCTRL || hasRoutedRequests(device, tileIDMap);
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

  // Create packet flows per col which moves control packets to and from shim
  // dma
  LogicalResult generatePacketFlowsForControl(
      OpBuilder builder, DeviceOp device, TileOp shimTile,
      WireBundle shimWireBundle, int shimChan,
      const SmallVector<AIE::TileOp> &ctrlTiles, WireBundle ctrlWireBundle,
      int coreOrMemChanId, DenseMap<TileID, int> tileIDMap, bool isShimMM2S) {
    int ctrlPktFlowID = 0;
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

      auto keep_pkt_header = builder.getBoolAttr(true);
      auto ctrl_pkt_flow = builder.getBoolAttr(true);
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
      } else if (!isShimMM2S || !hasRoutedRequest(device, shimTile, shimChan,
                                                  tOp, tileIDMap)) {
        // A request the switchboxes already route is not declared again; its
        // shim DMA allocation below is still emitted.
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
      std::string dma_name =
          getCtrlPktAllocName(shimTile.colIndex(), isShimMM2S, shimChan);

      // check to see if ShimDMAAllocationOp already exists
      Operation *alloc = device.lookupSymbol(dma_name);
      if (!alloc)
        alloc = AIE::ShimDMAAllocationOp::create(
            builder, tOp.getLoc(), StringRef(dma_name), shimTile.getResult(),
            dir, shimChan, false, nullptr);
      AIEDialect::IsCtrlPktOverlayAttrHelper(device.getContext())
          .setAttr(alloc, UnitAttr::get(device.getContext()));
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
