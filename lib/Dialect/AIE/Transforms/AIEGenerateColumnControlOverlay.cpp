//===- AIEGenerateColumnControlOverlay.cpp ----------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

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

static std::string getCtrlPktAllocName(int col, int chan) {
  return "ctrlpkt_col" + std::to_string(col) + "_mm2s_chan" +
         std::to_string(chan);
}

struct AIEAssignTileCtrlIDsPass
    : xilinx::AIE::impl::AIEAssignTileCtrlIDsBase<AIEAssignTileCtrlIDsPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();

    if (targetModel.getTargetArch() == AIEArch::AIE1)
      return; // Disable this pass for AIE1; AIE1 support NYI.

    auto tileIDMap =
        targetModel.getTileToControllerIdMap(clColumnWiseUniqueIDs);
    for (auto tile : device.getOps<TileOp>()) {
      if (!tile->hasAttr("controller_id"))
        tile->setAttr("controller_id",
                      PacketInfoAttr::get(tile.getContext(), 0,
                                          tileIDMap[tile.getTileID()]));
    }
  }
};

struct AIEGenerateColumnControlOverlayPass
    : xilinx::AIE::impl::AIEGenerateColumnControlOverlayBase<
          AIEGenerateColumnControlOverlayPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    SmallVector<DeviceOp> participating;
    for (auto device : module.getOps<DeviceOp>()) {
      if (device->hasAttr("aie.is_ctrl_overlay") || device.getOverlayAttr())
        continue;
      participating.push_back(device);
    }
    collectShimMM2SUses(participating);
    if (clEmitStandaloneOverlay) {
      if (failed(createOverlayDevice(module, builder, participating,
                                     collectTokenIssuingTiles(participating))))
        return signalPassFailure();
      return;
    }
    for (auto device : participating)
      if (failed(
              applyOverlayToDevice(device, collectTokenIssuingTiles(device))))
        return signalPassFailure();
  }

  // Per column, the shim MM2S channels the designs move data through, and the
  // channels control requests already declared in them use.
  llvm::DenseMap<int, llvm::SmallSet<int, 2>> dataChans, declaredCtrlChans;
  // getCtrlChan's cache.
  llvm::DenseMap<int, int> ctrlChans;
  llvm::DenseMap<std::pair<int, int>, unsigned> packetDemand;
  llvm::DenseMap<int, llvm::SmallSet<int, 2>> fixedCircuits;

  void collectShimMM2SUses(ArrayRef<DeviceOp> devices) {
    dataChans.clear();
    packetDemand.clear();
    fixedCircuits.clear();
    declaredCtrlChans.clear();
    ctrlChans.clear();
    AIEDialect::IsCtrlOverlayAttrHelper overlay(&getContext());
    auto shimCol = [](Value tile) -> std::optional<int> {
      auto tOp = dyn_cast_or_null<TileOp>(tile.getDefiningOp());
      if (!tOp || !tOp.isShimNOCorPLTile())
        return std::nullopt;
      return tOp.colIndex();
    };
    for (DeviceOp device : devices) {
      for (auto endpoint : device.getOps<RouteEndpointOp>()) {
        auto col = shimCol(endpoint.getTile());
        auto channel = endpoint.getChannelIndex();
        if (!col || endpoint.getBundle() != WireBundle::DMA ||
            endpoint.getRouteDirection() != DMAChannelDir::MM2S || !channel)
          continue;
        bool packet =
            llvm::any_of(device.getOps<RouteOp>(), [&](RouteOp route) {
              return route.namesAsSource(endpoint.getSymName()) &&
                     route.getPacket();
            });
        if (packet)
          ++packetDemand[{*col, *channel}];
        else
          fixedCircuits[*col].insert(*channel);
      }
      for (auto alloc : device.getOps<AIE::ShimDMAAllocationOp>()) {
        if (alloc.getChannelDir() != AIE::DMAChannelDir::MM2S)
          continue;
        int col = alloc.getTileOp().colIndex();
        int chan = static_cast<int>(alloc.getChannelIndex());
        // A design may declare the allocation by its conventional name.
        if (alloc.getSymName() == getCtrlPktAllocName(col, chan))
          overlay.setAttr(alloc, UnitAttr::get(alloc.getContext()));
        if (overlay.isAttrPresent(alloc))
          declaredCtrlChans[col].insert(chan);
        else {
          dataChans[col].insert(chan);
          fixedCircuits[col].insert(chan);
        }
      }
      for (auto flow : device.getOps<AIE::FlowOp>())
        if (flow.getSourceBundle() == WireBundle::DMA)
          if (auto col = shimCol(flow.getSource())) {
            dataChans[*col].insert(flow.getSourceChannel());
            fixedCircuits[*col].insert(flow.getSourceChannel());
          }
      for (auto flow : device.getOps<AIE::PacketFlowOp>()) {
        bool control = llvm::all_of(
            flow.getOps<AIE::PacketDestOp>(), [](AIE::PacketDestOp dest) {
              return dest.getBundle() == WireBundle::TileControl;
            });
        for (auto src : flow.getOps<AIE::PacketSourceOp>())
          if (src.getBundle() == WireBundle::DMA)
            if (auto col = shimCol(src.getTile())) {
              (control ? declaredCtrlChans : dataChans)[*col].insert(
                  src.channelIndex());
              if (!control)
                ++packetDemand[{*col, src.channelIndex()}];
            }
      }
      for (auto mux : device.getOps<AIE::ShimMuxOp>())
        for (auto connect :
             mux.getConnections().front().getOps<AIE::ConnectOp>())
          if (connect.getSourceBundle() == WireBundle::DMA) {
            int col = mux.getTileOp().colIndex();
            dataChans[col].insert(connect.sourceIndex());
            fixedCircuits[col].insert(connect.sourceIndex());
          }
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
    if (clPlanChannels) {
      if (!clAdaptiveChannels)
        return it->second = fixedCircuits.lookup(col).contains(0) ? -1 : 0;
      unsigned best = std::numeric_limits<unsigned>::max();
      for (int chan = 0; chan < numChans; ++chan) {
        if (fixedCircuits.lookup(col).contains(chan))
          continue;
        unsigned cost = packetDemand.lookup({col, chan});
        if (cost < best) {
          best = cost;
          it->second = chan;
        }
      }
      return it->second;
    }
    for (int chan = 0; chan < numChans; chan++)
      if (!dataChans.lookup(col).contains(chan)) {
        it->second = chan;
        break;
      }
    return it->second;
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
        builder, loc, refDevice, builder.getStringAttr("ctrl_pkt_overlay"),
        nullptr);
    overlayDevice.getRegion().emplaceBlock();
    builder.setInsertionPointToEnd(&overlayDevice.getRegion().front());
    AIE::EndOp::create(builder, loc);

    llvm::SmallSet<TileID, 8> tiles;
    builder.setInsertionPointToStart(overlayDevice.getBody());
    for (DeviceOp device : participating)
      for (TileOp tile : device.getOps<TileOp>())
        if (tiles.insert(tile.getTileID()).second)
          builder.clone(*tile);

    if (failed(applyOverlayToDevice(overlayDevice, tokenTiles)))
      return failure();

    overlayDevice->setAttr("aie.is_ctrl_overlay", builder.getUnitAttr());
    auto flows = collectExistingCtrlFlows(overlayDevice);
    for (DeviceOp device : participating) {
      device.setOverlayAttr(
          FlatSymbolRefAttr::get(overlayDevice.getSymNameAttr()));
      auto declared = collectExistingCtrlFlows(device);
      for (auto &[key, flow] : declared)
        if (flows.count(key))
          flow.erase();
    }
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

    llvm::MapVector<AIE::TileID, AIE::TileOp> tiles;
    for (auto tile : device.getOps<AIE::TileOp>())
      if (tile.isShimNOCorPLTile() || !isRoutedThroughOnly(tile))
        tiles[tile.getTileID()] = tile;
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

    auto tileIDMap = targetModel.getTileToControllerIdMap(true);
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

        generatePacketFlowsForControl(builder, device, shimTile,
                                      {WireBundle::South, 0}, tilesOnCol,
                                      tileIDMap, false);
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

        generatePacketFlowsForControl(builder, device, shimTile,
                                      {WireBundle::DMA, shimChan}, tilesOnCol,
                                      tileIDMap, true);
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
      return id.assignedId();
    return tileIDMap[{tile.colIndex(), tile.rowIndex()}];
  }

  // True when packets with ID `pktID` entering the switchbox of `from` on any
  // of `startPorts` are routed to `toPort` of `to`'s switchbox, through a
  // prioritized or overlay-owned connection. The ID is traced through the
  // routed switchboxes rather than inferred from the final rule.
  static bool hasRoutedPacket(DeviceOp device, int pktID, TileID from,
                              ArrayRef<Port> startPorts, TileID to,
                              Port toPort) {
    AIEDialect::IsCtrlOverlayAttrHelper overlay(device.getContext());
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
          if (overlay.isAttrPresent(op) || op->hasAttr("aie.priority_route"))
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
  // response, even though it is prioritized.
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

  void generatePacketFlowsForControl(OpBuilder builder, DeviceOp device,
                                     TileOp shimTile, Port shimPort,
                                     ArrayRef<TileOp> ctrlTiles,
                                     DenseMap<TileID, int> &tileIDMap,
                                     bool requests) {
    auto flows = collectExistingCtrlFlows(device);
    builder.setInsertionPoint(device.getBody()->getTerminator());
    for (TileOp tile : ctrlTiles) {
      int id = getControllerID(tile, tileIDMap);
      TileOp source = shimTile, dest = tile;
      Port sourcePort = shimPort, destPort{WireBundle::TileControl, 0};
      if (!requests) {
        std::swap(source, dest);
        std::swap(sourcePort, destPort);
      }
      auto key =
          tryGetCtrlFlowKey(id, source, sourcePort.bundle, sourcePort.channel,
                            dest, destPort.bundle, destPort.channel);
      auto it = key ? flows.find(*key) : flows.end();
      if (it != flows.end()) {
        it->second.setKeepPktHeader(true);
        it->second.setPriorityRoute(true);
      } else if (!requests ||
                 !hasRoutedRequest(device, shimTile, shimPort.channel, tile,
                                   tileIDMap)) {
        OpBuilder::InsertionGuard guard(builder);
        auto flow = PacketFlowOp::create(builder, tile.getLoc(), id,
                                         builder.getBoolAttr(true),
                                         builder.getBoolAttr(true));
        builder.createBlock(&flow.getPorts());
        PacketSourceOp::create(builder, tile.getLoc(), source,
                               sourcePort.bundle, sourcePort.channel);
        PacketDestOp::create(builder, tile.getLoc(), dest, destPort.bundle,
                             destPort.channel);
        EndOp::create(builder, tile.getLoc());
        if (key)
          flows.try_emplace(*key, flow);
      }

      if (!requests)
        continue;
      std::string name =
          getCtrlPktAllocName(shimTile.colIndex(), shimPort.channel);
      Operation *alloc = device.lookupSymbol(name);
      if (!alloc)
        alloc = ShimDMAAllocationOp::create(
            builder, tile.getLoc(), name, shimTile.getResult(),
            DMAChannelDir::MM2S, shimPort.channel, false, nullptr);
      AIEDialect::IsCtrlOverlayAttrHelper(device.getContext())
          .setAttr(alloc, builder.getUnitAttr());
      if (clPlanChannels)
        alloc->setAttr("aie.fixed", builder.getUnitAttr());
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEAssignTileCtrlIDsPass() {
  return std::make_unique<AIEAssignTileCtrlIDsPass>();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>>
AIE::createAIEGenerateColumnControlOverlayPass() {
  return std::make_unique<AIEGenerateColumnControlOverlayPass>();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>>
AIE::createAIEGenerateColumnControlOverlayPass(
    const AIEGenerateColumnControlOverlayOptions &options) {
  return std::make_unique<AIEGenerateColumnControlOverlayPass>(options);
}
