//===- AIEGenerateColumnControlOverlay.cpp ----------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"
#include "aie/Dialect/AIE/Transforms/AIEShimSharing.h"
#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/PassManager.h"

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

    if (!targetModel.hasProperty(AIETargetModel::SupportsControlPackets))
      return;

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
  DenseMap<std::pair<int, int>, unsigned> packetDemand;
  DenseMap<int, llvm::SmallSet<int, 2>> fixedCircuits;

  void runOnOperation() override {
    llvm::MapVector<AIEDevice, SmallVector<DeviceOp>> groups;
    for (DeviceOp device : getOperation().getOps<DeviceOp>())
      if (getsColumnControlOverlay(device, configuration))
        groups[device.getDevice()].push_back(device);
    for (auto &[target, users] : groups)
      if (failed(generate(users)))
        return signalPassFailure();
  }

  LogicalResult generate(ArrayRef<DeviceOp> users) {
    ModuleOp module = getOperation();
    AIEDialect::IsCtrlPktOverlayAttrHelper marker(&getContext());
    DeviceOp first = users.front();
    auto target = first.getDevice();
    auto tokenTiles = collectTokenIssuingTiles(users);
    std::set<int> columns;
    if (configuration)
      for (DeviceOp device : users)
        for (TileOp tile : device.getOps<TileOp>())
          columns.insert(tile.colIndex());
    for (TileID tile : tokenTiles)
      columns.insert(tile.col);
    if (configuration) {
      for (DeviceOp device : users)
        device.walk([&](AIEX::NpuControlPacketOp packet) {
          columns.insert(packet.getColumnFromAddr());
        });
      columns.insert(coverColumns.begin(), coverColumns.end());
      collectShimMM2SUses(users);
    }
    SymbolTable symbols(module);
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    DeviceOp overlay =
        DeviceOp::create(builder, module.getLoc(), target,
                         builder.getStringAttr("ctrl_pkt_overlay"), nullptr);
    symbols.insert(overlay);
    overlay.getRegion().emplaceBlock();
    builder.setInsertionPointToEnd(overlay.getBody());
    EndOp::create(builder, overlay.getLoc());
    marker.setAttr(overlay, builder.getUnitAttr());
    builder.setInsertionPointToStart(overlay.getBody());
    for (DeviceOp device : users)
      for (TileOp tile : device.getOps<TileOp>()) {
        TileOp dest = TileOp::getOrCreate(builder, overlay, tile.colIndex(),
                                          tile.rowIndex());
        if (auto id = tile->getAttr("controller_id")) {
          if (auto other = dest->getAttr("controller_id");
              other && other != id) {
            tile.emitOpError(
                "has a different controller ID in another configuration");
            return failure();
          }
          dest->setAttr("controller_id", id);
        }
      }
    const auto &model = overlay.getTargetModel();
    auto ids = model.getTileToControllerIdMap(true);
    auto flow = [&](TileOp source, Port src, TileOp dest, Port dst,
                    TileOp controller) {
      builder.setInsertionPoint(overlay.getBody()->getTerminator());
      int id = ids[controller.getTileID()];
      if (auto attr =
              controller->getAttrOfType<PacketInfoAttr>("controller_id"))
        id = attr.assignedId();
      auto packet = PacketFlowOp::create(builder, controller.getLoc(), id,
                                         builder.getBoolAttr(true),
                                         builder.getBoolAttr(true));
      OpBuilder::InsertionGuard guard(builder);
      builder.createBlock(&packet.getPorts());
      PacketSourceOp::create(builder, packet.getLoc(), source, src.bundle,
                             src.channel);
      PacketDestOp::create(builder, packet.getLoc(), dest, dst.bundle,
                           dst.channel);
      EndOp::create(builder, packet.getLoc());
    };
    for (int col : columns) {
      if (col < 0 || col >= model.columns()) {
        overlay.emitOpError("control coverage column is outside the target");
        return failure();
      }
      builder.setInsertionPointToStart(overlay.getBody());
      TileOp shim = TileOp::getOrCreate(builder, overlay, col, 0);
      flow(shim, {WireBundle::TileControl, 0}, shim, {WireBundle::South, 0},
           shim);
      for (TileID tile : tokenTiles)
        if (tile.col == col && tile.row != 0) {
          builder.setInsertionPointToStart(overlay.getBody());
          TileOp source = TileOp::getOrCreate(builder, overlay, col, tile.row);
          flow(source, {WireBundle::TileControl, 0}, shim,
               {WireBundle::South, 0}, source);
        }
      if (!configuration)
        continue;
      int channel = -1;
      unsigned best = std::numeric_limits<unsigned>::max();
      int count =
          adaptive
              ? model.getNumSourceShimMuxConnections(col, 0, WireBundle::DMA)
              : 1;
      for (int ch = 0; ch < count; ++ch)
        if (!fixedCircuits.lookup(col).contains(ch) &&
            packetDemand.lookup({col, ch}) < best) {
          channel = ch;
          best = packetDemand.lookup({col, ch});
        }
      if (channel < 0) {
        overlay.emitOpError("failed to generate control overlay: no available "
                            "shim MM2S channel in column ")
            << col;
        return failure();
      }
      builder.setInsertionPoint(overlay.getBody()->getTerminator());
      auto alloc = ShimDMAAllocationOp::create(
          builder, shim.getLoc(), getCtrlPktAllocName(col, channel), shim,
          DMAChannelDir::MM2S, channel, false, nullptr, nullptr);
      marker.setAttr(alloc, builder.getUnitAttr());
      for (int row = 0; row < model.rows(); ++row) {
        builder.setInsertionPointToStart(overlay.getBody());
        TileOp dest = TileOp::getOrCreate(builder, overlay, col, row);
        flow(shim, {WireBundle::DMA, channel}, dest,
             {WireBundle::TileControl, 0}, dest);
      }
    }
    for (DeviceOp device : users) {
      device.setOverlayAttr(FlatSymbolRefAttr::get(overlay.getSymNameAttr()));
      if (configuration)
        AIEDialect::ReserveControllerPacketIdsAttrHelper(&getContext())
            .setAttr(device, builder.getUnitAttr());
    }
    if (!configuration)
      return success();
    std::vector<std::pair<TileID, Port>> avoided;
    if (adaptive && failed(applicationPorts(module, avoided)))
      return failure();
    OpPassManager routing(DeviceOp::getOperationName());
    routing.addPass(createAIEAssignTileCtrlIDsPass());
    routing.addPass(createAIEPathfinderPass({}, avoided));
    if (failed(runPipeline(routing, overlay)))
      return failure();
    // Reprogramming another master set of an active arbiter can stall control
    // packets.
    for (auto box : overlay.getOps<SwitchboxOp>()) {
      auto amsels = llvm::to_vector(box.getConnections().getOps<AMSelOp>());
      std::set<std::pair<int, int>> slots;
      for (auto amsel : amsels)
        slots.emplace(amsel.arbiterIndex(), amsel.getMselValue());
      builder.setInsertionPoint(box.getConnections().front().getTerminator());
      for (auto amsel : amsels)
        for (unsigned msel = 0; msel < model.getNumMselsPerArbiter(); ++msel)
          if (slots.emplace(amsel.arbiterIndex(), msel).second)
            AMSelOp::create(builder, amsel.getLoc(), amsel.arbiterIndex(),
                            msel);
    }
    overlay.walk([&](Operation *op) {
      op->removeAttr("aie.priority_route");
      if (!isa<DeviceOp, EndOp, PacketRulesOp>(op))
        marker.setAttr(op, builder.getUnitAttr());
    });
    return success();
  }
  void collectShimMM2SUses(ArrayRef<DeviceOp> devices) {
    packetDemand.clear();
    fixedCircuits.clear();
    auto column = [](Operation *tile) {
      return cast<TileLike>(tile).tryGetCol();
    };
    for (DeviceOp device : devices) {
      ShimMM2SUses uses = shimMM2SUses(device, packetSwitched);
      for (auto &[tile, channels] : uses.circuits)
        if (std::optional<int> col = column(tile))
          fixedCircuits[*col].insert(channels.begin(), channels.end());
      for (auto &[at, ends] : uses.packetEnds)
        if (std::optional<int> col = column(at.first))
          packetDemand[{*col, at.second}] += ends.size();
      for (auto &[at, flows] : uses.packetFlows)
        if (std::optional<int> col = column(at.first))
          packetDemand[{*col, at.second}] += flows;
    }
  }

  static llvm::SmallSet<AIE::TileID, 4>
  collectTokenIssuingTiles(ArrayRef<DeviceOp> devices) {
    llvm::SmallSet<AIE::TileID, 4> tiles;
    for (DeviceOp device : devices) {
      for (TileOp tile : device.getOps<TileOp>())
        if (tile.isShimNOCorPLTile())
          tiles.insert(tile.getTileID());
      SymbolTable symbols(device);
      device.walk([&](Operation *op) {
        bool issuesToken = false;
        if (auto task = dyn_cast<AIEX::DMAConfigureTaskOp>(op))
          issuesToken = bool(task.getIssueToken());
        else if (auto task = dyn_cast<AIEX::DMAConfigureTaskForOp>(op))
          issuesToken = bool(task.getIssueToken());
        if (issuesToken)
          if (auto tile = dyn_cast_or_null<TileOp>(
                  AIEX::getRuntimeDmaTile(op, symbols)))
            tiles.insert(tile.getTileID());
        if (auto push = dyn_cast<AIEX::NpuPushQueueOp>(op))
          if (push.getIssueToken())
            tiles.insert({static_cast<int>(push.getColumn()),
                          static_cast<int>(push.getRow())});
      });
    }
    return tiles;
  }

  LogicalResult
  applicationPorts(ModuleOp module,
                   std::vector<std::pair<TileID, Port>> &avoided) {
    OwningOpRef<ModuleOp> trial = cast<ModuleOp>(module->clone());
    for (DeviceOp device :
         llvm::make_early_inc_range(trial->getOps<DeviceOp>())) {
      if (device->hasAttr("aie.is_ctrl_pkt_overlay"))
        device.erase();
      else
        device.removeOverlayAttr();
    }
    PassManager pm(&getContext());
    auto &dpm = pm.nest<DeviceOp>();
    dpm.addPass(createAIEAssignPacketIdsPass());
    dpm.addPass(createAIEObjectFifoAllocatePass());
    dpm.addPass(createAIEObjectFifoLowerDMAsPass());
    if (failed(pm.run(*trial))) {
      module.emitError("failed to analyze application resources for adaptive "
                       "control overlay");
      return failure();
    }
    std::set<std::pair<TileID, Port>> ports;
    for (DeviceOp device : trial->getOps<DeviceOp>()) {
      for (SwitchboxOp box : device.getOps<SwitchboxOp>()) {
        for (ConnectOp connect : box.getOps<ConnectOp>())
          ports.insert({box.getTileOp().getTileID(), connect.destPort()});
        for (MasterSetOp master : box.getOps<MasterSetOp>())
          ports.insert({box.getTileOp().getTileID(), master.destPort()});
      }
      DynamicTileAnalysis analysis;
      llvm::Error error = analysis.runAnalysis(device);
      if (error) {
        device.emitOpError("cannot plan adaptive control overlay: ")
            << llvm::toString(std::move(error));
        return failure();
      }
      auto collect = [&](const SwitchSettings &settings) {
        for (const auto &[tile, setting] : settings)
          for (Port output : setting.dsts)
            ports.insert({tile, output});
      };
      for (const auto &[_, settings] : analysis.routing.settings)
        collect(settings);
      for (const auto &[_, settings] : analysis.routing.idSettings)
        collect(settings);
    }
    avoided.assign(ports.begin(), ports.end());
    return success();
  }
};

ShimMM2SUses AIE::shimMM2SUses(DeviceOp device, bool packetSwitched) {
  ShimMM2SUses uses;
  auto shim = [](Value tile) -> Operation * {
    auto like = dyn_cast_or_null<TileLike>(tile.getDefiningOp());
    return like && (like.getTileType() == AIETileType::ShimNOCTile ||
                    like.getTileType() == AIETileType::ShimPLTile)
               ? like.getOperation()
               : nullptr;
  };
  for (auto endpoint : device.getOps<RouteEndpointOp>()) {
    Operation *tile = shim(endpoint.getTile());
    std::optional<int> channel = endpoint.getChannelIndex();
    if (!tile || endpoint.getBundle() != WireBundle::DMA ||
        endpoint.getRouteDirection() != DMAChannelDir::MM2S || !channel)
      continue;
    bool packet = llvm::any_of(device.getOps<RouteOp>(), [&](RouteOp route) {
      std::optional<PacketInfoAttr> header = route.getPacket();
      return route.namesAsSource(endpoint.getSymName()) &&
             isPacketSwitched(header.value_or(nullptr), packetSwitched);
    });
    if (packet)
      uses.packetEnds[{tile, *channel}].push_back(endpoint.getSymNameAttr());
    else
      uses.circuits[tile].insert(*channel);
  }
  for (auto alloc : device.getOps<ShimDMAAllocationOp>())
    if (alloc.getChannelDir() == DMAChannelDir::MM2S)
      uses.circuits[alloc.getTile().getDefiningOp()].insert(
          static_cast<int>(alloc.getChannelIndex()));
  for (auto flow : device.getOps<FlowOp>())
    if (flow.getSourceBundle() == WireBundle::DMA)
      if (Operation *tile = shim(flow.getSource()))
        uses.circuits[tile].insert(flow.getSourceChannel());
  for (auto flow : device.getOps<PacketFlowOp>())
    for (auto src : flow.getOps<PacketSourceOp>())
      if (src.getBundle() == WireBundle::DMA)
        if (Operation *tile = shim(src.getTile()))
          ++uses.packetFlows[{tile, src.channelIndex()}];
  for (auto mux : device.getOps<ShimMuxOp>())
    for (auto connect : mux.getConnections().front().getOps<ConnectOp>())
      if (connect.getSourceBundle() == WireBundle::DMA)
        uses.circuits[mux.getTile().getDefiningOp()].insert(
            connect.sourceIndex());
  return uses;
}

bool AIE::getsColumnControlOverlay(DeviceOp device, bool configuration) {
  return !AIEDialect::IsCtrlPktOverlayAttrHelper(device.getContext())
              .isAttrPresent(device) &&
         !device.getOverlayAttr() &&
         device.getTargetModel().hasProperty(
             AIETargetModel::SupportsControlPackets) &&
         (configuration ||
          !AIEGenerateColumnControlOverlayPass::collectTokenIssuingTiles(
               {device})
               .empty());
}

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
