//===- AIECreateControlOverlay.cpp -----------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"
#include "aie/Dialect/AIE/Transforms/AIEShimSharing.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIECREATECONTROLOVERLAY
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

namespace {
struct AIECreateControlOverlayPass
    : xilinx::AIEX::impl::AIECreateControlOverlayBase<
          AIECreateControlOverlayPass> {
  using Base::Base;

  LogicalResult sharingConstraints(ModuleOp module) {
    SmallVector<RouteEndpointOp> endpoints;
    DenseMap<Operation *, StringAttr> keys;
    for (DeviceOp device : module.getOps<DeviceOp>())
      for (auto endpoint : device.getOps<RouteEndpointOp>()) {
        if (endpoint.getBundle() != WireBundle::DMA ||
            !endpoint.getTileLike().isShimTile())
          continue;
        keys[endpoint] =
            StringAttr::get(&getContext(), std::to_string(endpoints.size()));
        endpoints.push_back(endpoint);
      }
    StringAttr config = StringAttr::get(&getContext(), "configuration");
    ShimTransferSpans spans(
        module,
        [&](DeviceOp device,
            StringAttr symbol) -> std::optional<ShimTransferSpans::End> {
          if (auto endpoint =
                  device.lookupSymbol<RouteEndpointOp>(symbol.getValue())) {
            auto key = keys.lookup(endpoint);
            if (!key)
              return std::nullopt;
            auto direction = endpoint.getRouteDirection();
            ShimTransferSpans::End end{key, direction == DMAChannelDir::MM2S};
            if (auto channel = endpoint.getChannelIndex())
              if (auto tile =
                      dyn_cast<TileOp>(endpoint.getTile().getDefiningOp()))
                end.channel =
                    std::make_tuple(tile.colIndex(), tile.rowIndex(),
                                    static_cast<int>(direction), *channel);
            return end;
          }
          auto alloc =
              device.lookupSymbol<ShimDMAAllocationOp>(symbol.getValue());
          AIEDialect::IsCtrlOverlayAttrHelper overlay(&getContext());
          if (!alloc || !overlay.isAttrPresent(alloc))
            return std::nullopt;
          return ShimTransferSpans::End{
              config, false,
              std::make_tuple(alloc.getTileOp().colIndex(),
                              alloc.getTileOp().rowIndex(),
                              static_cast<int>(alloc.getChannelDir()),
                              static_cast<int>(alloc.getChannelIndex()))};
        },
        config);
    for (auto endpoint : endpoints) {
      SmallVector<Attribute> allowed;
      for (auto other : endpoints)
        if (endpoint != other &&
            endpoint->getParentOp() == other->getParentOp() &&
            endpoint.getTile() == other.getTile() &&
            endpoint.getRouteDirection() == other.getRouteDirection() &&
            spans.apart(keys.lookup(endpoint), keys.lookup(other)))
          allowed.push_back(FlatSymbolRefAttr::get(other.getSymNameAttr()));
      AIEDialect::ShareableWithAttrHelper sharing(&getContext());
      if (!sharing.isAttrPresent(endpoint))
        sharing.setAttr(endpoint, ArrayAttr::get(&getContext(), allowed));
    }
    AIEDialect::IsCtrlOverlayAttrHelper overlay(&getContext());
    for (DeviceOp device : module.getOps<DeviceOp>()) {
      for (auto alloc : device.getOps<ShimDMAAllocationOp>()) {
        if (!overlay.isAttrPresent(alloc))
          continue;
        SmallVector<Attribute> allowed;
        for (auto endpoint : endpoints)
          if (shareChannels && spans.apart(keys.lookup(endpoint), config) &&
              endpoint->getParentOp() == device &&
              endpoint.getTile() == alloc.getTile() &&
              endpoint.getRouteDirection() == alloc.getChannelDir())
            allowed.push_back(
                FlatSymbolRefAttr::get(endpoint.getSymNameAttr()));
        AIEDialect::ShareableWithAttrHelper sharing(&getContext());
        if (!sharing.isAttrPresent(alloc))
          sharing.setAttr(alloc, ArrayAttr::get(&getContext(), allowed));
      }
    }
    return success();
  }

  std::vector<std::pair<TileID, Port>> applicationPorts(ModuleOp module) {
    OwningOpRef<ModuleOp> trial = cast<ModuleOp>(module->clone());
    for (DeviceOp device :
         llvm::make_early_inc_range(trial->getOps<DeviceOp>())) {
      if (device->hasAttr("aie.is_ctrl_overlay"))
        device.erase();
      else
        device.removeOverlayAttr();
    }
    PassManager pm(&getContext());
    auto &dpm = pm.nest<DeviceOp>();
    dpm.addPass(createAIEAssignPacketIdsPass());
    dpm.addPass(createAIEObjectFifoAllocatePass());
    dpm.addPass(createAIEObjectFifoLowerDMAsPass());
    ScopedDiagnosticHandler quiet(&getContext(),
                                  [](Diagnostic &) { return success(); });
    if (failed(pm.run(*trial)))
      return {};
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
        llvm::consumeError(std::move(error));
        continue;
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
    return {ports.begin(), ports.end()};
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    AIEGenerateColumnControlOverlayOptions options;
    options.clRouteShimDmaToTileCTRL = true;
    options.clEmitStandaloneOverlay = true;
    options.clPlanChannels = true;
    options.clAdaptiveChannels = adaptive;
    options.clCoverColumns = llvm::to_vector(coverColumns);
    OpPassManager pm(ModuleOp::getOperationName());
    pm.addPass(createAIEGenerateColumnControlOverlayPass(options));
    if (failed(runPipeline(pm, module)))
      return signalPassFailure();
    auto avoided = adaptive ? applicationPorts(module)
                            : std::vector<std::pair<TileID, Port>>{};
    for (DeviceOp overlay : module.getOps<DeviceOp>()) {
      if (!overlay->hasAttr("aie.is_ctrl_overlay") ||
          overlay.getOps<PacketFlowOp>().empty())
        continue;
      OpPassManager routing(DeviceOp::getOperationName());
      routing.addPass(createAIEAssignTileCtrlIDsPass());
      routing.addPass(createAIEPathfinderPass({}, avoided));
      if (failed(runPipeline(routing, overlay)))
        return signalPassFailure();
      // Adding master sets to an active arbiter can stall its in-flight
      // packets. Unused amsels reserve the rest of each overlay arbiter during
      // routing.
      for (auto switchbox : overlay.getOps<SwitchboxOp>()) {
        auto amsels =
            llvm::to_vector(switchbox.getConnections().getOps<AMSelOp>());
        std::set<std::pair<int, int>> slots;
        for (auto amsel : amsels)
          slots.emplace(amsel.arbiterIndex(), amsel.getMselValue());
        OpBuilder builder =
            OpBuilder::atBlockTerminator(&switchbox.getConnections().front());
        for (auto amsel : amsels)
          for (unsigned msel = 0;
               msel < overlay.getTargetModel().getNumMselsPerArbiter(); ++msel)
            if (slots.emplace(amsel.arbiterIndex(), msel).second)
              AMSelOp::create(builder, amsel.getLoc(), amsel.arbiterIndex(),
                              msel);
      }
      overlay.walk([&](Operation *op) {
        op->removeAttr("aie.priority_route");
        if (!isa<DeviceOp, EndOp, PacketRulesOp>(op))
          op->setAttr("aie.is_ctrl_overlay", UnitAttr::get(&getContext()));
      });
    }
    OpPassManager imports(ModuleOp::getOperationName());
    imports.addPass(createAIEImportOverlayPass());
    if (failed(runPipeline(imports, module)) ||
        failed(sharingConstraints(module)))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
AIEX::createAIECreateControlOverlayPass() {
  return std::make_unique<AIECreateControlOverlayPass>();
}
std::unique_ptr<OperationPass<ModuleOp>>
AIEX::createAIECreateControlOverlayPass(
    const AIECreateControlOverlayOptions &options) {
  return std::make_unique<AIECreateControlOverlayPass>(options);
}
