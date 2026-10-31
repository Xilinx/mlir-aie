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
#include "mlir/IR/IRMapping.h"
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
    IRMapping mapping;
    OwningOpRef<ModuleOp> schedules = cast<ModuleOp>(module->clone(mapping));
    SmallVector<RouteEndpointOp> endpoints;
    for (DeviceOp device : module.getOps<DeviceOp>())
      for (auto endpoint : device.getOps<RouteEndpointOp>()) {
        if (endpoint.getBundle() != WireBundle::DMA ||
            !endpoint.getTileLike().isShimTile())
          continue;
        auto *copy = mapping.lookup(endpoint.getOperation());
        copy->setAttr("aie.sharing_id",
                      IntegerAttr::get(IntegerType::get(&getContext(), 32),
                                       endpoints.size()));
        endpoints.push_back(endpoint);
      }
    PassManager materialize(&getContext());
    // Inlined symbolic endpoints have no local routes in this analysis copy.
    materialize.enableVerifier(false);
    AIEMaterializeRuntimeSequencesOptions options;
    options.analysisOnly = true;
    materialize.addPass(createAIEMaterializeRuntimeSequencesPass(options));
    if (failed(materialize.run(*schedules)))
      return failure();

    SmallVector<bool> seen(endpoints.size(), false),
        blocked(endpoints.size(), false);
    DenseSet<std::pair<unsigned, unsigned>> blockedPairs;
    SmallVector<StringAttr> keys;
    for (unsigned i = 0; i < endpoints.size(); ++i)
      keys.push_back(StringAttr::get(&getContext(), std::to_string(i)));
    StringAttr config = StringAttr::get(&getContext(), "configuration");
    for (DeviceOp device : schedules->getOps<DeviceOp>()) {
      auto resolve =
          [&](StringAttr symbol) -> std::optional<ShimTransferSpans::End> {
        auto endpoint = device.lookupSymbol<RouteEndpointOp>(symbol.getValue());
        if (!endpoint) {
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
        }
        auto id = endpoint->getAttrOfType<IntegerAttr>("aie.sharing_id");
        if (!id)
          return std::nullopt;
        auto direction = endpoints[id.getInt()].getRouteDirection();
        ShimTransferSpans::End end{keys[id.getInt()],
                                   direction == DMAChannelDir::MM2S};
        if (auto channel = endpoint.getChannelIndex()) {
          auto tile = dyn_cast<TileOp>(endpoint.getTile().getDefiningOp());
          if (tile)
            end.channel =
                std::make_tuple(tile.colIndex(), tile.rowIndex(),
                                static_cast<int>(direction), *channel);
        }
        return end;
      };
      ShimTransferSpans spans(device, resolve, config);
      for (unsigned i = 0; i < endpoints.size(); ++i) {
        auto key = keys[i];
        if (spans.isUsed(key)) {
          seen[i] = true;
          blocked[i] = blocked[i] || !spans.apart(key, config);
          for (unsigned j = i + 1; j < endpoints.size(); ++j)
            if (spans.isUsed(keys[j]) && !spans.apart(key, keys[j]))
              blockedPairs.insert({i, j});
        }
      }
    }
    for (auto [i, endpoint] : llvm::enumerate(endpoints)) {
      SmallVector<Attribute> allowed;
      for (auto [j, other] : llvm::enumerate(endpoints))
        if (i != j && seen[i] && seen[j] &&
            endpoint->getParentOp() == other->getParentOp() &&
            endpoint.getTile() == other.getTile() &&
            endpoint.getRouteDirection() == other.getRouteDirection() &&
            !blockedPairs.contains({std::min(i, j), std::max(i, j)}))
          allowed.push_back(FlatSymbolRefAttr::get(other.getSymNameAttr()));
      endpoint->setAttr("aie.shareable_with",
                        ArrayAttr::get(&getContext(), allowed));
    }
    AIEDialect::IsCtrlOverlayAttrHelper overlay(&getContext());
    for (DeviceOp device : module.getOps<DeviceOp>()) {
      for (auto alloc : device.getOps<ShimDMAAllocationOp>()) {
        if (!alloc->hasAttr("aie.fixed") || !overlay.isAttrPresent(alloc))
          continue;
        SmallVector<Attribute> allowed;
        for (auto [i, endpoint] : llvm::enumerate(endpoints))
          if (shareChannels && seen[i] && !blocked[i] &&
              endpoint->getParentOp() == device &&
              endpoint.getTile() == alloc.getTile() &&
              endpoint.getRouteDirection() == alloc.getChannelDir())
            allowed.push_back(
                FlatSymbolRefAttr::get(endpoint.getSymNameAttr()));
        alloc->setAttr("aie.shareable_with",
                       ArrayAttr::get(&getContext(), allowed));
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
