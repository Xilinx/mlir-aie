//===- AIEControlOverlayAvoidData.cpp ---------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIECONTROLOVERLAYAVOIDDATA
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

namespace {

// The router routes these alone.
static bool isOverlayFlow(PacketFlowOp flow) {
  return flow.getPriorityRoute().value_or(false) && touchesTileControl(flow);
}

struct AIEControlOverlayAvoidDataPass
    : xilinx::AIE::impl::AIEControlOverlayAvoidDataBase<
          AIEControlOverlayAvoidDataPass> {
  // Adds the switchbox output ports the device's data takes, routed without
  // the overlay flows.
  static LogicalResult
  collectDataPorts(DeviceOp device, std::set<std::pair<TileID, Port>> &ports) {
    OwningOpRef<ModuleOp> scratch = ModuleOp::create(device.getLoc());
    DeviceOp data = device.clone();
    scratch->push_back(data);
    for (PacketFlowOp flow :
         llvm::make_early_inc_range(data.getOps<PacketFlowOp>()))
      if (isOverlayFlow(flow))
        flow.erase();
    DynamicTileAnalysis analyzer;
    llvm::Error routed = analyzer.runAnalysis(data);
    while (routed && analyzer.pathfinder.relax()) {
      llvm::consumeError(std::move(routed));
      routed = analyzer.runAnalysis(data);
    }
    if (routed) {
      llvm::handleAllErrors(std::move(routed), [&](RoutingFailure &f) {
        device.emitError() << "the control overlay cannot route around this "
                              "configuration's data, which does not route "
                              "without it: "
                           << f.message();
      });
      return failure();
    }
    auto add = [&](const SwitchSettings &settings) {
      for (const auto &[coords, setting] : settings)
        for (Port dst : setting.dsts)
          ports.insert({coords, dst});
    };
    for (const auto &[_, settings] : analyzer.routing.settings)
      add(settings);
    for (const auto &[_, settings] : analyzer.routing.idSettings)
      add(settings);
    // Connections the design makes itself.
    for (SwitchboxOp box : device.getOps<SwitchboxOp>()) {
      TileID coords = box.getTileOp().getTileID();
      for (ConnectOp connect : box.getOps<ConnectOp>())
        ports.insert({coords, connect.destPort()});
      for (MasterSetOp master : box.getOps<MasterSetOp>())
        ports.insert({coords, master.destPort()});
    }
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto overlay = module.lookupSymbol<DeviceOp>("ctrl_pkt_overlay");
    if (!overlay)
      return;
    SmallVector<DeviceOp> devices;
    std::set<std::pair<TileID, Port>> ports;
    for (DeviceOp device : module.getOps<DeviceOp>()) {
      if (device == overlay ||
          llvm::none_of(device.getOps<PacketFlowOp>(), isOverlayFlow))
        continue;
      if (failed(collectDataPorts(device, ports)))
        return signalPassFailure();
      devices.push_back(device);
    }
    // Every device routes the overlay alone around the same ports, so all
    // take the route @ctrl_pkt_overlay installs.
    devices.push_back(overlay);
    MLIRContext *ctx = module.getContext();
    SmallVector<Attribute> avoided;
    for (const auto &[coords, port] : ports)
      avoided.push_back(SwitchPortAttr::get(ctx, coords.col, coords.row,
                                            port.bundle, port.channel));
    AIEDialect::CtrlPktOverlayAvoidAttrHelper avoid(ctx);
    for (DeviceOp device : devices)
      avoid.setAttr(device, ArrayAttr::get(ctx, avoided));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIEControlOverlayAvoidDataPass() {
  return std::make_unique<AIEControlOverlayAvoidDataPass>();
}
