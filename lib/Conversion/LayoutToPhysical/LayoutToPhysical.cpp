//===- LayoutToPhysical.cpp -----------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Conversion/LayoutToPhysical.h"

#include "phy/Connectivity/Implementation.h"
#include "phy/Connectivity/ResourceList.h"
#include "phy/Conversion/Passes.h"
#include "phy/Dialect/Layout/LayoutDialect.h"
#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Dialect/Spatial/SpatialDialect.h"
#include "phy/Rewrite/RemoveOp.h"

#include <memory>
#include <set>

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-layout-to-physical"

using namespace mlir;
using namespace xilinx::phy;
using namespace xilinx::phy::rewrite;
using namespace xilinx::phy::connectivity;

namespace {

// Run a pre pipeline of cleanup passes.
static void preCanonicalizeIR(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createInlinerPass());
  assert(!failed(pm.run(module)));
}

// Run a post pipeline of cleanup and optimization passes.
static void postCanonicalizeIR(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createSymbolDCEPass());
  assert(!failed(pm.run(module)));
}

struct LayoutToPhysical : public LayoutToPhysicalBase<LayoutToPhysical> {

  void runOnOperation() override {
    auto module = getOperation();
    preCanonicalizeIR(module);

    ImplementationContext context(module, device_option);
    collectImplementations(context);
    context.implementAll();

    cleanupSpatialLayoutOperations(context);

    postCanonicalizeIR(module);
  }

  void collectImplementations(ImplementationContext &context) {
    context.module.walk([&](Operation *op) {
      auto device = dyn_cast<layout::DeviceOp>(op);

      // skipping non device operations
      // TODO: support platforms
      if (!device)
        return;

      // skipping devices that are not selected.
      if (device.getDevice().str() != device_option)
        return;

      for (auto place : device.getOps<layout::PlaceOp>()) {
        auto *spatial = place.getVertex().getDefiningOp();
        ResourceList resources(place.getSlot().str());
        context.place(spatial, resources);
      }

      for (auto route : device.getOps<layout::RouteOp>()) {
        auto *src = route.getSrc().getDefiningOp();
        auto *dest = route.getDest().getDefiningOp();
        std::list<ResourceList> resources;
        for (auto wire : route.getWires()) {
          resources.emplace_back(wire.dyn_cast<StringAttr>().str());
        }
        context.route(src, dest, resources);
      }
    });
  }

  void cleanupSpatialLayoutOperations(ImplementationContext &context) {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<physical::PhysicalDialect>();
    target.addIllegalDialect<layout::LayoutDialect>();
    target.addIllegalDialect<spatial::SpatialDialect>();

    // Remove all functions with spatial arguments
    target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
      for (auto type : op.getFunctionType().getInputs())
        if (type.isa<spatial::QueueType>())
          return false;
      return true;
    });

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<OpRemover<layout::PlatformOp>>(&getContext());
    patterns.add<OpRemover<layout::DeviceOp>>(&getContext());
    patterns.add<OpRemover<layout::PlaceOp>>(&getContext());
    patterns.add<OpRemover<layout::RouteOp>>(&getContext());
    patterns.add<OpRemover<spatial::QueueOp>>(&getContext());
    patterns.add<OpRemover<spatial::NodeOp>>(&getContext());
    patterns.add<OpRemover<func::FuncOp>>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(context.module, target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> xilinx::phy::createLayoutToPhysical() {
  return std::make_unique<LayoutToPhysical>();
}
