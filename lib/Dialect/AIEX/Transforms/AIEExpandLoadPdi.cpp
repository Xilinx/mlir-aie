//===- AIEExpandLoadPdi.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//
//
// This pass expands `npu.load_pdi` operations that reference a device into:
// 1. An empty device PDI load (causes firmware to reset the device)
// 2. Explicit configuration operations (write32/blockwrite)
//
//===----------------------------------------------------------------------===//

#include "aie/Conversion/AIEToConfiguration/AIEToConfiguration.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "aie-expand-load-pdi"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;
using namespace xilinx::AIE;

namespace {

// Helper to transform a single load_pdi operation
static LogicalResult transformLoadPdi(NpuLoadPdiOp loadPdiOp, ModuleOp moduleOp,
                                      unsigned index) {
  static unsigned long i = 0;
  OpBuilder builder(loadPdiOp);

  // Only process load_pdi ops that reference a device
  auto deviceRefAttr = loadPdiOp.getDeviceRefAttr();
  if (!deviceRefAttr) {
    return success();
  }

  auto referencedDevice = moduleOp.lookupSymbol<AIE::DeviceOp>(deviceRefAttr);
  if (!referencedDevice) {
    loadPdiOp.emitError("Referenced symbol '")
        << deviceRefAttr.getValue() << "' is not a device";
    return failure();
  }

  // Create a unique empty device for this reset to avoid PDI address caching
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());

  // Find a unique name for the empty device
  std::string emptyName = "empty_" + std::to_string(index % 2);

  AIE::DeviceOp emptyDevice = moduleOp.lookupSymbol<AIE::DeviceOp>(emptyName);
  if (!emptyDevice) {
    auto deviceType = referencedDevice.getDevice();
    auto loc = builder.getUnknownLoc();
    emptyDevice = AIE::DeviceOp::create(builder, loc, deviceType,
                                        builder.getStringAttr(emptyName));
    emptyDevice.getRegion().emplaceBlock();
    Block *deviceBlock = &emptyDevice.getRegion().front();
    builder.setInsertionPointToEnd(deviceBlock);
    AIE::EndOp::create(builder, loc);
  }

  builder.setInsertionPoint(loadPdiOp);

  // Create new empty load_pdi operation; this triggers a device reset.
  // This is needed even for the first device configuration; without it, the
  // first iteration of the design would run, but subsequent ones might not.
  NpuLoadPdiOp::create(builder, loadPdiOp.getLoc(),
                       FlatSymbolRefAttr::get(emptyDevice.getSymNameAttr()),
                       loadPdiOp.getIdAttr(), loadPdiOp.getSizeAttr(),
                       loadPdiOp.getAddressAttr());

  // Generate and insert configuration operations
  if (failed(xilinx::AIE::generateAndInsertConfigOps(
          builder, referencedDevice, "",
          AIEToConfigurationOutputType::Transaction,
          "loadpdi_" + std::to_string(i)))) {
    loadPdiOp.emitError("Failed to generate configuration operations");
    return failure();
  }

  // Erase the original load_pdi operation
  loadPdiOp.erase();

  i++;

  return success();
}

struct AIEExpandLoadPdiPass
    : public AIEExpandLoadPdiBase<AIEExpandLoadPdiPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<memref::MemRefDialect, AIE::AIEDialect, AIEX::AIEXDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Collect all load_pdi operations in program order;
    // need to collect once, then transform all collected ops;
    // since the transform inserts a new empty load_pdi, we can't transform as
    // we walk or it'd infinitely recurse.
    SmallVector<NpuLoadPdiOp> loadPdiOps;

    module.walk(
        [&](NpuLoadPdiOp loadPdiOp) { loadPdiOps.push_back(loadPdiOp); });

    // Transform load_pdi ops
    unsigned idx = 0;
    for (auto loadPdiOp : loadPdiOps) {
      if (failed(transformLoadPdi(loadPdiOp, module, idx))) {
        signalPassFailure();
        return;
      }
      idx++;
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
xilinx::AIEX::createAIEExpandLoadPdiPass() {
  return std::make_unique<AIEExpandLoadPdiPass>();
}
