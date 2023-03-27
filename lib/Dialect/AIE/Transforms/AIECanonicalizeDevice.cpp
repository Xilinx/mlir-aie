//===- AIECanonicalizeDevice.cpp --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aie-canonicalize-device"
using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIECanonicalizeDevicePass
    : public AIECanonicalizeDeviceBase<AIECanonicalizeDevicePass> {
  void runOnOperation() override {

    ModuleOp moduleOp = getOperation();

    for (auto deviceOp : moduleOp.getOps<AIETarget>()) {
      (void)deviceOp;
      // We have a device..  Nothing to do.
      return;
    }

    // Builder with no insertion point, because we don't want to insert
    // the new op quite yet.
    OpBuilder builder(moduleOp->getContext());

    Location location = builder.getUnknownLoc();
    DeviceOp deviceOp = builder.create<DeviceOp>(
        location,
        AIEDeviceAttr::get(builder.getContext(), AIEDevice::xcvc1902));

    deviceOp.getRegion().takeBody(moduleOp.getBodyRegion());
    new (&moduleOp->getRegion(0)) Region(moduleOp);
    moduleOp->getRegion(0).emplaceBlock();
    OpBuilder builder2 = OpBuilder::atBlockBegin(moduleOp.getBody());
    builder2.insert(deviceOp);
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIECanonicalizeDevicePass() {
  return std::make_unique<AIECanonicalizeDevicePass>();
}