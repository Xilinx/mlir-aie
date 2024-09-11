//===- AIECoreToStandard.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/Regex.h"

namespace xilinx::AIE {

namespace {
struct DeviceNameAndLoc {
  std::string name;
  mlir::Location loc;
};

std::optional<DeviceNameAndLoc> getDeviceName(mlir::ModuleOp m) {
  std::optional<DeviceNameAndLoc> deviceName;
  m->walk([&](mlir::cir::CallOp call) {
    // Look at the member function call to aie::device::run()
    if (auto calleeName = call.getCallee();
        // \todo Make this mangle-free
        calleeName != "_ZN3aie6deviceILNS_3$_0E0EE3runEv")
      return;
    // Dive into the type of the object which is passed as the first parameter
    // to the member function
    auto aieDeviceInstance = call.getArgOperand(0);
    // Get to the aie::device type
    auto aieDeviceType = mlir::cast<mlir::cir::StructType>(
        mlir::cast<mlir::cir::PointerType>(aieDeviceInstance.getType())
            .getPointee());
    // The struct has a name like "aie::device<aie::npu1>" and the "npu1" is
    // used directly for the MLIR aie.device attribute
    static const llvm::Regex deviceNamePattern{"^aie::device<aie::([^>]+)>$"};
    if (llvm::SmallVector<llvm::StringRef> matches;
        deviceNamePattern.match(aieDeviceType.getName(), &matches))
      deviceName = {std::string{matches[1]}, call.getLoc()};
    // \todo optimize with early exit walk::interrupt()?
  });
  return deviceName;
}

} // namespace

struct CIRtoAIEidiomsPass : CIRtoAIEidiomsBase<CIRtoAIEidiomsPass> {
  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    if (std::optional<DeviceNameAndLoc> deviceName = getDeviceName(m);
        deviceName) {
      mlir::OpBuilder builder{m};
      // Parse the device name into the right integer attribute
      auto deviceId =
          *xilinx::AIE::symbolizeEnum<xilinx::AIE::AIEDevice>(deviceName->name);
      auto device =
          builder.create<xilinx::AIE::DeviceOp>(deviceName->loc, deviceId);
      // Move the module region block into the device region
      device.getRegion().takeBody(m.getBodyRegion());
      // Recreate an empty region in the module
      new (&m.getRegion()) mlir::Region{m};
      // Create an empty block in the region
      m.getRegion().emplaceBlock();
      // Insert the aie.device in the module block
      builder.setInsertionPointToStart(m.getBody());
      builder.insert(device);
    }
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCIRtoAIEidiomsPass() {
  return std::make_unique<CIRtoAIEidiomsPass>();
}

} // namespace xilinx::AIE
