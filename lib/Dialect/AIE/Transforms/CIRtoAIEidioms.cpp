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

#include <compare>
#include <map>
#include <optional>

namespace xilinx::AIE {

namespace {
struct DeviceNameAndLoc {
  std::string name;
  mlir::Location loc;
};

// Get a device name with its location in the C++ code by looking for a
// aie::device<aie::...>::run() call
//
// \todo Check that this aie::device appears only once in the code
std::optional<DeviceNameAndLoc> getDeviceName(mlir::ModuleOp m) {
  std::optional<DeviceNameAndLoc> deviceName;
  // The struct has a name like "aie::device<aie::npu1>" and the "npu1" is
  // used directly for the MLIR aie.device attribute
  static const llvm::Regex deviceNamePattern{"^aie::device<aie::([^>]+)>$"};
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
    if (llvm::SmallVector<llvm::StringRef> matches;
        deviceNamePattern.match(aieDeviceType.getName(), &matches))
      deviceName = {std::string{matches[1]}, call.getLoc()};
    // \todo optimize with early exit walk::interrupt()?
  });
  return deviceName;
}

// Wrap all the module code into a top-level aie.device operation if a
// aie::device<aie::...>::run() call is found
std::optional<xilinx::AIE::DeviceOp> generateAieDevice(mlir::ModuleOp m) {
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
    return device;
  }
  return {};
}

// Get a tile name coordinate with its location in the C++ code by looking for a
// aie::tile_t<...,...> cir.alloca
auto getTileCoordinates(mlir::ModuleOp m) {
  // Use a map to keep only the last use which is more likely to be in user code
  // rather than in implementation library
  std::map<xilinx::AIE::TileID, std::optional<mlir::Location>> tileCoordinates;
  // Look for a struct aie::tile_t<...,...>
  static const llvm::Regex tileNamePattern{
      "^aie::tile_t<([[:digit:]]+), ([[:digit:]]+)>$"};
  m->walk([&](mlir::cir::AllocaOp alloc) {
    if (const auto &st =
            mlir::dyn_cast<mlir::cir::StructType>(alloc.getAllocaType())) {
      st.dump();
      llvm::errs() << "getName: " << st.getName() << "\n";
      if (llvm::SmallVector<llvm::StringRef> matches;
          tileNamePattern.match(st.getName(), &matches))
        tileCoordinates[{std::stoi(std::string{matches[1]}),
                         std::stoi(std::string{matches[2]})}] = alloc.getLoc();
    }
  });
  return tileCoordinates;
}

// Declare the aie.tile from C++ aie::device<...>::tile<...,...>() calls
void generateAieTiles(mlir::ModuleOp m, xilinx::AIE::DeviceOp device) {
  auto tcs = getTileCoordinates(m);
  mlir::OpBuilder builder{device};
  builder.setInsertionPointToStart(device.getBody());
  for (auto &[coordinate, loc] : tcs) {
    llvm::errs() << coordinate.col << " " << coordinate.row << " " << *loc << "\n";
    builder.create<xilinx::AIE::TileOp>(*loc, coordinate.col, coordinate.row);
  }
} // namespace

struct CIRtoAIEidiomsPass : CIRtoAIEidiomsBase<CIRtoAIEidiomsPass> {
  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    if (auto device = generateAieDevice(m))
        generateAieTiles(m, *device);
  }
};
} // namespace
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCIRtoAIEidiomsPass() {
  return std::make_unique<CIRtoAIEidiomsPass>();
}

} // namespace xilinx::AIE
