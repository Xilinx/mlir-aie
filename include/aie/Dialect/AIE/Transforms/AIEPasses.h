//===- AIEPasses.h ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_PASSES_H
#define AIE_PASSES_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::AIE {

#define GEN_PASS_CLASSES
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"

std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEAssignBufferAddressesPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEAssignLockIDsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIECanonicalizeDevicePass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIECoreToStandardPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEFindFlowsPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIELocalizeLocksPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIENormalizeAddressSpacesPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAIERouteFlowsPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createAIEVectorOptPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEPathfinderPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEObjectFifoStatefulTransformPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEObjectFifoRegisterProcessPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIELowerCascadeFlowsPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEAssignBufferDescriptorIDsPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEGenerateColumnControlOverlayPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEAssignTileCtrlIDsPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIESetELFforCorePass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"

/// Overall Flow:
/// rewrite switchboxes to assign unassigned connections, ensure this can be
/// done concurrently ( by different threads)
/// 1. Goal is to rewrite all flows in the device into switchboxes + shim-mux
/// 2. multiple passes of the rewrite pattern rewriting streamswitch
/// configurations to routes
/// 3. rewrite flows to stream-switches using 'weights' from analysis pass.
/// 4. check a region is legal
/// 5. rewrite stream-switches (within a bounding box) back to flows
struct AIEPathfinderPass : AIERoutePathfinderFlowsBase<AIEPathfinderPass> {

  AIEPathfinderPass() = default;

  void runOnOperation() override;
  void runOnFlow(DeviceOp d, DynamicTileAnalysis &analyzer);
  void runOnPacketFlow(DeviceOp d, mlir::OpBuilder &builder, DynamicTileAnalysis &analyzer);

  typedef std::pair<TileID, Port> PhysPort;

  bool findPathToDest(SwitchSettings settings, TileID currTile,
                      WireBundle currDestBundle, int currDestChannel,
                      TileID finalTile, WireBundle finalDestBundle,
                      int finalDestChannel);
};

} // namespace xilinx::AIE

#endif // AIE_PASSES_H
