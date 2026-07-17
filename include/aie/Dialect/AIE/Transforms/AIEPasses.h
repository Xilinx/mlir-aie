//===- AIEPasses.h ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_PASSES_H
#define AIE_PASSES_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"
#include "aie/Dialect/AIE/Transforms/AIEPlacer.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::AIE {

/// Discardable attribute set by `AIEObjectFifoStatefulTransform` on `scf.for` 
/// loops containing ObjectFifo accesses -- the loop unroll factor (the least 
/// common multiple of the depths of the objectFifos accessed within the 
/// loop) consumed by the `AIEObjectFifoUnroll` pass.
inline constexpr llvm::StringLiteral kObjectFifoUnrollHintAttrName =
    "aie.unroll_hint";

#define GEN_PASS_DECL
#define GEN_PASS_DEF_AIEROUTEPATHFINDERFLOWS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"

std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEPlaceTilesPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEPlaceTilesPass(const AIEPlaceTilesOptions &options);
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEAssignBufferAddressesPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEAssignBufferAddressesPass(
    const AIEAssignBufferAddressesOptions &options);
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEAssignCoreLinkFilesPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEAssignLockIDsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIECanonicalizeDevicePass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIECoreToStandardPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIECoreToStandardPass(const AIECoreToStandardOptions &options);
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEFindFlowsPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIELocalizeLocksPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIENormalizeAddressSpacesPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAIERouteFlowsPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEVectorToPointerLoopsPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEVectorTransferLoweringPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIEHoistVectorTransferPointersPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEPathfinderPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEObjectFifoStatefulTransformPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEObjectFifoUnrollPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIELowerCascadeFlowsPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEAssignBufferDescriptorIDsPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEObjectFifoLivenessPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEGenerateColumnControlOverlayPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEGenerateColumnControlOverlayPass(
    const AIEGenerateColumnControlOverlayOptions &options);
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEAssignTileCtrlIDsPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIETraceToConfigPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIETraceRegPackWritesPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEInsertTraceFlowsPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"

/// \brief Routes flows in a device by lowering them to stream-switch
/// configurations.
///
/// Overall flow:
/// 1. Rewrite all flows in the device into switchboxes + shim-mux.
/// 2. Run multiple passes of the rewrite pattern, rewriting stream-switch
///    configurations to routes.
/// 3. Rewrite flows to stream-switches using 'weights' from the analysis pass.
/// 4. Check that a region is legal.
/// 5. Rewrite stream-switches (within a bounding box) back to flows.
struct AIEPathfinderPass
    : impl::AIERoutePathfinderFlowsBase<AIEPathfinderPass> {

  AIEPathfinderPass() = default;

  void runOnOperation() override;
  mlir::LogicalResult runOnFlow(DeviceOp d, DynamicTileAnalysis &analyzer);
  mlir::LogicalResult runOnPacketFlow(DeviceOp d, mlir::OpBuilder &builder,
                                      DynamicTileAnalysis &analyzer);

  typedef std::pair<TileID, Port> PhysPort;

  bool findPathToDest(SwitchSettings settings, TileID currTile,
                      WireBundle currDestBundle, int currDestChannel,
                      TileID finalTile, WireBundle finalDestBundle,
                      int finalDestChannel);
};

} // namespace xilinx::AIE

#endif // AIE_PASSES_H
