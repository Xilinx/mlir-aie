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
#include "aie/Dialect/AIE/Transforms/AIEPlacer.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::AIE {

/// Discardable attribute set by `--aie-objectfifo-lower-cores` on `scf.for`
/// loops containing ObjectFifo accesses -- the loop unroll factor (the least
/// common multiple of the depths of the objectFifos accessed within the
/// loop) consumed by the `AIEObjectFifoUnroll` pass.
inline constexpr llvm::StringLiteral kObjectFifoUnrollHintAttrName =
    "aie.unroll_hint";

#define GEN_PASS_DECL
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"

std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEPlaceTilesPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEPlaceTilesPass(const AIEPlaceTilesOptions &options);
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEPrepareBuffersPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEAssignBufferAddressesPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEAssignBufferAddressesPass(
    const AIEAssignBufferAddressesOptions &options);
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEAssignCoreLinkFilesPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEAssignLockIDsPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIENormalizeDmaBdDimsPass();
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
createAIEPathfinderPass(const AIERoutePathfinderFlowsOptions &options);
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEObjectFifoUnrollPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEObjectFifoSplitPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEObjectFifoVerifyPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEObjectFifoAllocatePass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEObjectFifoAllocatePass(bool packetSwitched);
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEObjectFifoLowerDMAsPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEObjectFifoLowerCoresPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEObjectFifoErasePoolsPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIELowerCascadeFlowsPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEAssignBufferDescriptorIDsPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEObjectFifoLivenessPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIEGenerateColumnControlOverlayPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
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

/// Register `aie-objectFifo-stateful-transform` as a pipeline over the passes
/// that lower `aie.objectfifo`.
void registerAIEObjectFifoPipeline();

} // namespace xilinx::AIE

#endif // AIE_PASSES_H
