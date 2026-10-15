//===- AIEXPasses.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIEX_PASSES_H
#define AIEX_PASSES_H

#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace xilinx::AIEX {

#define GEN_PASS_DECL
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"

std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>> createAIECreateCoresPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>> createAIECreateLocksPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>> createAIELowerMemcpyPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIELowerMulticastPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIEBroadcastPacketPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>> createAIEDmaToNpuPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIEDmaToNpuPass(const AIEDmaToNpuOptions &options);
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIEDecomposeLargeDmaBdPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAIENpuToCertPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>> createAIECertPagesPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>> createAIECertVerifyPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAIEXToStandardPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIESCFToControlFlowPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIEMaterializeBDChainsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIEMaterializeRuntimeSequencesPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIEFuseTraceBuffersPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIEResolveAddressPatchBuffersPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIEAssignRuntimeSequenceBDIDsPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIEAssignRuntimeSequenceBDIDsPass(
    const AIEAssignRuntimeSequenceBDIDsOptions &options);
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIESplitLongRepeatsPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIEReserveRuntimeBDIDsPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIEUnrollRuntimeSequenceLoopsPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIELowerDynamicBDPoolPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIELowerDynamicBDPoolPass(const AIELowerDynamicBDPoolOptions &options);
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIEDMATasksToNPUPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIESubstituteShimDMAAllocationsPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIECtrlPacketToDmaPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIECtrlPacketInferTilesPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>> createAIELowerSetLockPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIELowerDmaChannelResetPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIELowerCoreResetPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIEVerifyRuntimeRearmPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIETransformBfpTypesPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIETxnToControlPacketPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIELegalizeControlPacketPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIEExpandLoadPdiPass();
std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
createAIEXInlineTraceConfigPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIELowerScratchpadParametersPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIELowerScratchpadParametersPass(
    AIELowerScratchpadParametersOptions options);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"

/// Name of the runtime sequence aie-expand-load-pdi{init-entry} adds.
inline constexpr llvm::StringLiteral initEntrySequenceName = "reconfig_init";

/// aiecc's stages that give a design its tiles, locks, BD IDs, buffers and
/// routes, in its order. aiecc runs them as separate steps; the
/// `aie-resource-stages` pipeline runs them back to back. Each anchors on the
/// module.
void addPlacementStage(mlir::OpPassManager &pm,
                       const AIE::AIEPlaceTilesOptions &options);
void addTraceStage(mlir::OpPassManager &pm);
/// Affine lowering through `aie-prepare-buffers`.
void addLoweringStage(mlir::OpPassManager &pm, bool dynamicObjFifos,
                      bool packetSwObjFifos, bool ctrlPktOverlay,
                      bool loadPdiToCtrlPkt, bool skipObjectFifoVerify,
                      llvm::ArrayRef<int> coverColumns = {});
void addBufferAddressStage(mlir::OpPassManager &pm, int64_t placementBudget);
void addRoutingStage(mlir::OpPassManager &pm, bool allowDeadlockProne);

/// Options of `aie-resource-stages`, named and defaulted as aiecc's flags.
struct AIEResourceStagesOptions
    : public mlir::PassPipelineOptions<AIEResourceStagesOptions> {
  Option<AIE::PlacerType> placer{
      *this, "placer", llvm::cl::desc("Placement algorithm to use"),
      llvm::cl::values(clEnumValN(AIE::PlacerType::SequentialPlacer,
                                  "sequential_placer",
                                  "Sequential column-major placement"),
                       clEnumValN(AIE::PlacerType::SAPlacer, "sa_placer",
                                  "Simulated annealing placement")),
      llvm::cl::init(AIE::PlacerType::SequentialPlacer)};
  Option<int> coresPerCol{*this, "cores-per-col",
                          llvm::cl::desc("Max cores per column"),
                          llvm::cl::init(-1)};
  Option<int> saSeed{
      *this, "sa-seed",
      llvm::cl::desc("Random seed for SA placer (0 = non-deterministic)"),
      llvm::cl::init(1)};
  Option<double> saEffort{
      *this, "sa-effort",
      llvm::cl::desc("Scale on the SA placer's search budget"),
      llvm::cl::init(1.0)};
  Option<bool> dynamicObjFifos{*this, "dynamic-objFifos",
                               llvm::cl::desc("Dynamic objectFIFOs"),
                               llvm::cl::init(true)};
  Option<bool> packetSwObjFifos{*this, "packet-sw-objFifos",
                                llvm::cl::desc("Packet-switched objectFIFOs"),
                                llvm::cl::init(false)};
  Option<bool> ctrlPktOverlay{
      *this, "generate-ctrl-pkt-overlay",
      llvm::cl::desc("Route shim-to-tile control overlay"),
      llvm::cl::init(false)};
  Option<bool> loadPdiToCtrlPkt{
      *this, "load-pdi-to-ctrl-pkt",
      llvm::cl::desc("Load PDIs over control packets"), llvm::cl::init(false)};
  Option<bool> skipObjectFifoVerify{
      *this, "skip-objectFifo-verify",
      llvm::cl::desc("Skip structural verification of split objectFifo IR"),
      llvm::cl::init(false)};
  Option<int64_t> placementBudget{
      *this, "placement-budget",
      llvm::cl::desc("Buffer placements to try per tile before giving up"),
      llvm::cl::init(100000)};
  Option<bool> allowDeadlockProneRouting{
      *this, "allow-deadlock-prone-routing",
      llvm::cl::desc("Warn of, rather than fail on, deadlock-prone flows"),
      llvm::cl::init(false)};
};

void registerAIEResourceStagesPipeline();

} // namespace xilinx::AIEX

#endif // AIEX_PASSES_H
