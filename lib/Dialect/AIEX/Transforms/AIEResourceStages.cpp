//===- AIEResourceStages.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

void AIEX::addPlacementStage(OpPassManager &pm,
                             const AIEPlaceTilesOptions &options) {
  pm.nest<DeviceOp>().addPass(createAIEPlaceTilesPass(options));
}

void AIEX::addTraceStage(OpPassManager &pm) {
  OpPassManager &dpm = pm.nest<DeviceOp>();
  dpm.addPass(createAIEInsertTraceFlowsPass());
  dpm.addPass(createAIETraceToConfigPass());
  dpm.addPass(createAIETraceRegPackWritesPass());
  dpm.addPass(createAIEXInlineTraceConfigPass());
  // Module-level: it rewrites callers and callees together.
  pm.addPass(createAIEFuseTraceBuffersPass());
}

void AIEX::addLoweringStage(OpPassManager &pm, bool dynamicObjFifos,
                            bool packetSwObjFifos, bool ctrlPktOverlay,
                            bool loadPdiToCtrlPkt, bool skipObjectFifoVerify,
                            llvm::ArrayRef<int> coverColumns,
                            bool adaptiveOverlay, bool checkQueueSpace) {
  pm.addPass(createLowerAffinePass());
  pm.addPass(createAIECanonicalizeDevicePass());
  // The verifier checks a parameter-waiting Worker's fifo volume per dispatch
  // from the aiex.wait_scratchpad_parameters and parameter syncs that the
  // scratchpad lowering below replaces. Split is idempotent, so the lowering
  // after it splits again and does not verify twice.
  if (!skipObjectFifoVerify) {
    OpPassManager &verify = pm.nest<DeviceOp>();
    verify.addPass(createAIEObjectFifoSplitPass());
    verify.addPass(createAIEObjectFifoVerifyPass());
  }
  // Before lock IDs and buffer addresses, which its new locks and buffers need.
  pm.addPass(createAIELowerScratchpadParametersPass());

  pm.nest<DeviceOp>().addPass(createAIEObjectFifoSplitPass());
  AIEGenerateColumnControlOverlayOptions overlay;
  overlay.configuration = ctrlPktOverlay || loadPdiToCtrlPkt;
  overlay.adaptive = adaptiveOverlay;
  overlay.coverColumns = llvm::to_vector(coverColumns);
  pm.addPass(createAIEGenerateColumnControlOverlayPass(overlay));
  pm.addPass(createAIEImportOverlayPass());
  pm.addPass(createAIEAnnotateShimSharingPass());
  OpPassManager &dpm = pm.nest<DeviceOp>();
  dpm.addPass(createAIEAssignPacketIdsPass(packetSwObjFifos, ctrlPktOverlay));
  dpm.addPass(createAIEObjectFifoAllocatePass());
  AIEObjectFifoInsertCoreQueuePushesOptions pushOptions;
  pushOptions.checkQueueSpace = checkQueueSpace;
  dpm.addPass(createAIEObjectFifoInsertCoreQueuePushesPass(pushOptions));
  dpm.addPass(createAIEObjectFifoLowerDMAsPass());
  dpm.addPass(createAIEObjectFifoLowerCoresPass());
  dpm.addPass(createAIEObjectFifoErasePoolsPass());
  AIEObjectFifoUnrollOptions unroll;
  unroll.clDefaultDynamic = dynamicObjFifos;
  dpm.addPass(createAIEObjectFifoUnrollPass(unroll));
  dpm.addPass(createAIENormalizeDmaBdDimsPass());
  dpm.addPass(createAIEAssignLockIDsPass());
  dpm.addPass(createAIEReserveRuntimeBDIDsPass());
  dpm.addPass(createAIEAssignBufferDescriptorIDsPass());
  dpm.addPass(createAIELowerCoreIssuedConfigurationPass());
  dpm.addPass(createAIELowerCascadeFlowsPass());
  dpm.addPass(createAIEBroadcastPacketPass());
  dpm.addPass(createAIELowerMulticastPass());
  pm.nest<DeviceOp>().addPass(createAIEAssignTileCtrlIDsPass());

  // A buffer's name becomes a symbol in its core's object.
  pm.nest<DeviceOp>().addPass(createAIEPrepareBuffersPass());
}

void AIEX::addBufferAddressStage(OpPassManager &pm, int64_t placementBudget) {
  AIEAssignBufferAddressesOptions options;
  options.clPlacementBudget = placementBudget;
  pm.nest<DeviceOp>().addPass(createAIEAssignBufferAddressesPass(options));
}

void AIEX::addRoutingStage(OpPassManager &pm, bool allowDeadlockProne) {
  pm.nest<DeviceOp>().addPass(createAIESplitFlowViasPass());
  OpPassManager &dpm = pm.nest<DeviceOp>();
  AIERoutePathfinderFlowsOptions options;
  options.clAllowDeadlockProne = allowDeadlockProne;
  dpm.addPass(createAIEPathfinderPass(options));
}

void AIEX::registerAIEResourceStagesPipeline() {
  PassPipelineRegistration<AIEResourceStagesOptions>(
      "aie-resource-stages",
      "aiecc's placement, trace, lowering, buffer address and routing stages",
      [](OpPassManager &pm, const AIEResourceStagesOptions &options) {
        AIEPlaceTilesOptions place;
        place.clPlacerType = options.placer;
        place.clCoresPerCol = options.coresPerCol;
        place.clSASeed = options.saSeed;
        place.clSAEffort = options.saEffort;
        place.clCtrlPktOverlay =
            options.ctrlPktOverlay || options.loadPdiToCtrlPkt;
        addPlacementStage(pm, place);
        addTraceStage(pm);
        addLoweringStage(pm, options.dynamicObjFifos, options.packetSwObjFifos,
                         options.ctrlPktOverlay || options.loadPdiToCtrlPkt,
                         options.loadPdiToCtrlPkt, options.skipObjectFifoVerify,
                         {}, options.adaptiveOverlay);
        addBufferAddressStage(pm, options.placementBudget);
        addRoutingStage(pm, options.allowDeadlockProneRouting);
      });
}
