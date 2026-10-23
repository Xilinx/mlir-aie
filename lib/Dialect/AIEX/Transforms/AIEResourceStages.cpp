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
                            bool loadPdiToCtrlPkt, bool skipObjectFifoVerify) {
  pm.addPass(createLowerAffinePass());
  pm.addPass(createAIECanonicalizeDevicePass());
  // Before lock IDs and buffer addresses, which its new locks and buffers need.
  pm.addPass(createAIELowerScratchpadParametersPass());

  // With the overlay requested, it claims its shim DMA channels before
  // objectFIFO lowering works around them; otherwise it runs after tile
  // control IDs are assigned.
  AIEGenerateColumnControlOverlayOptions overlay;
  overlay.clRouteShimDmaToTileCTRL = ctrlPktOverlay;
  overlay.clEmitStandaloneOverlay = ctrlPktOverlay && loadPdiToCtrlPkt;
  if (ctrlPktOverlay)
    pm.addPass(createAIEGenerateColumnControlOverlayPass(overlay));

  OpPassManager &dpm = pm.nest<DeviceOp>();
  addObjectFifoLoweringPasses(dpm, packetSwObjFifos, skipObjectFifoVerify);
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
  dpm.addPass(createAIEAssignTileCtrlIDsPass());

  if (!ctrlPktOverlay)
    pm.addPass(createAIEGenerateColumnControlOverlayPass(overlay));

  // A buffer's name becomes a symbol in its core's object.
  pm.nest<DeviceOp>().addPass(createAIEPrepareBuffersPass());
}

void AIEX::addBufferAddressStage(OpPassManager &pm, int64_t placementBudget) {
  AIEAssignBufferAddressesOptions options;
  options.clPlacementBudget = placementBudget;
  pm.nest<DeviceOp>().addPass(createAIEAssignBufferAddressesPass(options));
}

void AIEX::addRoutingStage(OpPassManager &pm, bool allowDeadlockProne) {
  OpPassManager &dpm = pm.nest<DeviceOp>();
  dpm.addPass(createAIESplitFlowViasPass());
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
        addPlacementStage(pm, place);
        addTraceStage(pm);
        addLoweringStage(pm, options.dynamicObjFifos, options.packetSwObjFifos,
                         options.ctrlPktOverlay || options.loadPdiToCtrlPkt,
                         options.loadPdiToCtrlPkt,
                         options.skipObjectFifoVerify);
        addBufferAddressStage(pm, options.placementBudget);
        addRoutingStage(pm, options.allowDeadlockProneRouting);
      });
}
