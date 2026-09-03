//===- AIEXNpuPipelines.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace xilinx;

void xilinx::AIEX::buildNpuDmaLoweringPipeline(OpPassManager &pm) {
  OpPassManager &dpm = pm.nest<AIE::DeviceOp>();
  dpm.addPass(createAIEMaterializeBDChainsPass());
  dpm.addPass(createAIESubstituteShimDMAAllocationsPass());
  dpm.addPass(createAIEUnrollRuntimeSequenceLoopsPass());
  dpm.addPass(createCanonicalizerPass());
  // Decompose oversized non-contiguous ND transfers (wrap/stride exceeding the
  // hardware BD field limits) into legal sub-transfers before BD lowering.
  dpm.addPass(createAIEDecomposeLargeDmaBdPass());
  // A runtime-bound scf.for that survived unroll takes the dynamic BD pool path
  // (rewritten to pool pop/push, ids drawn at runtime); the static allocator
  // below skips it. Straight-line sequences fall through unchanged.
  dpm.addPass(createAIELowerDynamicBDPoolPass());
  dpm.addPass(createCanonicalizerPass());
  dpm.addPass(createAIEAssignRuntimeSequenceBDIDsPass());
  dpm.addPass(createAIEDMATasksToNPUPass());
  // Expands dma_channel_reset_for into its re-arm trio and lowers the
  // dma_channel_reset ops to maskwrite32. Must precede aie-dma-to-npu and
  // aie-lower-set-lock, which lower the push_queue and set_lock ops it emits.
  dpm.addPass(createAIELowerDmaChannelResetPass());
  dpm.addPass(createAIEDmaToNpuPass());
  dpm.addPass(createAIELowerSetLockPass());
  dpm.addPass(createAIELowerCoreResetPass());
}

void xilinx::AIEX::registerAIEXNpuPipelines() {
  PassPipelineRegistration<>(
      "aie-npu-dma-lowering",
      "Lower materialized runtime sequences to aiex.npu.* instructions",
      buildNpuDmaLoweringPipeline);
}
