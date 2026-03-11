//===- AIENpuLowering.cpp - Shared NPU lowering pipeline --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIENpuLowering.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace xilinx::AIE {

void populateNpuLoweringPipeline(PassManager &pm, bool skipMaterialize) {
  if (!skipMaterialize)
    pm.addPass(AIEX::createAIEMaterializeRuntimeSequencesPass());

  OpPassManager &devicePm = pm.nest<DeviceOp>();
  devicePm.addPass(AIEX::createAIEMaterializeBDChainsPass());
  devicePm.addPass(AIEX::createAIESubstituteShimDMAAllocationsPass());
  devicePm.addPass(AIEX::createAIEAssignRuntimeSequenceBDIDsPass());
  devicePm.addPass(AIEX::createAIEDMATasksToNPUPass());
  devicePm.addPass(AIEX::createAIEDmaToNpuPass());
  devicePm.addPass(AIEX::createAIELowerSetLockPass());
}

} // namespace xilinx::AIE
