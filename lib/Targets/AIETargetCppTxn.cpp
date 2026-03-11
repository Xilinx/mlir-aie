//===- AIETargetCppTxn.cpp - EmitC-based C++ TXN translation ------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Translates MLIR runtime sequences to compilable C++ code by:
// 1. Running the NPU lowering pipeline (same as aiecc) to lower high-level
//    DMA task ops to npu.write32/blockwrite/sync/address_patch
// 2. Running ConvertAIEXToEmitCPass to lower those to EmitC dialect
// 3. Calling translateToCpp() to emit C++ from EmitC IR
//
// The generated C++ code #includes TxnEncoding.h and calls its functions to
// build TXN instruction binaries at runtime.
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIETargets.h"

#include "aie/Conversion/AIEXToEmitC/AIEXToEmitC.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace xilinx {
namespace AIE {

LogicalResult AIETranslateToCppTxn(ModuleOp module, llvm::raw_ostream &output) {
  // Clone the module so we don't mutate the original.
  auto clonedModule = module.clone();
  auto *ctx = clonedModule.getContext();

  // Step 1: Run NPU lowering pipeline to lower high-level DMA task ops
  // (dma_configure_task_for, dma_start_task, dma_await_task, etc.)
  // down to npu.write32/blockwrite/sync/address_patch.
  // This mirrors the pipeline in aiecc.cpp.
  {
    PassManager pm(ctx);
    pm.addPass(AIEX::createAIEMaterializeRuntimeSequencesPass());
    OpPassManager &devicePm = pm.nest<AIE::DeviceOp>();
    devicePm.addPass(AIEX::createAIEMaterializeBDChainsPass());
    devicePm.addPass(AIEX::createAIESubstituteShimDMAAllocationsPass());
    devicePm.addPass(AIEX::createAIEAssignRuntimeSequenceBDIDsPass());
    devicePm.addPass(AIEX::createAIEDMATasksToNPUPass());
    devicePm.addPass(AIEX::createAIEDmaToNpuPass());
    devicePm.addPass(AIEX::createAIELowerSetLockPass());

    if (failed(pm.run(clonedModule))) {
      clonedModule->erase();
      return module.emitError("NPU lowering pipeline failed");
    }
  }

  // Step 2: Run the AIEX-to-EmitC conversion pass.
  {
    PassManager pm(ctx);
    pm.addPass(createConvertAIEXToEmitCPass());
    if (failed(pm.run(clonedModule))) {
      clonedModule->erase();
      return module.emitError("Failed to convert AIEX to EmitC");
    }
  }

  // Step 3: Translate EmitC IR to C++.
  auto result = emitc::translateToCpp(clonedModule, output,
                                      /*declareVariablesAtTop=*/false);
  clonedModule->erase();
  return result;
}

} // namespace AIE
} // namespace xilinx
