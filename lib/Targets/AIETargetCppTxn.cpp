//===- AIETargetCppTxn.cpp - Generate C++ TXN builder ----------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// aie-translate target that lowers an aie.runtime_sequence's npu transaction
// ops into a standalone C++ function (via convert-aiex-to-emitc +
// translateToCpp). The input is expected to be already lowered to npu ops, the
// same precondition as the aie-npu-to-binary target; this is its
// runtime-parameterizable C++ counterpart.
//
//===----------------------------------------------------------------------===//

#include "aie/Conversion/AIEXToEmitC/AIEXToEmitC.h"
#include "aie/Targets/AIETargets.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

LogicalResult xilinx::AIE::AIETranslateNpuToCpp(ModuleOp module,
                                                raw_ostream &output,
                                                bool foldDDRAddrOffset,
                                                bool emitDispatchShim) {
  PassManager pm(module.getContext());
  // A staged sequence repeats the same guard arithmetic at every use of a
  // scalar (each tap re-derives and re-checks its shape). CSE merges the
  // duplicated conditions and canonicalization then drops the repeated
  // requires, so the builder checks each constraint once.
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(xilinx::createConvertAIEXToEmitCPass(foldDDRAddrOffset,
                                                  emitDispatchShim));
  if (failed(pm.run(module)))
    return failure();
  return emitc::translateToCpp(module, output);
}
