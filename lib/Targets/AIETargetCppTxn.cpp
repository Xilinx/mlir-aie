//===- AIETargetCppTxn.cpp - Generate C++ TXN builder ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
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

using namespace mlir;

LogicalResult xilinx::AIE::AIETranslateNpuToCpp(ModuleOp module,
                                                raw_ostream &output) {
  PassManager pm(module.getContext());
  pm.addPass(xilinx::createConvertAIEXToEmitCPass());
  if (failed(pm.run(module)))
    return failure();
  return emitc::translateToCpp(module, output);
}
