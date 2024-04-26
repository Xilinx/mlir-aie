//===- AIECoreToStandard.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Pass/Pass.h"

namespace xilinx::AIE {

struct CIRtoAIEidiomsPass : CIRtoAIEidiomsBase<CIRtoAIEidiomsPass> {
  void runOnOperation() override {

    //ModuleOp m = getOperation();
    return signalPassFailure();
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCIRtoAIEidiomsPass() {
  return std::make_unique<CIRtoAIEidiomsPass>();
}

} // namespace xilinx::AIE
