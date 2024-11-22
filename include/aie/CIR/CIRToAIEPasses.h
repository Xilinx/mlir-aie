//===- CIRToAIEpasses.h ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//===----------------------------------------------------------------------===//

#ifndef CIR_AIE_PASSES_H
#define CIR_AIE_PASSES_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "mlir/Pass/Pass.h"

namespace xilinx::AIE::CIR {

#define GEN_PASS_CLASSES
#include "aie/CIR/CIRToAIEPasses.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCIRToAIEPreparePass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCIRToAIEPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCIRToAIEInlineKernelLambdaPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCIRToAIEDecaptureKernelPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aie/CIR/CIRToAIEPasses.h.inc"

}

#endif
