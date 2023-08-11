//===- Passes.h - AIE Vector Passes -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// Register all the AIE vectorization passes
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_ANALYSIS_PASSES_H
#define AIE_DIALECT_AIEVEC_ANALYSIS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include <limits>

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace xilinx {
namespace aievec {

#define GEN_PASS_DECL
#define GEN_PASS_CLASSES
#include "aie/Dialect/AIEVec/Analysis/Passes.h.inc"

std::unique_ptr<mlir::Pass> createAIEVecConvolutionAnalysisPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aie/Dialect/AIEVec/Analysis/Passes.h.inc"

} // end namespace aievec
} // end namespace xilinx

#endif // AIE_DIALECT_AIEVEC_ANALYSIS_PASSES_H
