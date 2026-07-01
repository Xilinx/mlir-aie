//===- Passes.h - AIE Vector Passes -----------------------------*- C++ -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "aie/Dialect/AIEVec/Analysis/Passes.h.inc"

std::unique_ptr<mlir::Pass> createAIEVecConvolutionAnalysisPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aie/Dialect/AIEVec/Analysis/Passes.h.inc"

} // end namespace aievec
} // end namespace xilinx

#endif // AIE_DIALECT_AIEVEC_ANALYSIS_PASSES_H
