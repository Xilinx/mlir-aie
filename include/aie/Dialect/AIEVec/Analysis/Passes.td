//=== Passes.td - AIE vector analysis pass definition file -*- tablegen -*-===//
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains definitions for passes within the AIEVec/ directory.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_ANALYSIS_PASSES
#define AIE_DIALECT_AIEVEC_ANALYSIS_PASSES

include "mlir/Pass/PassBase.td"

def AIEVecConvAnalysis : Pass<"aievec-convolution-analysis"> {
  let summary = "Find MAC chains that can be replaced by convolution ops in "
                "AIE2";
  let constructor = "xilinx::aievec::createAIEVecConvolutionAnalysisPass()";
  let options = [
    Option<"printResult", "print", "bool", /*default=*/"false",
      "Print the result of the analysis">,
  ];
}

#endif // AIE_DIALECT_AIEVEC_ANALYSIS_PASSES
