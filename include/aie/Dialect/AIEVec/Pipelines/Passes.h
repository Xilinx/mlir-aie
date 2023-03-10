//===- Passes.h - AIE Vector pipeline entry points --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all AIE vector pipelines.
//
//===----------------------------------------------------------------------===//
#ifndef AIE_DIALECT_AIEVEC_PIPELINES_PASSES_H
#define AIE_DIALECT_AIEVEC_PIPELINES_PASSES_H

#include "aie/Dialect/AIEVec/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"

namespace xilinx {
namespace aievec {

/// Options for the "convert-vector-to-aievec" pipeline.
struct ConvertVectorToAIEVecOptions
    : public PassPipelineOptions<ConvertVectorToAIEVecOptions> {
  // 'LowerVectorToAIEVec' options
  // TODO: Review the need for these options.
  PassOptions::Option<unsigned> shiftParam{
      *this, "shift",
      llvm::cl::desc("Shift parameter for rounding and saturation"),
      llvm::cl::init(0)};
  PassOptions::Option<unsigned> zeroOffset{
      *this, "zero-offset",
      llvm::cl::desc("Zero offset for indicating the location of zeroes in "
                     "convolution filter (useful for 16x16 scheme)"),
      llvm::cl::init(0)};
  PassOptions::Option<unsigned> dupFactor{
      *this, "dup-actor",
      llvm::cl::desc("Duplication factor for each value in convolution filter "
                     "(useful in 8x8 scheme)"),
      llvm::cl::init(2)};
  PassOptions::Option<std::string> aieTarget{
      *this, "aie-target",
      llvm::cl::desc("Select AIE version: \"aie\" or \"aieml\". This will "
                     "determine the vector size and available operations."),
      llvm::cl::init("aie")};

  LowerVectorToAIEVecOptions getLowerVectorToAIEVecOptions() const {
    return LowerVectorToAIEVecOptions{shiftParam, zeroOffset, dupFactor,
                                      aieTarget};
  }

  CanonicalizeForAIEVecOptions getCanonicalizeForAIEVecOptions() const {
    return CanonicalizeForAIEVecOptions{aieTarget};
  }

  AIEVecTransformationOptions getAIEVecTransformationOptions() const {
    return AIEVecTransformationOptions{aieTarget};
  }
};

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the "convert-vector-to-aievec" pipeline to the `OpPassManager`. This
/// pipeline takes `Vector` code, transforms it to make it compatible with the
/// selected `AIE` target, and lowers it to `AIEVec` dialect.
void buildConvertVectorToAIEVec(OpPassManager &pm,
                                const ConvertVectorToAIEVecOptions &options);

/// Register all pipelines for the AIE Vector dialect.
void registerAIEVecPipelines();

} // namespace aievec
} // namespace xilinx

#endif // AIE_DIALECT_AIEVEC_PIPELINES_PASSES_H
