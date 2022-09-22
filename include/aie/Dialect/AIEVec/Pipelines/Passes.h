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
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Pass/PassOptions.h"

namespace xilinx {
namespace aievec {

/// Options for the "aie-affine-vectorize" pipeline. The pass, as it is, only
/// accepts a small subset of the parameters for all the sub-passes
struct AIEAffineVectorizeOptions
    : public PassPipelineOptions<AIEAffineVectorizeOptions> {
  // Affine supervectorizer options

  // AIE vectorize options. Keep in sync with AIEVectorizePass options
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
      *this, "dup-factor",
      llvm::cl::desc("Duplication factor for each value in convolution filter "
                     "(useful for 8x8 scheme)"),
      llvm::cl::init(2)};

  /// Project out the options for `createAIEVectorizePass`
  AIEVectorizeOptions getAIEVectorizeOptions() const {
    return AIEVectorizeOptions{shiftParam, zeroOffset, dupFactor};
  }
};

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the "aie-affine-vectorize" pipeline to the `OpPassManager`. This
/// is pipeline takes an affine scalar code, vectorizes it, and lowers it to
/// the AIE vector representation.
void buildAIEAffineVectorizer(OpPassManager &pm,
                              const AIEAffineVectorizeOptions &options);

/// Registers all pipelines for the AIE Vector dialect.
void registerAIEVecPipelines();

} // end namespace aievec
} // end namespace xilinx

#endif // AIE_DIALECT_AIEVEC_PIPELINES_PASSES_H
