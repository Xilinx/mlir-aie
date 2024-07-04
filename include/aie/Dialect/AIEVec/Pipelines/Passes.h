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

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace xilinx {
enum class AIEArch {
  AIE,   // Original AIE
  AIE2,  // AIE-ML/V2 arch version of AIE
  AIE2P, // AIE2P arch version of AIE
};
enum class TargetBackend {
  CPP,    // Convert to aievec targeting C++ backend
  LLVMIR, // Convert to aievec targeting LLVM IR backend
};
} // namespace xilinx

namespace xilinx {
namespace aievec {

// TODO: Create a common base class for all AIEVec pipeline options.

/// Options for the "canonicalize-vector-for-aievec" pipeline.
struct CanonicalizeVectorForAIEVecOptions
    : public mlir::PassPipelineOptions<CanonicalizeVectorForAIEVecOptions> {
  PassOptions::Option<std::string> aieTarget{
      *this, "aie-target",
      llvm::cl::desc("Select AIE version: \"aie\" or \"aie2\". This will "
                     "determine the vector size and available operations."),
      llvm::cl::init("aie")};
  PassOptions::Option<std::string> targetBackend{
      *this, "target-backend",
      llvm::cl::desc("Select translation backend: \"cpp\" or \"llvmir\". This "
                     "will determine the aievec operations used to convert "
                     "from vector dialect."),
      llvm::cl::init("cpp")};
};

/// Options for the "lower-vector-to-aievec" pipeline.
struct LowerVectorToAIEVecOptions
    : public mlir::PassPipelineOptions<LowerVectorToAIEVecOptions> {
  PassOptions::Option<std::string> aieTarget{
      *this, "aie-target",
      llvm::cl::desc("Select AIE version: \"aie\" or \"aie2\". This will "
                     "determine the vector size and available operations."),
      llvm::cl::init("aie")};
  PassOptions::Option<std::string> targetBackend{
      *this, "target-backend",
      llvm::cl::desc("Select translation backend: \"cpp\" or \"llvmir\". This "
                     "will determine the aievec operations used to convert "
                     "from vector dialect."),
      llvm::cl::init("cpp")};
};

/// Options for the "optimize-aievec" pipeline.
struct OptimizeAIEVecOptions
    : public mlir::PassPipelineOptions<OptimizeAIEVecOptions> {
  PassOptions::Option<std::string> aieTarget{
      *this, "aie-target",
      llvm::cl::desc("Select AIE version: \"aie\" or \"aie2\". This will "
                     "determine the vector size and available operations."),
      llvm::cl::init("aie")};
  PassOptions::Option<std::string> targetBackend{
      *this, "target-backend",
      llvm::cl::desc("Select translation backend: \"cpp\" or \"llvmir\". This "
                     "will determine the aievec operations used to convert "
                     "from vector dialect."),
      llvm::cl::init("cpp")};
  PassOptions::Option<unsigned> shiftParam{
      *this, "shift",
      llvm::cl::desc("Shift parameter for rounding and saturation"),
      llvm::cl::init(0)};
};

/// Options for the "convert-vector-to-aievec" pipeline.
struct ConvertVectorToAIEVecOptions
    : public mlir::PassPipelineOptions<ConvertVectorToAIEVecOptions> {
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
      llvm::cl::desc("Select AIE version: \"aie\" or \"aie2\". This will "
                     "determine the vector size and available operations."),
      llvm::cl::init("aie")};
  PassOptions::Option<std::string> targetBackend{
      *this, "target-backend",
      llvm::cl::desc("Select translation backend: \"cpp\" or \"llvmir\". This "
                     "will determine the aievec operations used to convert "
                     "from vector dialect."),
      llvm::cl::init("cpp")};

  mlir::LogicalResult parseFromString(mlir::StringRef options) {
    auto res = PassPipelineOptions::parseFromString(options);
    if (!failed(res)) {
      lowerOptions.aieTarget = aieTarget;
      lowerOptions.targetBackend = targetBackend;
      canonicalizeOptions.aieTarget = aieTarget;
      canonicalizeOptions.targetBackend = targetBackend;
      optimizeOptions.aieTarget = aieTarget;
      optimizeOptions.targetBackend = targetBackend;
      optimizeOptions.shiftParam = shiftParam;
    }
    return res;
  }

  const LowerVectorToAIEVecOptions &getLowerVectorToAIEVecOptions() const {
    return lowerOptions;
  }

  const CanonicalizeVectorForAIEVecOptions &
  getCanonicalizeVectorForAIEVecOptions() const {
    return canonicalizeOptions;
  }

  const OptimizeAIEVecOptions &getOptimizeAIEVecOptions() const {
    return optimizeOptions;
  }

private:
  CanonicalizeVectorForAIEVecOptions canonicalizeOptions;
  LowerVectorToAIEVecOptions lowerOptions;
  OptimizeAIEVecOptions optimizeOptions;
};

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the "convert-vector-to-aievec" pipeline to the `OpPassManager`. This
/// pipeline takes `Vector` code, transforms it to make it compatible with the
/// selected `AIE` target, lowers it to `AIEVec` dialect, and performs some
/// optimizations based on the target AIE architecture.
void buildConvertVectorToAIEVec(mlir::OpPassManager &pm,
                                const ConvertVectorToAIEVecOptions &options);

void buildCanonicalizeVectorForAIEVec(
    mlir::OpPassManager &pm, const CanonicalizeVectorForAIEVecOptions &options);

void buildLowerVectorToAIEVec(mlir::OpPassManager &pm,
                              const LowerVectorToAIEVecOptions &options);

void buildOptimizeAIEVec(mlir::OpPassManager &pm,
                         const OptimizeAIEVecOptions &options);

/// Register all pipelines for the AIE Vector dialect.
void registerAIEVecPipelines();

/// Create a pass that removes unnecessary Copy operations.
std::unique_ptr<::mlir::Pass> createCopyRemovalPass();

// Create a pass that rewrites the arith dialect to enable the support of
// dynamic sized tensor/memref for the auto-vectorization to CPP flow.
std::unique_ptr<::mlir::Pass> createDynamicSizeNoImplicitBroadcastPass();

// Build a pipeline for CLI access to the pass
// `dynamic-size-no-implicit-broadcast`
void buildDynamicSizeNoImplicitBroadcastPass(mlir::OpPassManager &pm);

} // namespace aievec
} // namespace xilinx

#endif // AIE_DIALECT_AIEVEC_PIPELINES_PASSES_H
