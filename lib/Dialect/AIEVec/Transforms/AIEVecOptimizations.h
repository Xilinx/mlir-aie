//===----AIEVecOptimizations.h - Optimize AIEVec operations.---------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_AIEVECOPTIMIZATIONS_H
#define AIE_DIALECT_AIEVEC_AIEVECOPTIMIZATIONS_H

#include "aie/Dialect/AIEVec/Pipelines/Passes.h"

namespace xilinx {
namespace aievec {

/// Build a pipeline to replace simple AIEVec operations with complex, more
/// performant AIEVec operations.
void buildOptimizeAIEVec(mlir::OpPassManager &pm,
                         const OptimizeAIEVecOptions &options);

} // namespace aievec
} // namespace xilinx

#endif // AIE_DIALECT_AIEVEC_AIEVECOPTIMIZATIONS_H
