//===--FoldMulAddChainToConvOp.h - Fold Mul Add Chain To AIEVec Conv Op --===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.

#ifndef FOLDMULADDCHAINTOCONVOP_H
#define FOLDMULADDCHAINTOCONVOP_H

#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace xilinx {

enum class TargetBackend;

namespace aievec {
//===----------------------------------------------------------------------===//
// This is the implementation of the folding pass from mul add chain
// to AIEVec convolution operations, compatible with the AIE-ML architecture.
//===----------------------------------------------------------------------===//

// Configure the legalizations for aievec conv op transformation
void configureAIEVecConvOpTransformationLegalizations(
    mlir::ConversionTarget &target, mlir::AnalysisManager &am,
    TargetBackend backend);

// Populate the conversion pattern by FoldMulAddChainToConvOpPattern, which
// folds a mul add chain into mul_conv and mac_conv.
void populateAIEVecConvOpTransformationPatterns(
    mlir::RewritePatternSet &patterns, mlir::AnalysisManager &am,
    unsigned shiftParam, TargetBackend backend);

} // namespace aievec
} // namespace xilinx

#endif // FOLDMULADDCHAINTOCONVOP_H
