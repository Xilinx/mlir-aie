//===- BFP16MatMulOperands.h - Operands of bfp16 matmuls --------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_TRANSFORMS_BFP16MATMULOPERANDS_H
#define AIE_DIALECT_AIEVEC_TRANSFORMS_BFP16MATMULOPERANDS_H

#include "mlir/IR/PatternMatch.h"

namespace xilinx::aievec {

// Converts the bf16 tiles of a buffer that a loop does not modify to bfp16
// once, before the loop, and makes the loop load the converted blocks.
void populateHoistBFP16ConversionPatterns(mlir::RewritePatternSet &patterns);

// Makes matmuls load bfp16 blocks brought by DMA (aievec.load_bfp16) instead
// of the bf16 tiles that a core decodes from them, and removes the decode.
void populateReadDecodedBFP16Patterns(mlir::RewritePatternSet &patterns);

// Turns loads of consecutive bfp16 blocks in a loop into one block-vector
// load stream.
void populateBFP16LoadStreamPatterns(mlir::RewritePatternSet &patterns);

} // namespace xilinx::aievec

#endif // AIE_DIALECT_AIEVEC_TRANSFORMS_BFP16MATMULOPERANDS_H
