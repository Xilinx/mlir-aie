//===- AIEDialect.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIEX_DIALECT_H
#define MLIR_AIEX_DIALECT_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include <optional>

// Include dialect declarations such as parseAttributes, parseType
#include "aie/Dialect/AIEX/IR/AIEXDialect.h.inc"

// include TableGen generated Op definitions
#define GET_OP_CLASSES
#include "aie/Dialect/AIEX/IR/AIEX.h.inc"

namespace xilinx {
namespace AIEX {

uint64_t getBufferDescriptorAddressRegisterAddress(
    const AIE::AIETargetModel &tm, unsigned bd_id, unsigned col, unsigned row);
void getHardwareStridesWraps(const AIE::AIETargetModel &targetModel,
                             mlir::MemRefType referencedBufType,
                             llvm::SmallVector<int64_t, 4> inputSizes,
                             llvm::SmallVector<int64_t, 4> inputStrides,
                             llvm::SmallVector<int64_t, 4> &sizes,
                             llvm::SmallVector<int64_t, 4> &strides);
mlir::LogicalResult
verifyStridesWraps(mlir::Operation *forOp, mlir::MemRefType referencedBufType,
                   int tileCol, int tileRow,
                   llvm::SmallVector<int64_t, 4> inputSizes,
                   llvm::SmallVector<int64_t, 4> inputStrides,
                   llvm::SmallVector<int64_t, 4> hardwareSizes,
                   llvm::SmallVector<int64_t, 4> hardwareStrides,
                   bool skipTransformationChecks = false);

} // namespace AIEX
} // namespace xilinx

#endif
