//===- AIEDialect.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIEX_DIALECT_H
#define MLIR_AIEX_DIALECT_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/IR/BuiltinAttributes.h"

// Include dialect declarations such as parseAttributes, parseType
#include "aie/Dialect/AIEX/IR/AIEXDialect.h.inc"
#include "aie/Dialect/AIEX/IR/AIEXEnums.h"
#include "mlir/IR/Operation.h"

// include TableGen generated Op definitions
#define GET_OP_CLASSES
#include "aie/Dialect/AIEX/IR/AIEX.h.inc"

#define GET_TYPEDEF_CLASSES
#include "aie/Dialect/AIEX/IR/AIEXTypes.h.inc"

#include "llvm/ADT/StringRef.h"

#include <optional>

namespace xilinx {
namespace AIEX {

void getHardwareStridesWraps(const AIE::AIETargetModel &targetModel,
                             mlir::Operation *op,
                             mlir::BaseMemRefType referencedBufType,
                             llvm::SmallVector<int64_t, 4> inputSizes,
                             llvm::SmallVector<int64_t, 4> inputStrides,
                             llvm::SmallVector<int64_t, 4> &sizes,
                             llvm::SmallVector<int64_t, 4> &strides);
mlir::LogicalResult
verifyStridesWraps(mlir::Operation *forOp,
                   mlir::BaseMemRefType referencedBufType, int tileCol,
                   int tileRow, llvm::SmallVector<int64_t, 4> inputSizes,
                   llvm::SmallVector<int64_t, 4> inputStrides,
                   llvm::SmallVector<int64_t, 4> hardwareSizes,
                   llvm::SmallVector<int64_t, 4> hardwareStrides,
                   bool skipTransformationChecks = false);
bool isLinearTransfer(llvm::ArrayRef<int64_t> sizes,
                      llvm::ArrayRef<int64_t> strides);

// Returns true when sizes/strides (innermost-first, same convention as
// NpuDmaMemcpyNdOp) describe a contiguous row-major scan:
//   strides[0] == 1
//   for i in 1..2: if sizes[i] > 1 then strides[i] == product of sizes[0..i-1]
// The repeat dimension (index 3) is excluded.
// Size-1 dimensions are allowed to carry any stride value because that stride
// is never applied during the transfer (the loop runs only once).
// This is the vector-form counterpart of AIE::isContiguousBDTransfer.
bool isContiguousTransfer(llvm::ArrayRef<int64_t> sizes,
                          llvm::ArrayRef<int64_t> strides);

// Folds an SSA value to a compile-time uint32_t if it is a constant integer,
// otherwise returns nullopt. Used by ops whose operands carry
// compile-time-known values in the static lowering path.
std::optional<uint32_t> getConstantIntOperand(mlir::Value v);

// Materializes a 32-bit integer constant. The inverse of getConstantIntOperand:
// used when lowering passes feed a compile-time-known value into an npu op
// whose operand is an SSA i32. The bit pattern (including high-bit values such
// as 0x80000000) is preserved.
mlir::Value createConstantI32(mlir::OpBuilder &builder, mlir::Location loc,
                              uint32_t value);

} // namespace AIEX
} // namespace xilinx

#endif
