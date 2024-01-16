//===- Utils.h - Utilities to support AIE vectorization ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_UTILS_UTILS_H
#define AIE_DIALECT_AIEVEC_UTILS_UTILS_H

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include <cstdint>
#include <optional>
#include <type_traits>

namespace llvm {

class CallInst;
class IRBuilderBase;
class StringRef;

} // namespace llvm

namespace mlir {

class AffineExpr;
class AffineForOp;
class AffineMap;
class Operation;

namespace LLVM {
class ModuleTranslation;
} // namespace LLVM

} // namespace mlir

namespace xilinx::aievec {

template <
    typename TransferReadLikeOp,
    typename = std::enable_if_t<
        std::is_same_v<TransferReadLikeOp, mlir::vector::TransferReadOp> ||
        std::is_same_v<TransferReadLikeOp,
                       mlir::vector::TransferReadOp::Adaptor>>>
std::optional<int64_t> getTransferReadAlignmentOffset(TransferReadLikeOp readOp,
                                                      mlir::VectorType vType,
                                                      int64_t alignment);

llvm::CallInst *
createExternalIntrinsicCall(llvm::IRBuilderBase &builder,
                            mlir::LLVM::ModuleTranslation &moduleTranslation,
                            mlir::Operation *intrOp,
                            llvm::StringRef intrinsicName);

} // namespace xilinx::aievec

#endif // AIE_DIALECT_AIEVEC_UTILS_UTILS_H
