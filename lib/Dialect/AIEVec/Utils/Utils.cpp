//===- Utils.cpp - Utilities to support AIE vectorization -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the AIEVec dialect
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "aievec-utils"

using namespace mlir;

namespace xilinx::aievec {

static std::optional<int64_t> getLowerBoundValue(Value idx) {
  if (auto blkArg = dyn_cast<BlockArgument>(idx)) {
    auto parentOp = blkArg.getOwner()->getParentOp();
    return TypeSwitch<Operation *, std::optional<int64_t>>(parentOp)
        .Case<affine::AffineForOp>([&blkArg](affine::AffineForOp forOp) {
          if (forOp.getInductionVar() == blkArg &&
              forOp.hasConstantLowerBound())
            return std::optional<int64_t>(forOp.getConstantLowerBound());
          // If it's an iteration argument or the lower bound is an
          // affine expression.
          // TODO: Compute the value of the lower bound affine expression
          // TODO: if it's constant.
          return std::optional<int64_t>();
        })
        .Default([](auto) { return std::optional<int64_t>(); });
  }
  return TypeSwitch<Operation *, std::optional<int64_t>>(idx.getDefiningOp())
      .Case<arith::ConstantOp>([](auto constantOp) {
        return std::optional<int64_t>(
            cast<IntegerAttr>(constantOp.getValue()).getInt());
      })
      .Case<affine::AffineApplyOp>([](auto applyOp) {
        if (applyOp.getAffineMap().getNumResults() == 1) {
          SmallVector<int64_t, 4> srcIndices;
          for (auto index : applyOp.getMapOperands()) {
            std::optional<int64_t> lbv = getLowerBoundValue(index);
            // XXX: We assume block arguments to either have well-defined
            // XXX: compile-time values, or to be aligned.
            if (!lbv && !isa<BlockArgument>(index))
              return std::optional<int64_t>();
            srcIndices.push_back(lbv.value_or(0L));
          }
          return std::optional<int64_t>(
              applyOp.getAffineMap().compose(srcIndices)[0]);
        }
        return std::optional<int64_t>();
      })
      .Default([&](auto) { return std::optional<int64_t>(); });
}

// Return the offset of a given transfer read operation with regards to the
// specified vector type. If the read is aligned to the specified alignment
// parameter (in bits), then the offset is 0. Otherwise, the offset is the
// number of elements past the immediately preceding aligned vector length.
template <typename TransferReadLikeOp, typename>
std::optional<int64_t> getTransferReadAlignmentOffset(TransferReadLikeOp readOp,
                                                      VectorType vType,
                                                      int64_t alignment) {
  // TODO: Add support for cases where the index is not comming from an
  // TODO: `affine.apply` op or when the affine map has more than one
  // TODO: dimension. We also need to address the case where the index is an
  // TODO: induction variable.
  auto innerMostIndex = readOp.getIndices().back();
  auto vectorLength = vType.getShape().back();
  std::optional<int64_t> lbv = getLowerBoundValue(innerMostIndex);
  if (!lbv)
    return std::nullopt;
  int64_t vectorLengthAlignmentOffset = lbv.value() % vectorLength;
  int64_t absoluteAlignmentOffset = alignment / vType.getElementTypeBitWidth();
  if (vectorLengthAlignmentOffset % absoluteAlignmentOffset)
    return vectorLengthAlignmentOffset;
  return 0;
}

template std::optional<int64_t>
getTransferReadAlignmentOffset(vector::TransferReadOp readOp, VectorType vType,
                               int64_t alignment);
template std::optional<int64_t>
getTransferReadAlignmentOffset(vector::TransferReadOp::Adaptor readOp,
                               VectorType vType, int64_t alignment);

} // namespace xilinx::aievec
