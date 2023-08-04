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
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "aievec-utils"

using namespace mlir;

namespace xilinx::aievec {
// Return the offset of a given transfer read operation with regards to the
// specified vector type. If the read is aligned to the specified alignment
// parameter (in bits), then the offset is 0. Otherwise, the offset is the
// number of elements past the immediately preceding aligned vector length.
template <typename TransferReadLikeOp, typename>
int64_t getTransferReadAlignmentOffset(TransferReadLikeOp readOp,
                                       VectorType vType, int64_t alignment) {
  // TODO: Add support for cases where the index is not comming from an
  // TODO: `affine.apply` op or when the affine map has more than one
  // TODO: dimension. We also need to address the case where the index is an
  // TODO: induction variable.
  auto innerMostIndex = readOp.getIndices().back();
  auto vectorLength = vType.getShape().back();
  auto idxDefOp = innerMostIndex.getDefiningOp();
  if (!idxDefOp)
    return 0L;
  int64_t vectorLengthAlignmentOffset =
      TypeSwitch<Operation *, int64_t>(idxDefOp)
          .Case<arith::ConstantOp>([&](auto constantOp) {
            return cast<IntegerAttr>(constantOp.getValue()).getInt() %
                   vectorLength;
          })
          .template Case<AffineApplyOp>([&](auto applyOp) {
            if (applyOp.getAffineMap().getNumDims() == 1)
              return applyOp.getAffineMap().compose(ArrayRef<int64_t>{0})[0] %
                     vectorLength;
            return 0L;
          })
          .Default([&](auto) {
            // XXX: If we can't determine the offset, we assume the access is
            // XXX: aligned.
            return 0L;
          });
  int64_t absoluteAlignmentOffset = alignment / vType.getElementTypeBitWidth();
  if (vectorLengthAlignmentOffset % absoluteAlignmentOffset)
    return vectorLengthAlignmentOffset;
  return 0;
}

template int64_t getTransferReadAlignmentOffset(vector::TransferReadOp readOp,
                                                VectorType vType,
                                                int64_t alignment);
template int64_t
getTransferReadAlignmentOffset(vector::TransferReadOp::Adaptor readOp,
                               VectorType vType, int64_t alignment);

} // namespace xilinx::aievec
