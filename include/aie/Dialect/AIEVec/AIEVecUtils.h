//===- AIEVecUtils.h - AIE Vector Utility Operations ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// Utility functions for AIE vectorization
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_AIEVECUTILS_H
#define AIE_DIALECT_AIEVEC_AIEVECUTILS_H

#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "aie/Dialect/AIEVec/IR/AIEVecTypes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include <assert.h>

namespace xilinx {
namespace aievec {

// For input val, return its value in hex. Since we currently support each
// offset value to be only 4 bits, the val must be < 16
inline char getHexValue(int val) {
  assert(val >= 0 && val < 16);
  if (val <= 9)
    return '0' + val;
  return 'A' + (val - 10);
}

// Return true if the number is a power of 2
inline bool isPowerOfTwo(int32_t n) { return (n & (n - 1)) == 0; }

// Create a vector type, given the lanes and underlying element type
inline VectorType createVectorType(unsigned lanes, Type elementType) {
  SmallVector<int64_t, 4> vecShape = {lanes};
  return VectorType::get(vecShape, elementType);
}

// Return the size (in bits) of the underlying element type of the vector
inline int32_t getElementSizeInBits(VectorType type) {
  return type.cast<ShapedType>().getSizeInBits() / type.getNumElements();
}

// Return the number of lanes along the vectorized dimension for the vector
// type. For a multidimensional vector, return the innermost dimension size
inline unsigned getVectorLaneSize(VectorType type) {
  assert(type.getRank() > 0 && "Cannot handle rank-0 vectors");
  auto dimSize = type.getDimSize(type.getRank() - 1);
  assert(dimSize >= 0 && "Vector dimension cannot be negative");
  return std::max(1u, (unsigned)dimSize);
}

// For a 1D vector, return its size in bits. For an nD vector, return the size
// of the innerost dimension in bits.
inline int32_t getVectorSizeInBits(VectorType type) {
  int32_t veclen = getVectorLaneSize(type) * getElementSizeInBits(type);
  assert(veclen >= 128 && "AIE vector size should be greater than 128 bits");
  return veclen;
}

// Return true if this is an operation defined in AIE dialect
inline bool isAIEOp(Operation *op) {
  return llvm::isa<AIEVecDialect>(op->getDialect());
}

// Determine the output type for a vector operation based on whether
// it operates on integer or floating point data.
inline VectorType getVectorOpDestType(VectorType type, bool AIEML) {
  Type stype = type.getElementType();

  if (IntegerType itype = stype.dyn_cast<IntegerType>()) {
    // Integer vector types are sized for the appropriate accumulators
    assert(itype.getWidth() <= 64);
    unsigned width = 0;
    if (AIEML) {
      width = itype.getWidth() <= 16 ? 32 : 64;
    } else {
      width = itype.getWidth() <= 16 ? 48 : 80;
    }

    Type ctype = mlir::IntegerType::get(itype.getContext(), width);
    return VectorType::get(type.getShape(), ctype);
  } else if (FloatType ftype = stype.dyn_cast<FloatType>()) {
    if (AIEML && ftype.getWidth() == 16) {
      return VectorType::get(type.getShape(), ftype.getF32(ftype.getContext()));
    }

    // Floating point vector types for aie1 are returned as is since the
    // floating point operations write back to registers and not accumulators
    return type;
  } else
    llvm_unreachable("Unsupported destination type");
}

// Linearize the exprVec as a strided access, but do not simplify
inline AffineExpr flattenedStridedExpr(ArrayRef<int64_t> sizes,
                                       ArrayRef<AffineExpr> exprs,
                                       MLIRContext *context) {
  // Expect non-empty sizes and exprs
  if (sizes.empty() || exprs.empty())
    return nullptr;

  if (llvm::is_contained(sizes, 0))
    return getAffineConstantExpr(0, context);

  auto maps = AffineMap::inferFromExprList(exprs);
  if (maps.empty()) {
    return nullptr;
  }

  unsigned nSymbols = maps[0].getNumSymbols();

  AffineExpr expr;
  bool dynamicPoisonBit = false;
  int64_t runningSize = 1;
  for (auto en : llvm::zip(llvm::reverse(exprs), llvm::reverse(sizes))) {
    int64_t size = std::get<1>(en);

    if (size == 0)
      continue;
    AffineExpr dimExpr = std::get<0>(en);
    AffineExpr stride = dynamicPoisonBit
                            ? getAffineSymbolExpr(nSymbols++, context)
                            : getAffineConstantExpr(runningSize, context);
    expr = expr ? expr + dimExpr * stride : dimExpr * stride;
    if (size > 0) {
      runningSize *= size;
      if (runningSize <= 0) {
        return nullptr;
      }
    } else {
      dynamicPoisonBit = true;
    }
  }
  return expr;
}

// Construct a linearized affine expression for the upd op.
inline AffineExpr constructLinearizedAffineExprForUPDOp(aievec::UPDOp updOp) {
  SmallVector<Value, 4> indices(updOp.getIndices().begin(),
                                updOp.getIndices().end());
  MemRefType memRefType = updOp.getSource().getType().cast<MemRefType>();
  MLIRContext *context = memRefType.getContext();

  SmallVector<AffineExpr, 8> exprVec;
  DenseMap<Value, AffineExpr> indexToExprDimMap;
  for (auto idxAndValue : llvm::enumerate(indices)) {
    auto value = idxAndValue.value();
    if (AffineApplyOp apOf = value.getDefiningOp<AffineApplyOp>()) {
      AffineMap map = apOf.getAffineMap();
      // Cannot create linearized affineExpr for complicated index.
      if (map.getNumResults() != 1) {
        return nullptr;
      }
      SmallVector<AffineExpr, 4> indexExprs;

      for (auto index : apOf.getMapOperands()) {
        if (auto cIdx = index.getDefiningOp<arith::ConstantOp>()) {
          auto idxVal = cIdx.getValue().cast<IntegerAttr>().getValue();
          unsigned idx = idxVal.getSExtValue();
          indexExprs.push_back(getAffineConstantExpr(idx, context));
        } else {
          if (!indexToExprDimMap.count(index))
            indexToExprDimMap[index] =
                getAffineDimExpr(indexToExprDimMap.size(), context);
          indexExprs.push_back(indexToExprDimMap[index]);
        }
      }

      exprVec.push_back(map.getResult(0).replaceDims(indexExprs));
    } else if (auto cOp = value.getDefiningOp<arith::ConstantOp>()) {
      auto idxVal = cOp.getValue().cast<IntegerAttr>().getValue();
      unsigned idx = idxVal.getSExtValue();
      exprVec.push_back(getAffineConstantExpr(idx, context));
    } else {
      if (!indexToExprDimMap.count(value))
        indexToExprDimMap[value] =
            getAffineDimExpr(indexToExprDimMap.size(), context);
      exprVec.push_back(indexToExprDimMap[value]);
    }
  }

  if (exprVec.empty()) {
    return nullptr;
  }

  auto ret = flattenedStridedExpr(memRefType.getShape(), exprVec,
                                  memRefType.getContext());

  return ret;
}

// From a linearized affine expression, compute the base and the constant
// offset. If the access is A[i][j+2] for an N*N array A, the linearized
// expression will be A[i*N+j+2]. The base in this case will be (i*N+j), and the
// offset will be 2.
inline std::pair<AffineExpr, int32_t> extractBaseAndOffset(AffineExpr expr) {
  AffineExpr base = expr;
  int32_t offset = 0;

  if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
    base = nullptr;
    offset += constExpr.getValue();
  } else if (auto binopExpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
    if (binopExpr.getKind() == AffineExprKind::Add) {
      AffineExpr lhs = binopExpr.getLHS(), rhs = binopExpr.getRHS();
      if (auto constExpr = lhs.dyn_cast<AffineConstantExpr>()) {
        base = rhs;
        offset += constExpr.getValue();
      }
      if (auto constExpr = rhs.dyn_cast<AffineConstantExpr>()) {
        base = base == rhs ? nullptr : lhs;
        offset += constExpr.getValue();
      }
    }
  }
  return std::make_pair(base, offset);
}

// MLIR-AIE auto-vectorization to CPP flow currently doesn't support to
// implicitly broadcast a dynamic dimension of size `1`. Hence, we assume that
// dynamic dimensions are not with size '1' that can be interpreted to various
// broadcasting scenarios. We let lowerings assume this on a per-scope basis if
// the tosa.no_implicit_broadcast_of_dynamic_sizes attribute presents on any
// parent of the block.
inline bool isAssumingNoImplicitBroadcastOfDynamicSizes(Block *block) {
  for (Operation *parentOp = block->getParentOp(); parentOp;
       parentOp = parentOp->getParentOp()) {
    if (parentOp->hasAttr("tosa.no_implicit_broadcast_of_dynamic_sizes"))
      return true;
  }
  return false;
}

// Helper that uses the block from an OpBuilder for determining whether we
// are assuming no implict broadcast of dynamic sizes
inline bool isAssumingNoImplicitBroadcastOfDynamicSizes(OpBuilder &builder) {
  return isAssumingNoImplicitBroadcastOfDynamicSizes(builder.getBlock());
}

} // end namespace aievec
} // end namespace xilinx

#endif // AIE_DIALECT_AIEVEC_AIEVECUTILS_H
