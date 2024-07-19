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

#include <cassert>
#include <numeric>

namespace xilinx::aievec {

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
inline mlir::VectorType createVectorType(unsigned lanes,
                                         mlir::Type elementType) {
  llvm::SmallVector<int64_t, 4> vecShape = {lanes};
  return mlir::VectorType::get(vecShape, elementType);
}

// Return the size (in bits) of the underlying element type of the vector
inline int32_t getElementSizeInBits(mlir::VectorType type) {
  return llvm::cast<mlir::ShapedType>(type).getElementTypeBitWidth();
}

// Return the number of lanes along the vectorized dimension for the vector
// type. For a multidimensional vector, return the innermost dimension size
inline unsigned getVectorLaneSize(mlir::VectorType type) {
  assert(type.getRank() > 0 && "Cannot handle rank-0 vectors");
  auto vShape = type.getShape();
  assert(llvm::all_of(vShape, [](int64_t dim) { return dim > 0; }) &&
         "Vector dimensions cannot be dynamic");
  return std::accumulate(vShape.begin(), vShape.end(), 1,
                         std::multiplies<int64_t>());
}

// For a 1D vector, return its size in bits. For an nD vector, return the size
// of the innerost dimension in bits.
inline int32_t getVectorSizeInBits(mlir::VectorType type) {
  int32_t veclen = getVectorLaneSize(type) * getElementSizeInBits(type);
  assert(veclen >= 128 && "AIE vector size should be greater than 128 bits");
  return veclen;
}

// Determine the output type for a vector operation based on whether
// it operates on integer or floating point data.
inline mlir::VectorType getVectorOpDestType(mlir::VectorType type, bool AIE2) {
  mlir::Type stype = type.getElementType();

  if (auto itype = llvm::dyn_cast<mlir::IntegerType>(stype)) {
    // Integer vector types are sized for the appropriate accumulators
    assert(itype.getWidth() <= 64);
    unsigned width;
    if (AIE2)
      width = itype.getWidth() <= 16 ? 32 : 64;
    else
      width = itype.getWidth() <= 16 ? 48 : 80;

    mlir::Type ctype = mlir::IntegerType::get(itype.getContext(), width);
    return mlir::VectorType::get(type.getShape(), ctype);
  }

  if (auto ftype = llvm::dyn_cast<mlir::FloatType>(stype)) {
    if (AIE2 && ftype.getWidth() == 16)
      return mlir::VectorType::get(type.getShape(),
                                   mlir::FloatType::getF32(ftype.getContext()));

    // Floating point vector types for aie1 are returned as is since the
    // floating point operations write back to registers and not accumulators
    return type;
  }

  llvm::report_fatal_error("Unsupported destination type");
}

// Linearize the exprVec as a strided access, but do not simplify
inline mlir::AffineExpr
flattenedStridedExpr(llvm::ArrayRef<int64_t> sizes,
                     llvm::ArrayRef<mlir::AffineExpr> exprs,
                     mlir::MLIRContext *context) {
  // Expect non-empty sizes and exprs
  if (sizes.empty() || exprs.empty())
    return nullptr;

  if (is_contained(sizes, 0))
    return getAffineConstantExpr(0, context);

  auto maps = mlir::AffineMap::inferFromExprList(exprs, context);
  if (maps.empty())
    return nullptr;

  unsigned nSymbols = maps[0].getNumSymbols();

  mlir::AffineExpr expr;
  bool dynamicPoisonBit = false;
  int64_t runningSize = 1;
  for (auto en : zip(reverse(exprs), reverse(sizes))) {
    int64_t size = std::get<1>(en);
    if (size == 0)
      continue;

    mlir::AffineExpr dimExpr = std::get<0>(en);
    mlir::AffineExpr stride = dynamicPoisonBit
                                  ? getAffineSymbolExpr(nSymbols++, context)
                                  : getAffineConstantExpr(runningSize, context);
    expr = expr ? expr + dimExpr * stride : dimExpr * stride;
    if (size > 0) {
      runningSize *= size;
      if (runningSize <= 0)
        return nullptr;
    } else
      dynamicPoisonBit = true;
  }
  return expr;
}

// Construct a linearized affine expression for the upd op.
inline mlir::AffineExpr constructLinearizedAffineExprForUPDOp(UPDOp updOp) {
  auto memRefType = llvm::cast<mlir::MemRefType>(updOp.getSource().getType());
  mlir::MLIRContext *context = memRefType.getContext();

  llvm::SmallVector<mlir::AffineExpr, 8> exprVec;
  llvm::SmallDenseMap<mlir::Value, mlir::AffineExpr, 8> indexToExprDimMap;
  for (auto idxAndValue : llvm::enumerate(updOp.getIndices())) {
    auto value = idxAndValue.value();
    if (auto apOf = value.getDefiningOp<mlir::affine::AffineApplyOp>()) {
      mlir::AffineMap map = apOf.getAffineMap();
      // Cannot create linearized mlir::AffineExpr for complicated index.
      if (map.getNumResults() != 1)
        return nullptr;

      llvm::SmallVector<mlir::AffineExpr, 4> indexExprs;

      for (auto index : apOf.getMapOperands())
        if (auto cIdx = index.getDefiningOp<mlir::arith::ConstantOp>()) {
          auto idxVal =
              llvm::cast<mlir::IntegerAttr>(cIdx.getValue()).getValue();
          unsigned idx = idxVal.getSExtValue();
          indexExprs.push_back(getAffineConstantExpr(idx, context));
        } else {
          if (!indexToExprDimMap.count(index))
            indexToExprDimMap[index] =
                getAffineDimExpr(indexToExprDimMap.size(), context);
          indexExprs.push_back(indexToExprDimMap[index]);
        }

      exprVec.push_back(map.getResult(0).replaceDims(indexExprs));
    } else if (auto cOp = value.getDefiningOp<mlir::arith::ConstantOp>()) {
      auto idxVal = llvm::cast<mlir::IntegerAttr>(cOp.getValue()).getValue();
      unsigned idx = idxVal.getSExtValue();
      exprVec.push_back(getAffineConstantExpr(idx, context));
    } else {
      if (!indexToExprDimMap.count(value))
        indexToExprDimMap[value] =
            getAffineDimExpr(indexToExprDimMap.size(), context);
      exprVec.push_back(indexToExprDimMap[value]);
    }
  }

  if (exprVec.empty())
    return nullptr;

  auto ret = flattenedStridedExpr(memRefType.getShape(), exprVec,
                                  memRefType.getContext());
  return ret;
}

// From a linearized affine expression, compute the base and the constant
// offset. If the access is A[i][j+2] for an N*N array A, the linearized
// expression will be A[i*N+j+2]. The base in this case will be (i*N+j), and the
// offset will be 2.
inline std::pair<mlir::AffineExpr, int32_t>
extractBaseAndOffset(mlir::AffineExpr expr) {
  mlir::AffineExpr base = expr;
  int32_t offset = 0;

  if (auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
    base = nullptr;
    offset += constExpr.getValue();
  } else if (auto binopExpr = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    if (binopExpr.getKind() == mlir::AffineExprKind::Add) {
      mlir::AffineExpr lhs = binopExpr.getLHS(), rhs = binopExpr.getRHS();
      if (auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(lhs)) {
        base = rhs;
        offset += constExpr.getValue();
      }
      if (auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs)) {
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
inline bool isAssumingNoImplicitBroadcastOfDynamicSizes(mlir::Block *block) {
  for (mlir::Operation *parentOp = block->getParentOp(); parentOp;
       parentOp = parentOp->getParentOp())
    if (parentOp->hasAttr("tosa.no_implicit_broadcast_of_dynamic_sizes"))
      return true;
  return false;
}

// Helper that uses the block from an OpBuilder for determining whether we
// are assuming no implict broadcast of dynamic sizes
inline bool
isAssumingNoImplicitBroadcastOfDynamicSizes(mlir::OpBuilder &builder) {
  return isAssumingNoImplicitBroadcastOfDynamicSizes(builder.getBlock());
}

} // namespace xilinx::aievec
// end namespace xilinx

#endif // AIE_DIALECT_AIEVEC_AIEVECUTILS_H
