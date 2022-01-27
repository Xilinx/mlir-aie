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
#include "aie/Dialect/AIEVec/IR/AIEVecTypes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
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

// Return true if the incoming type is accumulator type
inline bool isAccType(Type type) { return type.isa<AccType>(); }

// Return true if this is an operation defined in AIE dialect
inline bool isAIEOp(Operation *op) {
  return llvm::isa<AIEVecDialect>(op->getDialect());
}

// Create an AIE accumulator type. If the incoming type is vector, then the
// #lanes are copied from it. Otherwise, a scalar accumulator type is created.
inline AccType getAccType(Type type) {
  Type stype = type;
  unsigned lanes = 1;
  // If the type was vector type, get the number of lanes and the underflying
  // element type
  if (VectorType vtype = type.dyn_cast<VectorType>()) {
    stype = vtype.getElementType();
    lanes = getVectorLaneSize(vtype);
  }

  // Now create the accumulator type. We currently only support i48 and i80
  // type accumulator
  AccType acc;
  if (IntegerType itype = stype.dyn_cast<IntegerType>()) {
    assert(itype.getWidth() <= 64);
    unsigned width = itype.getWidth() <= 16 ? 48 : 80;
    Type ctype = mlir::IntegerType::get(itype.getContext(), width);
    acc = AccType::get(lanes, ctype);
  } else if (stype.isa<FloatType>())
    acc = AccType::get(lanes, stype);
  else
    llvm_unreachable("Unsupported accumulator type");

  return acc;
}

} // end namespace aievec
} // end namespace xilinx

#endif // AIE_DIALECT_AIEVEC_AIEVECUTILS_H
