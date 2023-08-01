//===- BufferDescriptorAnalysis.h -----------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIE_ANALYSIS_BUFFERDESCRIPTORANALYSIS_H
#define MLIR_AIE_ANALYSIS_BUFFERDESCRIPTORANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include <iostream>

using namespace mlir;

namespace xilinx {
namespace AIE {

struct BufferDescriptorState {
public:
  BufferDescriptorState() = default;

  int64_t getTotalLengthInt() const {
    int64_t ret = 1;
    for (auto len : lengthInt) {
      ret *= len;
    }
    return ret;
  }

  void print(raw_ostream &os) const;
  void printInt(raw_ostream &os) const;

public:
  SmallVector<int64_t, 4> lengthInt;
  Value baseVal;
  SmallVector<int64_t, 4> stepsInt;
  SmallVector<int64_t, 4> wrapsInt;

  SmallVector<OpFoldResult> length;
  OpFoldResult base;
  SmallVector<OpFoldResult> steps;
  SmallVector<OpFoldResult> wraps;

  std::optional<int64_t> repetition;
  std::optional<int64_t> constantStep;

  Value source;
};

class BufferDescriptorAnalysis {
public:
  BufferDescriptorAnalysis() = default;

  static void visitOperand(Value operand, BufferDescriptorState &state);

  static void visitOperandReintCast(memref::ReinterpretCastOp reintCastOp,
                                    BufferDescriptorState &state);

  static void visitOperandSubView(memref::SubViewOp subViewOp,
                                  BufferDescriptorState &state);

  static void visitOperandCopy(memref::CopyOp copyOp,
                               BufferDescriptorState &state);

  static void visitOperandTensorStore(memref::TensorStoreOp tensorStoreOp,
                                      BufferDescriptorState &state);

  static void visitOperandCast(memref::CastOp castOp,
                               BufferDescriptorState &state);
};

} // namespace AIE
} // namespace xilinx

#endif
