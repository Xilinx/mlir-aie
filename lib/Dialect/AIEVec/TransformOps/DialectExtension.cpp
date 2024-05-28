//===- DialectExtension.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/TransformOps/DialectExtension.h"
#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"
#include "aie/Dialect/AIEVec/TransformOps/AIEVecTransformOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace xilinx;

namespace {

class AIEVecTransformDialectExtension
    : public transform::TransformDialectExtension<
          AIEVecTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareDependentDialect<linalg::LinalgDialect>();
    declareDependentDialect<vector::VectorDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "aie/Dialect/AIEVec/TransformOps/AIEVecTransformOps.cpp.inc"
        >();
  }
};

} // namespace

void aievec::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<AIEVecTransformDialectExtension>();
}
