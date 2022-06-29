//===-------- AIEVecTypes.cpp - AIE vector types ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// This file implements convenience types for AIE vectorization.
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/IR/AIEVecTypes.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOpsDialect.h.inc"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace xilinx::aievec;

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "aie/Dialect/AIEVec/IR/AIEVecOpsTypes.cpp.inc"

void AIEVecDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "aie/Dialect/AIEVec/IR/AIEVecOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// AIE Vector Types
//===----------------------------------------------------------------------===//

bool AIEVecType::classof(Type type) {
  return llvm::isa<AIEVecDialect>(type.getDialect());
}
