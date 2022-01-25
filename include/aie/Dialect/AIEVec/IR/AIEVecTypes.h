//===- AIEVecTypes.h - AIE Vector Type Classes ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECTYPES_H
#define AIE_DIALECT_AIEVEC_IR_AIEVECTYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

//===----------------------------------------------------------------------===//
// AIE Vector Dialect Types
//===----------------------------------------------------------------------===//

namespace xilinx {
namespace aievec {
using namespace mlir;

// Base class of all AIE types
class AIEVecType : public Type {
public:
  using Type::Type;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);
};

} // end namespace aievec
} // end namespace xilinx 

//===----------------------------------------------------------------------===//
// Tablegen Type Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "aie/Dialect/AIEVec/IR/AIEVecOpsTypes.h.inc"

#endif // AIE_DIALECT_AIEVEC_IR_AIEVECTYPES_H
