//===- AIEVecTypes.h - AIE Vector Type Classes ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECTYPES_H
#define AIE_DIALECT_AIEVEC_IR_AIEVECTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

//===----------------------------------------------------------------------===//
// AIE Vector Dialect Types
//===----------------------------------------------------------------------===//

namespace xilinx::aievec {

// Base class of all AIE types
class AIEVecType : public mlir::Type {
public:
  using Type::Type;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);
};

} // namespace xilinx::aievec

//===----------------------------------------------------------------------===//
// Tablegen Type Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "aie/Dialect/AIEVec/IR/AIEVecOpsTypes.h.inc"

#endif // AIE_DIALECT_AIEVEC_IR_AIEVECTYPES_H
