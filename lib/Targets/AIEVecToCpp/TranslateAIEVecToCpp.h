//===- TranslateAIEVecToCpp.h - C++ emitter for AIE vector code -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// This file defines helpers to emit C++ code for AIE vector dialect.
//===----------------------------------------------------------------------===//

#ifndef TARGET_TRANSLATEAIEVECTOCPP_H
#define TARGET_TRANSLATEAIEVECTOCPP_H

#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

namespace xilinx {
namespace aievec {
using namespace mlir;

/// Translates the AIE vector dialect MLIR to C++ code. 
LogicalResult translateAIEVecToCpp(Operation *op, raw_ostream &os);

} // namespace aievec 
} // namespace xilinx

#endif // TARGET_TRANSLATEAIEVECTOCPP_H
