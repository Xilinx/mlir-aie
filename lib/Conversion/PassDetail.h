//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
////===----------------------------------------------------------------------===//

#ifndef AIE_CONVERSION_PASSDETAIL_H_
#define AIE_CONVERSION_PASSDETAIL_H_

#include "aie/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace xilinx {

namespace aievec {

class AIEVecDialect;

} // namespace aievec

namespace xllvm {

class XLLVMDialect;

} // namespace xllvm

} // namespace xilinx

namespace mlir {

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

#define GEN_PASS_CLASSES
#include "aie/Conversion/Passes.h.inc"
} // namespace mlir

#endif // AIE_CONVERSION_PASSDETAIL_H_
