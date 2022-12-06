//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
// TODO
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_CONVERSION_PASSDETAIL_H_
#define AIE_CONVERSION_PASSDETAIL_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

// Forward declaration from Dialect.h
// template <typename mlir::ConcreteDialect>
// void mlir::registerDialect(DialectRegistry &registry);

namespace xilinx {
namespace aievec {
class AIEVecDialect;
} // namespace aievec
} // namespace xilinx

namespace mlir {

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

#define GEN_PASS_CLASSES
#include "aie/Conversion/Passes.h.inc"
} // namespace mlir

#endif // AIE_CONVERSION_PASSDETAIL_H_
