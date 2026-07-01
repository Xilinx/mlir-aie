//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

namespace AIE {

class AIEDialect;

} // namespace AIE

namespace AIEX {

class AIEXDialect;

} // namespace AIEX

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

} // namespace mlir

#endif // AIE_CONVERSION_PASSDETAIL_H_
