//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_CONVERSION_PASSES
#define AIE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// AIEVecToLLVM
//===----------------------------------------------------------------------===//
def ConvertAIEVecToLLVM : Pass<"convert-aievec-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert AIEVec dialect to LLVM dialect";
  let description = [{
    This pass converts AIEVec dialect ops to LLVM dialect calls to builtins.
  }];
  let constructor = "xilinx::aievec::createConvertAIEVecToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::vector::VectorDialect",
                           "xilinx::xllvm::XLLVMDialect"];
}

#endif // AIE_CONVERSION_PASSES
