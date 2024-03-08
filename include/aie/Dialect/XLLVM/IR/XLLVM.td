//===- XLLVM.td - XLLVM Dialect definition ----------------*- tablegen -*-====//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Defines External LLVM (XLLVM) dialect.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_XLLVM_IR_XLLVM_TD
#define AIE_DIALECT_XLLVM_IR_XLLVM_TD

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

def XLLVM_Dialect : Dialect {
  let name = "xllvm";
  let summary = "Types and operations for external LLVM dialect";
  let cppNamespace = "::xilinx::xllvm";
}

// Base clase for external LLVM intrinsic operations
class ExtIntrOpBase<string opName,
                    string enumName,
                    list<Trait> traits,
                    int numResults
> : LLVM_OpBase<XLLVM_Dialect, opName, traits>,
    Results<!if(!gt(numResults, 0), (outs LLVM_Type:$res), (outs))> {
    string llvmBuilder = [{
      auto *inst = ::xilinx::xllvm::createExternalLLVMIntrinsicCall(builder,
                        moduleTranslation, &opInst, "llvm.}] # enumName # [{");
      (void) inst;
      $res = inst;
    }];
    string mlirBuilder = "";
    list<int> llvmArgIndices = [];
}

#endif // AIE_DIALECT_XLLVM_IR_XLLVMDIALECT_TD
