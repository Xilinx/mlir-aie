//===- XLLVM.td - XLLVM Dialect definition ----------------*- tablegen -*-====//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
