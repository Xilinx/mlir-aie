//===- AIEVecAIE1Ops.td - AIE1 vector op definitions ------*- tablegen -*-====//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// Defines AIE1 vector operations.
//===----------------------------------------------------------------------===//

#ifndef AIEVEC_AIE1_OPS
#define AIEVEC_AIE1_OPS

// include "aie/Dialect/AIE/IR/AIEAttrs.td"
// include "aie/Dialect/AIEVec/IR/AIEVecAttributes.td"
// include "aie/Dialect/AIEVec/IR/AIEVecTypes.td"
// include "aie/Dialect/AIEVec/IR/AIEVecTypeConstraints.td"
include "aie/Dialect/AIEVec/AIE1/IR/AIEVecAIE1Dialect.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for AIE dialect ops.
class AIEVec_AIE1_Op<string mnemonic, list<Trait> traits = []> :
    Op<AIEVec_AIE1_Dialect, mnemonic, traits> {
  // For every AIE vector op, there needs to be a:
  //   * void ${C++ class of Op}::print(OpAsmPrinter &p)
  //   * LogicalResult ${C++ class of Op}::verify()
  //   * ParseResult ${C++ class of Op}::parse(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def AIEVec_AIE1_AddOp:
  AIEVec_AIE1_Op<"add", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$lhs, AnyVector:$rhs,
               DefaultValuedStrAttr<StrAttr, "">:$xstart,
               DefaultValuedStrAttr<StrAttr, "">:$xoffsets,
               DefaultValuedStrAttr<StrAttr, "">:$xoffsets_hi,
               DefaultValuedStrAttr<StrAttr, "">:$xsquare,
               DefaultValuedStrAttr<StrAttr, "">:$zstart,
               DefaultValuedStrAttr<StrAttr, "">:$zoffsets,
               DefaultValuedStrAttr<StrAttr, "">:$zoffsets_hi,
               DefaultValuedStrAttr<StrAttr, "">:$zsquare)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE1 vector add";
  let description = [{
    AMD-specific advanced add operation that adds two 1-D vectors 
    with lane selection. The vector sizes are at least 256 bits.
    `$result = `$lhs + $rhs`.
  }];
  let extraClassDeclaration = [{
    // Get the attributes
    llvm::StringRef getStart(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? getXstart() : getZstart(); }
    llvm::StringRef getOffset(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? getXoffsets() : getZoffsets(); }
    llvm::StringRef getOffsetHi(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? getXoffsetsHi() : getZoffsetsHi(); }
    llvm::StringRef getSquare(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? getXsquare() : getZsquare(); }
    // Get the attribute names
    llvm::StringRef getStartAttrName(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? "xstart" : "zstart"; }
    llvm::StringRef getOffsetAttrName(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? "xoffsets" : "zoffsets"; }
    llvm::StringRef getOffsetHiAttrName(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? "xoffsets_hi" : "zoffsets_hi"; }
    llvm::StringRef getSquareAttrName(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? "xsquare" : "zsquare"; }
  }];
}

def AIEVec_AIE1_SubOp:
  AIEVec_AIE1_Op<"sub", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$lhs, AnyVector:$rhs,
               DefaultValuedStrAttr<StrAttr, "">:$xstart,
               DefaultValuedStrAttr<StrAttr, "">:$xoffsets,
               DefaultValuedStrAttr<StrAttr, "">:$xoffsets_hi,
               DefaultValuedStrAttr<StrAttr, "">:$xsquare,
               DefaultValuedStrAttr<StrAttr, "">:$zstart,
               DefaultValuedStrAttr<StrAttr, "">:$zoffsets,
               DefaultValuedStrAttr<StrAttr, "">:$zoffsets_hi,
               DefaultValuedStrAttr<StrAttr, "">:$zsquare)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE1 vector subtract";
  let description = [{
    AMD-specific advanced sub operation that subtracts two 1-D vectors
    with lane selection. The vector sizes are at least 256 bits.
    `$result = `$lhs - $rhs`.
  }];
  let extraClassDeclaration = [{
    // Get the attributes
    llvm::StringRef getStart(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? getXstart() : getZstart(); }
    llvm::StringRef getOffset(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? getXoffsets() : getZoffsets(); }
    llvm::StringRef getOffsetHi(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? getXoffsetsHi() : getZoffsetsHi(); }
    llvm::StringRef getSquare(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? getXsquare() : getZsquare(); }
    // Get the attribute names
    llvm::StringRef getStartAttrName(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? "xstart" : "zstart"; }
    llvm::StringRef getOffsetAttrName(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? "xoffsets" : "zoffsets"; }
    llvm::StringRef getOffsetHiAttrName(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? "xoffsets_hi" : "zoffsets_hi"; }
    llvm::StringRef getSquareAttrName(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? "xsquare" : "zsquare"; }
  }];
}

#endif // AIEVEC_AIE1_OPS
