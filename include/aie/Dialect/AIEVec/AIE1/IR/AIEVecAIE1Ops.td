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

include "aie/Dialect/AIEVec/AIE1/IR/AIEVecAIE1Dialect.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for AIE dialect ops.
class AIEVecAIE1_Op<string mnemonic, list<Trait> traits = []> :
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

def AIEVecAIE1_AddOp:
  AIEVecAIE1_Op<"add", [
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

def AIEVecAIE1_SubOp:
  AIEVecAIE1_Op<"sub", [
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

def AIEVecAIE1_MulOp:
  AIEVecAIE1_Op<"mul", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$lhs, AnyVector:$rhs,
               DefaultValuedStrAttr<StrAttr, "">:$xstart,
               DefaultValuedStrAttr<StrAttr, "">:$xoffsets,
               DefaultValuedStrAttr<StrAttr, "">:$xoffsets_hi,
               DefaultValuedStrAttr<StrAttr, "">:$xstep,
               DefaultValuedStrAttr<StrAttr, "">:$xsquare,
               DefaultValuedStrAttr<StrAttr, "">:$zstart,
               DefaultValuedStrAttr<StrAttr, "">:$zoffsets,
               DefaultValuedStrAttr<StrAttr, "">:$zoffsets_hi,
               DefaultValuedStrAttr<StrAttr, "">:$zstep,
               DefaultValuedStrAttr<StrAttr, "">:$zsquare)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE vector multiply";
  let description = [{
    AMD-specific multiply operation that multiplies two 1-D vectors.
    The vector sizes are at least 256 bits, and the left operand vector 
    is at least twice the size of right operand vector. For integers, the
    lhs and rhs are 8/16/32 bits, and result is a 48-bit or 80-bit accumulator.
    `$result = `$lhs * $rhs`.
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs, "mlir::Type":$accType,
            "llvm::StringRef":$xstart,
            "llvm::StringRef":$xoffsets, "llvm::StringRef":$xoffsets_hi,
            "llvm::StringRef":$xstep, "llvm::StringRef":$xsquare,
            "llvm::StringRef":$zstart,
            "llvm::StringRef":$zoffsets, "llvm::StringRef":$zoffsets_hi,
            "llvm::StringRef":$zstep, "llvm::StringRef":$zsquare),
    [{build($_builder, $_state, accType, lhs, rhs, 
            xstart, xoffsets, xoffsets_hi,
            xstep, xsquare,
            zstart, zoffsets, zoffsets_hi,
            zstep, zsquare);}]>
  ];
  let extraClassDeclaration = [{
    // Get the attributes
    llvm::StringRef getStart(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? getXstart() : getZstart(); }
    llvm::StringRef getOffset(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? getXoffsets() : getZoffsets(); }
    llvm::StringRef getOffsetHi(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? getXoffsetsHi() : getZoffsetsHi(); }
    llvm::StringRef getStep(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? getXstep() : getZstep(); }
    llvm::StringRef getSquare(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? getXsquare() : getZsquare(); }
    // Get the attribute names
    llvm::StringRef getStartAttrName(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? "xstart" : "zstart"; }
    llvm::StringRef getOffsetAttrName(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? "xoffsets" : "zoffsets"; }
    llvm::StringRef getOffsetHiAttrName(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? "xoffsets_hi" : "zoffsets_hi"; }
    llvm::StringRef getStepAttrName(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? "xstep" : "zstep"; }
    llvm::StringRef getSquareAttrName(int idx) { assert(idx==0 || idx==1);
                            return idx==0 ? "xsquare" : "zsquare"; }
  }];
}

def AIEVecAIE1_FMAOp :
  AIEVecAIE1_Op<"mac", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$lhs, AnyVector:$rhs, AnyVector:$acc,
               DefaultValuedStrAttr<StrAttr, "">:$xstart,
               DefaultValuedStrAttr<StrAttr, "">:$xoffsets,
               DefaultValuedStrAttr<StrAttr, "">:$xoffsets_hi,
               DefaultValuedStrAttr<StrAttr, "">:$xstep,
               DefaultValuedStrAttr<StrAttr, "">:$xsquare,
               DefaultValuedStrAttr<StrAttr, "">:$zstart,
               DefaultValuedStrAttr<StrAttr, "">:$zoffsets,
               DefaultValuedStrAttr<StrAttr, "">:$zoffsets_hi,
               DefaultValuedStrAttr<StrAttr, "">:$zstep,
               DefaultValuedStrAttr<StrAttr, "">:$zsquare,
               DefaultValuedAttr<BoolAttr, "false">:$fmsub)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE vector fused multiply-add";
  let description = [{
    AMD-specific multiply-add operation. It multiplies two 1-D vectors,
    and adds the result to an accumulator. The vector sizes are at least
    256 bits, and the left operand vector is at least twice the size of
    right operand vector. For integers, the lhs and rhs are 8/16/32 bits;
    the result and acc are 48-bit or 80-bit accumulator.
    `$result = `$lhs * $rhs + $acc`.
    Note: the same operator can be used as fmsub operator by setting the
    'fmsub' bool to true.
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs, "mlir::Value":$acc,
            "llvm::StringRef":$xstart,
            "llvm::StringRef":$xoffsets, "llvm::StringRef":$xoffsets_hi,
            "llvm::StringRef":$xstep, "llvm::StringRef":$xsquare,
            "llvm::StringRef":$zstart,
            "llvm::StringRef":$zoffsets, "llvm::StringRef":$zoffsets_hi,
            "llvm::StringRef":$zstep, "llvm::StringRef":$zsquare,
            "bool":$fmsub),
    [{build($_builder, $_state, acc.getType(), lhs, rhs, acc,
            xstart, xoffsets, xoffsets_hi,
            xstep, xsquare,
            zstart, zoffsets, zoffsets_hi,
            zstep, zsquare,
            fmsub);}]>
  ];
  let extraClassDeclaration = [{
    // Get the attributes
    llvm::StringRef getStart(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? getXstart() : getZstart(); }
    llvm::StringRef getOffset(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? getXoffsets() : getZoffsets(); }
    llvm::StringRef getOffsetHi(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? getXoffsetsHi() : getZoffsetsHi(); }
    llvm::StringRef getStep(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? getXstep() : getZstep(); }
    llvm::StringRef getSquare(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? getXsquare() : getZsquare(); }
    // Get the attribute names
    llvm::StringRef getStartAttrName(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? "xstart" : "zstart"; }
    llvm::StringRef getOffsetAttrName(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? "xoffsets" : "zoffsets"; }
    llvm::StringRef getOffsetHiAttrName(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? "xoffsets_hi" : "zoffsets_hi"; }
    llvm::StringRef getStepAttrName(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? "xstep" : "zstep"; }
    llvm::StringRef getSquareAttrName(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? "xsquare" : "zsquare"; }
    llvm::StringRef getSubAttrName() { return "fmsub"; }
  }];
}

def AIEVecAIE1_SelectOp:
  AIEVecAIE1_Op<"select", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$xbuff,
               DefaultValuedStrAttr<StrAttr, "">:$select,
               DefaultValuedStrAttr<StrAttr, "">:$xstart,
               DefaultValuedStrAttr<StrAttr, "">:$xoffsets,
               DefaultValuedStrAttr<StrAttr, "">:$xoffsets_hi,
               DefaultValuedStrAttr<StrAttr, "">:$xsquare,
               DefaultValuedStrAttr<StrAttr, "">:$ystart,
               DefaultValuedStrAttr<StrAttr, "">:$yoffsets,
               DefaultValuedStrAttr<StrAttr, "">:$yoffsets_hi,
               DefaultValuedStrAttr<StrAttr, "">:$ysquare,
               Optional<AnyVector>:$ybuff)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE vector lane selection";
  let description = [{
    AMD-specific vector lane selection operation. It selects between the
    first set of lanes or the second one according to the value in 'select'. 
    If the bit in select is 0(1), it returns the value in the first(second) 
    set of lanes.
    `$result = `select32($select, $xbuff, $xstart, $xoffsets, $ystart, $yoffsets)`
  }];
  let builders = [
    OpBuilder<(ins "mlir::Type":$resultType, "mlir::Value":$xbuff, "llvm::StringRef":$select,
            "llvm::StringRef":$xstart,
            "llvm::StringRef":$xoffsets, "llvm::StringRef":$xoffsets_hi,
            "llvm::StringRef":$xsquare,
            "llvm::StringRef":$ystart,
            "llvm::StringRef":$yoffsets, "llvm::StringRef":$yoffsets_hi,
            "llvm::StringRef":$ysquare),
    [{build($_builder, $_state, resultType, xbuff, select,
            xstart, xoffsets, xoffsets_hi, xsquare,
            ystart, yoffsets, yoffsets_hi, ysquare, nullptr);}]>
  ];
  let extraClassDeclaration = [{
    // Get the attributes
    llvm::StringRef getStart(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? getXstart() : getYstart(); }
    llvm::StringRef getOffset(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? getXoffsets() : getYoffsets(); }
    llvm::StringRef getOffsetHi(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? getXoffsetsHi() : getYoffsetsHi(); }
    llvm::StringRef getSquare(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? getXsquare() : getYsquare(); }
    // Get the attribute names
    llvm::StringRef getStartAttrName(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? "xstart" : "ystart"; }
    llvm::StringRef getOffsetAttrName(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? "xoffsets" : "yoffsets"; }
    llvm::StringRef getOffsetHiAttrName(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? "xoffsets_hi" : "yoffsets_hi"; }
    llvm::StringRef getSquareAttrName(int idx) { assert(idx==0 || idx==1);
                        return idx==0 ? "xsquare" : "ysquare"; }
  }];
}

#endif // AIEVEC_AIE1_OPS
