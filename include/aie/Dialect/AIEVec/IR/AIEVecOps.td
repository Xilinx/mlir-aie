//===- AIEVecOps.td - AIE vector op definitions -----------*- tablegen -*-====//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// Defines AIE vector operations.
//===----------------------------------------------------------------------===//

#ifndef AIEVEC_OPS
#define AIEVEC_OPS

include "aie/Dialect/AIE/IR/AIEAttrs.td"
include "aie/Dialect/AIEVec/IR/AIEVecAttributes.td"
include "aie/Dialect/AIEVec/IR/AIEVecTypes.td"
include "aie/Dialect/AIEVec/IR/AIEVecTypeConstraints.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for AIE dialect ops.
class AIEVec_Op<string mnemonic, list<Trait> traits = []> :
    Op<AIEVec_Dialect, mnemonic, traits> {
  // For every AIE vector op, there needs to be a:
  //   * void ${C++ class of Op}::print(OpAsmPrinter &p)
  //   * LogicalResult ${C++ class of Op}::verify()
  //   * ParseResult ${C++ class of Op}::parse(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def AIEVec_AddOp:
  AIEVec_Op<"add", [
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
  let summary = "AIE vector add";
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

def AIEVec_AddElemOp:
  AIEVec_Op<"add_elem", [
    Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]>,
  Arguments<(ins AnyVector:$lhs, AnyVector:$rhs)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE vector add elem";
  let description = [{
    AMD-specific AIE2 intrinsic that allows you to perform addition operation
    on all types of vectors.`$result = `$lhs + $rhs`.
  }];
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_SubOp:
  AIEVec_Op<"sub", [
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
  let summary = "AIE vector subtract";
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

def AIEVec_SubElemOp:
  AIEVec_Op<"sub_elem", [
    Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]>,
  Arguments<(ins AnyVector:$lhs, AnyVector:$rhs)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE vector sub elem";
  let description = [{
    AMD-specific AIE2 intrinsic that allows you to perform substraction operation
    on all types of vectors.`$result = `$lhs - $rhs`.
  }];
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_FMAOp :
  AIEVec_Op<"mac", [
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

def AIEVec_FMAElemOp :
  AIEVec_Op<"mac_elem", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$lhs, AnyVector:$rhs, AnyVector:$acc,
               DefaultValuedAttr<BoolAttr, "false">:$fmsub)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE2 element-wise vector fused multiply-add";
  let description = [{
    AMD-specific multiply-add operation. It multiplies two 1-D vectors in the same channel,
    and adds the result to an accumulator.
    `$result = `$lhs * $rhs + $acc`.
    Note: the same operator can be used as fmsub operator by setting the
    'fmsub' bool to true.
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs, "mlir::Value":$acc,
            "bool":$fmsub),
    [{build($_builder, $_state, acc.getType(), lhs, rhs, acc,
            fmsub);}]>
  ];
  let extraClassDeclaration = [{
    // Get the attribute names
    llvm::StringRef getSubAttrName() { return "fmsub"; }
  }];
}

def AIEVec_MulOp:
  AIEVec_Op<"mul", [
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

def AIEVec_MulElemOp:
  AIEVec_Op<"mul_elem", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$lhs, AnyVector:$rhs)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE2 vector element-wise multiply";
  let description = [{
    AMD-specific multiply operation that multiplies two 1-D vectors in the same channel.
    The vector sizes are at least 512 bits.
    `$result = `$lhs * $rhs`.
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs, "mlir::Type":$accType),
    [{build($_builder, $_state, accType, lhs, rhs);}]>
  ];
}

def AIEVec_BroadcastOp:
  AIEVec_Op<"broadcast", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$source,
          DefaultValuedAttr<ConfinedAttr<AIEI8Attr, [IntNonNegative]>, "0">:$idx)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE2 broadcast";
  let description = [{
    AMD-specific broadcast intrinsic. Extract element index from vector and broadcasts its
    value to all lanes of the vector.
    `$result = broadcast($source, $idx)`
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$source, "int32_t":$idx),
    [{build($_builder, $_state, source.getType(), source, idx);}]>
  ];
}

def AIEVec_BroadcastScalarOp:
  AIEVec_Op<"broadcast_scalar", [
    Pure
  ]>,
  Arguments<(ins AnyTypeOf<[BF16, F32, I32, I16, I8]>:$source)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE2 broadcast scalar";
  let description = [{
    AMD-specific broadcast scalar intrinsic. Broadcasts input value to all vector lanes.
    `$result = broadcast_scalar($source)`
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$source),
    [{build($_builder, $_state, source.getType(), source);}]>
  ];
}

def AIEVec_UPSOp:
  AIEVec_Op<"ups", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$source,
          DefaultValuedAttr<ConfinedAttr<AIEI8Attr, [IntNonNegative]>, "0">:$shift)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE ups";
  let description = [{
    AMD-specific upshift intrinsic. Moves data from AIE vector data type
    to accumulator data type. The adjustment in precision is controlled by
    the shift parameter.
    `$result = ups($source, $shift)`
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$source, "int8_t":$shift),
    [{build($_builder, $_state, source.getType(), source, shift);}]>
  ];
  let hasFolder = 1;
}

def AIEVec_CastOp:
  AIEVec_Op<"cast", [
    Pure
  ]>, 
  Arguments<(ins AnyVector:$source, DefaultValuedAttr<BoolAttr, "false">:$isResAcc)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE cast";
  let description = [{
    AIE2 cast intrinsic. Cast values from source data type to result data types.
    `$result = cast($source, isResAcc)`
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$source, "bool":$isResAcc),
    [{build($_builder, $_state, source.getType(), source, isResAcc);}]>
  ];
  let hasFolder = 1;
}

def AIEVec_SRSOp:
  AIEVec_Op<"srs", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$source, AnyInteger:$shift)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE srs";
  let description = [{
    AMD-specific shift-round-saturate intrinsic. Moves values from
    accumulator data type to AIE vector data types. The adjustment in
    precision is controlled by the shift parameter.
    `$result = srs($source, $shift)`
  }];
  let hasFolder = 1;
}

def AIEVec_UPDOp:
  AIEVec_Op<"upd", [
    Pure,
    AttrSizedOperandSegments
  ]>,
  Arguments<(ins AnyShaped:$source,
                 Variadic<Index>:$indices,
                 DefaultValuedAttr<AIEI32Attr, "0">:$offset,
                 DefaultValuedAttr<ConfinedAttr<AIEI8Attr,
                        [IntMinValue<0>, IntMaxValue<1>]>, "0">:$index,
                 Optional<AnyVector>:$vector)>, 
  Results<(outs AnyVector:$result)> {
  let summary = "AIE upd";
  let description = [{
    AMD-specific update intrinsic. General upd intrinsic updates contiguous
    lanes of the result vector from a smaller source vector. This form of 
    upd intrinsic combines the load of data from memory into a vector 
    register, and then updating the lanes of the result vector using it. 
    `$result = upd($source[$indices], $offset, $index)`
  }];
  let builders = [
    OpBuilder<(ins "mlir::Type":$resultType, "mlir::Value":$source,
                   "mlir::ValueRange":$indices,
                   "int32_t":$offset, "int8_t":$index),
    [{build($_builder, $_state, resultType, source, indices, 
                   offset, index, nullptr);}]>
  ];
}

def AIEVec_ConcatOp:
  AIEVec_Op<"concat", [
    Pure, InferTypeOpAdaptor,
  ]>, 
  Arguments<(ins Variadic<AnyVector>:$sources)>, 
  Results<(outs AnyVector:$result)> {
  let summary = "AIE concat";
  let description = [{
    AMD-specific concat intrinsic. Concatenates two or more smaller
    vectors into a bigger vector. The verifier confirms that all the
    input vectors have the same number of lanes.
    `$result = concat($sources[0], $sources[1], ...)`
  }];
}

def AIEVec_ExtOp:
  AIEVec_Op<"ext", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$source,
             ConfinedAttr<AIEI8Attr, [IntMinValue<0>, IntMaxValue<8>]>:$index)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE ext";
  let description = [{
    AMD-specific vector extract intrinsic. Selects contiguous lanes from 
    the source vector, and transfers the data from those lanes to the 
    result. The lane selection is controlled by index. There are two cases:
    1. Extracted vector fills half of the original vector lanes (e.g. extract v64int8 from v128int8)
    2. Extracted vector fills a fourth of the original vector lanes (e.g. extract v32int8 from v128int8)
    In the first case, index can be 0 or 1. Index 0 extracts the lower half, and index 1 extracts the upper half.
    In the second case, index can be 0 to 3. Index 0 extracts the lowest quarter, index 1 the next quarter, and so on.
    `$result = ext($source, $index)`
  }];
}

def AIEVec_SelectOp:
  AIEVec_Op<"select", [
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

def AIEVec_PackOp:
  AIEVec_Op<"pack", [
    Pure
  ]>, 
  Arguments<(ins AnyVector:$source)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE pack";
  let description = [{
    AMD-specific pack intrinsic. Pack a vector of 16-bit values into
    a vector of 8-bit values.
    `$result = pack($source)`
  }];
}

def AIEVec_UnpackOp:
  AIEVec_Op<"unpack", [
    Pure
  ]>, 
  Arguments<(ins AnyVector:$source)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE unpack";
  let description = [{
    AMD-specific unpack intrinsic. Unpack a vector of 8-bit values into
    a vector of 16-bit values.
    `$result = unpack($source)`
  }];
}

def AIEVec_ShiftOp:
  AIEVec_Op<"shift", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$lhs, AnyVector:$rhs, I32:$shift, DefaultValuedAttr<BoolAttr, "false">:$isAcc)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE2 concat and shift";
  let description = [{
    AMD-specific shift intrinsic. Concatenates two
    vectors into a bigger vector, interprets them as a vector of 128 bytes
    and returns v1::v2[shift: shift+64]. `shift` is the number of bytes to 
    be shifted. The verifier confirms that all the input and result vectors 
    have the same number of lanes and element types.
    `$result = shift($lhs, $rhs, $shift)`
  }];
}

def AIEVec_LegacyShuffleOp:
  AIEVec_Op<"legacyshuffle", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$source,
             DefaultValuedAttr<AIEI32Attr, "0">:$mode)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE2 shuffle";
  let description = [{
    AMD-specific vector shuffle intrinsic by a specific shuffle mode.
    `$result = shuffle($source, $mode)`
  }];
}

def AIEVec_MulConvOp:
  AIEVec_Op<"mul_conv", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$lhs, AnyVector:$rhs,
             DefaultValuedAttr<AIEI32Attr, "0">:$M,
             DefaultValuedAttr<AIEI32Attr, "0">:$N)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE2 multiply convolution";
  let description = [{
    AMD-specific multiply convolution intrinsic. Multiply convolution operation
    of (M x N)matrix with (N x 1)kernel.
    `$result = mul_convMxN($lhs, $rhs)`
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs, "mlir::Type":$accType,
               "int32_t":$M, "int32_t":$N),
    [{build($_builder, $_state, accType, lhs, rhs, M, N);}]>
  ];
}

def AIEVec_FMAConvOp:
  AIEVec_Op<"fma_conv", [
    Pure
  ]>,
  Arguments<(ins AnyVector:$lhs, AnyVector:$rhs, AnyVector:$acc,
             DefaultValuedAttr<AIEI32Attr, "0">:$M,
             DefaultValuedAttr<AIEI32Attr, "0">:$N,
             DefaultValuedAttr<BoolAttr, "false">:$fmsub)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE2 multiply accumulate convolution";
  let description = [{
    AMD-specific multiply accumulate convolution intrinsic. Multiply accumulate convolution
    operation of (M x N)matrix with (N x 1)kernel.
    `$result = mac_convMxN($lhs, $rhs, $acc)`
  }];

   let extraClassDeclaration = [{
     // Get the attributes name
     llvm::StringRef getSubAttrName() { return "fmsub"; }
   }];
}

def AIEVec_MinOp:
  AIEVec_Op<"min", [
    Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]>,
  Arguments<(ins AnyVector:$lhs, AnyVector:$rhs)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE vector minimum";
  let description = [{
    AMD-specific intrinsic that calculates the minimum between two input vectors.
    `$result = min(`$lhs, $rhs`).
  }];
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_MaxOp:
  AIEVec_Op<"max", [
    Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]>,
  Arguments<(ins AnyVector:$lhs, AnyVector:$rhs)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE vector maximum";
  let description = [{
    AMD-specific intrinsic that calculates the maximum between two input vectors.
    `$result = max(`$lhs, $rhs`).
  }];
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_CmpOp:
  AIEVec_Op<"cmp", [
    Pure,
    AllTypesMatch<["lhs", "rhs"]>
  ]>,
  Arguments<(ins AnyVector:$lhs, AnyVector:$rhs, DefaultValuedStrAttr<StrAttr, "">:$pred)>,
  Results<(outs AnyUnsignedInteger:$result)> {
  let summary = "AIE vector comparison";
  let description = [{
    AMD-specific intrinsic that performs element-wise comparisonof two input vectors.
    The attribute predicate defines which type of comparison is
    performed. The following comparisons are supported:

    -   equal (mnemonic: `"eq"`)
    -   not equal (mnemonic: `"ne"`)
    -   signed less than (mnemonic: `"slt"`)
    -   unsigned less than (mnemonic: `"ult"`)
    -   signed less than or equal (mnemonic: `"sle"`)
    -   unsigned less than or equal (mnemonic: `"ule"`)
    -   signed greater than (mnemonic: `"sgt"`)
    -   unsigned greater than (mnemonic: `"ugt"`)
    -   signed greater than or equal (mnemonic: `"sge"`)
    -   unsigned greater than or equal (mnemonic: `"uge"`)
  }];

  let assemblyFormat = "$lhs `,` $rhs ` ` `{` `pred` `=` $pred attr-dict `}` `:` type($lhs) `,` type($rhs)  `,` type($result)";
  let hasVerifier = 0;
}

def AIEVec_SelOp:
  AIEVec_Op<"sel", [
    Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]>,
  Arguments<(ins AnyVector:$lhs, AnyVector:$rhs, AnyUnsignedInteger:$sel)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE vector lane wise selection";
  let description = [{
    AMD-specific intrinsic that performs lane wise selection between two input vectors, if a bit of sel is zero, the lane of vector lhs is selected, else the lane of vector rhs is selected.
    `$result = sel(`$lhs, $rhs, $sel`).
  }];
  let assemblyFormat = "$lhs `,` $rhs `,` $sel attr-dict `:` type($lhs) `,` type($rhs) `,` type($sel) `,` type($result)";
  let hasVerifier = 0;
}

def AIEVec_ExtElemOp:
  AIEVec_Op<"ext_elem", [
    Pure,
    AllElementTypesMatch<["source", "result"]>
  ]>,
  Arguments<(ins AnyVector:$source, I32:$index)>,
  Results<(outs AnyTypeOf<[BF16, F32, I32, I16, I8]>:$result)> {
  let summary = "AIE extract element";
  let description =
      [{AMD - specific extract element
                  intrinsic. Extract element determined by index from vector.
    `$result = ext_elem($source, $index)`.}];
  let assemblyFormat = "$source `,` $index attr-dict `:` type($source) `,` type($index) `,` type($result)";
  let hasVerifier = 1;
}

def AIEVec_NegOp:
  AIEVec_Op<"neg", [
    Pure,
    AllTypesMatch<["source", "result"]>
  ]>,
  Arguments<(ins AnyVector:$source)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE vector negative";
  let description = [{
    AMD-specific intrinsic that negates the vector and returns the result.
    `$result = neg(`$source`).
  }];
  let assemblyFormat = "$source attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_BxorOp:
  AIEVec_Op<"bxor", [
    Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]>,
  Arguments<(ins
      VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16]>:$lhs,
      VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16]>:$rhs)>,
  Results<(outs
      VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16]>:$result)> {
  let summary = "AIE vector bitwise xor";
  let description = [{
    AMD-specific intrinsic that computes bitwise xor of two vectors and returns
    the result.
    `$result = bxor(`$lhs, $rhs`).
  }];
  let assemblyFormat = [{$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
                         `,` type($result)}];
  let hasVerifier = 0;
}

def AIEVec_BnegOp:
  AIEVec_Op<"bneg", [
    Pure,
    AllTypesMatch<["source", "result"]>
  ]>,
  Arguments<(ins
      VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16]>:$source)>,
  Results<(outs
      VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16]>:$result)> {
  let summary = "AIE vector bitwise negation";
  let description = [{
    AMD-specific intrinsic that computes bitwise negation of a vector and
    returns the result.
    `$result = bneg(`$source`).
  }];
  let assemblyFormat = "$source attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def AIEVec_BorOp:
  AIEVec_Op<"bor", [
    Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]>,
  Arguments<(ins
      VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16]>:$lhs,
      VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16]>:$rhs)>,
  Results<(outs
      VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16]>:$result)> {
  let summary = "AIE vector bitwise or";
  let description = [{
    AMD-specific intrinsic that computes bitwise or of two vectors and returns
    the result.
    `$result = bor(`$lhs, $rhs`).
  }];
  let assemblyFormat = [{$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
                         `,` type($result)}];
  let hasVerifier = 0;
}

def AIEVec_BandOp:
  AIEVec_Op<"band", [
    Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]>,
  Arguments<(ins
      VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16]>:$lhs,
      VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16]>:$rhs)>,
  Results<(outs
      VectorOfBitWidthAndElementTypes<512, [I8, I16, I32, BF16]>:$result)> {
  let summary = "AIE vector bitwise and";
  let description = [{
    AMD-specific intrinsic that computes bitwise and of two vectors and returns
    the result.
    `$result = band(`$lhs, $rhs`).
  }];
  let assemblyFormat = [{$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
                         `,` type($result)}];
  let hasVerifier = 0;
}

def AIEVec_MatMulOp:
  AIEVec_Op<"matmul", [
    Pure,
    AllRanksMatch<["lhs", "rhs", "acc"]>,
    AllTypesMatch<["acc", "result"]>,
    ShapesCompatibleWithContraction<"lhs", "rhs", "acc">,
    IsValidAIE2MatMulShapeAndType<"lhs", "rhs", "acc">
  ]>,
  Arguments<(ins AIE2MatMulLHS:$lhs,
                 AIE2MatMulRHS:$rhs,
                 AIE2MatMulACC:$acc)>,
  Results<(outs AIE2MatMulACC:$result)> {
  let summary = "AIE2 matrix-multiply and accummulate";
  let description = [{
    AMD AIEv2-specific intrinsic that performs a matrix multiplications
    between `lhs` and `rhs`, and accumulates the result in `acc`.

    Currently, this intrinsic supports the following type combinations:

         lhs                | rhs                | Accumulator
        :------------------:|:------------------:|:-----------------:
         `vector<4x16xi8>`  | `vector<16x8xi4>`  | `vector<4x8xi32>`
         `vector<4x8xi8>`   | `vector<8x8xi8>`   | `vector<4x8xi32>`
         `vector<4x4xi16>`  | `vector<4x8xi8>`   | `vector<4x8xi32>`
         `vector<4x2xi16>`  | `vector<2x8xi16>`  | `vector<4x8xi32>`
         `vector<2x8xi16>`  | `vector<8x8xi8>`   | `vector<2x8xi64>`
         `vector<4x8xi16>`  | `vector<8x4xi8>`   | `vector<4x4xi64>`
         `vector<2x4xi16>`  | `vector<4x8xi16>`  | `vector<2x8xi64>`
         `vector<4x4xi16>`  | `vector<4x4xi16>`  | `vector<4x4xi64>`
         `vector<4x2xi32>`  | `vector<2x4xi16>`  | `vector<4x4xi64>`
         `vector<4x8xbf16>` | `vector<8x4xbf16>` | `vector<4x4xf32>`
  }];
  let assemblyFormat = [{$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,`
                         type($rhs) `into` type($acc)}];
  let hasVerifier = 0;
}

def AIEVec_ShuffleOp : AIEVec_Op<"shuffle",
    [Pure, AllTypesMatch<["lhs", "result"]>,
     OptionalTypesMatchWith<"result and rhs have the same type", "result", "rhs",
                            "::llvm::cast<Type>($_self)">]>,
  Arguments<(ins VectorOfBitWidthAndElementTypes<
                      512, [I8, I16, I32, I64, I128, I256,
                            I512, BF16, F32]>:$lhs,
                 Optional<VectorOfBitWidthAndElementTypes<
                      512, [I8, I16, I32, I64, I128, I256,
                            I512, BF16, F32]>>:$rhs,
                 AIEVec_ShuffleModeAttr:$mode)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE2 shuffle";
  let description = [{
    AMD AIEv2-specific vector shuffle. It performs a shuffle of the elements of
    1 or 2 input vectors using the specified shuffle mode. The shuffle mode is
    specified as:

      `t<width>_<r>x<c>(_(hi|lo))?`

    where `<width>` is the bitwidth of the vector element type, `<r>` and `<c>`
    are the number of rows and columns that will be transposed to perform the
    shuffle, and, for modes that require two 512-bit vectors, `hi` and `lo`
    indicate which part of the resulting extended 1024-bit vector will be
    assembled and returned.

    E.g.: `t32_4x8` would take two 512-bit vectors, `lhs` and `rhs`, with 16
    elements of 32 bits each. The resulting vector would contain either the
    least (`lo`) or most (`hi`) significant 16 elements of the 32 element vector
    that would result from selecting, out of the concatenated vectors `lhs:rhs`,
    8 blocks of 4 elements, each block taking one of every 8 elements starting
    from the block index.

    That is, for two `vector<16xi32>` operands containing:
    ```
    lhs = [0,   1,  2,  3, ..., 15]
    rhs = [17, 18, 19, 20, ..., 31]
    ```

    The first 8 blocks would be:
    ```
    b0 = [0,  8, 16, 24]
    b1 = [1,  9, 17, 25]
    b2 = [2, 10, 18, 26]
    b3 = [3, 11, 19, 27]
       ...
    b7 = [7, 15, 23, 31]
    ```

    `t32_4x8_lo` would return first four blocks:
    ```
    result = [0, 8, 16, 24, 1, 9, 17, 25, ..., 3, 11, 19, 27]
    ```

    And `t32_4x8_hi` would return the last four blocks:
    ```
    result = [4, 12, 20, 28, 5, 13, 21, 29, ..., 7, 15, 24, 31]
    ```

    It can be seen as flattened 4x8 matrix, split in two 16-element halfs, being
    tranposed to a 8x4 arrangement. In the example above:

    ```
    lhs = [ 0,  1,  2,  3,  4,  5,  6,  7]
          [ 8,  9, 10, 11, 12, 13, 14, 15]
    rhs = [16, 17, 18, 19, 20, 21, 22, 23]
          [24, 25, 26, 27, 28, 29, 30, 31]
    ```

    Would result in:
    ```
    t32_4x8_lo = [0,  8, 16, 24]
                 [1,  9, 17, 25]
                 [2, 10, 18, 26]
                 [3, 11, 19, 27]
    t32_4x8_hi = [4, 12, 20, 28]
                 [5, 13, 21, 29]
                 [6, 14, 22, 30]
                 [7, 15, 23, 31]
    ```

    A special mode, `t16_1x2_flip`, swaps each pair of elements in a vector with
    32 16-bit elements. E.g.:
    ```
    lhs = [0, 1, 2, 3, ..., 28, 29, 30, 31]
    ```
    Would result in:
    ```
    t16_1x2_flip = [1, 0, 3, 2, ..., 29, 28, 31, 30]
    ```

    The list of supported shuffle modes, required operands, and associated
    vector types are the following:

         Shuffle Mode       | Operands           | Types Supported
        :------------------:|:------------------:|:------------------:
         t8_8x4             | `lhs`              | `vector<64xi8>`
         t8_4x8             | ^                  | ^
         t8_8x8             | ^                  | ^
         t8_16x4            | ^                  | ^
         t8_4x16            | ^                  | ^
         t8_64x2_lo         | `lhs` & `rhs`      | ^
         t8_64x2_hi         | ^                  | ^
         t8_2x64_lo         | ^                  | ^
         t8_2x64_hi         | ^                  | ^
         t16_4x2            | `lhs`              | `vector<32xi16>` or `vector<32xbf16>`
         t16_2x4            | ^                  | ^
         t16_4x4            | ^                  | ^
         t16_8x2            | ^                  | ^
         t16_2x8            | ^                  | ^
         t16_8x4            | ^                  | ^
         t16_4x8            | ^                  | ^
         t16_16x2           | ^                  | ^
         t16_2x16           | ^                  | ^
         t16_1x2_flip       | ^                  | ^
         t16_32x2_lo        | `lhs` & `rhs`      | ^
         t16_32x2_hi        | ^                  | ^
         t16_2x32_lo        | ^                  | ^
         t16_2x32_hi        | ^                  | ^
         t16_16x4_lo        | ^                  | ^
         t16_16x4_hi        | ^                  | ^
         t16_4x16_lo        | ^                  | ^
         t16_4x16_hi        | ^                  | ^
         t32_4x4            | `lhs`              | `vector<16xi32>` or `vector<16xf32>`
         t32_16x2_lo        | `lhs` & `rhs`      | ^
         t32_16x2_hi        | ^                  | ^
         t32_2x16_lo        | ^                  | ^
         t32_2x16_hi        | ^                  | ^
         t32_8x4_lo         | ^                  | ^
         t32_8x4_hi         | ^                  | ^
         t32_4x8_lo         | ^                  | ^
         t32_4x8_hi         | ^                  | ^
         t64_8x2_lo         | ^                  | `vector<8xi64>`
         t64_8x2_hi         | ^                  | ^
         t64_2x8_lo         | ^                  | ^
         t64_2x8_hi         | ^                  | ^
         t128_4x2_lo        | ^                  | `vector<4xi128>`
         t128_4x2_hi        | ^                  | ^
         t128_2x4_lo        | ^                  | ^
         t128_2x4_hi        | ^                  | ^
         t256_2x2_lo        | ^                  | `vector<2xi256>`
         t256_2x2_hi        | ^                  | ^
         t512_1x2_lo        | ^                  | `vector<1xi512>`
         t512_1x2_hi        | ^                  | ^
  }];
  let assemblyFormat = [{$lhs (`,` $rhs^)? $mode attr-dict `:` type($result)}];
  let hasVerifier = 1;
}

#endif // AIEVEC_OPS
