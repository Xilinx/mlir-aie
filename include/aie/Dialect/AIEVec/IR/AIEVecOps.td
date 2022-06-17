//===- AIEVecOps.td - AIE vector op definitions -----------*- tablegen -*-====//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// Defines AIE vector operations.
//===----------------------------------------------------------------------===//

#ifndef AIEVEC_OPS
#define AIEVEC_OPS

include "aie/Dialect/AIEVec/IR/AIEVecTypes.td"
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
    NoSideEffect
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
    Xilinx-specific advanced add operation that adds two 1-D vectors 
    with lane selection. The vector sizes are at least 256 bits.
    `$result = `$lhs + $rhs`.
  }];
  let extraClassDeclaration = [{
    // Get the attributes
    StringRef getStart(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? xstart() : zstart(); }
    StringRef getOffset(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? xoffsets() : zoffsets(); }
    StringRef getOffsetHi(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? xoffsets_hi() : zoffsets_hi(); } 
    StringRef getSquare(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? xsquare() : zsquare(); }
    // Get the attribute names
    StringRef getStartAttrName(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? "xstart" : "zstart"; }
    StringRef getOffsetAttrName(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? "xoffsets" : "zoffsets"; }
    StringRef getOffsetHiAttrName(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? "xoffsets_hi" : "zoffsets_hi"; } 
    StringRef getSquareAttrName(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? "xsquare" : "zsquare"; }
  }];
}

def AIEVec_SubOp:
  AIEVec_Op<"sub", [
    NoSideEffect
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
    Xilinx-specific advanced sub operation that subtracts two 1-D vectors 
    with lane selection. The vector sizes are at least 256 bits.
    `$result = `$lhs - $rhs`.
  }];
  let extraClassDeclaration = [{
    // Get the attributes
    StringRef getStart(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? xstart() : zstart(); }
    StringRef getOffset(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? xoffsets() : zoffsets(); }
    StringRef getOffsetHi(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? xoffsets_hi() : zoffsets_hi(); } 
    StringRef getSquare(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? xsquare() : zsquare(); }
    // Get the attribute names
    StringRef getStartAttrName(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? "xstart" : "zstart"; }
    StringRef getOffsetAttrName(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? "xoffsets" : "zoffsets"; }
    StringRef getOffsetHiAttrName(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? "xoffsets_hi" : "zoffsets_hi"; } 
    StringRef getSquareAttrName(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? "xsquare" : "zsquare"; }
  }];
}

def AIEVec_FMAOp :
  AIEVec_Op<"mac", [
    NoSideEffect
  ]>,
  Arguments<(ins AnyVector:$lhs, AnyVector:$rhs,
               AnyTypeOf<[AIEVec_Acc, VectorOf<[F32]>]>:$acc,
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
  Results<(outs AnyTypeOf<[AIEVec_Acc, VectorOf<[F32]>]>:$result)> {
  let summary = "AIE vector fused multiply-add";
  let description = [{
    Xilinx-specific multiply-add operation. It multiplies two 1-D vectors,
    and adds the result to an accumulator. The vector sizes are at least 
    256 bits, and the left operand vector is at least twice the size of 
    right operand vector. For integers, the lhs and rhs are 8/16/32 bits;
    the result and acc are 48-bit or 80-bit accumulator.
    `$result = `$lhs * $rhs + $acc`.
    Note: the same operator can be used as fmsub operator by setting the 
    'fmsub' bool to true.
  }];
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$acc,
            "StringRef":$xstart,
            "StringRef":$xoffsets, "StringRef":$xoffsets_hi,
            "StringRef":$xstep, "StringRef":$xsquare,
            "StringRef":$zstart,
            "StringRef":$zoffsets, "StringRef":$zoffsets_hi,
            "StringRef":$zstep, "StringRef":$zsquare,
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
    StringRef getStart(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? xstart() : zstart(); }
    StringRef getOffset(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? xoffsets() : zoffsets(); }
    StringRef getOffsetHi(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? xoffsets_hi() : zoffsets_hi(); } 
    StringRef getStep(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? xstep() : zstep(); }
    StringRef getSquare(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? xsquare() : zsquare(); }
    // Get the attribute names
    StringRef getStartAttrName(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? "xstart" : "zstart"; }
    StringRef getOffsetAttrName(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? "xoffsets" : "zoffsets"; }
    StringRef getOffsetHiAttrName(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? "xoffsets_hi" : "zoffsets_hi"; } 
    StringRef getStepAttrName(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? "xstep" : "zstep"; }
    StringRef getSquareAttrName(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? "xsquare" : "zsquare"; }
    StringRef getSubAttrName() { return "fmsub"; }
  }];
}

def AIEVec_MulOp:
  AIEVec_Op<"mul", [
    NoSideEffect
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
  Results<(outs AnyTypeOf<[AIEVec_Acc, VectorOf<[F32]>]>:$result)> {
  let summary = "AIE vector multiply";
  let description = [{
    Xilinx-specific multiply operation that multiplies two 1-D vectors.
    The vector sizes are at least 256 bits, and the left operand vector 
    is at least twice the size of right operand vector. For integers, the
    lhs and rhs are 8/16/32 bits, and result is a 48-bit or 80-bit accumulator.
    `$result = `$lhs * $rhs`.
  }];
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Type":$accType,
            "StringRef":$xstart,
            "StringRef":$xoffsets, "StringRef":$xoffsets_hi,
            "StringRef":$xstep, "StringRef":$xsquare,
            "StringRef":$zstart,
            "StringRef":$zoffsets, "StringRef":$zoffsets_hi,
            "StringRef":$zstep, "StringRef":$zsquare),
    [{build($_builder, $_state, accType, lhs, rhs, 
            xstart, xoffsets, xoffsets_hi,
            xstep, xsquare,
            zstart, zoffsets, zoffsets_hi,
            zstep, zsquare);}]>
  ];
  let extraClassDeclaration = [{
    // Get the attributes
    StringRef getStart(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? xstart() : zstart(); }
    StringRef getOffset(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? xoffsets() : zoffsets(); }
    StringRef getOffsetHi(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? xoffsets_hi() : zoffsets_hi(); } 
    StringRef getStep(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? xstep() : zstep(); }
    StringRef getSquare(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? xsquare() : zsquare(); }
    // Get the attribute names
    StringRef getStartAttrName(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? "xstart" : "zstart"; }
    StringRef getOffsetAttrName(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? "xoffsets" : "zoffsets"; }
    StringRef getOffsetHiAttrName(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? "xoffsets_hi" : "zoffsets_hi"; } 
    StringRef getStepAttrName(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? "xstep" : "zstep"; }
    StringRef getSquareAttrName(int idx) { assert(idx==0 || idx==1); 
                            return idx==0 ? "xsquare" : "zsquare"; }
  }];
}

def AIEVec_UPSOp:
  AIEVec_Op<"ups", [
    NoSideEffect
  ]>, 
  Arguments<(ins AnyVector:$source, 
          DefaultValuedAttr<Confined<I8Attr, [IntNonNegative]>, "0">:$shift)>,
  Results<(outs AIEVec_Acc:$result)> {
  let summary = "AIE ups";
  let description = [{
    Xilinx-specific upshift intrinsic. Moves data from AIE vector data type
    to accumulator data type. The adjustment in precision is controlled by
    the shift parameter.
    `$result = ups($source, $shift)`
  }];
  let builders = [
    OpBuilder<(ins "Value":$source, "int8_t":$shift),
    [{build($_builder, $_state, source.getType(), source, shift);}]>
  ];
}

def AIEVec_SRSOp:
  AIEVec_Op<"srs", [
    NoSideEffect
  ]>, 
  Arguments<(ins AIEVec_Acc:$source, 
          DefaultValuedAttr<Confined<I8Attr, [IntNonNegative]>, "0">:$shift)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE srs";
  let description = [{
    Xilinx-specific shift-round-saturate intrinsic. Moves values from 
    accumulator data type to AIE vector data types. The adjustment in 
    precision is controlled by the shift parameter.
    `$result = srs($source, $shift)`
  }];
}

def AIEVec_UPDOp:
  AIEVec_Op<"upd", [
    NoSideEffect,
    AttrSizedOperandSegments
  ]>, 
  Arguments<(ins AnyShaped:$source, 
                 Variadic<Index>:$indices, 
                 DefaultValuedAttr<SI32Attr, "0">:$offset, 
                 DefaultValuedAttr<Confined<I8Attr, 
                        [IntMinValue<0>, IntMaxValue<1>]>, "0">:$index,
                 Optional<AnyVector>:$vector)>, 
  Results<(outs AnyVector:$result)> {
  let summary = "AIE upd";
  let description = [{
    Xilinx-specific update intrinsic. General upd intrinsic updates contiguous
    lanes of the result vector from a smaller source vector. This form of 
    upd intrinsic combines the load of data from memory into a vector 
    register, and then updating the lanes of the result vector using it. 
    `$result = upd($source[$indices], $offset, $index)`
  }];
  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$source, 
                   "ValueRange":$indices, 
                   "int32_t":$offset, "int8_t":$index),
    [{build($_builder, $_state, resultType, source, indices, 
                   offset, index, nullptr);}]>
  ];
}

def AIEVec_ConcatOp:
  AIEVec_Op<"concat", [
    NoSideEffect
  ]>, 
  Arguments<(ins Variadic<AnyVector>:$sources)>, 
  Results<(outs AnyVector:$result)> {
  let summary = "AIE concat";
  let description = [{
    Xilinx-specific concat intrinsic. Concatenates two or more smaller
    vectors into a bigger vector. The verifier confirms that all the
    input vectors have the same number of lanes.
    `$result = concat($sources[0], $sources[1], ...)`
  }];
}

def AIEVec_ExtOp:
  AIEVec_Op<"ext", [
    NoSideEffect
  ]>, 
  Arguments<(ins AnyVector:$source, 
             Confined<I8Attr, [IntMinValue<0>, IntMaxValue<8>]>:$index)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE ext";
  let description = [{
    Xilinx-specific vector extract intrinsic. Selects contiguous lanes from 
    the source vector, and transfers the data from those lanes to the 
    result. The lane selection is controlled by index.
    `$result = ext($source, $index)`
  }];
}

def AIEVec_SelectOp:
  AIEVec_Op<"select", [
    NoSideEffect
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
    Xilinx-specific vector lane selection operation. It selects between the
    first set of lanes or the second one according to the value in 'select'. 
    If the bit in select is 0(1), it returns the value in the first(second) 
    set of lanes.
    `$result = `select32($select, $xbuff, $xstart, $xoffsets, $ystart, $yoffsets)`
  }];
  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$xbuff, "StringRef":$select, 
            "StringRef":$xstart,
            "StringRef":$xoffsets, "StringRef":$xoffsets_hi,
            "StringRef":$xsquare,
            "StringRef":$ystart,
            "StringRef":$yoffsets, "StringRef":$yoffsets_hi,
            "StringRef":$ysquare),
    [{build($_builder, $_state, resultType, xbuff, select,
            xstart, xoffsets, xoffsets_hi, xsquare,
            ystart, yoffsets, yoffsets_hi, ysquare, nullptr);}]>
  ];
  let extraClassDeclaration = [{
    // Get the attributes
    StringRef getStart(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? xstart() : ystart(); }
    StringRef getOffset(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? xoffsets() : yoffsets(); }
    StringRef getOffsetHi(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? xoffsets_hi() : yoffsets_hi(); } 
    StringRef getSquare(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? xsquare() : ysquare(); }
    // Get the attribute names
    StringRef getStartAttrName(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? "xstart" : "ystart"; }
    StringRef getOffsetAttrName(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? "xoffsets" : "yoffsets"; }
    StringRef getOffsetHiAttrName(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? "xoffsets_hi" : "yoffsets_hi"; } 
    StringRef getSquareAttrName(int idx) { assert(idx==0 || idx==1); 
                        return idx==0 ? "xsquare" : "ysquare"; }
  }];
}

def AIEVec_PackOp:
  AIEVec_Op<"pack", [
    NoSideEffect
  ]>, 
  Arguments<(ins AnyVector:$source)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE pack";
  let description = [{
    Xilinx-specific pack intrinsic. Pack a vector of 16-bit values into
    a vector of 8-bit values.
    `$result = pack($source)`
  }];
}

def AIEVec_UnpackOp:
  AIEVec_Op<"unpack", [
    NoSideEffect
  ]>, 
  Arguments<(ins AnyVector:$source)>,
  Results<(outs AnyVector:$result)> {
  let summary = "AIE unpack";
  let description = [{
    Xilinx-specific unpack intrinsic. Unpack a vector of 8-bit values into
    a vector of 16-bit values.
    `$result = unpack($source)`
  }];
}

#endif // AIEVEC_OPS
