//===- XLLVMAIE2IntrOps.td - XLLVM AIE2 intr. op defs. ----*- tablegen -*-====//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Defines external LLVM (XLLVM) intrinsic operations for AIE2 devices.
//===----------------------------------------------------------------------===//


#ifndef AIE_DIALECT_XLLVM_IR_XLLVMAIE2INTROPS_TD
#define AIE_DIALECT_XLLVM_IR_XLLVMAIE2INTROPS_TD

include "aie/Dialect/XLLVM/IR/XLLVM.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// For AIE2 only
class AIEVec2_IntrOp<string mnemonic,
                     list<Trait> traits = [],
                     int numResults = 1> :
    ExtIntrOpBase</*opName =*/"intr.aie2." # mnemonic,
                  /*enumName =*/"aie2." # mnemonic,
                  traits,
                  numResults>;

// TODO: Find better names for these

class AIE2bf16MACConf : 
    Arguments<(ins VectorOfLengthAndType<[32], [BF16]>:$lhs,
                   VectorOfLengthAndType<[32], [BF16]>:$rhs,
                   VectorOfLengthAndType<[8], [I64]>:$acc,
                   I32:$conf)>;

// ----- MAC ----- 

def MacConfAcc32IntrOp :
    AIEVec2_IntrOp<"I512.I512.ACC1024.acc32.mac.conf",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[64], [I8]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs,
                   VectorOfLengthAndType<[16], [I64]>:$acc,
                   I32:$conf)>;

def MacConfAcc64IntrOp :
    AIEVec2_IntrOp<"I512.I512.ACC1024.acc64.mac.conf",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[64], [I8]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs,
                   VectorOfLengthAndType<[16], [I64]>:$acc,
                   I32:$conf)>;

def MacConfBF16IntrOp :
    AIEVec2_IntrOp<"bf.mac16.conf",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I64]>>]>,
        AIE2bf16MACConf;

// ----- MSC ----- 

def MscConfBF16IntrOp :
    AIEVec2_IntrOp<"bf.msc16.conf",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I64]>>]>,
        AIE2bf16MACConf;

// ----- MUL ----- 

def MulConfAcc32IntrOp :
    AIEVec2_IntrOp<"I512.I512.acc32.mul.conf",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[64], [I8]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs,
                   I32:$conf)>;

def MulConfAcc64IntrOp :
    AIEVec2_IntrOp<"I512.I512.acc64.mul.conf",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[64], [I8]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs,
                   I32:$conf)>;

def MulConfBF16IntrOp :
    AIEVec2_IntrOp<"bf.mul16.conf",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [BF16]>:$lhs,
                   VectorOfLengthAndType<[32], [BF16]>:$rhs,
                   I32:$conf)>;

// ----- SET ----- 

def VectorSetI512I128IntrOp :
    AIEVec2_IntrOp<"set.I512.I128",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[4], [I32]>:$src)>;

def VectorSetI512I256IntrOp :
    AIEVec2_IntrOp<"set.I512.I256",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I32]>:$src,
                   I32:$pos)>;

// ----- SRS ----- 

def I256V16Acc32SrsIntrOp :
    AIEVec2_IntrOp<"I256.v16.acc32.srs",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I16]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def I256V16Acc64SrsIntrOp :
    AIEVec2_IntrOp<"I256.v16.acc64.srs",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I16]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def I256V32Acc32SrsIntrOp :
    AIEVec2_IntrOp<"I256.v32.acc32.srs",
        [TypeIs<"res", VectorOfLengthAndType<[32], [I8]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def I256V8Acc64SrsIntrOp :
    AIEVec2_IntrOp<"I256.v8.acc64.srs",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def I512V16Acc64SrsIntrOp :
    AIEVec2_IntrOp<"I512.v16.acc64.srs",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def I512V32Acc32SrsIntrOp :
    AIEVec2_IntrOp<"I512.v32.acc32.srs",
        [TypeIs<"res", VectorOfLengthAndType<[32], [I16]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def Vector16AccFloatToV16BF16IntrOp :
    AIEVec2_IntrOp<"v16accfloat.to.v16bf16",
        [TypeIs<"res", VectorOfLengthAndType<[16], [BF16]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I64]>:$src)>;

// ----- BROADCAST ----- 

def VectorBroadcast8I512IntrOp :
    AIEVec2_IntrOp<"vbroadcast8.I512",
        [TypeIs<"res", VectorOfLengthAndType<[64], [I8]>>]>,
    Arguments<(ins I32:$src)>;

def VectorBroadcast16I512IntrOp :
    AIEVec2_IntrOp<"vbroadcast16.I512",
        [TypeIs<"res", VectorOfLengthAndType<[32], [I16]>>]>,
    Arguments<(ins I32:$src)>;

def VectorBroadcast32I512IntrOp :
    AIEVec2_IntrOp<"vbroadcast32.I512",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins I32:$src)>;

def VectorBroadcast16BF512IntrOp :
    AIEVec2_IntrOp<"vbroadcast16.bf512",
        [TypeIs<"res", VectorOfLengthAndType<[32], [BF16]>>]>,
    Arguments<(ins BF16:$src)>;

def VectorBroadcastfloatI512IntrOp :
    AIEVec2_IntrOp<"vbroadcastfloat.I512",
        [TypeIs<"res", VectorOfLengthAndType<[16], [F32]>>]>,
    Arguments<(ins F32:$src)>;

// ----- EXT ----- 

def ExtI256I512IntrOp :
    AIEVec2_IntrOp<"ext.I256.I512",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$src,
                   I32:$idx)>;

// ----- CONCAT ----- 

def ConcatI512I256IntrOp :
    AIEVec2_IntrOp<"concat.I512.I256",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I32]>:$lhs,
                   VectorOfLengthAndType<[8], [I32]>:$rhs)>;

def ConcatI1024I256IntrOp :
    AIEVec2_IntrOp<"concat.I1024.I256",
        [TypeIs<"res", VectorOfLengthAndType<[32], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I32]>:$src0,
                   VectorOfLengthAndType<[8], [I32]>:$src1,
                   VectorOfLengthAndType<[8], [I32]>:$src2,
                   VectorOfLengthAndType<[8], [I32]>:$src3)>;

def ConcatI1024I512IntrOp :
    AIEVec2_IntrOp<"concat.I1024.I512",
        [TypeIs<"res", VectorOfLengthAndType<[32], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs)>;

// ----- SHUFFLE ----- 

def VectorShuffleIntrOp :
    AIEVec2_IntrOp<"vshuffle",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs,
                   I32:$mode)>;

// ----- UNDEF ----- 

def UndefV16I32IntrOp :
    AIEVec2_IntrOp<"v16int32",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>;

// ----- UPD ----- 

def UpdBF512BF256IntrOp :
    AIEVec2_IntrOp<"upd.bf512.bf256",
        [TypeIs<"res", VectorOfLengthAndType<[32], [BF16]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [BF16]>:$dst,
                   VectorOfLengthAndType<[16], [BF16]>:$src,
                   I32:$idx)>;

// ----- SHIFT ----- 

def VectorShiftI512I512IntrOp :
    AIEVec2_IntrOp<"vshift.I512.I512",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs,
                   I32:$step,
                   I32:$shift)>;

def VectorShiftBF512BF512IntrOp :
    AIEVec2_IntrOp<"vshift.bf512.bf512",
        [TypeIs<"res", VectorOfLengthAndType<[32], [BF16]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [BF16]>:$lhs,
                   VectorOfLengthAndType<[32], [BF16]>:$rhs,
                   I32:$step,
                   I32:$shift)>;

// ----- EXTRACT ELEMENT ----- 

def VectorExtractElem8I512IntrOp :
    AIEVec2_IntrOp<"vextract.elem8.I512",
        [TypeIs<"res", I32>]>,
    Arguments<(ins VectorOfLengthAndType<[64], [I8]>:$src,
                   I32:$idx,
                   I32:$sign)>;

def VectorExtractElem16I512IntrOp :
    AIEVec2_IntrOp<"vextract.elem16.I512",
        [TypeIs<"res", I32>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [I16]>:$src,
                   I32:$idx,
                   I32:$sign)>;

def VectorExtractElem32I512IntrOp :
    AIEVec2_IntrOp<"vextract.elem32.I512",
        [TypeIs<"res", I32>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$src,
                   I32:$idx,
                   I32:$sign)>;

#endif // AIE_DIALECT_XLLVM_IR_XLLVMAIE2INTROPS_TD
