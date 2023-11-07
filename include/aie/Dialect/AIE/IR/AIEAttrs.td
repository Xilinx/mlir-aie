//===- AIEAttrs.td -----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_ATTRS
#define AIE_ATTRS

include "aie/Dialect/AIE/IR/AIE.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Overridden upstream attributes that return int instead of uint
//===----------------------------------------------------------------------===//

class AIETypedSignlessIntegerAttrBase<I attrValType, string retType, string descr>
    : SignlessIntegerAttrBase<attrValType, descr> {
  let returnType = retType;
  let convertFromStorage = "$_self.getValue().getSExtValue()";
}

def AIEI8Attr  : AIETypedSignlessIntegerAttrBase<
    I8,  "int8_t",  "8-bit signless integer attribute">;
def AIEI16Attr : AIETypedSignlessIntegerAttrBase<
    I16, "int16_t", "16-bit signless integer attribute">;
def AIEI32Attr : AIETypedSignlessIntegerAttrBase<
    I32, "int32_t", "32-bit signless integer attribute">;
def AIEI64Attr : AIETypedSignlessIntegerAttrBase<
    I64, "int64_t", "64-bit signless integer attribute">;

//===----------------------------------------------------------------------===//
// AIE attributes.
//===----------------------------------------------------------------------===//

def Core: I32EnumAttrCase<"Core", 0>;
def DMA: I32EnumAttrCase<"DMA", 1>;
def FIFO: I32EnumAttrCase<"FIFO", 2>;
def South: I32EnumAttrCase<"South", 3>;
def West: I32EnumAttrCase<"West", 4>;
def North: I32EnumAttrCase<"North", 5>;
def East: I32EnumAttrCase<"East", 6>;
def PLIO: I32EnumAttrCase<"PLIO", 7>;
def NOC: I32EnumAttrCase<"NOC", 8>;
def Trace: I32EnumAttrCase<"Trace", 9>;

def WireBundle: I32EnumAttr<"WireBundle", "Bundle of wires",
  [
    Core,
    DMA,
    FIFO,
    South,
    West,
    North,
    East,
    PLIO,
    NOC,
    Trace
  ]> {

  let cppNamespace = "xilinx::AIE";
}

def Produce: I32EnumAttrCase<"Produce", 0>;
def Consume: I32EnumAttrCase<"Consume", 1>;

def Acquire: I32EnumAttrCase<"Acquire", 0>;
def Release: I32EnumAttrCase<"Release", 1>;
def AcquireGreaterEqual: I32EnumAttrCase<"AcquireGreaterEqual", 2>;

def NonBlocking: I32EnumAttrCase<"NonBlocking", 0>;
def Blocking: I32EnumAttrCase<"Blocking", 1>;

def AIE1: I32EnumAttrCase<"AIE1", 1>;
def AIE2: I32EnumAttrCase<"AIE2", 2>;

def xcvc1902: I32EnumAttrCase<"xcvc1902", 1>;
def xcve2302: I32EnumAttrCase<"xcve2302", 2>;
def xcve2802: I32EnumAttrCase<"xcve2802", 3>;
def ipu: I32EnumAttrCase<"ipu", 4>;

def LockAction: I32EnumAttr<"LockAction", "lock acquire/release",
  [
    Acquire,
    AcquireGreaterEqual,
    Release
  ]> {

  let cppNamespace = "xilinx::AIE";
}

def LockBlocking: I32EnumAttr<"LockBlocking", "lock operation is blocking",
  [
    NonBlocking,
    Blocking
  ]> {

  let cppNamespace = "xilinx::AIE";
}
def AIEArch: I32EnumAttr<"AIEArch", "AIE Architecture",
  [
    AIE1,
    AIE2
  ]> {

  let cppNamespace = "xilinx::AIE";
}
def AIEDevice: I32EnumAttr<"AIEDevice", "AIE Device",
  [
    xcvc1902,
    xcve2302,
    xcve2802,
    ipu
  ]> {

  let cppNamespace = "xilinx::AIE";
}

def ObjectFifoPort: I32EnumAttr<"ObjectFifoPort",
  "Ports of an object FIFO",
  [
    Produce,
    Consume
  ]
  > {
  let cppNamespace = "xilinx::AIE";
}

def S2MM:  I32EnumAttrCase<"S2MM", 0>;
def MM2S:  I32EnumAttrCase<"MM2S", 1>;

def DMAChannelDir: I32EnumAttr<"DMAChannelDir",
  "DMA Channel direction",
  [
    S2MM,
    MM2S
  ]> {
  let cppNamespace = "xilinx::AIE";
}

def AIE_DimTupleAttr : AttrDef<AIE_Dialect, "DimTuple", []> {
  let mnemonic = "DimTuple";
  let summary = [{
    Tuple encoding the stride and wrap of one dimension in an AIE2 n-dimensional
    buffer descriptor;
  }];

  let parameters = (ins
    "uint32_t" : $stepsize,
    "uint16_t" : $wrap
  );

  let assemblyFormat = "`<` $wrap `,` $stepsize `>`";
}

def AIE_DimTupleArrayAttr : ArrayOfAttr<
    /*dialect*/AIE_Dialect,
    /*attrName*/"DimTupleArray",
    /*attrMnemonic*/"DimTupleArray",
    /*eltName*/AIE_DimTupleAttr.cppClassName
>;

def AIE_DimTupleArrayArrayAttr : ArrayOfAttr<
    /*dialect*/AIE_Dialect,
    /*attrName*/"DimTupleArrayArray",
    /*attrMnemonic*/"DimTupleArrayArray",
    /*eltName*/AIE_DimTupleArrayAttr.cppClassName
>;

#endif // AIE_ATTRS