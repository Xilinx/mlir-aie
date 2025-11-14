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

def CoreWire:   I32EnumAttrCase<"Core", 0>;
def DMAWire:    I32EnumAttrCase<"DMA", 1>;
def FIFOWire:   I32EnumAttrCase<"FIFO", 2>;
def SouthWire:  I32EnumAttrCase<"South", 3>;
def WestWire:   I32EnumAttrCase<"West", 4>;
def NorthWire:  I32EnumAttrCase<"North", 5>;
def EastWire:   I32EnumAttrCase<"East", 6>;
def PLIOWire:   I32EnumAttrCase<"PLIO", 7>;
def NOCWire:    I32EnumAttrCase<"NOC", 8>;
def TraceWire:  I32EnumAttrCase<"Trace", 9>;
def TileControlWire:  I32EnumAttrCase<"TileControl", 10>;

def WireBundle: I32EnumAttr<"WireBundle", "Bundle of wires",
  [
    CoreWire, DMAWire, FIFOWire, SouthWire, WestWire, NorthWire,
    EastWire, PLIOWire, NOCWire, TraceWire, TileControlWire
  ]> {

  let cppNamespace = "xilinx::AIE";
}

def CascadeDir: I32EnumAttr<"CascadeDir", "Directions for cascade",
  [
    SouthWire, WestWire, NorthWire, EastWire
  ]> {

  let cppNamespace = "xilinx::AIE";
}

def LockAction: I32EnumAttr<"LockAction", "lock acquire/release",
  [
    I32EnumAttrCase<"Acquire", 0>,
    I32EnumAttrCase<"AcquireGreaterEqual", 2>,
    I32EnumAttrCase<"Release", 1>,
  ]> {

  let cppNamespace = "xilinx::AIE";
}

def LockBlocking: I32EnumAttr<"LockBlocking", "lock operation is blocking",
  [
    I32EnumAttrCase<"NonBlocking", 0>,
    I32EnumAttrCase<"Blocking", 1>
  ]> {

  let cppNamespace = "xilinx::AIE";
}

def AIEArch: I32EnumAttr<"AIEArch", "AIE Architecture",
  [
    I32EnumAttrCase<"AIE1", 1>,
    I32EnumAttrCase<"AIE2", 2>,
    I32EnumAttrCase<"AIE2p", 3>,
  ]> {

  let cppNamespace = "xilinx::AIE";
}

def AIEDevice: I32EnumAttr<"AIEDevice", "AIE Device",
  [
    I32EnumAttrCase<"xcvc1902", 1>,
    I32EnumAttrCase<"xcve2302", 2>,
    I32EnumAttrCase<"xcve2802", 3>,
    I32EnumAttrCase<"npu1", 4>,
    I32EnumAttrCase<"npu1_1col", 5>,
    I32EnumAttrCase<"npu1_2col", 6>,
    I32EnumAttrCase<"npu1_3col", 7>,
    I32EnumAttrCase<"npu2", 8>,
    I32EnumAttrCase<"npu2_1col", 9>,
    I32EnumAttrCase<"npu2_2col", 10>,
    I32EnumAttrCase<"npu2_3col", 11>,
    I32EnumAttrCase<"npu2_4col", 12>,
    I32EnumAttrCase<"npu2_5col", 13>,
    I32EnumAttrCase<"npu2_6col", 14>,
    I32EnumAttrCase<"npu2_7col", 15>,
  ]> {

  let cppNamespace = "xilinx::AIE";
}

def ObjectFifoPort: I32EnumAttr<"ObjectFifoPort",
  "Ports of an object FIFO",
  [
    I32EnumAttrCase<"Produce", 0>,
    I32EnumAttrCase<"Consume", 1>
  ]
  > {
  let cppNamespace = "xilinx::AIE";
}

def DMAChannelDir: I32EnumAttr<"DMAChannelDir",
  "DMA Channel direction",
  [
    I32EnumAttrCase<"S2MM", 0>,
    I32EnumAttrCase<"MM2S", 1>
  ]> {
  let cppNamespace = "xilinx::AIE";
}

def BDDimLayoutAttr : AttrDef<AIE_Dialect, "BDDimLayout", []> {
  let mnemonic = "bd_dim_layout";
  let summary = [{
    Tuple encoding the stride and size of one dimension in an AIE2 n-dimensional
    buffer descriptor;
  }];

  let parameters = (ins
    "uint32_t" : $size,
    "uint32_t" : $stride
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def BDDimLayoutArrayAttr : ArrayOfAttr<
    /*dialect*/AIE_Dialect,
    /*attrName*/"BDDimLayoutArray",
    /*attrMnemonic*/"bd_dim_layout_array",
    /*eltName*/BDDimLayoutAttr.cppClassName
>;

def BDDimLayoutArrayArrayAttr : ArrayOfAttr<
    /*dialect*/AIE_Dialect,
    /*attrName*/"BDDimLayoutArrayArray",
    /*attrMnemonic*/"bd_dim_layout_array_array",
    /*eltName*/BDDimLayoutArrayAttr.cppClassName
>;

def BDPadLayoutAttr : AttrDef<AIE_Dialect, "BDPadLayout", []> {
  let mnemonic = "bd_pad_layout";
  let summary = [{
    Tuple encoding number of zeros before and after on that dimension in an AIE2 
    n-dimensional buffer descriptor;
  }];

  let parameters = (ins
    "uint16_t" : $const_pad_before,
    "uint16_t" : $const_pad_after
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def BDPadLayoutArrayAttr : ArrayOfAttr<
    /*dialect*/AIE_Dialect,
    /*attrName*/"BDPadLayoutArray",
    /*attrMnemonic*/"bd_pad_layout_array",
    /*eltName*/BDPadLayoutAttr.cppClassName
>;

def PacketInfoAttr : AttrDef<AIE_Dialect, "PacketInfo", []> {
  let mnemonic = "packet_info";
  let summary = [{
    Tuple encoding the type and header of a packet;
  }];

  let parameters = (ins
    "uint16_t" : $pkt_type,
    "uint16_t" : $pkt_id
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def InitValuesArrayAttr
  : OptionalAttr<TypedArrayAttrBase<ElementsAttr, "array of ElementsAttr">>;

//===----------------------------------------------------------------------===//
// Generated AIE event enumerations
//===----------------------------------------------------------------------===//

include "AIEEvents.td.inc"
include "AIEEventsAIE2.td.inc"
include "AIEEventsAIE2P.td.inc"

#endif // AIE_ATTRS
