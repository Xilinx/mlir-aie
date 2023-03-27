//===- AIEInterfaces.td ------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/IR/EnumAttr.td"

def FlowEndPoint : OpInterface<"FlowEndPoint"> {
  let description = [{
    Interface for operations that have interconnect-like properties,
    enabling them to host flows for routing.
  }];
  let cppNamespace = "::xilinx::AIE";
  let methods = [
    InterfaceMethod<[{
      }],
      "int", "colIndex", (ins )
    >,
    InterfaceMethod<[{
      }],
      "int", "rowIndex", (ins )
    >
  ];
}

def Interconnect : OpInterface<"Interconnect"> {
  let description = [{
    Interface for operations that have interconnect-like properties,
    enabling them to host flows for routing.
  }];
  let cppNamespace = "::xilinx::AIE";

  let methods = [
    InterfaceMethod<[{
      }],
      "::mlir::Region &", "getConnections", (ins )
    >,
    InterfaceMethod<[{
      }],
      "int", "colIndex", (ins )
    >,
    InterfaceMethod<[{
      }],
      "int", "rowIndex", (ins )
    >,
    InterfaceMethod<[{
        int FooOp::getNumSourceConnections(WireBundle bundle) {
      }],
      "int", "getNumSourceConnections", (ins "WireBundle":$bundle)
    >,
    InterfaceMethod<[{
        int FooOp::getNumDestConnections(WireBundle bundle) {
      }],
      "int", "getNumDestConnections", (ins "WireBundle":$bundle)
    >
  ];
}

def TileElement : OpInterface<"TileElement"> {
  let description = [{
    Interface for operations that exist in a TileOp.
  }];
  let cppNamespace = "::xilinx::AIE";
  let methods = [
    InterfaceMethod<[{
        Return the location of the Tile where the element is located.
      }],
      "xilinx::AIE::TileID", "getTileID", (ins ),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getTileOp().getTileID();
      }]
    >
  ];
}

def AIETarget : OpInterface<"AIETarget"> {
  let description = [{
    Interface for operations that model an array of AIEngine cores.
  }];
  let cppNamespace = "::xilinx::AIE";
  let methods = [
    InterfaceMethod<[{
        Return the target model describing the characteristics of how this operation will be implemented.
      }],
      "const ::xilinx::AIE::AIETargetModel&", "getTargetModel", (ins )
    >
  ];
}

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
  [Core, DMA, FIFO, South, West, North, East, PLIO, NOC, Trace]> {

  let cppNamespace = "xilinx::AIE";
}

def Produce: I32EnumAttrCase<"Produce", 0>;
def Consume: I32EnumAttrCase<"Consume", 1>;

def Acquire: I32EnumAttrCase<"Acquire", 0>;
def Release: I32EnumAttrCase<"Release", 1>;
def NonBlocking: I32EnumAttrCase<"NonBlocking", 0>;
def Blocking: I32EnumAttrCase<"Blocking", 1>;

def xcvc1902: I32EnumAttrCase<"xcvc1902", 1>;
def xcve2302: I32EnumAttrCase<"xcve2302", 2>;
def xcve2802: I32EnumAttrCase<"xcve2802", 3>;

def LockAction: I32EnumAttr<"LockAction", "lock acquire/release",
  [Acquire, Release]> {

  let cppNamespace = "xilinx::AIE";
}
def LockBlocking: I32EnumAttr<"LockBlocking", "lock operation is blocking",
  [NonBlocking, Blocking]> {

  let cppNamespace = "xilinx::AIE";
}

def AIEDevice: I32EnumAttr<"AIEDevice", "AIE Device",
  [xcvc1902, xcve2302, xcve2802]> {

  let cppNamespace = "xilinx::AIE";
}