//===- ADF.td - ADF dialect operation definitions ----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
//
// Defines the operations and types of the ADF dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ADF_TD
#define ADF_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Pass/PassBase.td"

def ADF_Dialect : Dialect {
  let name = "ADF";
  let summary = "Types and operations for the ADF dialect";
  let description = [{
    This is an MLIR dialect designed to capture the Xilinx Adaptive Data Flow
    (ADF) Graph semantics.  This dialect is intended to be converible to/from the
    C++ graph syntax, but to represent the graph abstractly, rather than at
    the C++ syntactic level.
  }];
  let cppNamespace = "::xilinx::ADF";
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base ADF operation definition.
//===----------------------------------------------------------------------===//
class ADF_Op<string mnemonic, list<Trait> traits = []> :
    Op<ADF_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// ADF Types
//===----------------------------------------------------------------------===//
def ADF_InterfaceType : TypeDef<ADF_Dialect, "Interface"> {
  let mnemonic = "interface";
  let parameters = (ins "mlir::Type":$type);
  let assemblyFormat = "`<` $type `>`";
}

def ADF_WindowType : TypeDef<ADF_Dialect, "Window"> {
  let mnemonic = "window";
  let parameters = (ins "mlir::Type":$type, "int":$size, "int":$overlap);
  let assemblyFormat = "`<` $type `,` $size `,` $overlap `>`";
}
def ADF_StreamType : TypeDef<ADF_Dialect, "Stream"> {
  let mnemonic = "stream";
  let parameters = (ins "mlir::Type":$type);
  let assemblyFormat = "`<` $type `>`";
}
def ADF_ParameterType : TypeDef<ADF_Dialect, "Parameter"> {
  let mnemonic = "parameter";
  let parameters = (ins "mlir::Type":$type);
  let assemblyFormat = "`<` $type `>`";
}

class ADF_DataType<string name> : TypeDef<ADF_Dialect, name> {
  let mnemonic = name;
}

def int8 : ADF_DataType<"int8">;
def int16 : ADF_DataType<"int16">;
def int32 : ADF_DataType<"int32">;
def int64 : ADF_DataType<"int64">;
def uint8 : ADF_DataType<"uint8">;
def uint16 : ADF_DataType<"uint16">;
def uint32 : ADF_DataType<"uint32">;
def uint64 : ADF_DataType<"uint64">;
def cint16 : ADF_DataType<"cint16">;
def cint32 : ADF_DataType<"cint32">;
def float : ADF_DataType<"float">;
def cfloat : ADF_DataType<"cfloat">;
def accfloat : ADF_DataType<"accfloat">;
def caccfloat : ADF_DataType<"caccfloat">;
def acc48 : ADF_DataType<"acc48">;
def cacc48 : ADF_DataType<"cacc48">;
def acc80 : ADF_DataType<"acc80">;
def cacc80 : ADF_DataType<"cacc80">;

//===----------------------------------------------------------------------===//
// ADF Operations
//===----------------------------------------------------------------------===//

def GraphOp : ADF_Op<"graph", [Pure, SingleBlock, NoTerminator]> {
  let summary = "An ADF graph operation";
  let description = [{
    An operation that represents an ADF graph.  This operation determines
    the name of the graph and contains a single region containing ADF
    dialect operations. For example, 

    ```mlir
    ADF.graph("name") {
      your_graph_operations
    }
    ```
  }];

  let arguments = (ins StrAttr:$name);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = "`(` $name `)` regions attr-dict";
}

def GraphInputOp : ADF_Op<"input_port", [ConstantLike, Pure]> {
  let summary = "Graph input port operation";
  let description = [{
    An operation representing a primary input of an ADF graph. 
    It takes an array attribute as the input
    argument. The constant values of the array attribute defines the interface
    of whole ADF graph. The first index is a bool integer and 1:i1 means the ADF
    graph is based on stream data access mechanism; 0:i1 means it's an adf
    window data access. If it's the stream mechanism, then the second index
    should set to -1, otherwise, the second index should set to the window size.
    e.g. 128 byte.  

    The input_port will return an output with "interface" type to propagate the
    interface values defined by the argument array attributes.
    
    ```mlir
      %0 = ADF.input_port("name")  [1:i1, -1:i32] -> !ADF.interface<!ADF.int32>  // for ADF stream access mechanism
    ```
      or 
    ```mlir
      %1 = ADF.input_port("name")  [0:i1, 128:i32] -> !ADF.interface<!ADF.int32> // for ADF window access mechanism 
    ```
  }];

  let arguments = (ins ArrayAttr:$value, StrAttr:$name);
  let results = (outs ADF_InterfaceType:$output);

  let assemblyFormat = "`(` $name `)` $value attr-dict `->` type($output)";
}


def KernelOp : ADF_Op<"kernel", [Pure]> {
  let summary = "call and instantiate a kernel node with proper connections";
  let description = [{
    An operation representing a primitive node in the graph.
    This operation refers to a function implementing the kernel.  It takes "window"
    types for both its variadic inputs and the op_return to propagate the
    window size through entire ADF graph. 

    ```mlir
      %3 = ADF.kernel @kfunc2(%1, %2) :
              (!ADF.interface<!ADF.int32>, !ADF.interface<!ADF.int32>) -> !ADF.interface<!ADF.int32>
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<ADF_InterfaceType>:$kernel_inputs);
  let results = (outs ADF_InterfaceType);

  let assemblyFormat = [{
    $callee `(` $kernel_inputs `)` attr-dict `:` functional-type($kernel_inputs, results)
  }];
}


def GraphOutputOp : ADF_Op<"output_port", [Pure]> {
  let summary = "Graph output port operation";
  let description = [{
    An operation representing a primary input of an ADF graph. 
    The operation takes an argument with "window" type and also returns an
    "window" type output for final type checking. 

    ```mlir
      %2 = ADF.output_port("name") %3 : (!ADF.interface<!ADF.int32>) -> !ADF.interface<!ADF.int32>                                           
    ```
  }];

  let arguments = (ins ADF_InterfaceType:$inp, StrAttr:$name);
  let results = (outs ADF_InterfaceType);
  let assemblyFormat = [{
    `(` $name `)` $inp attr-dict `:` `(`type($inp) `)` `->` type(results)
  }];
}

def GraphInOutOp : ADF_Op<"inout_port", [Pure]> {
  let summary = "Graph input/output port operation";
  let description = [{
    An operation representing a dual input/output of an ADF graph.
    This is usually used to represent inout parameter val=ues

    ```mlir
      %2 = ADF.inout_port("name") %3 : (!ADF.interface<!ADF.int32>) -> !ADF.interface<!ADF.int32>                                           
    ```
  }];

  let arguments = (ins ADF_InterfaceType:$inp, StrAttr:$name);
  let results = (outs ADF_InterfaceType);
  let assemblyFormat = [{
    `(` $name `)` $inp attr-dict `:` `(`type($inp) `)` `->` type(results)
  }];
}


#endif // ADF_TD


