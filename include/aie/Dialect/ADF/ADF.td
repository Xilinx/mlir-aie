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

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Pass/PassBase.td"

def ADF_Dialect : Dialect {
  let name = "ADF";
  let summary = "Types and operations for the ADF dialect";
  let description = [{
    This is a mlir dialect designed to capture the Xilinx Adaptive Data Flow
    (ADF) Graph semantics. The dialect is designed to perform C++ ADF graph in
    mind; it uses operations like create graph input, create graph output, and
    create kernel along with an new interface type to reflect the real
    connections in the C++ ADF graph.
  }];
  let cppNamespace = "::xilinx::ADF";
}

//===----------------------------------------------------------------------===//
// Base ADF operation definition.
//===----------------------------------------------------------------------===//
class ADF_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ADF_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// ADF Types
//===----------------------------------------------------------------------===//
def ADF_InterfaceType :
    DialectType<ADF_Dialect, CPred<"$_self.isa<InterfaceType>()">,
                "ADF interface type">;


//===----------------------------------------------------------------------===//
// ADF Operations
//===----------------------------------------------------------------------===//

def CreateADFGraphOp : ADF_Op<"graph", [NoSideEffect, NoTerminator]> {
  let summary = "create an ADF graph";
  let description = [{
    Create an ADF graph region which consists of ADF operations. The graph
    operation also need to specify several string attributes for C++ ADF graph
    generation, including "graph_name", "graph_input_name", and
    "graph_output_name". For example, 

    ```mlir
    ADF.graph {
      your_graph_operations
    } {graph_name = "simpleStream", graph_input_name = "gin", graph_output_name = "gout"} 
    ```
  }];

  let arguments = (StrAttr:$graph_name);

  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{ 
     regions attr-dict 
  }];
}

def CreateGraphInputOp : ADF_Op<"create_ginp", [ConstantLike, NoSideEffect]> {
  let summary = "create_ginp operation";
  let description = [{
    Create an input for the ADF. It takes an array attribute as the input
    argument. The constant values of the array attribute defines the interface
    of whole ADF graph. The first index is a bool integer and 1:i1 means the ADF
    graph is based on stream data access mechanism; 0:i1 means it's an adf
    window data access. If it's the stream mechanism, then the second index
    should set to -1, otherwise, the second index should set to the window size.
    e.g. 128 byte.  

    The create_ginp will returns an outputs with "itf" type to propagate the
    interface values defined by the argument array attributes. "itf" is the
    abbreviation of "interface". 
    
    ```mlir
      %0 = ADF.create_ginp  [1:i1, -1:i32] -> !ADF.itf<i1, i32>  // for ADF stream access mechanism

      or 

      %1 = ADF.create_ginp  [0:i1, 128:i32] -> !ADF.itf<i1, i32> // for ADF window access mechanism 
    ```
  }];

  let arguments = (ins ArrayAttr:$value);
  let results = (outs ADF_InterfaceType:$output);

  let assemblyFormat = "$value attr-dict `->` type($output)";
}


def CreateKernelOp : ADF_Op<"create_kernel", [NoSideEffect]> {
  let summary = "call and instantiate a kernel node with proper connections";
  let description = [{
    Perform a generic kernel instantiattion (i.e. function call). This Op will
    instantiate a kernel node with a specific external kernel fucntion. The
    operation takes "itf" type for both its variadic inputs and the op_return to propagate the
    window size through entire adf graph. 


    ```mlir

      %3 = ADF.create_kernel @kfunc2(%1, %2) : (!ADF.itf<i1, i32>, !ADF.itf<i1, i32>) 
                                             -> !ADF.itf<i1, i32>
    ```
  }];


  // The kernel call operation takes a symbol reference attribute as the callee,
  // and inputs for the call
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<ADF_InterfaceType>:$kernel_inputs);


  // The create_kernel operation returns a single value of Integer type (or wireWindow).
  let results = (outs ADF_InterfaceType);

  // no need parse and printer if you have defined a assemblyFormat
  // let parser = [{ return ::parseConstantOp(parser, result); }];
  // let printer = [{ return ::print(p, *this); }];
  let assemblyFormat = [{
    $callee `(` $kernel_inputs `)` attr-dict `:` functional-type($kernel_inputs, results)
  }];
}


def CreateGraphOutputOp : ADF_Op<"create_gout", [NoSideEffect]> {
  let summary = "create_graph_out operation";
  let description = [{
    create an adf graph output. The operation takes an argument with "itf" type and also returns an
    "itf" type output for final type checking. 

    ```mlir
      %2 = ADF.create_gout %3 : (!ADF.itf<i1, i32>) 
                              -> !ADF.itf<i1, i32>                                            
    ```
  }];

  // The create_graph_out operation takes an integer type as the only input.
  let arguments = (ins ADF_InterfaceType:$inp);

  // The create_graph_out operation returns a single value of Integer type (or wireWindow).
  let results = (outs ADF_InterfaceType);

  let assemblyFormat = [{
    $inp attr-dict `:` `(`type($inp) `)` `->` type(results)
  }];
}


#endif // ADF_TD


