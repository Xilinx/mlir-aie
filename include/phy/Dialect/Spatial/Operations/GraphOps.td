//===- GraphOps.td -----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_SPATIAL_GRAPH_OPS_TD
#define MLIR_PHY_DIALECT_SPATIAL_GRAPH_OPS_TD

include "phy/Dialect/Spatial/SpatialBase.td"

include "mlir/Interfaces/CallInterfaces.td"

//===----------------------------------------------------------------------===//
// NodeOp
//===----------------------------------------------------------------------===//

def NodeOp : Spatial_Op<"node",
    [CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {

  let summary = "compute node creation operation";
  let description = [{
    The `spatial.node` operation represents a creation of a logical compute node
    that has the function argument as its entry point.  Each node invokes the
    software function `callee` free-running since system startup.  The function
    interacts with queues connected to the node as in `operands`.  The function
    must be within the same symbol scope as the operation.  The `operands` must
    match the specified function type.

    Example:

    ```mlir
    %node = spatial.node @func(%queue)
          : (!spatial.queue<memref<1024xi32>>) -> !spatial.node
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                   Variadic<AnyQueueType>:$operands);
  let results = (outs NodeType:$node);

  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    ::mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<::mlir::SymbolRefAttr>("callee");
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, $node)
  }];
}

//===----------------------------------------------------------------------===//
// QueueOp
//===----------------------------------------------------------------------===//

def QueueOp : Spatial_Op<"queue"> {
  let summary = "message queue creation operation";
  let description = [{
    The `spatial.queue` operation represents a creation of a queue that has
    the type argument as its element's datatype.  The created queue can be used
    as an operand in `spatial.node` operations for the software function to
    access and communicate.

    Example:

    ```mlir
    %queue = spatial.queue<2>(): !spatial.queue<memref<16xi32>>
    ```
  }];

  let arguments = (ins I64Attr:$depth);
  let results = (outs AnyQueueType:$queue);

  let assemblyFormat = [{
    `<` $depth `>` `(` `)` attr-dict `:` type($queue)
  }];
}

//===----------------------------------------------------------------------===//
// BridgeOp
//===----------------------------------------------------------------------===//

def BridgeOp : Spatial_Op<"bridge", [SameTypeOperands]> {
  let summary = "create a compute node that connectes two queues";
  let description = [{
    The `spatial.bridge` operation represents a creation of a bridge, which is
    a simple special node, which bridges two queues, one input, and one output.
    A bridge, whenever the output queue is not full and the input queue is not
    empty, reads the data from the input queue and writes the data into the
    output queue.

    Example:

    ```mlir
    %bridge = spatial.bridge(%q1 -> %q2: !spatial.queue<memref<8xi32>>)
    ```
  }];

  let arguments = (ins AnyQueueType:$src, AnyQueueType:$dest);
  let results = (outs NodeType:$node);

  let assemblyFormat = [{
    `(` $src `->` $dest `:` type($dest) `)` attr-dict
  }];
}

#endif  // MLIR_PHY_DIALECT_SPATIAL_GRAPH_OPS_TD
