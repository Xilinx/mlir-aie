//===- CoreOps.td ------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_CORE_OPS_TD
#define MLIR_PHY_DIALECT_PHYSICAL_CORE_OPS_TD

include "phy/Dialect/Physical/PhysicalBase.td"

include "mlir/Interfaces/CallInterfaces.td"

//===----------------------------------------------------------------------===//
// CoreOp
//===----------------------------------------------------------------------===//

def CoreOp : Physical_Op<"core",
    [CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {

  let summary = "processing core creation operation";
  let description = [{
    The `physical.core` operation represents a creation of a processing core
    that has the function argument as its entry point.  The processing core
    will be free-running and the function will be invoked.  The function must
    be within the same symbol scope as the operation.  The operands must match
    the specified function type. The function is encoded as a symbol reference
    attribute named `callee`.

    Example:

    ```mlir
    %core = physical.core @my_adder(%buf)
          : (memref<1024xi32>) -> !physical.core
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                   Variadic<AnyCoreArgsType>:$operands);
  let results = (outs CoreType:$core);

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
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, $core)
  }];
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_CORE_OPS_TD
