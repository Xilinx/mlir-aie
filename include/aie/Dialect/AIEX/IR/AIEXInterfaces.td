//===- AIEXInterfaces.td -----------------------------------*- tablegen -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIEX_INTERFACES
#define AIEX_INTERFACES

include "mlir/IR/OpBase.td"

def DMAConfigureTaskLike : OpInterface<"DMAConfigureTaskLike"> {
  let description = [{
    Interface for the ops that configure a runtime DMA task: the region holding
    its buffer descriptor chain, and how a start of it runs.
  }];
  let cppNamespace = "::xilinx::AIEX";
  let methods =
      [InterfaceMethod<[{Return the region holding the task's descriptors.}],
                       "mlir::Region &", "getBody", (ins), [{}],
                       [{ return $_op.getBody(); }]>,
       InterfaceMethod<[{Return the compile-time repeat count.}], "uint32_t",
                       "getRepeatCount", (ins), [{}],
                       [{ return $_op.getRepeatCount(); }]>,
       InterfaceMethod<[{Set the compile-time repeat count.}], "void",
                       "setRepeatCount", (ins "uint32_t":$count), [{}],
                       [{ $_op.setRepeatCount(count); }]>,
       InterfaceMethod<[{Return the runtime repeat count, or null.}],
                       "mlir::Value", "getRepeatCountVal", (ins), [{}],
                       [{ return $_op.getRepeatCountVal(); }]>,
       InterfaceMethod<[{Return whether the task issues a completion token.}],
                       "bool", "getIssueToken", (ins), [{}],
                       [{ return $_op.getIssueToken(); }]>,
       InterfaceMethod<[{Set whether the task issues a completion token.}],
                       "void", "setIssueToken", (ins "bool":$token), [{}],
                       [{ $_op.setIssueToken(token); }]>,
  ];
}

#endif // AIEX_INTERFACES
