//===- AIEXAttrs.td ----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIEX_ATTRS
#define AIEX_ATTRS

include "mlir/IR/EnumAttr.td"

def StateTableFuncMul  : I32EnumAttrCase<"Mul", 0, "mul">;
def StateTableFuncIncr : I32EnumAttrCase<"Incr", 1, "incr">;
def StateTableFuncDecr : I32EnumAttrCase<"Decr", 2, "decr">;

def StateTableFunc : I32EnumAttr<"StateTableFunc",
    "Function to apply between scratchpad value and register pair",
    [StateTableFuncMul, StateTableFuncIncr, StateTableFuncDecr]> {
  let cppNamespace = "::xilinx::AIEX";
}

def ParameterKindCore : I32EnumAttrCase<"Core", 0, "core">;
def ParameterKindAddr : I32EnumAttrCase<"Addr", 1, "addr">;

def ParameterKind : I32EnumAttr<"ParameterKind",
    "Usage kind of a scratchpad runtime parameter",
    [ParameterKindCore, ParameterKindAddr]> {
  let cppNamespace = "::xilinx::AIEX";
}

#endif // AIEX_ATTRS
