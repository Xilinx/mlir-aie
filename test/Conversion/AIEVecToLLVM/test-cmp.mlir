//===- test-cmp.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie2" | FileCheck %s -DT=aie2
// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s -DT=aie2p

// aievec.cmp lowers to vmax.lt (lanes where a < b) and vmin.ge (lanes where
// a >= b). gt and le swap the operands; eq and ne combine both directions.
// The integer intrinsics take a last operand that is 0 for an unsigned and 1
// for a signed compare. eq and ne use the signed form. The bf16 intrinsics
// have no such operand, so the unordered (u) and ordered (s) float predicates
// lower alike.

// CHECK-LABEL: func.func @i32_slt
// CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[B:.*]]: vector<16xi32>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmax.lt32"(%[[A]], %[[B]], %[[SIGN]])
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i32_slt(%a: vector<16xi32>, %b: vector<16xi32>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "slt"} : vector<16xi32>, vector<16xi32>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i32_ult
// CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[B:.*]]: vector<16xi32>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmax.lt32"(%[[A]], %[[B]], %[[SIGN]])
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i32_ult(%a: vector<16xi32>, %b: vector<16xi32>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "ult"} : vector<16xi32>, vector<16xi32>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i32_sge
// CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[B:.*]]: vector<16xi32>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmin.ge32"(%[[A]], %[[B]], %[[SIGN]])
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i32_sge(%a: vector<16xi32>, %b: vector<16xi32>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "sge"} : vector<16xi32>, vector<16xi32>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i32_uge
// CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[B:.*]]: vector<16xi32>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmin.ge32"(%[[A]], %[[B]], %[[SIGN]])
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i32_uge(%a: vector<16xi32>, %b: vector<16xi32>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "uge"} : vector<16xi32>, vector<16xi32>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i32_sgt
// CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[B:.*]]: vector<16xi32>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmax.lt32"(%[[B]], %[[A]], %[[SIGN]])
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i32_sgt(%a: vector<16xi32>, %b: vector<16xi32>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "sgt"} : vector<16xi32>, vector<16xi32>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i32_ugt
// CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[B:.*]]: vector<16xi32>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmax.lt32"(%[[B]], %[[A]], %[[SIGN]])
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i32_ugt(%a: vector<16xi32>, %b: vector<16xi32>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "ugt"} : vector<16xi32>, vector<16xi32>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i32_sle
// CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[B:.*]]: vector<16xi32>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmin.ge32"(%[[B]], %[[A]], %[[SIGN]])
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i32_sle(%a: vector<16xi32>, %b: vector<16xi32>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "sle"} : vector<16xi32>, vector<16xi32>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i32_ule
// CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[B:.*]]: vector<16xi32>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmin.ge32"(%[[B]], %[[A]], %[[SIGN]])
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i32_ule(%a: vector<16xi32>, %b: vector<16xi32>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "ule"} : vector<16xi32>, vector<16xi32>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i32_eq
// CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[B:.*]]: vector<16xi32>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[AB:.*]] = "xllvm.intr.[[T]].vmin.ge32"(%[[A]], %[[B]], %[[SIGN]])
// CHECK: %[[MAB:.*]] = llvm.extractvalue %[[AB]][1]
// CHECK: %[[BA:.*]] = "xllvm.intr.[[T]].vmin.ge32"(%[[B]], %[[A]], %[[SIGN]])
// CHECK: %[[MBA:.*]] = llvm.extractvalue %[[BA]][1]
// CHECK: %[[M:.*]] = llvm.and %[[MAB]], %[[MBA]] : i32
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i32_eq(%a: vector<16xi32>, %b: vector<16xi32>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "eq"} : vector<16xi32>, vector<16xi32>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i32_ne
// CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[B:.*]]: vector<16xi32>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[AB:.*]] = "xllvm.intr.[[T]].vmax.lt32"(%[[A]], %[[B]], %[[SIGN]])
// CHECK: %[[MAB:.*]] = llvm.extractvalue %[[AB]][1]
// CHECK: %[[BA:.*]] = "xllvm.intr.[[T]].vmax.lt32"(%[[B]], %[[A]], %[[SIGN]])
// CHECK: %[[MBA:.*]] = llvm.extractvalue %[[BA]][1]
// CHECK: %[[M:.*]] = llvm.or %[[MAB]], %[[MBA]] : i32
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i32_ne(%a: vector<16xi32>, %b: vector<16xi32>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "ne"} : vector<16xi32>, vector<16xi32>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i16_slt
// CHECK-SAME: %[[A:.*]]: vector<32xi16>, %[[B:.*]]: vector<32xi16>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmax.lt16"(%[[A]], %[[B]], %[[SIGN]])
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i16_slt(%a: vector<32xi16>, %b: vector<32xi16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "slt"} : vector<32xi16>, vector<32xi16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i16_ult
// CHECK-SAME: %[[A:.*]]: vector<32xi16>, %[[B:.*]]: vector<32xi16>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmax.lt16"(%[[A]], %[[B]], %[[SIGN]])
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i16_ult(%a: vector<32xi16>, %b: vector<32xi16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "ult"} : vector<32xi16>, vector<32xi16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i16_sge
// CHECK-SAME: %[[A:.*]]: vector<32xi16>, %[[B:.*]]: vector<32xi16>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmin.ge16"(%[[A]], %[[B]], %[[SIGN]])
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i16_sge(%a: vector<32xi16>, %b: vector<32xi16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "sge"} : vector<32xi16>, vector<32xi16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i16_uge
// CHECK-SAME: %[[A:.*]]: vector<32xi16>, %[[B:.*]]: vector<32xi16>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmin.ge16"(%[[A]], %[[B]], %[[SIGN]])
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i16_uge(%a: vector<32xi16>, %b: vector<32xi16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "uge"} : vector<32xi16>, vector<32xi16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i16_sgt
// CHECK-SAME: %[[A:.*]]: vector<32xi16>, %[[B:.*]]: vector<32xi16>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmax.lt16"(%[[B]], %[[A]], %[[SIGN]])
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i16_sgt(%a: vector<32xi16>, %b: vector<32xi16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "sgt"} : vector<32xi16>, vector<32xi16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i16_ugt
// CHECK-SAME: %[[A:.*]]: vector<32xi16>, %[[B:.*]]: vector<32xi16>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmax.lt16"(%[[B]], %[[A]], %[[SIGN]])
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i16_ugt(%a: vector<32xi16>, %b: vector<32xi16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "ugt"} : vector<32xi16>, vector<32xi16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i16_sle
// CHECK-SAME: %[[A:.*]]: vector<32xi16>, %[[B:.*]]: vector<32xi16>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmin.ge16"(%[[B]], %[[A]], %[[SIGN]])
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i16_sle(%a: vector<32xi16>, %b: vector<32xi16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "sle"} : vector<32xi16>, vector<32xi16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i16_ule
// CHECK-SAME: %[[A:.*]]: vector<32xi16>, %[[B:.*]]: vector<32xi16>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmin.ge16"(%[[B]], %[[A]], %[[SIGN]])
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i16_ule(%a: vector<32xi16>, %b: vector<32xi16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "ule"} : vector<32xi16>, vector<32xi16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i16_eq
// CHECK-SAME: %[[A:.*]]: vector<32xi16>, %[[B:.*]]: vector<32xi16>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[AB:.*]] = "xllvm.intr.[[T]].vmin.ge16"(%[[A]], %[[B]], %[[SIGN]])
// CHECK: %[[MAB:.*]] = llvm.extractvalue %[[AB]][1]
// CHECK: %[[BA:.*]] = "xllvm.intr.[[T]].vmin.ge16"(%[[B]], %[[A]], %[[SIGN]])
// CHECK: %[[MBA:.*]] = llvm.extractvalue %[[BA]][1]
// CHECK: %[[M:.*]] = llvm.and %[[MAB]], %[[MBA]] : i32
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i16_eq(%a: vector<32xi16>, %b: vector<32xi16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "eq"} : vector<32xi16>, vector<32xi16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @i16_ne
// CHECK-SAME: %[[A:.*]]: vector<32xi16>, %[[B:.*]]: vector<32xi16>
// CHECK: %[[SIGN:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[AB:.*]] = "xllvm.intr.[[T]].vmax.lt16"(%[[A]], %[[B]], %[[SIGN]])
// CHECK: %[[MAB:.*]] = llvm.extractvalue %[[AB]][1]
// CHECK: %[[BA:.*]] = "xllvm.intr.[[T]].vmax.lt16"(%[[B]], %[[A]], %[[SIGN]])
// CHECK: %[[MBA:.*]] = llvm.extractvalue %[[BA]][1]
// CHECK: %[[M:.*]] = llvm.or %[[MAB]], %[[MBA]] : i32
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @i16_ne(%a: vector<32xi16>, %b: vector<32xi16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "ne"} : vector<32xi16>, vector<32xi16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @bf16_slt
// CHECK-SAME: %[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>
// CHECK-NOT: llvm.mlir.constant
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmax.ltbf16"(%[[A]], %[[B]]) :
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @bf16_slt(%a: vector<32xbf16>, %b: vector<32xbf16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "slt"} : vector<32xbf16>, vector<32xbf16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @bf16_ult
// CHECK-SAME: %[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>
// CHECK-NOT: llvm.mlir.constant
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmax.ltbf16"(%[[A]], %[[B]]) :
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @bf16_ult(%a: vector<32xbf16>, %b: vector<32xbf16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "ult"} : vector<32xbf16>, vector<32xbf16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @bf16_sge
// CHECK-SAME: %[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>
// CHECK-NOT: llvm.mlir.constant
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmin.gebf16"(%[[A]], %[[B]]) :
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @bf16_sge(%a: vector<32xbf16>, %b: vector<32xbf16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "sge"} : vector<32xbf16>, vector<32xbf16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @bf16_uge
// CHECK-SAME: %[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>
// CHECK-NOT: llvm.mlir.constant
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmin.gebf16"(%[[A]], %[[B]]) :
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @bf16_uge(%a: vector<32xbf16>, %b: vector<32xbf16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "uge"} : vector<32xbf16>, vector<32xbf16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @bf16_sgt
// CHECK-SAME: %[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>
// CHECK-NOT: llvm.mlir.constant
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmax.ltbf16"(%[[B]], %[[A]]) :
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @bf16_sgt(%a: vector<32xbf16>, %b: vector<32xbf16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "sgt"} : vector<32xbf16>, vector<32xbf16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @bf16_ugt
// CHECK-SAME: %[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>
// CHECK-NOT: llvm.mlir.constant
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmax.ltbf16"(%[[B]], %[[A]]) :
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @bf16_ugt(%a: vector<32xbf16>, %b: vector<32xbf16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "ugt"} : vector<32xbf16>, vector<32xbf16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @bf16_sle
// CHECK-SAME: %[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>
// CHECK-NOT: llvm.mlir.constant
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmin.gebf16"(%[[B]], %[[A]]) :
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @bf16_sle(%a: vector<32xbf16>, %b: vector<32xbf16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "sle"} : vector<32xbf16>, vector<32xbf16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @bf16_ule
// CHECK-SAME: %[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>
// CHECK-NOT: llvm.mlir.constant
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmin.gebf16"(%[[B]], %[[A]]) :
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @bf16_ule(%a: vector<32xbf16>, %b: vector<32xbf16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "ule"} : vector<32xbf16>, vector<32xbf16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @bf16_eq
// CHECK-SAME: %[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>
// CHECK-NOT: llvm.mlir.constant
// CHECK: %[[AB:.*]] = "xllvm.intr.[[T]].vmin.gebf16"(%[[A]], %[[B]]) :
// CHECK: %[[MAB:.*]] = llvm.extractvalue %[[AB]][1]
// CHECK: %[[BA:.*]] = "xllvm.intr.[[T]].vmin.gebf16"(%[[B]], %[[A]]) :
// CHECK: %[[MBA:.*]] = llvm.extractvalue %[[BA]][1]
// CHECK: %[[M:.*]] = llvm.and %[[MAB]], %[[MBA]] : i32
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @bf16_eq(%a: vector<32xbf16>, %b: vector<32xbf16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "eq"} : vector<32xbf16>, vector<32xbf16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: func.func @bf16_ne
// CHECK-SAME: %[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>
// CHECK-NOT: llvm.mlir.constant
// CHECK: %[[AB:.*]] = "xllvm.intr.[[T]].vmax.ltbf16"(%[[A]], %[[B]]) :
// CHECK: %[[MAB:.*]] = llvm.extractvalue %[[AB]][1]
// CHECK: %[[BA:.*]] = "xllvm.intr.[[T]].vmax.ltbf16"(%[[B]], %[[A]]) :
// CHECK: %[[MBA:.*]] = llvm.extractvalue %[[BA]][1]
// CHECK: %[[M:.*]] = llvm.or %[[MAB]], %[[MBA]] : i32
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[M]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @bf16_ne(%a: vector<32xbf16>, %b: vector<32xbf16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "ne"} : vector<32xbf16>, vector<32xbf16>, ui32
  return %0 : ui32
}

// A 16-lane bf16 compare runs on 32 lanes; the padding lanes are masked off.
// CHECK-LABEL: func.func @bf16_v16_ugt
// CHECK-SAME: %[[A:.*]]: vector<16xbf16>, %[[B:.*]]: vector<16xbf16>
// CHECK: %[[PA:.*]] = vector.shuffle %[[A]]
// CHECK-SAME: vector<16xbf16>, vector<16xbf16>
// CHECK: %[[PB:.*]] = vector.shuffle %[[B]]
// CHECK-SAME: vector<16xbf16>, vector<16xbf16>
// CHECK: %[[R:.*]] = "xllvm.intr.[[T]].vmax.ltbf16"(%[[PB]], %[[PA]]) :
// CHECK: %[[M:.*]] = llvm.extractvalue %[[R]][1]
// CHECK: %[[LOW:.*]] = llvm.mlir.constant(65535 : i32) : i32
// CHECK: %[[MASKED:.*]] = llvm.and %[[M]], %[[LOW]] : i32
// CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[MASKED]] : i32 to ui32
// CHECK: return %[[RES]]
func.func @bf16_v16_ugt(%a: vector<16xbf16>, %b: vector<16xbf16>) -> ui32 {
  %0 = aievec.cmp %a, %b {pred = "ugt"} : vector<16xbf16>, vector<16xbf16>, ui32
  return %0 : ui32
}
