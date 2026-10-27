//===- test-native-lanes-unchanged-aie2p.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" -split-input-file | FileCheck %s

// Elementwise ops on AIE2P that are not brought to another lane count.

// 24 lanes of f32 neither divide nor are divided by a native width of the
// add (16, 32, 64).
// CHECK-LABEL: func.func @addf_v24f32
// CHECK-NOT: aievec
// CHECK: arith.addf %{{.*}}, %{{.*}} : vector<24xf32>
func.func @addf_v24f32(%a: vector<24xf32>, %b: vector<24xf32>) -> vector<24xf32> {
  %0 = arith.addf %a, %b : vector<24xf32>
  return %0 : vector<24xf32>
}

// -----

// 48 lanes of bf16 are wider than the max's 32 and not a multiple of it.
// CHECK-LABEL: func.func @maximumf_v48bf16
// CHECK-NOT: aievec
// CHECK: arith.maximumf %{{.*}}, %{{.*}} : vector<48xbf16>
func.func @maximumf_v48bf16(%a: vector<48xbf16>, %b: vector<48xbf16>) -> vector<48xbf16> {
  %0 = arith.maximumf %a, %b : vector<48xbf16>
  return %0 : vector<48xbf16>
}

// -----

// No native width: f32 max and f16 add.
// CHECK-LABEL: func.func @maximumf_v64f32
// CHECK-NOT: aievec
// CHECK: arith.maximumf %{{.*}}, %{{.*}} : vector<64xf32>
func.func @maximumf_v64f32(%a: vector<64xf32>, %b: vector<64xf32>) -> vector<64xf32> {
  %0 = arith.maximumf %a, %b : vector<64xf32>
  return %0 : vector<64xf32>
}

// -----

// CHECK-LABEL: func.func @addf_v64f16
// CHECK-NOT: aievec
// CHECK: arith.addf %{{.*}}, %{{.*}} : vector<64xf16>
func.func @addf_v64f16(%a: vector<64xf16>, %b: vector<64xf16>) -> vector<64xf16> {
  %0 = arith.addf %a, %b : vector<64xf16>
  return %0 : vector<64xf16>
}

// -----

// A wide compare is split only with a select that uses it.
// CHECK-LABEL: func.func @cmpi_v64i32
// CHECK-NOT: aievec
// CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : vector<64xi32>
func.func @cmpi_v64i32(%a: vector<64xi32>, %b: vector<64xi32>) -> vector<64xi1> {
  %0 = arith.cmpi slt, %a, %b : vector<64xi32>
  return %0 : vector<64xi1>
}

// -----

// A select on a mask that is not a compare, and one on an f32 compare, which
// has no native width.
// CHECK-LABEL: func.func @select_mask_v64i32
// CHECK-NOT: aievec
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<64xi1>, vector<64xi32>
func.func @select_mask_v64i32(%m: vector<64xi1>, %x: vector<64xi32>,
                              %y: vector<64xi32>) -> vector<64xi32> {
  %0 = arith.select %m, %x, %y : vector<64xi1>, vector<64xi32>
  return %0 : vector<64xi32>
}

// -----

// CHECK-LABEL: func.func @cmpf_select_v64f32
// CHECK-NOT: aievec
// CHECK: arith.cmpf olt, %{{.*}}, %{{.*}} : vector<64xf32>
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<64xi1>, vector<64xf32>
func.func @cmpf_select_v64f32(%a: vector<64xf32>, %b: vector<64xf32>,
                              %x: vector<64xf32>, %y: vector<64xf32>) -> vector<64xf32> {
  %c = arith.cmpf olt, %a, %b : vector<64xf32>
  %0 = arith.select %c, %x, %y : vector<64xi1>, vector<64xf32>
  return %0 : vector<64xf32>
}

// -----

// A select with a scalar condition.
// CHECK-LABEL: func.func @select_scalar_v64i32
// CHECK-NOT: aievec
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<64xi32>
func.func @select_scalar_v64i32(%a: i32, %b: i32, %x: vector<64xi32>,
                                %y: vector<64xi32>) -> vector<64xi32> {
  %c = arith.cmpi slt, %a, %b : i32
  %0 = arith.select %c, %x, %y : vector<64xi32>
  return %0 : vector<64xi32>
}

// -----

// Integer adds and subtracts of one 512-bit register or less are left to
// LLVM as they are, after flattening.
// CHECK-LABEL: func.func @addi_narrow
// CHECK-NOT: aievec
// CHECK: arith.addi %{{.*}}, %{{.*}} : vector<8xi32>
// CHECK: arith.subi %{{.*}}, %{{.*}} : vector<32xi16>
func.func @addi_narrow(%a: vector<8xi32>, %b: vector<8xi32>,
                       %c: vector<2x16xi16>, %d: vector<2x16xi16>)
    -> (vector<8xi32>, vector<2x16xi16>) {
  %0 = arith.addi %a, %b : vector<8xi32>
  %1 = arith.subi %c, %d : vector<2x16xi16>
  return %0, %1 : vector<8xi32>, vector<2x16xi16>
}

// -----

// Integer max is not split: wider ones may be part of a clamp the srs
// lowering takes as a whole.
// CHECK-LABEL: func.func @maxsi_v64i32
// CHECK-NOT: aievec
// CHECK: arith.maxsi %{{.*}}, %{{.*}} : vector<64xi32>
func.func @maxsi_v64i32(%a: vector<64xi32>, %b: vector<64xi32>) -> vector<64xi32> {
  %0 = arith.maxsi %a, %b : vector<64xi32>
  return %0 : vector<64xi32>
}

// -----

// A one-lane vector is a scalar to the lowering.
// CHECK-LABEL: func.func @addf_v1f32
// CHECK-NOT: aievec
// CHECK: arith.addf %{{.*}}, %{{.*}} : vector<1xf32>
func.func @addf_v1f32(%a: vector<1xf32>, %b: vector<1xf32>) -> vector<1xf32> {
  %0 = arith.addf %a, %b : vector<1xf32>
  return %0 : vector<1xf32>
}

// -----

// An ord or uno compare has no lowering, so neither it nor its select is split.
// CHECK-LABEL: func.func @cmpf_ord_select_v64bf16
// CHECK-NOT: aievec
// CHECK: arith.cmpf ord, %{{.*}}, %{{.*}} : vector<64xbf16>
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<64xi1>, vector<64xbf16>
func.func @cmpf_ord_select_v64bf16(%a: vector<64xbf16>, %b: vector<64xbf16>,
                                   %x: vector<64xbf16>, %y: vector<64xbf16>) -> vector<64xbf16> {
  %c = arith.cmpf ord, %a, %b : vector<64xbf16>
  %0 = arith.select %c, %x, %y : vector<64xi1>, vector<64xbf16>
  return %0 : vector<64xbf16>
}

// -----

// CHECK-LABEL: func.func @cmpf_uno_select_v64bf16
// CHECK-NOT: aievec
// CHECK: arith.cmpf uno, %{{.*}}, %{{.*}} : vector<64xbf16>
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<64xi1>, vector<64xbf16>
func.func @cmpf_uno_select_v64bf16(%a: vector<64xbf16>, %b: vector<64xbf16>,
                                   %x: vector<64xbf16>, %y: vector<64xbf16>) -> vector<64xbf16> {
  %c = arith.cmpf uno, %a, %b : vector<64xbf16>
  %0 = arith.select %c, %x, %y : vector<64xi1>, vector<64xbf16>
  return %0 : vector<64xbf16>
}

// -----

// The LLVM lowering has no i8 compare or select.
// CHECK-LABEL: func.func @cmpi_select_v128i8
// CHECK-NOT: aievec
// CHECK: arith.cmpi sge, %{{.*}}, %{{.*}} : vector<128xi8>
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<128xi1>, vector<128xi8>
func.func @cmpi_select_v128i8(%a: vector<128xi8>, %b: vector<128xi8>,
                              %x: vector<128xi8>, %y: vector<128xi8>) -> vector<128xi8> {
  %c = arith.cmpi sge, %a, %b : vector<128xi8>
  %0 = arith.select %c, %x, %y : vector<128xi1>, vector<128xi8>
  return %0 : vector<128xi8>
}

// -----

// Nor a part width an i8 compare and an i32 select would share.
// CHECK-LABEL: func.func @cmpi_v128i8_select_v128i32
// CHECK-NOT: aievec
// CHECK: arith.cmpi sge, %{{.*}}, %{{.*}} : vector<128xi8>
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<128xi1>, vector<128xi32>
func.func @cmpi_v128i8_select_v128i32(%a: vector<128xi8>, %b: vector<128xi8>,
                                      %x: vector<128xi32>, %y: vector<128xi32>) -> vector<128xi32> {
  %c = arith.cmpi sge, %a, %b : vector<128xi8>
  %0 = arith.select %c, %x, %y : vector<128xi1>, vector<128xi32>
  return %0 : vector<128xi32>
}

// -----

// The LLVM lowering has no f32 select.
// CHECK-LABEL: func.func @cmpf_v64bf16_select_v64f32
// CHECK-NOT: aievec
// CHECK: arith.cmpf olt, %{{.*}}, %{{.*}} : vector<64xbf16>
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<64xi1>, vector<64xf32>
func.func @cmpf_v64bf16_select_v64f32(%a: vector<64xbf16>, %b: vector<64xbf16>,
                                      %x: vector<64xf32>, %y: vector<64xf32>) -> vector<64xf32> {
  %c = arith.cmpf olt, %a, %b : vector<64xbf16>
  %0 = arith.select %c, %x, %y : vector<64xi1>, vector<64xf32>
  return %0 : vector<64xf32>
}

// -----

// 24 or 80 lanes of bf16 max (native 32), and 24 lanes of tanh (16 and 32):
// no native width divides the lane count or is a multiple of it.
// CHECK-LABEL: func.func @no_part_width
// CHECK-NOT: aievec
// CHECK: arith.maximumf %{{.*}}, %{{.*}} : vector<24xbf16>
// CHECK: arith.maximumf %{{.*}}, %{{.*}} : vector<80xbf16>
// CHECK: math.tanh %{{.*}} : vector<24xbf16>
func.func @no_part_width(%a: vector<24xbf16>, %b: vector<80xbf16>,
                         %c: vector<24xbf16>, %d: vector<80xbf16>)
    -> (vector<24xbf16>, vector<80xbf16>, vector<24xbf16>) {
  %0 = arith.maximumf %a, %c : vector<24xbf16>
  %1 = arith.maximumf %b, %d : vector<80xbf16>
  %2 = math.tanh %a : vector<24xbf16>
  return %0, %1, %2 : vector<24xbf16>, vector<80xbf16>, vector<24xbf16>
}

// -----

// An integer add that fits in a register is left to LLVM whatever its lane
// count.
// CHECK-LABEL: func.func @addi_v24i16
// CHECK-NOT: aievec
// CHECK: arith.addi %{{.*}}, %{{.*}} : vector<24xi16>
func.func @addi_v24i16(%a: vector<24xi16>, %b: vector<24xi16>) -> vector<24xi16> {
  %0 = arith.addi %a, %b : vector<24xi16>
  return %0 : vector<24xi16>
}

// -----

// The LLVM lowering compares bf16 as ordered, false if either is NaN, so an
// unordered compare and its select are left as they are.
// CHECK-LABEL: func.func @cmpf_unordered_select_v64bf16
// CHECK-NOT: aievec
// CHECK-COUNT-6: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<64xi1>, vector<64xbf16>
func.func @cmpf_unordered_select_v64bf16(%a: vector<64xbf16>, %b: vector<64xbf16>,
                                         %x: vector<64xbf16>, %y: vector<64xbf16>)
    -> (vector<64xbf16>, vector<64xbf16>, vector<64xbf16>, vector<64xbf16>,
        vector<64xbf16>, vector<64xbf16>) {
  %c0 = arith.cmpf ueq, %a, %b : vector<64xbf16>
  %0 = arith.select %c0, %x, %y : vector<64xi1>, vector<64xbf16>
  %c1 = arith.cmpf une, %a, %b : vector<64xbf16>
  %1 = arith.select %c1, %x, %y : vector<64xi1>, vector<64xbf16>
  %c2 = arith.cmpf ult, %a, %b : vector<64xbf16>
  %2 = arith.select %c2, %x, %y : vector<64xi1>, vector<64xbf16>
  %c3 = arith.cmpf ule, %a, %b : vector<64xbf16>
  %3 = arith.select %c3, %x, %y : vector<64xi1>, vector<64xbf16>
  %c4 = arith.cmpf ugt, %a, %b : vector<64xbf16>
  %4 = arith.select %c4, %x, %y : vector<64xi1>, vector<64xbf16>
  %c5 = arith.cmpf uge, %a, %b : vector<64xbf16>
  %5 = arith.select %c5, %x, %y : vector<64xi1>, vector<64xbf16>
  return %0, %1, %2, %3, %4, %5 : vector<64xbf16>, vector<64xbf16>, vector<64xbf16>,
                                  vector<64xbf16>, vector<64xbf16>, vector<64xbf16>
}
