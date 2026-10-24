//===- test-native-lanes-aie2p.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" -split-input-file | FileCheck %s
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" -split-input-file | FileCheck %s --check-prefix=AIE2

// On AIE2P, an elementwise op whose lane count its lowering does not take is
// rewritten to lane counts it does take: a wider vector is split into equal
// parts of the widest native width that divides it, a narrower one is
// repeated to the narrowest native width that holds a whole number of
// copies, and the low lanes are kept. AIE2 is left as it was.

// f32 add and sub: 16, 32 and 64 lanes (the 2048-bit accumulator).

// CHECK-LABEL: func.func @addf_v8f32
// CHECK-SAME: (%[[A:.*]]: vector<8xf32>, %[[B:.*]]: vector<8xf32>)
// CHECK-DAG: %[[WA:.*]] = aievec.concat %[[A]], %[[A]] : vector<8xf32>, vector<16xf32>
// CHECK-DAG: %[[WB:.*]] = aievec.concat %[[B]], %[[B]] : vector<8xf32>, vector<16xf32>
// CHECK-DAG: %[[CA:.*]] = aievec.cast %[[WA]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
// CHECK-DAG: %[[CB:.*]] = aievec.cast %[[WB]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
// CHECK: %[[S:.*]] = aievec.add_elem %[[CA]], %[[CB]] : vector<16xf32>
// CHECK: %[[R:.*]] = aievec.cast %[[S]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
// CHECK: %[[E:.*]] = aievec.ext %[[R]] {index = 0 : i8} : vector<16xf32>, vector<8xf32>
// CHECK: return %[[E]] : vector<8xf32>
// AIE2-LABEL: func.func @addf_v8f32
// AIE2-NOT: aievec
// AIE2: arith.addf %{{.*}}, %{{.*}} : vector<8xf32>
func.func @addf_v8f32(%a: vector<8xf32>, %b: vector<8xf32>) -> vector<8xf32> {
  %0 = arith.addf %a, %b : vector<8xf32>
  return %0 : vector<8xf32>
}

// -----

// CHECK-LABEL: func.func @subf_v4f32
// CHECK-SAME: (%[[A:.*]]: vector<4xf32>, %[[B:.*]]: vector<4xf32>)
// CHECK-DAG: aievec.concat %[[A]], %[[A]], %[[A]], %[[A]] : vector<4xf32>, vector<16xf32>
// CHECK-DAG: aievec.concat %[[B]], %[[B]], %[[B]], %[[B]] : vector<4xf32>, vector<16xf32>
// CHECK: %[[S:.*]] = aievec.sub_elem %{{.*}}, %{{.*}} : vector<16xf32>
// CHECK: %[[R:.*]] = aievec.cast %[[S]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
// CHECK: aievec.ext %[[R]] {index = 0 : i8} : vector<16xf32>, vector<4xf32>
func.func @subf_v4f32(%a: vector<4xf32>, %b: vector<4xf32>) -> vector<4xf32> {
  %0 = arith.subf %a, %b : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: func.func @addf_v64f32
// CHECK-SAME: (%[[A:.*]]: vector<64xf32>, %[[B:.*]]: vector<64xf32>)
// CHECK-NOT: aievec.ext
// CHECK-DAG: %[[CA:.*]] = aievec.cast %[[A]] {isResAcc = true} : vector<64xf32>, vector<64xf32>
// CHECK-DAG: %[[CB:.*]] = aievec.cast %[[B]] {isResAcc = true} : vector<64xf32>, vector<64xf32>
// CHECK: %[[S:.*]] = aievec.add_elem %[[CA]], %[[CB]] : vector<64xf32>
// CHECK: %[[R:.*]] = aievec.cast %[[S]] {isResAcc = false} : vector<64xf32>, vector<64xf32>
// CHECK: return %[[R]] : vector<64xf32>
// AIE2-LABEL: func.func @addf_v64f32
// AIE2-NOT: aievec
// AIE2: arith.addf %{{.*}}, %{{.*}} : vector<64xf32>
func.func @addf_v64f32(%a: vector<64xf32>, %b: vector<64xf32>) -> vector<64xf32> {
  %0 = arith.addf %a, %b : vector<64xf32>
  return %0 : vector<64xf32>
}

// -----

// CHECK-LABEL: func.func @subf_v128f32
// CHECK-SAME: (%[[A:.*]]: vector<128xf32>, %[[B:.*]]: vector<128xf32>)
// CHECK-DAG: aievec.ext %[[A]] {index = 0 : i8} : vector<128xf32>, vector<64xf32>
// CHECK-DAG: aievec.ext %[[B]] {index = 0 : i8} : vector<128xf32>, vector<64xf32>
// CHECK: aievec.sub_elem %{{.*}}, %{{.*}} : vector<64xf32>
// CHECK: %[[R0:.*]] = aievec.cast %{{.*}} {isResAcc = false} : vector<64xf32>, vector<64xf32>
// CHECK-DAG: aievec.ext %[[A]] {index = 1 : i8} : vector<128xf32>, vector<64xf32>
// CHECK-DAG: aievec.ext %[[B]] {index = 1 : i8} : vector<128xf32>, vector<64xf32>
// CHECK: aievec.sub_elem %{{.*}}, %{{.*}} : vector<64xf32>
// CHECK: %[[R1:.*]] = aievec.cast %{{.*}} {isResAcc = false} : vector<64xf32>, vector<64xf32>
// CHECK: %[[R:.*]] = aievec.concat %[[R0]], %[[R1]] : vector<64xf32>, vector<128xf32>
// CHECK: return %[[R]] : vector<128xf32>
func.func @subf_v128f32(%a: vector<128xf32>, %b: vector<128xf32>) -> vector<128xf32> {
  %0 = arith.subf %a, %b : vector<128xf32>
  return %0 : vector<128xf32>
}

// -----

// 96 lanes: three parts of 32, the widest native width that divides 96.
// CHECK-LABEL: func.func @addf_v96f32
// CHECK-SAME: (%[[A:.*]]: vector<96xf32>, %[[B:.*]]: vector<96xf32>)
// CHECK-DAG: aievec.ext %[[A]] {index = 0 : i8} : vector<96xf32>, vector<32xf32>
// CHECK-DAG: aievec.ext %[[A]] {index = 1 : i8} : vector<96xf32>, vector<32xf32>
// CHECK-DAG: aievec.ext %[[A]] {index = 2 : i8} : vector<96xf32>, vector<32xf32>
// CHECK: %[[R:.*]] = aievec.concat %{{.*}}, %{{.*}}, %{{.*}} : vector<32xf32>, vector<96xf32>
// CHECK: return %[[R]] : vector<96xf32>
func.func @addf_v96f32(%a: vector<96xf32>, %b: vector<96xf32>) -> vector<96xf32> {
  %0 = arith.addf %a, %b : vector<96xf32>
  return %0 : vector<96xf32>
}

// -----

// bf16 add and sub: 16 and 32 lanes.

// CHECK-LABEL: func.func @addf_v8bf16
// CHECK-SAME: (%[[A:.*]]: vector<8xbf16>, %[[B:.*]]: vector<8xbf16>)
// CHECK-DAG: %[[WA:.*]] = aievec.concat %[[A]], %[[A]] : vector<8xbf16>, vector<16xbf16>
// CHECK-DAG: %[[WB:.*]] = aievec.concat %[[B]], %[[B]] : vector<8xbf16>, vector<16xbf16>
// CHECK-DAG: %[[UA:.*]] = aievec.ups %[[WA]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
// CHECK-DAG: %[[UB:.*]] = aievec.ups %[[WB]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
// CHECK: %[[S:.*]] = aievec.add_elem %[[UA]], %[[UB]] : vector<16xf32>
// CHECK: %[[R:.*]] = aievec.srs %[[S]], %{{.*}} : vector<16xf32>, i32, vector<16xbf16>
// CHECK: %[[E:.*]] = aievec.ext %[[R]] {index = 0 : i8} : vector<16xbf16>, vector<8xbf16>
// CHECK: return %[[E]] : vector<8xbf16>
func.func @addf_v8bf16(%a: vector<8xbf16>, %b: vector<8xbf16>) -> vector<8xbf16> {
  %0 = arith.addf %a, %b : vector<8xbf16>
  return %0 : vector<8xbf16>
}

// -----

// CHECK-LABEL: func.func @subf_v64bf16
// CHECK-SAME: (%[[A:.*]]: vector<64xbf16>, %[[B:.*]]: vector<64xbf16>)
// CHECK: aievec.ext %[[A]] {index = 0 : i8} : vector<64xbf16>, vector<32xbf16>
// CHECK-COUNT-2: aievec.sub_elem %{{.*}}, %{{.*}} : vector<16xf32>
// CHECK: aievec.ext %[[A]] {index = 1 : i8} : vector<64xbf16>, vector<32xbf16>
// CHECK-COUNT-2: aievec.sub_elem %{{.*}}, %{{.*}} : vector<16xf32>
// CHECK: %[[R:.*]] = aievec.concat %{{.*}}, %{{.*}} : vector<32xbf16>, vector<64xbf16>
// CHECK: return %[[R]] : vector<64xbf16>
func.func @subf_v64bf16(%a: vector<64xbf16>, %b: vector<64xbf16>) -> vector<64xbf16> {
  %0 = arith.subf %a, %b : vector<64xbf16>
  return %0 : vector<64xbf16>
}

// -----

// 48 lanes: three parts of 16.
// CHECK-LABEL: func.func @addf_v48bf16
// CHECK-SAME: (%[[A:.*]]: vector<48xbf16>, %[[B:.*]]: vector<48xbf16>)
// CHECK: aievec.ext %[[A]] {index = 0 : i8} : vector<48xbf16>, vector<16xbf16>
// CHECK: aievec.add_elem %{{.*}}, %{{.*}} : vector<16xf32>
// CHECK: aievec.ext %[[A]] {index = 1 : i8} : vector<48xbf16>, vector<16xbf16>
// CHECK: aievec.add_elem %{{.*}}, %{{.*}} : vector<16xf32>
// CHECK: aievec.ext %[[A]] {index = 2 : i8} : vector<48xbf16>, vector<16xbf16>
// CHECK: aievec.add_elem %{{.*}}, %{{.*}} : vector<16xf32>
// CHECK: %[[R:.*]] = aievec.concat %{{.*}}, %{{.*}}, %{{.*}} : vector<16xbf16>, vector<48xbf16>
// CHECK: return %[[R]] : vector<48xbf16>
func.func @addf_v48bf16(%a: vector<48xbf16>, %b: vector<48xbf16>) -> vector<48xbf16> {
  %0 = arith.addf %a, %b : vector<48xbf16>
  return %0 : vector<48xbf16>
}

// -----

// bf16 max and min: one 512-bit register, 32 lanes.

// CHECK-LABEL: func.func @maximumf_v8bf16
// CHECK-SAME: (%[[A:.*]]: vector<8xbf16>, %[[B:.*]]: vector<8xbf16>)
// CHECK-DAG: %[[WA:.*]] = aievec.concat %[[A]], %[[A]], %[[A]], %[[A]] : vector<8xbf16>, vector<32xbf16>
// CHECK-DAG: %[[WB:.*]] = aievec.concat %[[B]], %[[B]], %[[B]], %[[B]] : vector<8xbf16>, vector<32xbf16>
// CHECK: %[[M:.*]] = aievec.max %[[WA]], %[[WB]] : vector<32xbf16>
// CHECK: %[[E:.*]] = aievec.ext %[[M]] {index = 0 : i8} : vector<32xbf16>, vector<8xbf16>
// CHECK: return %[[E]] : vector<8xbf16>
// AIE2-LABEL: func.func @maximumf_v8bf16
// AIE2-NOT: aievec
// AIE2: arith.maximumf %{{.*}}, %{{.*}} : vector<8xbf16>
func.func @maximumf_v8bf16(%a: vector<8xbf16>, %b: vector<8xbf16>) -> vector<8xbf16> {
  %0 = arith.maximumf %a, %b : vector<8xbf16>
  return %0 : vector<8xbf16>
}

// -----

// CHECK-LABEL: func.func @maxnumf_v16bf16
// CHECK-SAME: (%[[A:.*]]: vector<16xbf16>, %[[B:.*]]: vector<16xbf16>)
// CHECK-DAG: %[[WA:.*]] = aievec.concat %[[A]], %[[A]] : vector<16xbf16>, vector<32xbf16>
// CHECK-DAG: %[[WB:.*]] = aievec.concat %[[B]], %[[B]] : vector<16xbf16>, vector<32xbf16>
// CHECK: %[[M:.*]] = aievec.max %[[WA]], %[[WB]] : vector<32xbf16>
// CHECK: %[[E:.*]] = aievec.ext %[[M]] {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
// CHECK: return %[[E]] : vector<16xbf16>
// AIE2-LABEL: func.func @maxnumf_v16bf16
// AIE2: %[[Z:.*]] = arith.constant dense<0.000000e+00> : vector<16xbf16>
// AIE2: aievec.concat %{{.*}}, %[[Z]] : vector<16xbf16>, vector<32xbf16>
// AIE2: aievec.max %{{.*}}, %{{.*}} : vector<32xbf16>
func.func @maxnumf_v16bf16(%a: vector<16xbf16>, %b: vector<16xbf16>) -> vector<16xbf16> {
  %0 = arith.maxnumf %a, %b : vector<16xbf16>
  return %0 : vector<16xbf16>
}

// -----

// CHECK-LABEL: func.func @minimumf_v32bf16
// CHECK-SAME: (%[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>)
// CHECK-NOT: aievec.concat
// CHECK: %[[M:.*]] = aievec.min %[[A]], %[[B]] : vector<32xbf16>
// CHECK: return %[[M]] : vector<32xbf16>
func.func @minimumf_v32bf16(%a: vector<32xbf16>, %b: vector<32xbf16>) -> vector<32xbf16> {
  %0 = arith.minimumf %a, %b : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

// CHECK-LABEL: func.func @maximumf_v128bf16
// CHECK-SAME: (%[[A:.*]]: vector<128xbf16>, %[[B:.*]]: vector<128xbf16>)
// CHECK-DAG: %[[A0:.*]] = aievec.ext %[[A]] {index = 0 : i8} : vector<128xbf16>, vector<32xbf16>
// CHECK-DAG: %[[B0:.*]] = aievec.ext %[[B]] {index = 0 : i8} : vector<128xbf16>, vector<32xbf16>
// CHECK-DAG: %[[A3:.*]] = aievec.ext %[[A]] {index = 3 : i8} : vector<128xbf16>, vector<32xbf16>
// CHECK-DAG: %[[B3:.*]] = aievec.ext %[[B]] {index = 3 : i8} : vector<128xbf16>, vector<32xbf16>
// CHECK-DAG: %[[M0:.*]] = aievec.max %[[A0]], %[[B0]] : vector<32xbf16>
// CHECK-DAG: %[[M3:.*]] = aievec.max %[[A3]], %[[B3]] : vector<32xbf16>
// CHECK: %[[R:.*]] = aievec.concat %[[M0]], %{{.*}}, %{{.*}}, %[[M3]] : vector<32xbf16>, vector<128xbf16>
// CHECK: return %[[R]] : vector<128xbf16>
// AIE2-LABEL: func.func @maximumf_v128bf16
// AIE2-NOT: aievec
// AIE2: arith.maximumf %{{.*}}, %{{.*}} : vector<128xbf16>
func.func @maximumf_v128bf16(%a: vector<128xbf16>, %b: vector<128xbf16>) -> vector<128xbf16> {
  %0 = arith.maximumf %a, %b : vector<128xbf16>
  return %0 : vector<128xbf16>
}

// -----

// CHECK-LABEL: func.func @minimumf_v96bf16
// CHECK-COUNT-3: aievec.min %{{.*}}, %{{.*}} : vector<32xbf16>
// CHECK: aievec.concat %{{.*}}, %{{.*}}, %{{.*}} : vector<32xbf16>, vector<96xbf16>
func.func @minimumf_v96bf16(%a: vector<96xbf16>, %b: vector<96xbf16>) -> vector<96xbf16> {
  %0 = arith.minimumf %a, %b : vector<96xbf16>
  return %0 : vector<96xbf16>
}

// -----

// Rank 2: flattened first, then split.
// CHECK-LABEL: func.func @maxnumf_2x32bf16
// CHECK-SAME: (%[[A:.*]]: vector<2x32xbf16>, %[[B:.*]]: vector<2x32xbf16>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<2x32xbf16> to vector<64xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<2x32xbf16> to vector<64xbf16>
// CHECK-DAG: %[[A0:.*]] = aievec.ext %[[FA]] {index = 0 : i8} : vector<64xbf16>, vector<32xbf16>
// CHECK-DAG: %[[B0:.*]] = aievec.ext %[[FB]] {index = 0 : i8} : vector<64xbf16>, vector<32xbf16>
// CHECK-DAG: %[[A1:.*]] = aievec.ext %[[FA]] {index = 1 : i8} : vector<64xbf16>, vector<32xbf16>
// CHECK-DAG: %[[B1:.*]] = aievec.ext %[[FB]] {index = 1 : i8} : vector<64xbf16>, vector<32xbf16>
// CHECK-DAG: %[[M0:.*]] = aievec.max %[[A0]], %[[B0]] : vector<32xbf16>
// CHECK-DAG: %[[M1:.*]] = aievec.max %[[A1]], %[[B1]] : vector<32xbf16>
// CHECK: %[[C:.*]] = aievec.concat %[[M0]], %[[M1]] : vector<32xbf16>, vector<64xbf16>
// CHECK: %[[R:.*]] = vector.shape_cast %[[C]] : vector<64xbf16> to vector<2x32xbf16>
// CHECK: return %[[R]] : vector<2x32xbf16>
func.func @maxnumf_2x32bf16(%a: vector<2x32xbf16>, %b: vector<2x32xbf16>) -> vector<2x32xbf16> {
  %0 = arith.maxnumf %a, %b : vector<2x32xbf16>
  return %0 : vector<2x32xbf16>
}

// -----

// A splat operand is built again at the width of a part instead of being
// extracted from.
// CHECK-LABEL: func.func @maximumf_splat_v64bf16
// CHECK-SAME: (%[[A:.*]]: vector<64xbf16>)
// CHECK: %[[Z:.*]] = arith.constant dense<0.000000e+00> : vector<32xbf16>
// CHECK: %[[A0:.*]] = aievec.ext %[[A]] {index = 0 : i8} : vector<64xbf16>, vector<32xbf16>
// CHECK: aievec.max %[[A0]], %[[Z]] : vector<32xbf16>
// CHECK: %[[A1:.*]] = aievec.ext %[[A]] {index = 1 : i8} : vector<64xbf16>, vector<32xbf16>
// CHECK: aievec.max %[[A1]], %[[Z]] : vector<32xbf16>
func.func @maximumf_splat_v64bf16(%a: vector<64xbf16>) -> vector<64xbf16> {
  %z = arith.constant dense<0.0> : vector<64xbf16>
  %0 = arith.maximumf %a, %z : vector<64xbf16>
  return %0 : vector<64xbf16>
}

// -----

// tanh and erf: 16 and 32 lanes.

// CHECK-LABEL: func.func @tanh_v8f32
// CHECK-SAME: (%[[A:.*]]: vector<8xf32>)
// CHECK: %[[W:.*]] = aievec.concat %[[A]], %[[A]] : vector<8xf32>, vector<16xf32>
// CHECK: %[[T:.*]] = aievec.tanh %[[W]] : vector<16xf32>
// CHECK: %[[E:.*]] = aievec.ext %[[T]] {index = 0 : i8} : vector<16xf32>, vector<8xf32>
// CHECK: return %[[E]] : vector<8xf32>
func.func @tanh_v8f32(%a: vector<8xf32>) -> vector<8xf32> {
  %0 = math.tanh %a : vector<8xf32>
  return %0 : vector<8xf32>
}

// -----

// CHECK-LABEL: func.func @tanh_v96bf16
// CHECK-SAME: (%[[A:.*]]: vector<96xbf16>)
// CHECK: %[[A0:.*]] = aievec.ext %[[A]] {index = 0 : i8} : vector<96xbf16>, vector<32xbf16>
// CHECK: %[[T0:.*]] = aievec.tanh %[[A0]] : vector<32xbf16>
// CHECK: %[[A1:.*]] = aievec.ext %[[A]] {index = 1 : i8} : vector<96xbf16>, vector<32xbf16>
// CHECK: %[[T1:.*]] = aievec.tanh %[[A1]] : vector<32xbf16>
// CHECK: %[[A2:.*]] = aievec.ext %[[A]] {index = 2 : i8} : vector<96xbf16>, vector<32xbf16>
// CHECK: %[[T2:.*]] = aievec.tanh %[[A2]] : vector<32xbf16>
// CHECK: %[[R:.*]] = aievec.concat %[[T0]], %[[T1]], %[[T2]] : vector<32xbf16>, vector<96xbf16>
// CHECK: return %[[R]] : vector<96xbf16>
// AIE2-LABEL: func.func @tanh_v96bf16
// AIE2-NOT: aievec
// AIE2: math.tanh %{{.*}} : vector<96xbf16>
func.func @tanh_v96bf16(%a: vector<96xbf16>) -> vector<96xbf16> {
  %0 = math.tanh %a : vector<96xbf16>
  return %0 : vector<96xbf16>
}

// -----

// CHECK-LABEL: func.func @erf_v64f32
// CHECK-SAME: (%[[A:.*]]: vector<64xf32>)
// CHECK: aievec.ext %[[A]] {index = 0 : i8} : vector<64xf32>, vector<32xf32>
// CHECK: aievec.tanh %{{.*}} : vector<32xf32>
// CHECK: aievec.ext %[[A]] {index = 1 : i8} : vector<64xf32>, vector<32xf32>
// CHECK: aievec.tanh %{{.*}} : vector<32xf32>
// CHECK: aievec.concat %{{.*}}, %{{.*}} : vector<32xf32>, vector<64xf32>
func.func @erf_v64f32(%a: vector<64xf32>) -> vector<64xf32> {
  %0 = math.erf %a : vector<64xf32>
  return %0 : vector<64xf32>
}

// -----

// A compare is split together with the select that uses it. The parts are as
// wide as both take: 16 lanes for an i32 compare.

// CHECK-LABEL: func.func @cmpi_select_v64i32
// CHECK-SAME: (%[[A:.*]]: vector<64xi32>, %[[B:.*]]: vector<64xi32>, %[[X:.*]]: vector<64xi32>, %[[Y:.*]]: vector<64xi32>)
// CHECK: %[[A0:.*]] = aievec.ext %[[A]] {index = 0 : i8} : vector<64xi32>, vector<16xi32>
// CHECK: %[[B0:.*]] = aievec.ext %[[B]] {index = 0 : i8} : vector<64xi32>, vector<16xi32>
// CHECK: %[[C0:.*]] = aievec.cmp %[[A0]], %[[B0]] {pred = "slt"} : vector<16xi32>, vector<16xi32>, ui32
// CHECK: %[[X0:.*]] = aievec.ext %[[X]] {index = 0 : i8} : vector<64xi32>, vector<16xi32>
// CHECK: %[[Y0:.*]] = aievec.ext %[[Y]] {index = 0 : i8} : vector<64xi32>, vector<16xi32>
// CHECK: %[[S0:.*]] = aievec.sel %[[Y0]], %[[X0]], %[[C0]] : vector<16xi32>, vector<16xi32>, ui32, vector<16xi32>
// CHECK-COUNT-3: aievec.sel
// CHECK: %[[R:.*]] = aievec.concat %[[S0]], %{{.*}}, %{{.*}}, %{{.*}} : vector<16xi32>, vector<64xi32>
// CHECK: return %[[R]] : vector<64xi32>
// AIE2-LABEL: func.func @cmpi_select_v64i32
// AIE2-NOT: aievec
// AIE2: arith.cmpi slt, %{{.*}}, %{{.*}} : vector<64xi32>
// AIE2: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<64xi1>, vector<64xi32>
func.func @cmpi_select_v64i32(%a: vector<64xi32>, %b: vector<64xi32>,
                              %x: vector<64xi32>, %y: vector<64xi32>) -> vector<64xi32> {
  %c = arith.cmpi slt, %a, %b : vector<64xi32>
  %0 = arith.select %c, %x, %y : vector<64xi1>, vector<64xi32>
  return %0 : vector<64xi32>
}

// -----

// An i32 compare selecting bf16: parts of 16 lanes, the i32 compare's width.
// CHECK-LABEL: func.func @cmpi_select_v64bf16
// CHECK-COUNT-4: aievec.cmp %{{.*}}, %{{.*}} {pred = "ult"} : vector<16xi32>, vector<16xi32>, ui32
// CHECK: aievec.concat %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<16xbf16>, vector<64xbf16>
func.func @cmpi_select_v64bf16(%a: vector<64xi32>, %b: vector<64xi32>,
                               %x: vector<64xbf16>, %y: vector<64xbf16>) -> vector<64xbf16> {
  %c = arith.cmpi ult, %a, %b : vector<64xi32>
  %0 = arith.select %c, %x, %y : vector<64xi1>, vector<64xbf16>
  return %0 : vector<64xbf16>
}

// -----

// Rank 2 bf16: 32-lane parts.
// CHECK-LABEL: func.func @cmpf_select_2x32bf16
// CHECK-COUNT-2: aievec.cmp %{{.*}}, %{{.*}} {pred = "sgt"} : vector<32xbf16>, vector<32xbf16>, ui32
// CHECK: %[[C:.*]] = aievec.concat %{{.*}}, %{{.*}} : vector<32xbf16>, vector<64xbf16>
// CHECK: vector.shape_cast %[[C]] : vector<64xbf16> to vector<2x32xbf16>
func.func @cmpf_select_2x32bf16(%a: vector<2x32xbf16>, %b: vector<2x32xbf16>,
                                %x: vector<2x32xbf16>, %y: vector<2x32xbf16>) -> vector<2x32xbf16> {
  %c = arith.cmpf ogt, %a, %b : vector<2x32xbf16>
  %0 = arith.select %c, %x, %y : vector<2x32xi1>, vector<2x32xbf16>
  return %0 : vector<2x32xbf16>
}

// -----

// CHECK-LABEL: func.func @cmpi_select_v128i8
// CHECK-COUNT-2: aievec.sel %{{.*}}, %{{.*}}, %{{.*}} : vector<64xi8>, vector<64xi8>, ui64, vector<64xi8>
// CHECK: aievec.concat %{{.*}}, %{{.*}} : vector<64xi8>, vector<128xi8>
func.func @cmpi_select_v128i8(%a: vector<128xi8>, %b: vector<128xi8>,
                              %x: vector<128xi8>, %y: vector<128xi8>) -> vector<128xi8> {
  %c = arith.cmpi sge, %a, %b : vector<128xi8>
  %0 = arith.select %c, %x, %y : vector<128xi1>, vector<128xi8>
  return %0 : vector<128xi8>
}

// -----

// Integer add and sub are left to LLVM, which takes one 512-bit register;
// wider ones are split into registers.

// CHECK-LABEL: func.func @addi_v64i32
// CHECK-SAME: (%[[A:.*]]: vector<64xi32>, %[[B:.*]]: vector<64xi32>)
// CHECK: %[[A0:.*]] = aievec.ext %[[A]] {index = 0 : i8} : vector<64xi32>, vector<16xi32>
// CHECK: %[[B0:.*]] = aievec.ext %[[B]] {index = 0 : i8} : vector<64xi32>, vector<16xi32>
// CHECK: %[[S0:.*]] = arith.addi %[[A0]], %[[B0]] : vector<16xi32>
// CHECK-COUNT-3: arith.addi %{{.*}}, %{{.*}} : vector<16xi32>
// CHECK: %[[R:.*]] = aievec.concat %[[S0]], %{{.*}}, %{{.*}}, %{{.*}} : vector<16xi32>, vector<64xi32>
// CHECK: return %[[R]] : vector<64xi32>
// AIE2-LABEL: func.func @addi_v64i32
// AIE2-NOT: aievec
// AIE2: arith.addi %{{.*}}, %{{.*}} : vector<64xi32>
func.func @addi_v64i32(%a: vector<64xi32>, %b: vector<64xi32>) -> vector<64xi32> {
  %0 = arith.addi %a, %b : vector<64xi32>
  return %0 : vector<64xi32>
}

// -----

// CHECK-LABEL: func.func @subi_8x16xi8
// CHECK-SAME: (%[[A:.*]]: vector<8x16xi8>, %[[B:.*]]: vector<8x16xi8>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x16xi8> to vector<128xi8>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x16xi8> to vector<128xi8>
// CHECK: arith.subi %{{.*}}, %{{.*}} : vector<64xi8>
// CHECK: arith.subi %{{.*}}, %{{.*}} : vector<64xi8>
// CHECK: %[[C:.*]] = aievec.concat %{{.*}}, %{{.*}} : vector<64xi8>, vector<128xi8>
// CHECK: vector.shape_cast %[[C]] : vector<128xi8> to vector<8x16xi8>
func.func @subi_8x16xi8(%a: vector<8x16xi8>, %b: vector<8x16xi8>) -> vector<8x16xi8> {
  %0 = arith.subi %a, %b : vector<8x16xi8>
  return %0 : vector<8x16xi8>
}

// -----

// The parts of an integer add feed the parts of a compare split with its
// select directly; a broadcast operand is built again at the part width.
// CHECK-LABEL: func.func @addi_cmpi_select_v64i32
// CHECK-SAME: (%[[A:.*]]: vector<64xi32>, %[[N:.*]]: i32, %[[X:.*]]: vector<64xi32>, %[[Y:.*]]: vector<64xi32>)
// CHECK: %[[A0:.*]] = aievec.ext %[[A]] {index = 0 : i8} : vector<64xi32>, vector<16xi32>
// CHECK: %[[NB:.*]] = aievec.broadcast_scalar %[[N]] : i32, vector<16xi32>
// CHECK: %[[S0:.*]] = arith.addi %[[A0]], %[[NB]] : vector<16xi32>
// CHECK-COUNT-3: arith.addi %{{.*}}, %[[NB]] : vector<16xi32>
// CHECK-NOT: aievec.concat
// CHECK: %[[X0:.*]] = aievec.ext %[[X]] {index = 0 : i8} : vector<64xi32>, vector<16xi32>
// CHECK: aievec.cmp %[[S0]], %[[X0]] {pred = "slt"} : vector<16xi32>, vector<16xi32>, ui32
// CHECK-NOT: vector<64xi1>
// CHECK: return
func.func @addi_cmpi_select_v64i32(%a: vector<64xi32>, %n: i32,
                                   %x: vector<64xi32>, %y: vector<64xi32>) -> vector<64xi32> {
  %nb = vector.broadcast %n : i32 to vector<64xi32>
  %s = arith.addi %a, %nb : vector<64xi32>
  %c = arith.cmpi slt, %s, %x : vector<64xi32>
  %0 = arith.select %c, %x, %y : vector<64xi1>, vector<64xi32>
  return %0 : vector<64xi32>
}

// -----

// Rank 2, with an integer subtract feeding the compare and a splat selected:
// all of it is split into 16-lane parts.
// CHECK-LABEL: func.func @subi_cmpi_select_8x8
// CHECK-NOT: vector<64xi1>
// CHECK-COUNT-4: aievec.sel %{{.*}}, %{{.*}}, %{{.*}} : vector<32xbf16>, vector<32xbf16>, ui32, vector<32xbf16>
// CHECK: %[[C:.*]] = aievec.concat %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<16xbf16>, vector<64xbf16>
// CHECK: vector.shape_cast %[[C]] : vector<64xbf16> to vector<8x8xbf16>
func.func @subi_cmpi_select_8x8(%r: vector<8x8xi32>, %c: vector<8x8xi32>,
                                %s: vector<8x8xbf16>) -> vector<8x8xbf16> {
  %k = arith.constant dense<512> : vector<8x8xi32>
  %ninf = arith.constant dense<0xFF80> : vector<8x8xbf16>
  %d = arith.subi %r, %c : vector<8x8xi32>
  %m = arith.cmpi ult, %d, %k : vector<8x8xi32>
  %o = arith.select %m, %s, %ninf : vector<8x8xi1>, vector<8x8xbf16>
  return %o : vector<8x8xbf16>
}

// -----

// A compare with another user stays for that user; the select is split with
// compares of its own.
// CHECK-LABEL: func.func @cmpi_two_users
// CHECK: %[[M:.*]] = arith.cmpi ult, %{{.*}}, %{{.*}} : vector<64xi32>
// CHECK-COUNT-4: aievec.cmp %{{.*}}, %{{.*}} {pred = "ult"} : vector<16xi32>, vector<16xi32>, ui32
// CHECK: %[[O:.*]] = aievec.concat %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<16xbf16>, vector<64xbf16>
// CHECK: return %[[O]], %[[M]]
func.func @cmpi_two_users(%d: vector<64xi32>, %k: vector<64xi32>, %s: vector<64xbf16>,
                          %t: vector<64xbf16>) -> (vector<64xbf16>, vector<64xi1>) {
  %m = arith.cmpi ult, %d, %k : vector<64xi32>
  %o = arith.select %m, %s, %t : vector<64xi1>, vector<64xbf16>
  return %o, %m : vector<64xbf16>, vector<64xi1>
}
