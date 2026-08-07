// Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm | FileCheck %s

func.func @matmul(%A : vector<4x8xbf16>, %B : vector<8x4xbf16>,
                  %C : vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xbf16>, vector<8x4xbf16>
                                  into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// CHECK-LABEL: @matmul
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>
// CHECK-SAME: %[[B:.*]]: vector<8x4xbf16>
// CHECK-SAME: %[[C:.*]]: vector<4x4xf32>
// CHECK:      %[[FA:.*]] = vector.shape_cast %[[A]] :
// CHECK-SAME:                      vector<4x8xbf16> to vector<32xbf16>
// CHECK:      %[[FB:.*]] = vector.shape_cast %[[B]] :
// CHECK-SAME:                      vector<8x4xbf16> to vector<32xbf16>
// CHECK:      %[[FC:.*]] = vector.shape_cast %[[C]] :
// CHECK-SAME:                      vector<4x4xf32> to vector<16xf32>
// CHECK:      %[[CONF:.*]] = llvm.mlir.constant(28 : i32) : i32
// CHECK:      %[[BCACC:.*]] = llvm.bitcast %[[FC]] : vector<16xf32> to vector<8xi64>
// CHECK:      %[[RACC:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(
// CHECK-SAME:         %[[FA]], %[[FB]], %[[BCACC]], %[[CONF]]) :
// CHECK-SAME:         (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32)
// CHECK-SAME:         -> vector<8xi64>
// CHECK:      %[[BCR:.*]] = llvm.bitcast %[[RACC]] : vector<8xi64> to vector<16xf32>
// CHECK:      %[[R:.*]] = vector.shape_cast %[[BCR]] :
// CHECK-SAME:                      vector<16xf32> to vector<4x4xf32>
// CHECK:      return %[[R]] : vector<4x4xf32>

// -----

// Signless element types are treated as signed, so this matches the
// si8 x si8 case below (conf 776).
func.func @matmul(%A : vector<4x8xi8>, %B : vector<8x8xi8>,
                  %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xi8>, vector<8x8xi8>
                                  into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// CHECK-LABEL: @matmul
// CHECK-SAME: %[[A:.*]]: vector<4x8xi8>
// CHECK-SAME: %[[B:.*]]: vector<8x8xi8>
// CHECK-SAME: %[[C:.*]]: vector<4x8xi32>
// CHECK:      %[[FA:.*]] = vector.shape_cast %[[A]] :
// CHECK-SAME:                      vector<4x8xi8> to vector<32xi8>
// CHECK:      %[[FB:.*]] = vector.shape_cast %[[B]] :
// CHECK-SAME:                      vector<8x8xi8> to vector<64xi8>
// CHECK:      %[[FC:.*]] = vector.shape_cast %[[C]] :
// CHECK-SAME:                      vector<4x8xi32> to vector<32xi32>
// CHECK:      %[[CONF:.*]] = llvm.mlir.constant(776 : i32) : i32
// CHECK:      %[[C0I32:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:      %[[IFA2512b:.*]] = llvm.bitcast %[[FA]] : vector<32xi8> to vector<8xi32>
// CHECK:      %[[IFA:.*]] = "xllvm.intr.aie2.set.I512.I256"(%[[IFA2512b]],
// CHECK-SAME:               %[[C0I32]]) : (vector<8xi32>, i32) -> vector<16xi32>
// CHECK:      %[[BCA:.*]] = llvm.bitcast %[[IFA]] : vector<16xi32> to vector<64xi8>
// CHECK:      %[[BCB:.*]] = llvm.bitcast %[[FB]] : vector<64xi8> to vector<16xi32>
// CHECK:      %[[BCC:.*]] = llvm.bitcast %[[FC]] : vector<32xi32> to vector<16xi64>
// CHECK:      %[[RACC:.*]] =
// CHECK-SAME:         "xllvm.intr.aie2.I512.I512.ACC1024.acc32.mac.conf"(
// CHECK-SAME:           %[[BCA]], %[[BCB]], %[[BCC]], %[[CONF]]) :
// CHECK-SAME:           (vector<64xi8>, vector<16xi32>, vector<16xi64>, i32)
// CHECK-SAME:           -> vector<16xi64>
// CHECK:      %[[BCR:.*]] = llvm.bitcast %[[RACC]] : vector<16xi64> to vector<32xi32>
// CHECK:      %[[R:.*]] = vector.shape_cast %[[BCR]] :
// CHECK-SAME:                      vector<32xi32> to vector<4x8xi32>
// CHECK:      return %[[R]] : vector<4x8xi32>

// -----

func.func @matmul(%A : vector<4x2xi32>, %B : vector<2x4xi16>,
                  %C : vector<4x4xi64>) -> vector<4x4xi64> {
  %0 = aievec.matmul %A, %B, %C : vector<4x2xi32>, vector<2x4xi16>
                                  into vector<4x4xi64>
  return %0 : vector<4x4xi64>
}

// CHECK-LABEL: @matmul
// CHECK-SAME: %[[A:.*]]: vector<4x2xi32>
// CHECK-SAME: %[[B:.*]]: vector<2x4xi16>
// CHECK-SAME: %[[C:.*]]: vector<4x4xi64>
// CHECK:      %[[FA:.*]] = vector.shape_cast %[[A]] :
// CHECK-SAME:                      vector<4x2xi32> to vector<8xi32>
// CHECK:      %[[FB:.*]] = vector.shape_cast %[[B]] :
// CHECK-SAME:                      vector<2x4xi16> to vector<8xi16>
// CHECK:      %[[FC:.*]] = vector.shape_cast %[[C]] :
// CHECK-SAME:                      vector<4x4xi64> to vector<16xi64>
// CHECK:      %[[CONF:.*]] = llvm.mlir.constant(770 : i32) : i32
// CHECK:      %[[C0I32:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:      %[[IFA2512b:.*]] = llvm.bitcast %[[FA]] : vector<8xi32> to
// CHECK-SAME:                      vector<8xi32>
// CHECK:      %[[IFA:.*]] = "xllvm.intr.aie2.set.I512.I256"(%[[IFA2512b]],
// CHECK-SAME:                      %[[C0I32]]) : (vector<8xi32>, i32) ->
// CHECK-SAME:                      vector<16xi32>
// CHECK:      %[[BCA:.*]] = llvm.bitcast %[[IFA]] : vector<16xi32> to
// CHECK-SAME:                      vector<64xi8>
// CHECK:      %[[IFB2512b:.*]] = llvm.bitcast %[[FB]] : vector<8xi16> to
// CHECK-SAME:                      vector<4xi32>
// CHECK:      %[[IFB:.*]] = "xllvm.intr.aie2.set.I512.I128"(%[[IFB2512b]]) :
// CHECK-SAME:                      (vector<4xi32>) -> vector<16xi32>
// CHECK:      %[[BCB:.*]] = llvm.bitcast %[[IFB]] : vector<16xi32> to
// CHECK-SAME:                      vector<16xi32>
// CHECK:      %[[RACC:.*]] =
// CHECK-SAME:         "xllvm.intr.aie2.I512.I512.ACC1024.acc64.mac.conf"(
// CHECK-SAME:           %[[BCA]], %[[BCB]], %[[FC]], %[[CONF]]) :
// CHECK-SAME:           (vector<64xi8>, vector<16xi32>, vector<16xi64>, i32)
// CHECK-SAME:           -> vector<16xi64>
// CHECK:      %[[BCR:.*]] = llvm.bitcast %[[RACC]] : vector<16xi64> to vector<16xi64>
// CHECK:      %[[R:.*]] = vector.shape_cast %[[BCR]] :
// CHECK-SAME:                      vector<16xi64> to vector<4x4xi64>
// CHECK:      return %[[R]] : vector<4x4xi64>

// -----

// The AIE2 MAC takes its operand signedness from bits 9 (signX, lhs) and
// 8 (signY, rhs) of the configuration word. Operand signedness is carried in
// the element type (si8/ui8). For the i8 4x8x8 shape the remaining config bits
// are amode=0, bmode=1 (BMODE_8x8), giving a base of 0x008. The four
// combinations below therefore pin 0x008/0x108/0x208/0x308.

func.func @matmul_i8i8_signed_signed(%A : vector<4x8xsi8>, %B : vector<8x8xsi8>,
                                     %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xsi8>, vector<8x8xsi8>
                                  into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// CHECK-LABEL: @matmul_i8i8_signed_signed
// signX = 1, signY = 1 -> 0x308
// CHECK:      %[[CONF:.*]] = llvm.mlir.constant(776 : i32) : i32
// CHECK:      "xllvm.intr.aie2.I512.I512.ACC1024.acc32.mac.conf"(
// CHECK-SAME:   %{{.*}}, %{{.*}}, %{{.*}}, %[[CONF]])

// -----

func.func @matmul_i8i8_unsigned_unsigned(%A : vector<4x8xui8>, %B : vector<8x8xui8>,
                                         %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xui8>, vector<8x8xui8>
                                  into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// CHECK-LABEL: @matmul_i8i8_unsigned_unsigned
// signX = 0, signY = 0 -> 0x008
// CHECK:      %[[CONF:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK:      "xllvm.intr.aie2.I512.I512.ACC1024.acc32.mac.conf"(
// CHECK-SAME:   %{{.*}}, %{{.*}}, %{{.*}}, %[[CONF]])

// -----

func.func @matmul_i8i8_unsigned_signed(%A : vector<4x8xui8>, %B : vector<8x8xsi8>,
                                       %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xui8>, vector<8x8xsi8>
                                  into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// CHECK-LABEL: @matmul_i8i8_unsigned_signed
// signX = 0, signY = 1 -> 0x108 (uint8 activations x int8 weights)
// CHECK:      %[[CONF:.*]] = llvm.mlir.constant(264 : i32) : i32
// CHECK:      "xllvm.intr.aie2.I512.I512.ACC1024.acc32.mac.conf"(
// CHECK-SAME:   %{{.*}}, %{{.*}}, %{{.*}}, %[[CONF]])

// -----

func.func @matmul_i8i8_signed_unsigned(%A : vector<4x8xsi8>, %B : vector<8x8xui8>,
                                       %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xsi8>, vector<8x8xui8>
                                  into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// CHECK-LABEL: @matmul_i8i8_signed_unsigned
// signX = 1, signY = 0 -> 0x208
// CHECK:      %[[CONF:.*]] = llvm.mlir.constant(520 : i32) : i32
// CHECK:      "xllvm.intr.aie2.I512.I512.ACC1024.acc32.mac.conf"(
// CHECK-SAME:   %{{.*}}, %{{.*}}, %{{.*}}, %[[CONF]])


// -----

// Signedness is resolved per operand, so a signless operand does not disturb an
// explicitly-typed partner. conf base for this shape is 0x008; signX is bit 9,
// signY bit 8.

func.func @matmul_i8i8_signless_lhs_unsigned_rhs(%A : vector<4x8xi8>, %B : vector<8x8xui8>,
                                                 %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xi8>, vector<8x8xui8>
                                  into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// CHECK-LABEL: @matmul_i8i8_signless_lhs_unsigned_rhs
// signless lhs -> signed (signX=1), ui8 rhs -> unsigned (signY=0) -> 0x208
// CHECK:      %[[CONF:.*]] = llvm.mlir.constant(520 : i32) : i32
// CHECK:      "xllvm.intr.aie2.I512.I512.ACC1024.acc32.mac.conf"(
// CHECK-SAME:   %{{.*}}, %{{.*}}, %{{.*}}, %[[CONF]])

// -----

func.func @matmul_i8i8_unsigned_lhs_signless_rhs(%A : vector<4x8xui8>, %B : vector<8x8xi8>,
                                                 %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xui8>, vector<8x8xi8>
                                  into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// CHECK-LABEL: @matmul_i8i8_unsigned_lhs_signless_rhs
// ui8 lhs -> unsigned (signX=0), signless rhs -> signed (signY=1) -> 0x108
// CHECK:      %[[CONF:.*]] = llvm.mlir.constant(264 : i32) : i32
// CHECK:      "xllvm.intr.aie2.I512.I512.ACC1024.acc32.mac.conf"(
// CHECK-SAME:   %{{.*}}, %{{.*}}, %{{.*}}, %[[CONF]])

// -----

// The two mixed-precision shapes the VectorToAIEVec contraction pattern
// actually emits: the lhs is already a legal AIE2 narrow type so nothing is
// extended and it stays signless, while the rhs picks up si8/si16 from the
// arith.extsi that was peeled. See @contracti16i8i32 and @contracti32i16i64 in
// test/Conversion/VectorToAIEVec/test-contract.mlir.

func.func @matmul_i16si8_signless_lhs(%A : vector<4x4xi16>, %B : vector<4x8xsi8>,
                                      %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x4xi16>, vector<4x8xsi8>
                                  into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// CHECK-LABEL: @matmul_i16si8_signless_lhs
// CHECK:      %[[CONF:.*]] = llvm.mlir.constant(784 : i32) : i32

// -----

func.func @matmul_i32si16_signless_lhs(%A : vector<4x2xi32>, %B : vector<2x4xsi16>,
                                       %C : vector<4x4xi64>) -> vector<4x4xi64> {
  %0 = aievec.matmul %A, %B, %C : vector<4x2xi32>, vector<2x4xsi16>
                                  into vector<4x4xi64>
  return %0 : vector<4x4xi64>
}

// CHECK-LABEL: @matmul_i32si16_signless_lhs
// CHECK:      %[[CONF:.*]] = llvm.mlir.constant(770 : i32) : i32
