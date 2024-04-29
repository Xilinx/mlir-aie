// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm | FileCheck %s

func.func @i8_broadcast_scalar(%arg0 : i8) -> vector<64xi8> {
  %0 = aievec.broadcast_scalar %arg0 : i8, vector<64xi8>
  return %0 : vector<64xi8>
}

// CHECK-LABEL: @i8_broadcast_scalar
// CHECK-SAME: %[[ARG0:.*]]: i8
// CHECK: %[[VAL:.*]]  = llvm.sext %[[ARG0]] : i8 to i32
// CHECK-NEXT: %[[VBROADCAST:.*]] = "xllvm.intr.aie2.vbroadcast8.I512"(
// CHECK-SAME: %[[VAL]]) : 
// CHECK-SAME: (i32) -> vector<64xi8>
// CHECK-NEXT: return %[[VBROADCAST]] : vector<64xi8>

// -----

func.func @i32_broadcast_scalar(%arg0 : i32) -> vector<16xi32> {
  %0 = aievec.broadcast_scalar %arg0 : i32, vector<16xi32>
  return %0 : vector<16xi32>
}

// CHECK-LABEL: @i32_broadcast_scalar
// CHECK-SAME: %[[ARG0:.*]]: i32
// CHECK-NEXT: %[[VBROADCAST:.*]] = "xllvm.intr.aie2.vbroadcast32.I512"(
// CHECK-SAME: %[[ARG0]]) : 
// CHECK-SAME: (i32) -> vector<16xi32>
// CHECK-NEXT: return %[[VBROADCAST]] : vector<16xi32>

// -----

func.func @bf16_broadcast_scalar(%arg0 : bf16) -> vector<32xbf16> {
  %0 = aievec.broadcast_scalar %arg0 : bf16, vector<32xbf16>
  return %0 : vector<32xbf16>
}

// CHECK-LABEL: @bf16_broadcast_scalar
// CHECK-SAME: %[[ARG0:.*]]: bf16
// CHECK: %[[VBROADCAST:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(
// CHECK-SAME: %[[ARG0]]) : 
// CHECK-SAME: (bf16) -> vector<32xbf16>
// CHECK-NEXT: return %[[VBROADCAST]] : vector<32xbf16>