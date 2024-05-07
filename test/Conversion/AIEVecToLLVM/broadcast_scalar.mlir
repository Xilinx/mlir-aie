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

func.func @i16_broadcast_scalar(%arg0 : i16) -> vector<32xi16> {
  %0 = aievec.broadcast_scalar %arg0 : i16, vector<32xi16>
  return %0 : vector<32xi16>
}

// CHECK-LABEL: @i16_broadcast_scalar
// CHECK-SAME: %[[ARG0:.*]]: i16
// CHECK: %[[VAL:.*]]  = llvm.sext %[[ARG0]] : i16 to i32
// CHECK-NEXT: %[[VBROADCAST:.*]] = "xllvm.intr.aie2.vbroadcast16.I512"(
// CHECK-SAME: %[[VAL]]) : 
// CHECK-SAME: (i32) -> vector<32xi16>
// CHECK-NEXT: return %[[VBROADCAST]] : vector<32xi16>

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

// -----

func.func @f32_broadcast_scalar(%arg0 : f32) -> vector<16xf32> {
  %0 = aievec.broadcast_scalar %arg0 : f32, vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: @f32_broadcast_scalar
// CHECK-SAME: %[[ARG0:.*]]: f32
// CHECK: %[[VBROADCAST:.*]] = "xllvm.intr.aie2.vbroadcastfloat.I512"(
// CHECK-SAME: %[[ARG0]]) : 
// CHECK-SAME: (f32) -> vector<16xf32>
// CHECK-NEXT: return %[[VBROADCAST]] : vector<16xf32>
