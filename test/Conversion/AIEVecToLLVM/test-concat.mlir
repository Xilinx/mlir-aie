// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm | FileCheck %s

func.func @v64i8_concat_v32i8(%arg0 : vector<32xi8>, %arg1 : vector<32xi8>) -> vector<64xi8> {
  %0 = aievec.concat %arg0, %arg1 : vector<32xi8>, vector<64xi8>
  return %0 : vector<64xi8>
}

// CHECK-LABEL: @v64i8_concat_v32i8
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi8>,
// CHECK-SAME: %[[ARG1:.*]]: vector<32xi8>
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi8> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<32xi8> to vector<8xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I512.I256"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]]) : 
// CHECK-SAME: (vector<8xi32>, vector<8xi32>) -> vector<16xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<16xi32> to vector<64xi8>
// CHECK-NEXT: return %[[RES]] : vector<64xi8>

// -----

func.func @v128i8_concat_v32i8(%arg0 : vector<32xi8>, %arg1 : vector<32xi8>, 
                               %arg2 : vector<32xi8>, %arg3 : vector<32xi8>) -> vector<128xi8> {
  %0 = aievec.concat %arg0, %arg1, %arg2, %arg3 : vector<32xi8>, vector<128xi8>
  return %0 : vector<128xi8>
}

// CHECK-LABEL: @v128i8_concat_v32i8
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi8>, %[[ARG1:.*]]: vector<32xi8>, %[[ARG2:.*]]: vector<32xi8>, %[[ARG3:.*]]: vector<32xi8>
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi8> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<32xi8> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST2:.*]] = llvm.bitcast %[[ARG2]] : vector<32xi8> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST3:.*]] = llvm.bitcast %[[ARG3]] : vector<32xi8> to vector<8xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I1024.I256"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]], %[[BITCAST2]], %[[BITCAST3]]) : 
// CHECK-SAME: (vector<8xi32>, vector<8xi32>, vector<8xi32>, vector<8xi32>) -> vector<32xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<32xi32> to vector<128xi8>
// CHECK-NEXT: return %[[RES]] : vector<128xi8>

// -----

func.func @v128i8_concat_v64i8(%arg0 : vector<64xi8>, %arg1 : vector<64xi8>) -> vector<128xi8> {
  %0 = aievec.concat %arg0, %arg1 : vector<64xi8>, vector<128xi8>
  return %0 : vector<128xi8>
}

// CHECK-LABEL: @v128i8_concat_v64i8
// CHECK-SAME: %[[ARG0:.*]]: vector<64xi8>,
// CHECK-SAME: %[[ARG1:.*]]: vector<64xi8>
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<64xi8> to vector<16xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<64xi8> to vector<16xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I1024.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]]) : 
// CHECK-SAME: (vector<16xi32>, vector<16xi32>) -> vector<32xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<32xi32> to vector<128xi8>
// CHECK-NEXT: return %[[RES]] : vector<128xi8>

// -----

func.func @v32i16_concat_v16i16(%arg0 : vector<16xi16>, %arg1 : vector<16xi16>) -> vector<32xi16> {
  %0 = aievec.concat %arg0, %arg1 : vector<16xi16>, vector<32xi16>
  return %0 : vector<32xi16>
}

// CHECK-LABEL: @v32i16_concat_v16i16
// CHECK-SAME: %[[ARG0:.*]]: vector<16xi16>,
// CHECK-SAME: %[[ARG1:.*]]: vector<16xi16>
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xi16> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<16xi16> to vector<8xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I512.I256"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]]) : 
// CHECK-SAME: (vector<8xi32>, vector<8xi32>) -> vector<16xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<16xi32> to vector<32xi16>
// CHECK-NEXT: return %[[RES]] : vector<32xi16>

// -----

func.func @v64i16_concat_v16i16(%arg0 : vector<16xi16>, %arg1 : vector<16xi16>,
                                  %arg2 : vector<16xi16>, %arg3 : vector<16xi16>) -> vector<64xi16> {
  %0 = aievec.concat %arg0, %arg1, %arg2, %arg3 : vector<16xi16>, vector<64xi16>
  return %0 : vector<64xi16>
}

// CHECK-LABEL: @v64i16_concat_v16i16
// CHECK-SAME: %[[ARG0:.*]]: vector<16xi16>, %[[ARG1:.*]]: vector<16xi16>, %[[ARG2:.*]]: vector<16xi16>, %[[ARG3:.*]]: vector<16xi16>
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xi16> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<16xi16> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST2:.*]] = llvm.bitcast %[[ARG2]] : vector<16xi16> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST3:.*]] = llvm.bitcast %[[ARG3]] : vector<16xi16> to vector<8xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I1024.I256"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]], %[[BITCAST2]], %[[BITCAST3]]) : 
// CHECK-SAME: (vector<8xi32>, vector<8xi32>, vector<8xi32>, vector<8xi32>) -> vector<32xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<32xi32> to vector<64xi16>
// CHECK-NEXT: return %[[RES]] : vector<64xi16>

// -----

func.func @v64i16_concat_v32i16(%arg0 : vector<32xi16>, %arg1 : vector<32xi16>) -> vector<64xi16> {
  %0 = aievec.concat %arg0, %arg1 : vector<32xi16>, vector<64xi16>
  return %0 : vector<64xi16>
}

// CHECK-LABEL: @v64i16_concat_v32i16
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi16>,
// CHECK-SAME: %[[ARG1:.*]]: vector<32xi16>
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi16> to vector<16xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<32xi16> to vector<16xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I1024.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]]) : 
// CHECK-SAME: (vector<16xi32>, vector<16xi32>) -> vector<32xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<32xi32> to vector<64xi16>
// CHECK-NEXT: return %[[RES]] : vector<64xi16>

// -----

func.func @v16i32_concat_v8i32(%arg0 : vector<8xi32>, %arg1 : vector<8xi32>) -> vector<16xi32> {
  %0 = aievec.concat %arg0, %arg1 : vector<8xi32>, vector<16xi32>
  return %0 : vector<16xi32>
}

// CHECK-LABEL: @v16i32_concat_v8i32
// CHECK-SAME: %[[ARG0:.*]]: vector<8xi32>,
// CHECK-SAME: %[[ARG1:.*]]: vector<8xi32>
// CHECK: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I512.I256"(
// CHECK-SAME: %[[ARG0]], %[[ARG1]]) : 
// CHECK-SAME: (vector<8xi32>, vector<8xi32>) -> vector<16xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<16xi32> to vector<16xi32>
// CHECK-NEXT: return %[[RES]] : vector<16xi32>

// -----

func.func @v32i32_concat_v8i32(%arg0 : vector<8xi32>, %arg1 : vector<8xi32>,
                                  %arg2 : vector<8xi32>, %arg3 : vector<8xi32>) -> vector<32xi32> {
  %0 = aievec.concat %arg0, %arg1, %arg2, %arg3 : vector<8xi32>, vector<32xi32>
  return %0 : vector<32xi32>
}

// CHECK-LABEL: @v32i32_concat_v8i32
// CHECK-SAME: %[[ARG0:.*]]: vector<8xi32>, %[[ARG1:.*]]: vector<8xi32>, %[[ARG2:.*]]: vector<8xi32>, %[[ARG3:.*]]: vector<8xi32>
// CHECK: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I1024.I256"(
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]) : 
// CHECK-SAME: (vector<8xi32>, vector<8xi32>, vector<8xi32>, vector<8xi32>) -> vector<32xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<32xi32> to vector<32xi32>
// CHECK-NEXT: return %[[RES]] : vector<32xi32>

// -----

func.func @v32i32_concat_v16i32(%arg0 : vector<16xi32>, %arg1 : vector<16xi32>) -> vector<32xi32> {
  %0 = aievec.concat %arg0, %arg1 : vector<16xi32>, vector<32xi32>
  return %0 : vector<32xi32>
}

// CHECK-LABEL: @v32i32_concat_v16i32
// CHECK-SAME: %[[ARG0:.*]]: vector<16xi32>,
// CHECK-SAME: %[[ARG1:.*]]: vector<16xi32>
// CHECK: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I1024.I512"(
// CHECK-SAME: %[[ARG0]], %[[ARG1]]) : 
// CHECK-SAME: (vector<16xi32>, vector<16xi32>) -> vector<32xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<32xi32> to vector<32xi32>
// CHECK-NEXT: return %[[RES]] : vector<32xi32>

// -----

func.func @v32bf16_concat_v16bf16(%arg0 : vector<16xbf16>, %arg1 : vector<16xbf16>) -> vector<32xbf16> {
  %0 = aievec.concat %arg0, %arg1 : vector<16xbf16>, vector<32xbf16>
  return %0 : vector<32xbf16>
}

// CHECK-LABEL: @v32bf16_concat_v16bf16
// CHECK-SAME: %[[ARG0:.*]]: vector<16xbf16>,
// CHECK-SAME: %[[ARG1:.*]]: vector<16xbf16>
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xbf16> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<16xbf16> to vector<8xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I512.I256"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]]) : 
// CHECK-SAME: (vector<8xi32>, vector<8xi32>) -> vector<16xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<16xi32> to vector<32xbf16>
// CHECK-NEXT: return %[[RES]] : vector<32xbf16>

// -----

func.func @v64bf16_concat_v16bf16(%arg0 : vector<16xbf16>, %arg1 : vector<16xbf16>,
                                  %arg2 : vector<16xbf16>, %arg3 : vector<16xbf16>) -> vector<64xbf16> {
  %0 = aievec.concat %arg0, %arg1, %arg2, %arg3 : vector<16xbf16>, vector<64xbf16>
  return %0 : vector<64xbf16>
}

// CHECK-LABEL: @v64bf16_concat_v16bf16
// CHECK-SAME: %[[ARG0:.*]]: vector<16xbf16>, %[[ARG1:.*]]: vector<16xbf16>, %[[ARG2:.*]]: vector<16xbf16>, %[[ARG3:.*]]: vector<16xbf16>
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xbf16> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<16xbf16> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST2:.*]] = llvm.bitcast %[[ARG2]] : vector<16xbf16> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST3:.*]] = llvm.bitcast %[[ARG3]] : vector<16xbf16> to vector<8xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I1024.I256"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]], %[[BITCAST2]], %[[BITCAST3]]) : 
// CHECK-SAME: (vector<8xi32>, vector<8xi32>, vector<8xi32>, vector<8xi32>) -> vector<32xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<32xi32> to vector<64xbf16>
// CHECK-NEXT: return %[[RES]] : vector<64xbf16>

// -----

func.func @v64bf16_concat_v32bf16(%arg0 : vector<32xbf16>, %arg1 : vector<32xbf16>) -> vector<64xbf16> {
  %0 = aievec.concat %arg0, %arg1 : vector<32xbf16>, vector<64xbf16>
  return %0 : vector<64xbf16>
}

// CHECK-LABEL: @v64bf16_concat_v32bf16
// CHECK-SAME: %[[ARG0:.*]]: vector<32xbf16>,
// CHECK-SAME: %[[ARG1:.*]]: vector<32xbf16>
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xbf16> to vector<16xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<32xbf16> to vector<16xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I1024.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]]) : 
// CHECK-SAME: (vector<16xi32>, vector<16xi32>) -> vector<32xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<32xi32> to vector<64xbf16>
// CHECK-NEXT: return %[[RES]] : vector<64xbf16>

// -----

func.func @v16f32_concat_v8f32(%arg0 : vector<8xf32>, %arg1 : vector<8xf32>) -> vector<16xf32> {
  %0 = aievec.concat %arg0, %arg1 : vector<8xf32>, vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: @v16f32_concat_v8f32
// CHECK-SAME: %[[ARG0:.*]]: vector<8xf32>,
// CHECK-SAME: %[[ARG1:.*]]: vector<8xf32>
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<8xf32> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<8xf32> to vector<8xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I512.I256"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]]) : 
// CHECK-SAME: (vector<8xi32>, vector<8xi32>) -> vector<16xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<16xi32> to vector<16xf32>
// CHECK-NEXT: return %[[RES]] : vector<16xf32>

// -----

func.func @v32f32_concat_v8f32(%arg0 : vector<8xf32>, %arg1 : vector<8xf32>,
                                  %arg2 : vector<8xf32>, %arg3 : vector<8xf32>) -> vector<32xf32> {
  %0 = aievec.concat %arg0, %arg1, %arg2, %arg3 : vector<8xf32>, vector<32xf32>
  return %0 : vector<32xf32>
}

// CHECK-LABEL: @v32f32_concat_v8f32
// CHECK-SAME: %[[ARG0:.*]]: vector<8xf32>, %[[ARG1:.*]]: vector<8xf32>, %[[ARG2:.*]]: vector<8xf32>, %[[ARG3:.*]]: vector<8xf32>
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<8xf32> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<8xf32> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST2:.*]] = llvm.bitcast %[[ARG2]] : vector<8xf32> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST3:.*]] = llvm.bitcast %[[ARG3]] : vector<8xf32> to vector<8xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I1024.I256"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]], %[[BITCAST2]], %[[BITCAST3]]) : 
// CHECK-SAME: (vector<8xi32>, vector<8xi32>, vector<8xi32>, vector<8xi32>) -> vector<32xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<32xi32> to vector<32xf32>
// CHECK-NEXT: return %[[RES]] : vector<32xf32>

// -----

func.func @v32f32_concat_v16f32(%arg0 : vector<16xf32>, %arg1 : vector<16xf32>) -> vector<32xf32> {
  %0 = aievec.concat %arg0, %arg1 : vector<16xf32>, vector<32xf32>
  return %0 : vector<32xf32>
}

// CHECK-LABEL: @v32f32_concat_v16f32
// CHECK-SAME: %[[ARG0:.*]]: vector<16xf32>,
// CHECK-SAME: %[[ARG1:.*]]: vector<16xf32>
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xf32> to vector<16xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<16xf32> to vector<16xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I1024.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]]) : 
// CHECK-SAME: (vector<16xi32>, vector<16xi32>) -> vector<32xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<32xi32> to vector<32xf32>
// CHECK-NEXT: return %[[RES]] : vector<32xf32>
