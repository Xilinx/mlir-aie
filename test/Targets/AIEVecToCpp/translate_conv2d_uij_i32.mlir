//===- translate_conv2d_uij_i32.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// RUN: aie-translate --aievec-to-cpp %s -split-input-file | FileCheck %s

// CHECK-LABEL: void conv2d_0(int32_t * restrict v1, int32_t * restrict v2, int32_t * restrict v3) {
func.func @conv2d_0(%arg0: memref<2048x2048xi32>, %arg1: memref<9xi32>, %arg2: memref<2046x2046xi32>) {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<9xi32>, vector<8xi32>
  %1 = aievec.upd %arg1[%c8] {index = 0 : i8, offset = 0 : i32} : memref<9xi32>, vector<8xi32>
  %c0_0 = arith.constant 0 : index
  %c2046 = arith.constant 2046 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0_0 to %c2046 step %c1 {
    %c1_1 = arith.constant 1 : index
    %2 = arith.addi %arg3, %c1_1 : index
    %c2 = arith.constant 2 : index
    %3 = arith.addi %arg3, %c2 : index
    %c0_2 = arith.constant 0 : index
    %c2046_3 = arith.constant 2046 : index
    %c8_4 = arith.constant 8 : index
    scf.for %arg4 = %c0_2 to %c2046_3 step %c8_4 {
      %4 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2046x2046xi32>, vector<8xi32>
      %5 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xi32>, vector<16xi32>
      %6 = aievec.ups %4 {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
      %7 = aievec_aie1.mac %5, %0, %6 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %c1_5 = arith.constant 1 : index
      %8 = arith.addi %arg4, %c1_5 : index
      %9 = aievec.upd %arg0[%arg3, %8], %5 {index = 1 : i8, offset = 224 : i32} : memref<2048x2048xi32>, vector<16xi32>
      %10 = aievec_aie1.mac %9, %0, %7 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %11 = aievec_aie1.mac %9, %0, %10 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %12 = aievec.upd %arg0[%2, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xi32>, vector<16xi32>
      %13 = aievec_aie1.mac %12, %0, %11 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %14 = aievec.upd %arg0[%2, %8], %12 {index = 1 : i8, offset = 224 : i32} : memref<2048x2048xi32>, vector<16xi32>
      %15 = aievec_aie1.mac %14, %0, %13 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %16 = aievec_aie1.mac %14, %0, %15 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %17 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xi32>, vector<16xi32>
      %18 = aievec_aie1.mac %17, %0, %16 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %19 = aievec.upd %arg0[%3, %8], %17 {index = 1 : i8, offset = 224 : i32} : memref<2048x2048xi32>, vector<16xi32>
      %20 = aievec_aie1.mac %19, %0, %18 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %21 = aievec_aie1.mac %19, %1, %20 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %22 = aievec.srs %21, %c0_i32: vector<8xi80>, i32, vector<8xi32>
      vector.transfer_write %22, %arg2[%arg3, %arg4] : vector<8xi32>, memref<2046x2046xi32>
    }
  }
  return
}

//CHECK-NEXT:  size_t v4 = 8;
//CHECK-NEXT:  size_t v5 = 0;
//CHECK-NEXT:  int32_t v6 = 0;
//CHECK-NEXT:  v8int32 v7 = *(v8int32 *)(v2 + v5);
//CHECK-NEXT:  v8int32 v8 = *(v8int32 *)(v2 + v4);
//CHECK-NEXT:  size_t v9 = 0;
//CHECK-NEXT:  size_t v10 = 2046;
//CHECK-NEXT:  size_t v11 = 1;
//CHECK-NEXT:  for (size_t v12 = v9; v12 < v10; v12 += v11)
//CHECK-NEXT:  chess_prepare_for_pipelining
//CHECK-NEXT:  chess_loop_range(2046, 2046)
//CHECK-NEXT:  {
//CHECK-NEXT:    size_t v13 = 1;
//CHECK-NEXT:    size_t v14 = v12 + v13;
//CHECK-NEXT:    size_t v15 = 2;
//CHECK-NEXT:    size_t v16 = v12 + v15;
//CHECK-NEXT:    size_t v17 = 0;
//CHECK-NEXT:    size_t v18 = 2046;
//CHECK-NEXT:    size_t v19 = 8;
//CHECK-NEXT:    for (size_t v20 = v17; v20 < v18; v20 += v19)
//CHECK-NEXT:    chess_prepare_for_pipelining
//CHECK-NEXT:    chess_loop_range(255, 256)
//CHECK-NEXT:    {
//CHECK-NEXT:      v8int32 v21 = *(v8int32 *)(v3 + 2046*v12+v20);
//CHECK-NEXT:      v16int32 v22;
//CHECK-NEXT:      int32_t * restrict r_v22_v1 = v1;
//CHECK-NEXT:      v22 = upd_w(v22, 0, *(v8int32 *)(r_v22_v1 + 2048*v12+v20));
//CHECK-NEXT:      v8acc80 v23 = lups(v21, 0);
//CHECK-NEXT:      v23 = lmac8(v23, v22, 0, 0x76543210, v7, 0, 0x00000000);
//CHECK-NEXT:      size_t v24 = 1;
//CHECK-NEXT:      size_t v25 = v20 + v24;
//CHECK-NEXT:      v22 = upd_w(v22, 1, *(v8int32 *)(r_v22_v1 + 2048*v12+v25 + 7));
//CHECK-NEXT:      v23 = lmac8(v23, v22, 1, 0x76543210, v7, 1, 0x00000000);
//CHECK-NEXT:      v23 = lmac8(v23, v22, 2, 0x76543210, v7, 2, 0x00000000);
//CHECK-NEXT:      v16int32 v26;
//CHECK-NEXT:      int32_t * restrict r_v26_v1 = v1;
//CHECK-NEXT:      v26 = upd_w(v26, 0, *(v8int32 *)(r_v26_v1 + 2048*v14+v20));
//CHECK-NEXT:      v23 = lmac8(v23, v26, 0, 0x76543210, v7, 3, 0x00000000);
//CHECK-NEXT:      v26 = upd_w(v26, 1, *(v8int32 *)(r_v26_v1 + 2048*v14+v25 + 7));
//CHECK-NEXT:      v23 = lmac8(v23, v26, 1, 0x76543210, v7, 4, 0x00000000);
//CHECK-NEXT:      v23 = lmac8(v23, v26, 2, 0x76543210, v7, 5, 0x00000000);
//CHECK-NEXT:      v16int32 v27;
//CHECK-NEXT:      int32_t * restrict r_v27_v1 = v1;
//CHECK-NEXT:      v27 = upd_w(v27, 0, *(v8int32 *)(r_v27_v1 + 2048*v16+v20));
//CHECK-NEXT:      v23 = lmac8(v23, v27, 0, 0x76543210, v7, 6, 0x00000000);
//CHECK-NEXT:      v27 = upd_w(v27, 1, *(v8int32 *)(r_v27_v1 + 2048*v16+v25 + 7));
//CHECK-NEXT:      v23 = lmac8(v23, v27, 1, 0x76543210, v7, 7, 0x00000000);
//CHECK-NEXT:      v23 = lmac8(v23, v27, 2, 0x76543210, v8, 0, 0x00000000);
//CHECK-NEXT:      v8int32 v28 = srs(v23, v6);
//CHECK-NEXT:      *(v8int32 *)(v3 + 2046*v12+v20) = v28;

// CHECK-LABEL: void conv2d_1(int32_t * restrict v6, size_t m1, size_t m2, int32_t * restrict v7, size_t m3, int32_t * restrict v8, size_t m4, size_t m5, size_t v9, size_t v10) {
func.func @conv2d_1(%arg0: memref<?x?xi32>, %arg1: memref<?xi32>, %arg2: memref<?x?xi32>, %arg3: index, %arg4: index) {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<?xi32>, vector<8xi32>
  %1 = aievec.upd %arg1[%c8] {index = 0 : i8, offset = 0 : i32} : memref<?xi32>, vector<8xi32>
  %c0_0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %arg5 = %c0_0 to %arg3 step %c1 {
    %c1_1 = arith.constant 1 : index
    %2 = arith.addi %arg5, %c1_1 : index
    %c2 = arith.constant 2 : index
    %3 = arith.addi %arg5, %c2 : index
    %c0_2 = arith.constant 0 : index
    %c8_3 = arith.constant 8 : index
    scf.for %arg6 = %c0_2 to %arg4 step %c8_3 {
      %4 = aievec.upd %arg2[%arg5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi32>, vector<8xi32>
      %5 = aievec.upd %arg0[%arg5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi32>, vector<16xi32>
      %6 = aievec.ups %4 {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
      %7 = aievec_aie1.mac %5, %0, %6 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %c1_4 = arith.constant 1 : index
      %8 = arith.addi %arg6, %c1_4 : index
      %9 = aievec.upd %arg0[%arg5, %8], %5 {index = 1 : i8, offset = 224 : i32} : memref<?x?xi32>, vector<16xi32>
      %10 = aievec_aie1.mac %9, %0, %7 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %11 = aievec_aie1.mac %9, %0, %10 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %12 = aievec.upd %arg0[%2, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi32>, vector<16xi32>
      %13 = aievec_aie1.mac %12, %0, %11 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %14 = aievec.upd %arg0[%2, %8], %12 {index = 1 : i8, offset = 224 : i32} : memref<?x?xi32>, vector<16xi32>
      %15 = aievec_aie1.mac %14, %0, %13 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %16 = aievec_aie1.mac %14, %0, %15 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %17 = aievec.upd %arg0[%3, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi32>, vector<16xi32>
      %18 = aievec_aie1.mac %17, %0, %16 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %19 = aievec.upd %arg0[%3, %8], %17 {index = 1 : i8, offset = 224 : i32} : memref<?x?xi32>, vector<16xi32>
      %20 = aievec_aie1.mac %19, %0, %18 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %21 = aievec_aie1.mac %19, %1, %20 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %22 = aievec.srs %21, %c0_i32 : vector<8xi80>, i32, vector<8xi32>
      vector.transfer_write %22, %arg2[%arg5, %arg6] : vector<8xi32>, memref<?x?xi32>
    }
  }
  return
}

//CHECK-NEXT:  size_t v11 = 8;
//CHECK-NEXT:  size_t v12 = 0;
//CHECK-NEXT:  int32_t v13 = 0;
//CHECK-NEXT:  v8int32 v14 = *(v8int32 *)(v7 + v12);
//CHECK-NEXT:  v8int32 v15 = *(v8int32 *)(v7 + v11);
//CHECK-NEXT:  size_t v16 = 0;
//CHECK-NEXT:  size_t v17 = 1;
//CHECK-NEXT:  for (size_t v18 = v16; v18 < v9; v18 += v17)
//CHECK-NEXT:  chess_prepare_for_pipelining
//CHECK-NEXT:  {
//CHECK-NEXT:    size_t v19 = 1;
//CHECK-NEXT:    size_t v20 = v18 + v19;
//CHECK-NEXT:    size_t v21 = 2;
//CHECK-NEXT:    size_t v22 = v18 + v21;
//CHECK-NEXT:    size_t v23 = 0;
//CHECK-NEXT:    size_t v24 = 8;
//CHECK-NEXT:    for (size_t v25 = v23; v25 < v10; v25 += v24)
//CHECK-NEXT:    chess_prepare_for_pipelining
//CHECK-NEXT:    {
//CHECK-NEXT:      v8int32 v26 = *(v8int32 *)(v8 + m5*v18+v25);
//CHECK-NEXT:      v16int32 v27;
//CHECK-NEXT:      int32_t * restrict r_v27_v6 = v6;
//CHECK-NEXT:      v27 = upd_w(v27, 0, *(v8int32 *)(r_v27_v6 + m2*v18+v25));
//CHECK-NEXT:      v8acc80 v28 = lups(v26, 0);
//CHECK-NEXT:      v28 = lmac8(v28, v27, 0, 0x76543210, v14, 0, 0x00000000);
//CHECK-NEXT:      size_t v29 = 1;
//CHECK-NEXT:      size_t v30 = v25 + v29;
//CHECK-NEXT:      v27 = upd_w(v27, 1, *(v8int32 *)(r_v27_v6 + m2*v18+v30 + 7));
//CHECK-NEXT:      v28 = lmac8(v28, v27, 1, 0x76543210, v14, 1, 0x00000000);
//CHECK-NEXT:      v28 = lmac8(v28, v27, 2, 0x76543210, v14, 2, 0x00000000);
//CHECK-NEXT:      v16int32 v31;
//CHECK-NEXT:      int32_t * restrict r_v31_v6 = v6;
//CHECK-NEXT:      v31 = upd_w(v31, 0, *(v8int32 *)(r_v31_v6 + m2*v20+v25));
//CHECK-NEXT:      v28 = lmac8(v28, v31, 0, 0x76543210, v14, 3, 0x00000000);
//CHECK-NEXT:      v31 = upd_w(v31, 1, *(v8int32 *)(r_v31_v6 + m2*v20+v30 + 7));
//CHECK-NEXT:      v28 = lmac8(v28, v31, 1, 0x76543210, v14, 4, 0x00000000);
//CHECK-NEXT:      v28 = lmac8(v28, v31, 2, 0x76543210, v14, 5, 0x00000000);
//CHECK-NEXT:      v16int32 v32;
//CHECK-NEXT:      int32_t * restrict r_v32_v6 = v6;
//CHECK-NEXT:      v32 = upd_w(v32, 0, *(v8int32 *)(r_v32_v6 + m2*v22+v25));
//CHECK-NEXT:      v28 = lmac8(v28, v32, 0, 0x76543210, v14, 6, 0x00000000);
//CHECK-NEXT:      v32 = upd_w(v32, 1, *(v8int32 *)(r_v32_v6 + m2*v22+v30 + 7));
//CHECK-NEXT:      v28 = lmac8(v28, v32, 1, 0x76543210, v14, 7, 0x00000000);
//CHECK-NEXT:      v28 = lmac8(v28, v32, 2, 0x76543210, v15, 0, 0x00000000);
//CHECK-NEXT:      v8int32 v33 = srs(v28, v13);
//CHECK-NEXT:      *(v8int32 *)(v8 + m5*v18+v25) = v33;

// CHECK-LABEL: void conv2d_2(int32_t * restrict v6, size_t m1, size_t m2, int32_t * restrict v7, size_t m3, int32_t * restrict v8, size_t m4, size_t m5) {
func.func @conv2d_2(%arg0: memref<?x?xi32>, %arg1: memref<?xi32>, %arg2: memref<?x?xi32>) {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = memref.dim %arg0, %c0 : memref<?x?xi32>
  %1 = memref.dim %arg0, %c1 : memref<?x?xi32>
  %2 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<?xi32>, vector<8xi32>
  %3 = aievec.upd %arg1[%c8] {index = 0 : i8, offset = 0 : i32} : memref<?xi32>, vector<8xi32>
  %c0_0 = arith.constant 0 : index
  %c1_1 = arith.constant 1 : index
  scf.for %arg3 = %c0_0 to %0 step %c1_1 {
    %c1_2 = arith.constant 1 : index
    %4 = arith.addi %arg3, %c1_2 : index
    %c2 = arith.constant 2 : index
    %5 = arith.addi %arg3, %c2 : index
    %c0_3 = arith.constant 0 : index
    %c8_4 = arith.constant 8 : index
    scf.for %arg4 = %c0_3 to %1 step %c8_4 {
      %6 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi32>, vector<8xi32>
      %7 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi32>, vector<16xi32>
      %8 = aievec.ups %6 {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
      %9 = aievec_aie1.mac %7, %2, %8 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %c1_5 = arith.constant 1 : index
      %10 = arith.addi %arg4, %c1_5 : index
      %11 = aievec.upd %arg0[%arg3, %10], %7 {index = 1 : i8, offset = 224 : i32} : memref<?x?xi32>, vector<16xi32>
      %12 = aievec_aie1.mac %11, %2, %9 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %13 = aievec_aie1.mac %11, %2, %12 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %14 = aievec.upd %arg0[%4, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi32>, vector<16xi32>
      %15 = aievec_aie1.mac %14, %2, %13 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %16 = aievec.upd %arg0[%4, %10], %14 {index = 1 : i8, offset = 224 : i32} : memref<?x?xi32>, vector<16xi32>
      %17 = aievec_aie1.mac %16, %2, %15 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %18 = aievec_aie1.mac %16, %2, %17 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %19 = aievec.upd %arg0[%5, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi32>, vector<16xi32>
      %20 = aievec_aie1.mac %19, %2, %18 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %21 = aievec.upd %arg0[%5, %10], %19 {index = 1 : i8, offset = 224 : i32} : memref<?x?xi32>, vector<16xi32>
      %22 = aievec_aie1.mac %21, %2, %20 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %23 = aievec_aie1.mac %21, %3, %22 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %24 = aievec.srs %23, %c0_i32 : vector<8xi80>, i32, vector<8xi32>
      vector.transfer_write %24, %arg2[%arg3, %arg4] : vector<8xi32>, memref<?x?xi32>
    }
  }
  return
}

//CHECK-NEXT:  size_t v9 = 8;
//CHECK-NEXT:  size_t v10 = 0;
//CHECK-NEXT:  int32_t v11 = 0;
//CHECK-NEXT:  v8int32 v12 = *(v8int32 *)(v7 + v10);
//CHECK-NEXT:  v8int32 v13 = *(v8int32 *)(v7 + v9);
//CHECK-NEXT:  size_t v14 = 0;
//CHECK-NEXT:  size_t v15 = 1;
//CHECK-NEXT:  for (size_t v16 = v14; v16 < m1; v16 += v15)
//CHECK-NEXT:  chess_prepare_for_pipelining
//CHECK-NEXT:  {
//CHECK-NEXT:    size_t v17 = 1;
//CHECK-NEXT:    size_t v18 = v16 + v17;
//CHECK-NEXT:    size_t v19 = 2;
//CHECK-NEXT:    size_t v20 = v16 + v19;
//CHECK-NEXT:    size_t v21 = 0;
//CHECK-NEXT:    size_t v22 = 8;
//CHECK-NEXT:    for (size_t v23 = v21; v23 < m2; v23 += v22)
//CHECK-NEXT:    chess_prepare_for_pipelining
//CHECK-NEXT:    {
//CHECK-NEXT:      v8int32 v24 = *(v8int32 *)(v8 + m5*v16+v23);
//CHECK-NEXT:      v16int32 v25;
//CHECK-NEXT:      int32_t * restrict r_v25_v6 = v6;
//CHECK-NEXT:      v25 = upd_w(v25, 0, *(v8int32 *)(r_v25_v6 + m2*v16+v23));
//CHECK-NEXT:      v8acc80 v26 = lups(v24, 0);
//CHECK-NEXT:      v26 = lmac8(v26, v25, 0, 0x76543210, v12, 0, 0x00000000);
//CHECK-NEXT:      size_t v27 = 1;
//CHECK-NEXT:      size_t v28 = v23 + v27;
//CHECK-NEXT:      v25 = upd_w(v25, 1, *(v8int32 *)(r_v25_v6 + m2*v16+v28 + 7));
//CHECK-NEXT:      v26 = lmac8(v26, v25, 1, 0x76543210, v12, 1, 0x00000000);
//CHECK-NEXT:      v26 = lmac8(v26, v25, 2, 0x76543210, v12, 2, 0x00000000);
//CHECK-NEXT:      v16int32 v29;
//CHECK-NEXT:      int32_t * restrict r_v29_v6 = v6;
//CHECK-NEXT:      v29 = upd_w(v29, 0, *(v8int32 *)(r_v29_v6 + m2*v18+v23));
//CHECK-NEXT:      v26 = lmac8(v26, v29, 0, 0x76543210, v12, 3, 0x00000000);
//CHECK-NEXT:      v29 = upd_w(v29, 1, *(v8int32 *)(r_v29_v6 + m2*v18+v28 + 7));
//CHECK-NEXT:      v26 = lmac8(v26, v29, 1, 0x76543210, v12, 4, 0x00000000);
//CHECK-NEXT:      v26 = lmac8(v26, v29, 2, 0x76543210, v12, 5, 0x00000000);
//CHECK-NEXT:      v16int32 v30;
//CHECK-NEXT:      int32_t * restrict r_v30_v6 = v6;
//CHECK-NEXT:      v30 = upd_w(v30, 0, *(v8int32 *)(r_v30_v6 + m2*v20+v23));
//CHECK-NEXT:      v26 = lmac8(v26, v30, 0, 0x76543210, v12, 6, 0x00000000);
//CHECK-NEXT:      v30 = upd_w(v30, 1, *(v8int32 *)(r_v30_v6 + m2*v20+v28 + 7));
//CHECK-NEXT:      v26 = lmac8(v26, v30, 1, 0x76543210, v12, 7, 0x00000000);
//CHECK-NEXT:      v26 = lmac8(v26, v30, 2, 0x76543210, v13, 0, 0x00000000);
//CHECK-NEXT:      v8int32 v31 = srs(v26, v11);
//CHECK-NEXT:      *(v8int32 *)(v8 + m5*v16+v23) = v31;

// CHECK-LABEL: void conv2d_3(int32_t * restrict v4, size_t m1, int32_t * restrict v5, size_t m2, int32_t * restrict v6, size_t m3) {
func.func @conv2d_3(%arg0: memref<?x256xi32>, %arg1: memref<?xi32>, %arg2: memref<?x256xi32>) {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = memref.dim %arg0, %c0 : memref<?x256xi32>
  %1 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<?xi32>, vector<8xi32>
  %2 = aievec.upd %arg1[%c8] {index = 0 : i8, offset = 0 : i32} : memref<?xi32>, vector<8xi32>
  %c0_0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0_0 to %0 step %c1 {
    %c1_1 = arith.constant 1 : index
    %3 = arith.addi %arg3, %c1_1 : index
    %c2 = arith.constant 2 : index
    %4 = arith.addi %arg3, %c2 : index
    %c0_2 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c8_3 = arith.constant 8 : index
    scf.for %arg4 = %c0_2 to %c256 step %c8_3 {
      %5 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi32>, vector<8xi32>
      %6 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi32>, vector<16xi32>
      %7 = aievec.ups %5 {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
      %8 = aievec_aie1.mac %6, %1, %7 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %c1_4 = arith.constant 1 : index
      %9 = arith.addi %arg4, %c1_4 : index
      %10 = aievec.upd %arg0[%arg3, %9], %6 {index = 1 : i8, offset = 224 : i32} : memref<?x256xi32>, vector<16xi32>
      %11 = aievec_aie1.mac %10, %1, %8 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %12 = aievec_aie1.mac %10, %1, %11 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %13 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi32>, vector<16xi32>
      %14 = aievec_aie1.mac %13, %1, %12 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %15 = aievec.upd %arg0[%3, %9], %13 {index = 1 : i8, offset = 224 : i32} : memref<?x256xi32>, vector<16xi32>
      %16 = aievec_aie1.mac %15, %1, %14 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %17 = aievec_aie1.mac %15, %1, %16 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %18 = aievec.upd %arg0[%4, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi32>, vector<16xi32>
      %19 = aievec_aie1.mac %18, %1, %17 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %20 = aievec.upd %arg0[%4, %9], %18 {index = 1 : i8, offset = 224 : i32} : memref<?x256xi32>, vector<16xi32>
      %21 = aievec_aie1.mac %20, %1, %19 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %22 = aievec_aie1.mac %20, %2, %21 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %23 = aievec.srs %22, %c0_i32 : vector<8xi80>, i32, vector<8xi32>
      vector.transfer_write %23, %arg2[%arg3, %arg4] : vector<8xi32>, memref<?x256xi32>
    }
  }
  return
}

//CHECK-NEXT:  size_t v7 = 8;
//CHECK-NEXT:  size_t v8 = 0;
//CHECK-NEXT:  int32_t v9 = 0;
//CHECK-NEXT:  v8int32 v10 = *(v8int32 *)(v5 + v8);
//CHECK-NEXT:  v8int32 v11 = *(v8int32 *)(v5 + v7);
//CHECK-NEXT:  size_t v12 = 0;
//CHECK-NEXT:  size_t v13 = 1;
//CHECK-NEXT:  for (size_t v14 = v12; v14 < m1; v14 += v13)
//CHECK-NEXT:  chess_prepare_for_pipelining
//CHECK-NEXT:  {
//CHECK-NEXT:    size_t v15 = 1;
//CHECK-NEXT:    size_t v16 = v14 + v15;
//CHECK-NEXT:    size_t v17 = 2;
//CHECK-NEXT:    size_t v18 = v14 + v17;
//CHECK-NEXT:    size_t v19 = 0;
//CHECK-NEXT:    size_t v20 = 256;
//CHECK-NEXT:    size_t v21 = 8;
//CHECK-NEXT:    for (size_t v22 = v19; v22 < v20; v22 += v21)
//CHECK-NEXT:    chess_prepare_for_pipelining
//CHECK-NEXT:    chess_loop_range(32, 32)
//CHECK-NEXT:    {
//CHECK-NEXT:      v8int32 v23 = *(v8int32 *)(v6 + 256*v14+v22);
//CHECK-NEXT:      v16int32 v24;
//CHECK-NEXT:      int32_t * restrict r_v24_v4 = v4;
//CHECK-NEXT:      v24 = upd_w(v24, 0, *(v8int32 *)(r_v24_v4 + 256*v14+v22));
//CHECK-NEXT:      v8acc80 v25 = lups(v23, 0);
//CHECK-NEXT:      v25 = lmac8(v25, v24, 0, 0x76543210, v10, 0, 0x00000000);
//CHECK-NEXT:      size_t v26 = 1;
//CHECK-NEXT:      size_t v27 = v22 + v26;
//CHECK-NEXT:      v24 = upd_w(v24, 1, *(v8int32 *)(r_v24_v4 + 256*v14+v27 + 7));
//CHECK-NEXT:      v25 = lmac8(v25, v24, 1, 0x76543210, v10, 1, 0x00000000);
//CHECK-NEXT:      v25 = lmac8(v25, v24, 2, 0x76543210, v10, 2, 0x00000000);
//CHECK-NEXT:      v16int32 v28;
//CHECK-NEXT:      int32_t * restrict r_v28_v4 = v4;
//CHECK-NEXT:      v28 = upd_w(v28, 0, *(v8int32 *)(r_v28_v4 + 256*v16+v22));
//CHECK-NEXT:      v25 = lmac8(v25, v28, 0, 0x76543210, v10, 3, 0x00000000);
//CHECK-NEXT:      v28 = upd_w(v28, 1, *(v8int32 *)(r_v28_v4 + 256*v16+v27 + 7));
//CHECK-NEXT:      v25 = lmac8(v25, v28, 1, 0x76543210, v10, 4, 0x00000000);
//CHECK-NEXT:      v25 = lmac8(v25, v28, 2, 0x76543210, v10, 5, 0x00000000);
//CHECK-NEXT:      v16int32 v29;
//CHECK-NEXT:      int32_t * restrict r_v29_v4 = v4;
//CHECK-NEXT:      v29 = upd_w(v29, 0, *(v8int32 *)(r_v29_v4 + 256*v18+v22));
//CHECK-NEXT:      v25 = lmac8(v25, v29, 0, 0x76543210, v10, 6, 0x00000000);
//CHECK-NEXT:      v29 = upd_w(v29, 1, *(v8int32 *)(r_v29_v4 + 256*v18+v27 + 7));
//CHECK-NEXT:      v25 = lmac8(v25, v29, 1, 0x76543210, v10, 7, 0x00000000);
//CHECK-NEXT:      v25 = lmac8(v25, v29, 2, 0x76543210, v11, 0, 0x00000000);
//CHECK-NEXT:      v8int32 v30 = srs(v25, v9);
//CHECK-NEXT:      *(v8int32 *)(v6 + 256*v14+v22) = v30;
