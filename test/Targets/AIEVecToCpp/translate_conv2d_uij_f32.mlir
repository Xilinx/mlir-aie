//===- translate_conv2d_uij_f32.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// RUN: aie-translate --aievec-to-cpp %s -split-input-file | FileCheck %s

// CHECK-LABEL: void conv2d_0(float * restrict v6, size_t m1, size_t m2, float * restrict v7, size_t m3, float * restrict v8, size_t m4, size_t m5) {
func.func @conv2d_0(%arg0: memref<?x?xf32>, %arg1: memref<?xf32>, %arg2: memref<?x?xf32>) {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %2 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<?xf32>, vector<8xf32>
  %3 = aievec.upd %arg1[%c8] {index = 0 : i8, offset = 0 : i32} : memref<?xf32>, vector<8xf32>
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
      %6 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x?xf32>, vector<8xf32>
      %7 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x?xf32>, vector<16xf32>
      %8 = aievec.mac %7, %2, %6 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      %c1_5 = arith.constant 1 : index
      %9 = arith.addi %arg4, %c1_5 : index
      %10 = aievec.upd %arg0[%arg3, %9], %7 {index = 1 : i8, offset = 224 : i32} : memref<?x?xf32>, vector<16xf32>
      %11 = aievec.mac %10, %2, %8 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      %12 = aievec.mac %10, %2, %11 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      %13 = aievec.upd %arg0[%4, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x?xf32>, vector<16xf32>
      %14 = aievec.mac %13, %2, %12 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      %15 = aievec.upd %arg0[%4, %9], %13 {index = 1 : i8, offset = 224 : i32} : memref<?x?xf32>, vector<16xf32>
      %16 = aievec.mac %15, %2, %14 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      %17 = aievec.mac %15, %2, %16 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      %18 = aievec.upd %arg0[%5, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x?xf32>, vector<16xf32>
      %19 = aievec.mac %18, %2, %17 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      %20 = aievec.upd %arg0[%5, %9], %18 {index = 1 : i8, offset = 224 : i32} : memref<?x?xf32>, vector<16xf32>
      %21 = aievec.mac %20, %2, %19 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      %22 = aievec.mac %20, %3, %21 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      vector.transfer_write %22, %arg2[%arg3, %arg4] : vector<8xf32>, memref<?x?xf32>
    }
  }
  return
}

//CHECK-NEXT: size_t v9 = 8;
//CHECK-NEXT: size_t v10 = 0;
//CHECK-NEXT: v8float v11 = *(v8float *)(v7 + v10);
//CHECK-NEXT: v8float v12 = *(v8float *)(v7 + v9);
//CHECK-NEXT: size_t v13 = 0;
//CHECK-NEXT: size_t v14 = 1;
//CHECK-NEXT: for (size_t v15 = v13; v15 < m1; v15 += v14)
//CHECK-NEXT: chess_prepare_for_pipelining
//CHECK-NEXT: {
//CHECK-NEXT:   size_t v16 = 1;
//CHECK-NEXT:   size_t v17 = v15 + v16;
//CHECK-NEXT:   size_t v18 = 2;
//CHECK-NEXT:   size_t v19 = v15 + v18;
//CHECK-NEXT:   size_t v20 = 0;
//CHECK-NEXT:   size_t v21 = 8;
//CHECK-NEXT:   for (size_t v22 = v20; v22 < m2; v22 += v21)
//CHECK-NEXT:   chess_prepare_for_pipelining
//CHECK-NEXT:   {
//CHECK-NEXT:     v8float v23 = *(v8float *)(v8 + m5*v15+v22);
//CHECK-NEXT:     v16float v24;
//CHECK-NEXT:     float * restrict r_v24_v6 = v6;
//CHECK-NEXT:     v24 = upd_w(v24, 0, *(v8float *)(r_v24_v6 + m2*v15+v22));
//CHECK-NEXT:     v23 = fpmac(v23, v24, 0, 0x76543210, v11, 0, 0x00000000);
//CHECK-NEXT:     size_t v25 = 1;
//CHECK-NEXT:     size_t v26 = v22 + v25;
//CHECK-NEXT:     v24 = upd_w(v24, 1, *(v8float *)(r_v24_v6 + m2*v15+v26 + 7));
//CHECK-NEXT:     v23 = fpmac(v23, v24, 1, 0x76543210, v11, 1, 0x00000000);
//CHECK-NEXT:     v23 = fpmac(v23, v24, 2, 0x76543210, v11, 2, 0x00000000);
//CHECK-NEXT:     v16float v27;
//CHECK-NEXT:     float * restrict r_v27_v6 = v6;
//CHECK-NEXT:     v27 = upd_w(v27, 0, *(v8float *)(r_v27_v6 + m2*v17+v22));
//CHECK-NEXT:     v23 = fpmac(v23, v27, 0, 0x76543210, v11, 3, 0x00000000);
//CHECK-NEXT:     v27 = upd_w(v27, 1, *(v8float *)(r_v27_v6 + m2*v17+v26 + 7));
//CHECK-NEXT:     v23 = fpmac(v23, v27, 1, 0x76543210, v11, 4, 0x00000000);
//CHECK-NEXT:     v23 = fpmac(v23, v27, 2, 0x76543210, v11, 5, 0x00000000);
//CHECK-NEXT:     v16float v28;
//CHECK-NEXT:     float * restrict r_v28_v6 = v6;
//CHECK-NEXT:     v28 = upd_w(v28, 0, *(v8float *)(r_v28_v6 + m2*v19+v22));
//CHECK-NEXT:     v23 = fpmac(v23, v28, 0, 0x76543210, v11, 6, 0x00000000);
//CHECK-NEXT:     v28 = upd_w(v28, 1, *(v8float *)(r_v28_v6 + m2*v19+v26 + 7));
//CHECK-NEXT:     v23 = fpmac(v23, v28, 1, 0x76543210, v11, 7, 0x00000000);
//CHECK-NEXT:     v23 = fpmac(v23, v28, 2, 0x76543210, v12, 0, 0x00000000);
//CHECK-NEXT:     *(v8float *)(v8 + m5*v15+v22) = v23;
//CHECK-NEXT:   }
//CHECK-NEXT: }


// CHECK-LABEL: void conv2d_1(float * restrict v4, size_t m1, float * restrict v5, size_t m2, float * restrict v6, size_t m3) {
func.func @conv2d_1(%arg0: memref<?x256xf32>, %arg1: memref<?xf32>, %arg2: memref<?x256xf32>) {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?x256xf32>
  %1 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<?xf32>, vector<8xf32>
  %2 = aievec.upd %arg1[%c8] {index = 0 : i8, offset = 0 : i32} : memref<?xf32>, vector<8xf32>
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
      %5 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x256xf32>, vector<8xf32>
      %6 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x256xf32>, vector<16xf32>
      %7 = aievec.mac %6, %1, %5 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      %c1_4 = arith.constant 1 : index
      %8 = arith.addi %arg4, %c1_4 : index
      %9 = aievec.upd %arg0[%arg3, %8], %6 {index = 1 : i8, offset = 224 : i32} : memref<?x256xf32>, vector<16xf32>
      %10 = aievec.mac %9, %1, %7 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      %11 = aievec.mac %9, %1, %10 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      %12 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x256xf32>, vector<16xf32>
      %13 = aievec.mac %12, %1, %11 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      %14 = aievec.upd %arg0[%3, %8], %12 {index = 1 : i8, offset = 224 : i32} : memref<?x256xf32>, vector<16xf32>
      %15 = aievec.mac %14, %1, %13 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      %16 = aievec.mac %14, %1, %15 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      %17 = aievec.upd %arg0[%4, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x256xf32>, vector<16xf32>
      %18 = aievec.mac %17, %1, %16 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      %19 = aievec.upd %arg0[%4, %8], %17 {index = 1 : i8, offset = 224 : i32} : memref<?x256xf32>, vector<16xf32>
      %20 = aievec.mac %19, %1, %18 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      %21 = aievec.mac %19, %2, %20 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      vector.transfer_write %21, %arg2[%arg3, %arg4] : vector<8xf32>, memref<?x256xf32>
    }
  }
  return
}

//CHECK-NEXT: size_t v7 = 8;
//CHECK-NEXT: size_t v8 = 0;
//CHECK-NEXT: v8float v9 = *(v8float *)(v5 + v8);
//CHECK-NEXT: v8float v10 = *(v8float *)(v5 + v7);
//CHECK-NEXT: size_t v11 = 0;
//CHECK-NEXT: size_t v12 = 1;
//CHECK-NEXT: for (size_t v13 = v11; v13 < m1; v13 += v12)
//CHECK-NEXT: chess_prepare_for_pipelining
//CHECK-NEXT: {
//CHECK-NEXT:   size_t v14 = 1;
//CHECK-NEXT:   size_t v15 = v13 + v14;
//CHECK-NEXT:   size_t v16 = 2;
//CHECK-NEXT:   size_t v17 = v13 + v16;
//CHECK-NEXT:   size_t v18 = 0;
//CHECK-NEXT:   size_t v19 = 256;
//CHECK-NEXT:   size_t v20 = 8;
//CHECK-NEXT:   for (size_t v21 = v18; v21 < v19; v21 += v20)
//CHECK-NEXT:   chess_prepare_for_pipelining
//CHECK-NEXT:   chess_loop_range(32, 32)
//CHECK-NEXT:   {
//CHECK-NEXT:     v8float v22 = *(v8float *)(v6 + 256*v13+v21);
//CHECK-NEXT:     v16float v23;
//CHECK-NEXT:     float * restrict r_v23_v4 = v4;
//CHECK-NEXT:     v23 = upd_w(v23, 0, *(v8float *)(r_v23_v4 + 256*v13+v21));
//CHECK-NEXT:     v22 = fpmac(v22, v23, 0, 0x76543210, v9, 0, 0x00000000);
//CHECK-NEXT:     size_t v24 = 1;
//CHECK-NEXT:     size_t v25 = v21 + v24;
//CHECK-NEXT:     v23 = upd_w(v23, 1, *(v8float *)(r_v23_v4 + 256*v13+v25 + 7));
//CHECK-NEXT:     v22 = fpmac(v22, v23, 1, 0x76543210, v9, 1, 0x00000000);
//CHECK-NEXT:     v22 = fpmac(v22, v23, 2, 0x76543210, v9, 2, 0x00000000);
//CHECK-NEXT:     v16float v26;
//CHECK-NEXT:     float * restrict r_v26_v4 = v4;
//CHECK-NEXT:     v26 = upd_w(v26, 0, *(v8float *)(r_v26_v4 + 256*v15+v21));
//CHECK-NEXT:     v22 = fpmac(v22, v26, 0, 0x76543210, v9, 3, 0x00000000);
//CHECK-NEXT:     v26 = upd_w(v26, 1, *(v8float *)(r_v26_v4 + 256*v15+v25 + 7));
//CHECK-NEXT:     v22 = fpmac(v22, v26, 1, 0x76543210, v9, 4, 0x00000000);
//CHECK-NEXT:     v22 = fpmac(v22, v26, 2, 0x76543210, v9, 5, 0x00000000);
//CHECK-NEXT:     v16float v27;
//CHECK-NEXT:     float * restrict r_v27_v4 = v4;
//CHECK-NEXT:     v27 = upd_w(v27, 0, *(v8float *)(r_v27_v4 + 256*v17+v21));
//CHECK-NEXT:     v22 = fpmac(v22, v27, 0, 0x76543210, v9, 6, 0x00000000);
//CHECK-NEXT:     v27 = upd_w(v27, 1, *(v8float *)(r_v27_v4 + 256*v17+v25 + 7));
//CHECK-NEXT:     v22 = fpmac(v22, v27, 1, 0x76543210, v9, 7, 0x00000000);
//CHECK-NEXT:     v22 = fpmac(v22, v27, 2, 0x76543210, v10, 0, 0x00000000);
//CHECK-NEXT:     *(v8float *)(v6 + 256*v13+v21) = v22;
//CHECK-NEXT:   }
//CHECK-NEXT: }
