//===- translate_conv2d_uij_i16.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// RUN: aie-translate --aievec-to-cpp %s -split-input-file | FileCheck %s

// CHECK-LABEL: void conv2d(int16_t * restrict v1, int16_t * restrict v2, int16_t * restrict v3) {
module  {
  func.func @conv2d(%arg0: memref<2048x2048xi16>, %arg1: memref<12xi16>, %arg2: memref<2046x2046xi16>) {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = aievec_aie1.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<12xi16>, vector<16xi16>
    %c0_0 = arith.constant 0 : index
    %c2046 = arith.constant 2046 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0_0 to %c2046 step %c1 {
      %c1_1 = arith.constant 1 : index
      %1 = arith.addi %arg3, %c1_1 : index
      %c2 = arith.constant 2 : index
      %2 = arith.addi %arg3, %c2 : index
      %c0_2 = arith.constant 0 : index
      %c2046_3 = arith.constant 2046 : index
      %c16 = arith.constant 16 : index
      scf.for %arg4 = %c0_2 to %c2046_3 step %c16 {
        %3 = aievec_aie1.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2046x2046xi16>, vector<16xi16>
        %4 = aievec_aie1.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xi16>, vector<32xi16>
        %5 = aievec.ups %3 {shift = 0 : i8} : vector<16xi16>, vector<16xi48>
        %6 = aievec_aie1.mac %4, %0, %5 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "0", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %c1_4 = arith.constant 1 : index
        %7 = arith.addi %arg4, %c1_4 : index
        %8 = aievec_aie1.upd %arg0[%arg3, %7], %4 {index = 1 : i8, offset = 240 : i32} : memref<2048x2048xi16>, vector<32xi16>
        %9 = aievec_aie1.mac %8, %0, %6 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "2", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %10 = aievec_aie1.upd %arg0[%1, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xi16>, vector<32xi16>
        %11 = aievec_aie1.mac %10, %0, %9 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "4", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %12 = aievec_aie1.upd %arg0[%1, %7], %10 {index = 1 : i8, offset = 240 : i32} : memref<2048x2048xi16>, vector<32xi16>
        %13 = aievec_aie1.mac %12, %0, %11 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "6", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %14 = aievec_aie1.upd %arg0[%2, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xi16>, vector<32xi16>
        %15 = aievec_aie1.mac %14, %0, %13 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "8", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %16 = aievec_aie1.upd %arg0[%2, %7], %14 {index = 1 : i8, offset = 240 : i32} : memref<2048x2048xi16>, vector<32xi16>
        %17 = aievec_aie1.mac %16, %0, %15 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "10", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %18 = aievec.srs %17, %c0_i32 : vector<16xi48>, i32, vector<16xi16>
        vector.transfer_write %18, %arg2[%arg3, %arg4] : vector<16xi16>, memref<2046x2046xi16>
      }
    }
    return
  }
}

//CHECK-NEXT:  size_t v4 = 0;
//CHECK-NEXT:  int32_t v5 = 0;
//CHECK-NEXT:  v16int16 v6 = *(v16int16 *)(v2 + v4);
//CHECK-NEXT:  size_t v7 = 0;
//CHECK-NEXT:  size_t v8 = 2046;
//CHECK-NEXT:  size_t v9 = 1;
//CHECK-NEXT:  for (size_t v10 = v7; v10 < v8; v10 += v9)
//CHECK-NEXT:  chess_prepare_for_pipelining
//CHECK-NEXT:  chess_loop_range(2046, 2046)
//CHECK-NEXT:  {
//CHECK-NEXT:    size_t v11 = 1;
//CHECK-NEXT:    size_t v12 = v10 + v11;
//CHECK-NEXT:    size_t v13 = 2;
//CHECK-NEXT:    size_t v14 = v10 + v13;
//CHECK-NEXT:    size_t v15 = 0;
//CHECK-NEXT:    size_t v16 = 2046;
//CHECK-NEXT:    size_t v17 = 16;
//CHECK-NEXT:    for (size_t v18 = v15; v18 < v16; v18 += v17)
//CHECK-NEXT:    chess_prepare_for_pipelining
//CHECK-NEXT:    chess_loop_range(127, 128)
//CHECK-NEXT:    {
//CHECK-NEXT:      v16int16 v19 = *(v16int16 *)(v3 + 2046*v10+v18);
//CHECK-NEXT:      v32int16 v20;
//CHECK-NEXT:      int16_t * restrict r_v20_v1 = v1;
//CHECK-NEXT:      v20 = upd_w(v20, 0, *(v16int16 *)(r_v20_v1 + 2048*v10+v18));
//CHECK-NEXT:      v16acc48 v21 = ups(v19, 0);
//CHECK-NEXT:      v21 = mac16(v21, v20, 0, 0x03020100, 0x07060504, 0x2110, v6, 0, 0, 0, 1);
//CHECK-NEXT:      size_t v22 = 1;
//CHECK-NEXT:      size_t v23 = v18 + v22;
//CHECK-NEXT:      v20 = upd_w(v20, 1, *(v16int16 *)(r_v20_v1 + 2048*v10+v23 + 15));
//CHECK-NEXT:      v21 = mac16(v21, v20, 2, 0x03020100, 0x07060504, 0x2110, v6, 2, 0, 0, 1);
//CHECK-NEXT:      v32int16 v24;
//CHECK-NEXT:      int16_t * restrict r_v24_v1 = v1;
//CHECK-NEXT:      v24 = upd_w(v24, 0, *(v16int16 *)(r_v24_v1 + 2048*v12+v18));
//CHECK-NEXT:      v21 = mac16(v21, v24, 0, 0x03020100, 0x07060504, 0x2110, v6, 4, 0, 0, 1);
//CHECK-NEXT:      v24 = upd_w(v24, 1, *(v16int16 *)(r_v24_v1 + 2048*v12+v23 + 15));
//CHECK-NEXT:      v21 = mac16(v21, v24, 2, 0x03020100, 0x07060504, 0x2110, v6, 6, 0, 0, 1);
//CHECK-NEXT:      v32int16 v25;
//CHECK-NEXT:      int16_t * restrict r_v25_v1 = v1;
//CHECK-NEXT:      v25 = upd_w(v25, 0, *(v16int16 *)(r_v25_v1 + 2048*v14+v18));
//CHECK-NEXT:      v21 = mac16(v21, v25, 0, 0x03020100, 0x07060504, 0x2110, v6, 8, 0, 0, 1);
//CHECK-NEXT:      v25 = upd_w(v25, 1, *(v16int16 *)(r_v25_v1 + 2048*v14+v23 + 15));
//CHECK-NEXT:      v21 = mac16(v21, v25, 2, 0x03020100, 0x07060504, 0x2110, v6, 10, 0, 0, 1);
//CHECK-NEXT:      v16int16 v26 = srs(v21, v5);
//CHECK-NEXT:      *(v16int16 *)(v3 + 2046*v10+v18) = v26;

// CHECK-LABEL: void conv2d(int16_t * restrict v6, size_t m1, size_t m2, int16_t * restrict v7, size_t m3, int16_t * restrict v8, size_t m4, size_t m5, size_t v9, size_t v10) {
module  {
  func.func @conv2d(%arg0: memref<?x?xi16>, %arg1: memref<?xi16>, %arg2: memref<?x?xi16>, %arg3: index, %arg4: index) {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = aievec_aie1.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<?xi16>, vector<16xi16>
    %c0_0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg5 = %c0_0 to %arg3 step %c1 {
      %c1_1 = arith.constant 1 : index
      %1 = arith.addi %arg5, %c1_1 : index
      %c2 = arith.constant 2 : index
      %2 = arith.addi %arg5, %c2 : index
      %c0_2 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      scf.for %arg6 = %c0_2 to %arg4 step %c16 {
        %3 = aievec_aie1.upd %arg2[%arg5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<16xi16>
        %4 = aievec_aie1.upd %arg0[%arg5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<32xi16>
        %5 = aievec.ups %3 {shift = 0 : i8} : vector<16xi16>, vector<16xi48>
        %6 = aievec_aie1.mac %4, %0, %5 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "0", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %c1_3 = arith.constant 1 : index
        %7 = arith.addi %arg6, %c1_3 : index
        %8 = aievec_aie1.upd %arg0[%arg5, %7], %4 {index = 1 : i8, offset = 240 : i32} : memref<?x?xi16>, vector<32xi16>
        %9 = aievec_aie1.mac %8, %0, %6 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "2", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %10 = aievec_aie1.upd %arg0[%1, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<32xi16>
        %11 = aievec_aie1.mac %10, %0, %9 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "4", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %12 = aievec_aie1.upd %arg0[%1, %7], %10 {index = 1 : i8, offset = 240 : i32} : memref<?x?xi16>, vector<32xi16>
        %13 = aievec_aie1.mac %12, %0, %11 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "6", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %14 = aievec_aie1.upd %arg0[%2, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<32xi16>
        %15 = aievec_aie1.mac %14, %0, %13 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "8", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %16 = aievec_aie1.upd %arg0[%2, %7], %14 {index = 1 : i8, offset = 240 : i32} : memref<?x?xi16>, vector<32xi16>
        %17 = aievec_aie1.mac %16, %0, %15 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "10", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %18 = aievec.srs %17, %c0_i32 : vector<16xi48>, i32, vector<16xi16>
        vector.transfer_write %18, %arg2[%arg5, %arg6] : vector<16xi16>, memref<?x?xi16>
      }
    }
    return
  }
}

//CHECK-NEXT:  size_t v11 = 0;
//CHECK-NEXT:  int32_t v12 = 0;
//CHECK-NEXT:  v16int16 v13 = *(v16int16 *)(v7 + v11);
//CHECK-NEXT:  size_t v14 = 0;
//CHECK-NEXT:  size_t v15 = 1;
//CHECK-NEXT:  for (size_t v16 = v14; v16 < v9; v16 += v15)
//CHECK-NEXT:  chess_prepare_for_pipelining
//CHECK-NEXT:  {
//CHECK-NEXT:    size_t v17 = 1;
//CHECK-NEXT:    size_t v18 = v16 + v17;
//CHECK-NEXT:    size_t v19 = 2;
//CHECK-NEXT:    size_t v20 = v16 + v19;
//CHECK-NEXT:    size_t v21 = 0;
//CHECK-NEXT:    size_t v22 = 16;
//CHECK-NEXT:    for (size_t v23 = v21; v23 < v10; v23 += v22)
//CHECK-NEXT:    chess_prepare_for_pipelining
//CHECK-NEXT:    {
//CHECK-NEXT:      v16int16 v24 = *(v16int16 *)(v8 + m5*v16+v23);
//CHECK-NEXT:      v32int16 v25;
//CHECK-NEXT:      int16_t * restrict r_v25_v6 = v6;
//CHECK-NEXT:      v25 = upd_w(v25, 0, *(v16int16 *)(r_v25_v6 + m2*v16+v23));
//CHECK-NEXT:      v16acc48 v26 = ups(v24, 0);
//CHECK-NEXT:      v26 = mac16(v26, v25, 0, 0x03020100, 0x07060504, 0x2110, v13, 0, 0, 0, 1);
//CHECK-NEXT:      size_t v27 = 1;
//CHECK-NEXT:      size_t v28 = v23 + v27;
//CHECK-NEXT:      v25 = upd_w(v25, 1, *(v16int16 *)(r_v25_v6 + m2*v16+v28 + 15));
//CHECK-NEXT:      v26 = mac16(v26, v25, 2, 0x03020100, 0x07060504, 0x2110, v13, 2, 0, 0, 1);
//CHECK-NEXT:      v32int16 v29;
//CHECK-NEXT:      int16_t * restrict r_v29_v6 = v6;
//CHECK-NEXT:      v29 = upd_w(v29, 0, *(v16int16 *)(r_v29_v6 + m2*v18+v23));
//CHECK-NEXT:      v26 = mac16(v26, v29, 0, 0x03020100, 0x07060504, 0x2110, v13, 4, 0, 0, 1);
//CHECK-NEXT:      v29 = upd_w(v29, 1, *(v16int16 *)(r_v29_v6 + m2*v18+v28 + 15));
//CHECK-NEXT:      v26 = mac16(v26, v29, 2, 0x03020100, 0x07060504, 0x2110, v13, 6, 0, 0, 1);
//CHECK-NEXT:      v32int16 v30;
//CHECK-NEXT:      int16_t * restrict r_v30_v6 = v6;
//CHECK-NEXT:      v30 = upd_w(v30, 0, *(v16int16 *)(r_v30_v6 + m2*v20+v23));
//CHECK-NEXT:      v26 = mac16(v26, v30, 0, 0x03020100, 0x07060504, 0x2110, v13, 8, 0, 0, 1);
//CHECK-NEXT:      v30 = upd_w(v30, 1, *(v16int16 *)(r_v30_v6 + m2*v20+v28 + 15));
//CHECK-NEXT:      v26 = mac16(v26, v30, 2, 0x03020100, 0x07060504, 0x2110, v13, 10, 0, 0, 1);
//CHECK-NEXT:      v16int16 v31 = srs(v26, v12);
//CHECK-NEXT:      *(v16int16 *)(v8 + m5*v16+v23) = v31;

// CHECK-LABEL: void conv2d(int16_t * restrict v6, size_t m1, size_t m2, int16_t * restrict v7, size_t m3, int16_t * restrict v8, size_t m4, size_t m5) {
module  {
  func.func @conv2d(%arg0: memref<?x?xi16>, %arg1: memref<?xi16>, %arg2: memref<?x?xi16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.dim %arg0, %c0 : memref<?x?xi16>
    %1 = memref.dim %arg0, %c1 : memref<?x?xi16>
    %2 = aievec_aie1.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<?xi16>, vector<16xi16>
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    scf.for %arg3 = %c0_0 to %0 step %c1_1 {
      %c1_2 = arith.constant 1 : index
      %3 = arith.addi %arg3, %c1_2 : index
      %c2 = arith.constant 2 : index
      %4 = arith.addi %arg3, %c2 : index
      %c0_3 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      scf.for %arg4 = %c0_3 to %1 step %c16 {
        %5 = aievec_aie1.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<16xi16>
        %6 = aievec_aie1.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<32xi16>
        %7 = aievec_aie1.upd %arg0[%arg3, %arg4], %6 {index = 1 : i8, offset = 256 : i32} : memref<?x?xi16>, vector<32xi16>
        %8 = aievec.ups %5 {shift = 0 : i8} : vector<16xi16>, vector<16xi48>
        %9 = aievec_aie1.mac %7, %2, %8 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "0", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %10 = aievec_aie1.mac %7, %2, %9 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "2", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %11 = aievec_aie1.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<32xi16>
        %12 = aievec_aie1.upd %arg0[%3, %arg4], %11 {index = 1 : i8, offset = 256 : i32} : memref<?x?xi16>, vector<32xi16>
        %13 = aievec_aie1.mac %12, %2, %10 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "4", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %14 = aievec_aie1.mac %12, %2, %13 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "6", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %15 = aievec_aie1.upd %arg0[%4, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<32xi16>
        %16 = aievec_aie1.upd %arg0[%4, %arg4], %15 {index = 1 : i8, offset = 256 : i32} : memref<?x?xi16>, vector<32xi16>
        %17 = aievec_aie1.mac %16, %2, %14 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "8", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %18 = aievec_aie1.mac %16, %2, %17 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "10", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %19 = aievec.srs %18, %c0_i32 : vector<16xi48>, i32, vector<16xi16>
        vector.transfer_write %19, %arg2[%arg3, %arg4] : vector<16xi16>, memref<?x?xi16>
      }
    }
    return
  }
}

//CHECK-NEXT:  size_t v9 = 0;
//CHECK-NEXT:  int32_t v10 = 0;
//CHECK-NEXT:  v16int16 v11 = *(v16int16 *)(v7 + v9);
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
//CHECK-NEXT:    size_t v20 = 16;
//CHECK-NEXT:    for (size_t v21 = v19; v21 < m2; v21 += v20)
//CHECK-NEXT:    chess_prepare_for_pipelining
//CHECK-NEXT:    {
//CHECK-NEXT:      v16int16 v22 = *(v16int16 *)(v8 + m5*v14+v21);
//CHECK-NEXT:      v32int16 v23;
//CHECK-NEXT:      int16_t * restrict r_v23_v6 = v6;
//CHECK-NEXT:      v23 = upd_w(v23, 0, *(v16int16 *)(r_v23_v6 + m2*v14+v21));
//CHECK-NEXT:      v23 = upd_w(v23, 1, *(v16int16 *)(r_v23_v6 + m2*v14+v21 + 16));
//CHECK-NEXT:      v16acc48 v24 = ups(v22, 0);
//CHECK-NEXT:      v24 = mac16(v24, v23, 0, 0x03020100, 0x07060504, 0x2110, v11, 0, 0, 0, 1);
//CHECK-NEXT:      v24 = mac16(v24, v23, 2, 0x03020100, 0x07060504, 0x2110, v11, 2, 0, 0, 1);
//CHECK-NEXT:      v32int16 v25;
//CHECK-NEXT:      int16_t * restrict r_v25_v6 = v6;
//CHECK-NEXT:      v25 = upd_w(v25, 0, *(v16int16 *)(r_v25_v6 + m2*v16+v21));
//CHECK-NEXT:      v25 = upd_w(v25, 1, *(v16int16 *)(r_v25_v6 + m2*v16+v21 + 16));
//CHECK-NEXT:      v24 = mac16(v24, v25, 0, 0x03020100, 0x07060504, 0x2110, v11, 4, 0, 0, 1);
//CHECK-NEXT:      v24 = mac16(v24, v25, 2, 0x03020100, 0x07060504, 0x2110, v11, 6, 0, 0, 1);
//CHECK-NEXT:      v32int16 v26;
//CHECK-NEXT:      int16_t * restrict r_v26_v6 = v6;
//CHECK-NEXT:      v26 = upd_w(v26, 0, *(v16int16 *)(r_v26_v6 + m2*v18+v21));
//CHECK-NEXT:      v26 = upd_w(v26, 1, *(v16int16 *)(r_v26_v6 + m2*v18+v21 + 16));
//CHECK-NEXT:      v24 = mac16(v24, v26, 0, 0x03020100, 0x07060504, 0x2110, v11, 8, 0, 0, 1);
//CHECK-NEXT:      v24 = mac16(v24, v26, 2, 0x03020100, 0x07060504, 0x2110, v11, 10, 0, 0, 1);
//CHECK-NEXT:      v16int16 v27 = srs(v24, v10);
//CHECK-NEXT:      *(v16int16 *)(v8 + m5*v14+v21) = v27;

// CHECK-LABEL: void conv2d(int16_t * restrict v4, size_t m1, int16_t * restrict v5, size_t m2, int16_t * restrict v6, size_t m3) {
module  {
  func.func @conv2d(%arg0: memref<?x256xi16>, %arg1: memref<?xi16>, %arg2: memref<?x256xi16>) {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.dim %arg0, %c0 : memref<?x256xi16>
    %1 = aievec_aie1.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<?xi16>, vector<16xi16>
    %c0_0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0_0 to %0 step %c1 {
      %c1_1 = arith.constant 1 : index
      %2 = arith.addi %arg3, %c1_1 : index
      %c2 = arith.constant 2 : index
      %3 = arith.addi %arg3, %c2 : index
      %c0_2 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      scf.for %arg4 = %c0_2 to %c256 step %c16 {
        %4 = aievec_aie1.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi16>, vector<16xi16>
        %5 = aievec_aie1.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi16>, vector<32xi16>
        %6 = aievec_aie1.upd %arg0[%arg3, %arg4], %5 {index = 1 : i8, offset = 256 : i32} : memref<?x256xi16>, vector<32xi16>
        %7 = aievec.ups %4 {shift = 0 : i8} : vector<16xi16>, vector<16xi48>
        %8 = aievec_aie1.mac %6, %1, %7 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "0", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %9 = aievec_aie1.mac %6, %1, %8 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "2", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %10 = aievec_aie1.upd %arg0[%2, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi16>, vector<32xi16>
        %11 = aievec_aie1.upd %arg0[%2, %arg4], %10 {index = 1 : i8, offset = 256 : i32} : memref<?x256xi16>, vector<32xi16>
        %12 = aievec_aie1.mac %11, %1, %9 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "4", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %13 = aievec_aie1.mac %11, %1, %12 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "6", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %14 = aievec_aie1.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi16>, vector<32xi16>
        %15 = aievec_aie1.upd %arg0[%3, %arg4], %14 {index = 1 : i8, offset = 256 : i32} : memref<?x256xi16>, vector<32xi16>
        %16 = aievec_aie1.mac %15, %1, %13 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "8", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %17 = aievec_aie1.mac %15, %1, %16 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "10", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %18 = aievec.srs %17, %c0_i32 : vector<16xi48>, i32, vector<16xi16>
        vector.transfer_write %18, %arg2[%arg3, %arg4] : vector<16xi16>, memref<?x256xi16>
      }
    }
    return
  }
}

//CHECK-NEXT:  size_t v7 = 0;
//CHECK-NEXT:  int32_t v8 = 0;
//CHECK-NEXT:  v16int16 v9 = *(v16int16 *)(v5 + v7);
//CHECK-NEXT:  size_t v10 = 0;
//CHECK-NEXT:  size_t v11 = 1;
//CHECK-NEXT:  for (size_t v12 = v10; v12 < m1; v12 += v11)
//CHECK-NEXT:  chess_prepare_for_pipelining
//CHECK-NEXT:  {
//CHECK-NEXT:    size_t v13 = 1;
//CHECK-NEXT:    size_t v14 = v12 + v13;
//CHECK-NEXT:    size_t v15 = 2;
//CHECK-NEXT:    size_t v16 = v12 + v15;
//CHECK-NEXT:    size_t v17 = 0;
//CHECK-NEXT:    size_t v18 = 256;
//CHECK-NEXT:    size_t v19 = 16;
//CHECK-NEXT:    for (size_t v20 = v17; v20 < v18; v20 += v19)
//CHECK-NEXT:    chess_prepare_for_pipelining
//CHECK-NEXT:    chess_loop_range(16, 16)
//CHECK-NEXT:    {
//CHECK-NEXT:      v16int16 v21 = *(v16int16 *)(v6 + 256*v12+v20);
//CHECK-NEXT:      v32int16 v22;
//CHECK-NEXT:      int16_t * restrict r_v22_v4 = v4;
//CHECK-NEXT:      v22 = upd_w(v22, 0, *(v16int16 *)(r_v22_v4 + 256*v12+v20));
//CHECK-NEXT:      v22 = upd_w(v22, 1, *(v16int16 *)(r_v22_v4 + 256*v12+v20 + 16));
//CHECK-NEXT:      v16acc48 v23 = ups(v21, 0);
//CHECK-NEXT:      v23 = mac16(v23, v22, 0, 0x03020100, 0x07060504, 0x2110, v9, 0, 0, 0, 1);
//CHECK-NEXT:      v23 = mac16(v23, v22, 2, 0x03020100, 0x07060504, 0x2110, v9, 2, 0, 0, 1);
//CHECK-NEXT:      v32int16 v24;
//CHECK-NEXT:      int16_t * restrict r_v24_v4 = v4;
//CHECK-NEXT:      v24 = upd_w(v24, 0, *(v16int16 *)(r_v24_v4 + 256*v14+v20));
//CHECK-NEXT:      v24 = upd_w(v24, 1, *(v16int16 *)(r_v24_v4 + 256*v14+v20 + 16));
//CHECK-NEXT:      v23 = mac16(v23, v24, 0, 0x03020100, 0x07060504, 0x2110, v9, 4, 0, 0, 1);
//CHECK-NEXT:      v23 = mac16(v23, v24, 2, 0x03020100, 0x07060504, 0x2110, v9, 6, 0, 0, 1);
//CHECK-NEXT:      v32int16 v25;
//CHECK-NEXT:      int16_t * restrict r_v25_v4 = v4;
//CHECK-NEXT:      v25 = upd_w(v25, 0, *(v16int16 *)(r_v25_v4 + 256*v16+v20));
//CHECK-NEXT:      v25 = upd_w(v25, 1, *(v16int16 *)(r_v25_v4 + 256*v16+v20 + 16));
//CHECK-NEXT:      v23 = mac16(v23, v25, 0, 0x03020100, 0x07060504, 0x2110, v9, 8, 0, 0, 1);
//CHECK-NEXT:      v23 = mac16(v23, v25, 2, 0x03020100, 0x07060504, 0x2110, v9, 10, 0, 0, 1);
//CHECK-NEXT:      v16int16 v26 = srs(v23, v8);
//CHECK-NEXT:      *(v16int16 *)(v6 + 256*v12+v20) = v26;
