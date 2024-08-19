//===- translate_conv2d_uij_i8.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// RUN: aie-translate --aievec-to-cpp %s -split-input-file | FileCheck %s

// CHECK-LABEL: void conv2d_0(int8_t * restrict v1, int8_t * restrict v2, int8_t * restrict v3) {
func.func @conv2d_0(%arg0: memref<18x288xi8>, %arg1: memref<48xi8>, %arg2: memref<16x256xi8>) {
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : i32
  %0 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<48xi8>, vector<64xi8>
  %1 = aievec.upd %arg1[%c32], %0 {index = 1 : i8, offset = 0 : i32} : memref<48xi8>, vector<64xi8>
  %c0_0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0_0 to %c16 step %c1 {
    %c1_1 = arith.constant 1 : index
    %2 = arith.addi %arg3, %c1_1 : index
    %c2 = arith.constant 2 : index
    %3 = arith.addi %arg3, %c2 : index
    %c0_2 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c16_3 = arith.constant 16 : index
    scf.for %arg4 = %c0_2 to %c256 step %c16_3 {
      %4 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<16x256xi8>, vector<16xi8>
      %5 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi8>, vector<32xi8>
      %6 = aievec.ups %4 {shift = 10 : i8} : vector<16xi8>, vector<16xi48>
      %7 = aievec_aie1.mac %0, %5, %6 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "0", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
      %8 = aievec_aie1.mac %0, %5, %6 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "0", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "4", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
      %9 = aievec.upd %arg0[%2, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi8>, vector<32xi8>
      %10 = aievec_aie1.mac %0, %9, %7 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "16", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
      %11 = aievec_aie1.mac %0, %9, %8 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "16", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "4", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
      %12 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi8>, vector<32xi8>
      %13 = aievec_aie1.mac %1, %12, %10 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "32", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
      %14 = aievec_aie1.mac %1, %12, %11 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "32", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "4", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
      %15 = aievec.srs %13, %c10 : vector<16xi48>, i32, vector<16xi16>
      %16 = aievec.srs %14, %c10 : vector<16xi48>, i32, vector<16xi16>
      %17 = aievec.concat %15, %16 : vector<16xi16>, vector<32xi16>
      %18 = aievec_aie1.select %17 {select = "0xcccccccc", xoffsets = "0x0c080400", xoffsets_hi = "0x0", xsquare = "0x1010", xstart = "0", yoffsets = "0x0c080400", yoffsets_hi = "0x0", ysquare = "0x1010", ystart = "4"} : vector<32xi16>, vector<32xi16>
      %19 = aievec_aie1.ext %18 {index = 0 : i8} : vector<32xi16>, vector<16xi16>
      %20 = aievec.pack %19 : vector<16xi16>, vector<16xi8>
      vector.transfer_write %20, %arg2[%arg3, %arg4] : vector<16xi8>, memref<16x256xi8>
    }
  }
  return
}

//CHECK-NEXT:  size_t v4 = 32;
//CHECK-NEXT:  size_t v5 = 0;
//CHECK-NEXT:  int32_t v6 = 10;
//CHECK-NEXT:  v64int8 v7;
//CHECK-NEXT:  int8_t * restrict r_v7_v2 = v2;
//CHECK-NEXT:  v7 = upd_w(v7, 0, *(v32int8 *)(r_v7_v2 + v5));
//CHECK-NEXT:  v7 = upd_w(v7, 1, *(v32int8 *)(r_v7_v2 + v4));
//CHECK-NEXT:  size_t v8 = 0;
//CHECK-NEXT:  size_t v9 = 16;
//CHECK-NEXT:  size_t v10 = 1;
//CHECK-NEXT:  for (size_t v11 = v8; v11 < v9; v11 += v10)
//CHECK-NEXT:  chess_prepare_for_pipelining
//CHECK-NEXT:  chess_loop_range(16, 16)
//CHECK-NEXT:  {
//CHECK-NEXT:    size_t v12 = 1;
//CHECK-NEXT:    size_t v13 = v11 + v12;
//CHECK-NEXT:    size_t v14 = 2;
//CHECK-NEXT:    size_t v15 = v11 + v14;
//CHECK-NEXT:    size_t v16 = 0;
//CHECK-NEXT:    size_t v17 = 256;
//CHECK-NEXT:    size_t v18 = 16;
//CHECK-NEXT:    for (size_t v19 = v16; v19 < v17; v19 += v18)
//CHECK-NEXT:    chess_prepare_for_pipelining
//CHECK-NEXT:    chess_loop_range(16, 16)
//CHECK-NEXT:    {
//CHECK-NEXT:      v16int8 v20 = *(v16int8 *)(v3 + 256*v11+v19);
//CHECK-NEXT:      v32int8 v21 = *(v32int8 *)(v1 + 288*v11+v19);
//CHECK-NEXT:      v16acc48 v22 = ups(v20, 10);
//CHECK-NEXT:      v22 = mac16(v22, v7, 0, 0x00000000, 4, 0x1010, v21, 0, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v22 = mac16(v22, v7, 0, 0x00000000, 4, 0x1010, v21, 4, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v32int8 v23 = *(v32int8 *)(v1 + 288*v13+v19);
//CHECK-NEXT:      v22 = mac16(v22, v7, 16, 0x00000000, 4, 0x1010, v23, 0, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v22 = mac16(v22, v7, 16, 0x00000000, 4, 0x1010, v23, 4, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v32int8 v24 = *(v32int8 *)(v1 + 288*v15+v19);
//CHECK-NEXT:      v22 = mac16(v22, v7, 32, 0x00000000, 4, 0x1010, v24, 0, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v22 = mac16(v22, v7, 32, 0x00000000, 4, 0x1010, v24, 4, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v16int16 v25 = srs(v22, v6);
//CHECK-NEXT:      v16int16 v26 = srs(v22, v6);
//CHECK-NEXT:     v32int16 v27 = concat(v25, v26);
//CHECK-NEXT:      v32int16 v28 = select32(0xcccccccc, v27, 0, 0x0c080400, 0x0, 0x1010, 4, 0x0c080400, 0x0, 0x1010);
//CHECK-NEXT:      v16int16 v29 = ext_w(v28, 0);
//CHECK-NEXT:      v16int8 v30 = pack(v29);
//CHECK-NEXT:      *(v16int8 *)(v3 + 256*v11+v19) = v30;

// CHECK-LABEL: void conv2d_1(int8_t * restrict v1, int8_t * restrict v2, int8_t * restrict v3) {
func.func @conv2d_1(%arg0: memref<18x288xi8>, %arg1: memref<48xi8>, %arg2: memref<16x256xi8>) {
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : i32
  %0 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<48xi8>, vector<64xi8>
  %1 = aievec.upd %arg1[%c32], %0 {index = 1 : i8, offset = 0 : i32} : memref<48xi8>, vector<64xi8>
  %c0_0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0_0 to %c16 step %c1 {
    %c1_1 = arith.constant 1 : index
    %2 = arith.addi %arg3, %c1_1 : index
    %c2 = arith.constant 2 : index
    %3 = arith.addi %arg3, %c2 : index
    %c0_2 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c16_3 = arith.constant 16 : index
    scf.for %arg4 = %c0_2 to %c256 step %c16_3 {
      %4 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi8>, vector<32xi8>
      %5 = aievec_aie1.mul %0, %4 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "0", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
      %6 = aievec_aie1.mul %0, %4 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "0", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "4", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
      %7 = aievec.upd %arg0[%2, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi8>, vector<32xi8>
      %8 = aievec_aie1.mac %0, %7, %5 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "16", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
      %9 = aievec_aie1.mac %0, %7, %6 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "16", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "4", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
      %10 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi8>, vector<32xi8>
      %11 = aievec_aie1.mac %1, %10, %8 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "32", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
      %12 = aievec_aie1.mac %1, %10, %9 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "32", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "4", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
      %13 = aievec.srs %11, %c10 : vector<16xi48>, i32, vector<16xi16>
      %14 = aievec.srs %12, %c10 : vector<16xi48>, i32, vector<16xi16>
      %15 = aievec.concat %13, %14 : vector<16xi16>, vector<32xi16>
      %16 = aievec_aie1.select %15 {select = "0xcccccccc", xoffsets = "0x0c080400", xoffsets_hi = "0x0", xsquare = "0x1010", xstart = "0", yoffsets = "0x0c080400", yoffsets_hi = "0x0", ysquare = "0x1010", ystart = "4"} : vector<32xi16>, vector<32xi16>
      %17 = aievec_aie1.ext %16 {index = 0 : i8} : vector<32xi16>, vector<16xi16>
      %18 = aievec.pack %17 : vector<16xi16>, vector<16xi8>
      vector.transfer_write %18, %arg2[%arg3, %arg4] : vector<16xi8>, memref<16x256xi8>
    }
  }
  return
}

//CHECK-NEXT:  size_t v4 = 32;
//CHECK-NEXT:  size_t v5 = 0;
//CHECK-NEXT:  int32_t v6 = 10;
//CHECK-NEXT:  v64int8 v7;
//CHECK-NEXT:  int8_t * restrict r_v7_v2 = v2;
//CHECK-NEXT:  v7 = upd_w(v7, 0, *(v32int8 *)(r_v7_v2 + v5));
//CHECK-NEXT:  v7 = upd_w(v7, 1, *(v32int8 *)(r_v7_v2 + v4));
//CHECK-NEXT:  size_t v8 = 0;
//CHECK-NEXT:  size_t v9 = 16;
//CHECK-NEXT:  size_t v10 = 1;
//CHECK-NEXT:  for (size_t v11 = v8; v11 < v9; v11 += v10)
//CHECK-NEXT:  chess_prepare_for_pipelining
//CHECK-NEXT:  chess_loop_range(16, 16)
//CHECK-NEXT:  {
//CHECK-NEXT:    size_t v12 = 1;
//CHECK-NEXT:    size_t v13 = v11 + v12;
//CHECK-NEXT:    size_t v14 = 2;
//CHECK-NEXT:    size_t v15 = v11 + v14;
//CHECK-NEXT:    size_t v16 = 0;
//CHECK-NEXT:    size_t v17 = 256;
//CHECK-NEXT:    size_t v18 = 16;
//CHECK-NEXT:    for (size_t v19 = v16; v19 < v17; v19 += v18)
//CHECK-NEXT:    chess_prepare_for_pipelining
//CHECK-NEXT:    chess_loop_range(16, 16)
//CHECK-NEXT:    {
//CHECK-NEXT:      v32int8 v20 = *(v32int8 *)(v1 + 288*v11+v19);
//CHECK-NEXT:      v16acc48 v21 = mul16(v7, 0, 0x00000000, 4, 0x1010, v20, 0, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v16acc48 v22 = mul16(v7, 0, 0x00000000, 4, 0x1010, v20, 4, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v32int8 v23 = *(v32int8 *)(v1 + 288*v13+v19);
//CHECK-NEXT:      v21 = mac16(v21, v7, 16, 0x00000000, 4, 0x1010, v23, 0, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v22 = mac16(v22, v7, 16, 0x00000000, 4, 0x1010, v23, 4, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v32int8 v24 = *(v32int8 *)(v1 + 288*v15+v19);
//CHECK-NEXT:      v21 = mac16(v21, v7, 32, 0x00000000, 4, 0x1010, v24, 0, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v22 = mac16(v22, v7, 32, 0x00000000, 4, 0x1010, v24, 4, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v16int16 v25 = srs(v21, v6);
//CHECK-NEXT:      v16int16 v26 = srs(v22, v6);
//CHECK-NEXT:      v32int16 v27 = concat(v25, v26);
//CHECK-NEXT:      v32int16 v28 = select32(0xcccccccc, v27, 0, 0x0c080400, 0x0, 0x1010, 4, 0x0c080400, 0x0, 0x1010);
//CHECK-NEXT:      v16int16 v29 = ext_w(v28, 0);
//CHECK-NEXT:      v16int8 v30 = pack(v29);
//CHECK-NEXT:      *(v16int8 *)(v3 + 256*v11+v19) = v30;
