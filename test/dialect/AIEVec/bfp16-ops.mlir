//===- bfp16-ops.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | aie-opt | FileCheck %s

// CHECK-LABEL: func.func @convert_and_multiply
// CHECK: aievec.to_bfp16 %{{.*}} : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
// CHECK: aievec.to_bfp16 %{{.*}} transpose : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
// CHECK: aievec.to_bfp16 %{{.*}} : vector<4x8xbf16> -> vector<64xi8>, vector<8xi8>
// CHECK: aievec.to_bfp16 %{{.*}} transpose : vector<8x4xbf16> -> vector<64xi8>, vector<8xi8>
// CHECK: aievec.matmul_aie2p_bfp16 {{.*}} into vector<8x8xf32>
// CHECK: aievec.matmul_aie2p_bfp16 {{.*}} into vector<4x8xf32>
func.func @convert_and_multiply(%a: vector<8x8xbf16>, %b: vector<8x8xbf16>,
                                %c: vector<4x8xbf16>, %d: vector<8x4xbf16>,
                                %acc8: vector<8x8xf32>, %acc4: vector<4x8xf32>)
    -> (vector<8x8xf32>, vector<4x8xf32>) {
  %am, %ae = aievec.to_bfp16 %a : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
  %bm, %be = aievec.to_bfp16 %b transpose : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
  %cm, %ce = aievec.to_bfp16 %c : vector<4x8xbf16> -> vector<64xi8>, vector<8xi8>
  %dm, %de = aievec.to_bfp16 %d transpose : vector<8x4xbf16> -> vector<64xi8>, vector<8xi8>
  %r8 = aievec.matmul_aie2p_bfp16 %am, %ae, %bm, %be, %acc8 : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
  %r4 = aievec.matmul_aie2p_bfp16 %cm, %ce, %bm, %be, %acc4 : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<4x8xf32>
  return %r8, %r4 : vector<8x8xf32>, vector<4x8xf32>
}

// CHECK-LABEL: func.func @blocks
// CHECK: aievec.load_bfp16 %{{.*}}[%{{.*}}] : memref<576xi8> -> vector<64xi8>, vector<8xi8>
// CHECK: aievec.fifo_start %{{.*}}[%{{.*}}] : memref<576xi8> -> vector<32xi32>
// CHECK: aievec.fifo_ld_pop_bfp16 %{{.*}}, %{{.*}}, %{{.*}} : vector<32xi32> -> vector<32xi32>, vector<64xi8>, vector<8xi8>
// CHECK: aievec.fifo_ld_pop_bfp16 %{{.*}}, %{{.*}}, %{{.*}} no_fill : vector<32xi32> -> vector<32xi32>, vector<64xi8>, vector<8xi8>
// CHECK: aievec.fifo_start %{{.*}}[%{{.*}}] : memref<576xi8> -> vector<32xi32>
// CHECK: aievec.fifo_st_push_bfp16 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<32xi32>, vector<64xi8>, vector<8xi8> -> vector<32xi32>
func.func @blocks(%b: memref<576xi8>, %o: index) -> (vector<64xi8>, vector<64xi8>) {
  %m0, %e0 = aievec.load_bfp16 %b[%o] : memref<576xi8> -> vector<64xi8>, vector<8xi8>
  %a, %s, %n = aievec.fifo_start %b[%o] : memref<576xi8> -> vector<32xi32>
  %a1, %s1, %n1, %m1, %e1 = aievec.fifo_ld_pop_bfp16 %a, %s, %n : vector<32xi32> -> vector<32xi32>, vector<64xi8>, vector<8xi8>
  %a2, %s2, %n2, %m2, %e2 = aievec.fifo_ld_pop_bfp16 %a1, %s1, %n1 no_fill : vector<32xi32> -> vector<32xi32>, vector<64xi8>, vector<8xi8>
  %w, %ws, %wn = aievec.fifo_start %b[%o] : memref<576xi8> -> vector<32xi32>
  %w1, %ws1, %wn1 = aievec.fifo_st_push_bfp16 %w, %ws, %wn, %m0, %e0 : vector<32xi32>, vector<64xi8>, vector<8xi8> -> vector<32xi32>
  return %m1, %m2 : vector<64xi8>, vector<64xi8>
}
