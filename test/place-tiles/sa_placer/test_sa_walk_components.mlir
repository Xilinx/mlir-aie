//===- test_sa_walk_components.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t 2> %t.err
// RUN: FileCheck %s --input-file=%t.err
// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie.device(aie-objectFifo-stateful-transform))' %t | FileCheck %s --check-prefix=ALLOC

// Walk sets are searched per spill component, every choice of sets for its
// mem tiles in order of bytes walked, until its buffers place, however the
// fifos are written. Every case here needs a later choice than the cheapest.
// Allocation searches the choices; the SA tries one per annealing step, so
// it counts these placements as over capacity, never as fitting.
// CHECK-COUNT-4: warning: SA placer's resource estimate is over capacity

// (0, 1) and (3, 1) each place only with their second walk set. Their buffers
// share no tile, so neither one's search waits on the other's.
// ALLOC-LABEL: @joint
// ALLOC-DAG: {sym_name = "a__cons__buf0"} : memref<4096xi32>
// ALLOC-DAG: {sym_name = "c__cons__buf0"} : memref<16384xi32>
// ALLOC-DAG: {sym_name = "d__cons__buf0"} : memref<4096xi32>
// ALLOC-DAG: {sym_name = "f__cons__buf0"} : memref<16384xi32>
// ALLOC-DAG: {sym_name = "b__cons__buf15"} : memref<512xi32>
// ALLOC-DAG: {sym_name = "e__cons__buf15"} : memref<512xi32>
// ALLOC-LABEL: @joint_reversed
// ALLOC-DAG: {sym_name = "a__cons__buf0"} : memref<4096xi32>
// ALLOC-DAG: {sym_name = "c__cons__buf0"} : memref<16384xi32>
// ALLOC-DAG: {sym_name = "d__cons__buf0"} : memref<4096xi32>
// ALLOC-DAG: {sym_name = "f__cons__buf0"} : memref<16384xi32>

// Eight sets are cheaper than {a, c}, each @a and @b with some of the tiny
// t1..t3, and none places; {a, c} is the ninth and does.
// ALLOC-LABEL: @cheap_sets_fail
// ALLOC-DAG: {sym_name = "a__cons__buf0"} : memref<4096xi32>
// ALLOC-DAG: {sym_name = "c__cons__buf0"} : memref<16384xi32>
// ALLOC-DAG: {sym_name = "b__cons__buf15"} : memref<512xi32>
// ALLOC-LABEL: @cheap_sets_fail_reversed
// ALLOC-DAG: {sym_name = "a__cons__buf0"} : memref<4096xi32>
// ALLOC-DAG: {sym_name = "c__cons__buf0"} : memref<16384xi32>
// ALLOC-DAG: {sym_name = "b__cons__buf15"} : memref<512xi32>

module @joint {
  aie.device(npu2) {
    %s0 = aie.logical_tile<ShimNOCTile>(0, 0)
    %s1 = aie.logical_tile<ShimNOCTile>(1, 0)
    %s3 = aie.logical_tile<ShimNOCTile>(3, 0)
    %s4 = aie.logical_tile<ShimNOCTile>(4, 0)
    %m0 = aie.logical_tile<MemTile>(0, 1)
    %m1 = aie.logical_tile<MemTile>(1, 1)
    %m2 = aie.logical_tile<MemTile>(2, 1)
    %m3 = aie.logical_tile<MemTile>(3, 1)
    %m4 = aie.logical_tile<MemTile>(4, 1)
    %fill0 = aie.buffer(%m0) {sym_name = "fill0"} : memref<440320xi8>
    %fill1 = aie.buffer(%m1) {sym_name = "fill1"} : memref<493568xi8>
    %fill2 = aie.buffer(%m2) {sym_name = "fill2"} : memref<493568xi8>
    %fill3 = aie.buffer(%m3) {sym_name = "fill3"} : memref<440320xi8>
    %fill4 = aie.buffer(%m4) {sym_name = "fill4"} : memref<524288xi8>
    aie.objectfifo @a(%s0, {%m0}, 16 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @a_out(%m0, {%s0}, 16 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [@a] -> [@a_out] ([] [])
    aie.objectfifo @b(%s0, {%m0}, 16 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @b_out(%m0, {%s0}, 16 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo.link [@b] -> [@b_out] ([] [])
    aie.objectfifo @c(%s1, {%m0}, 16 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @c_out(%m0, {%s1}, 16 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@c] -> [@c_out] ([] [])
    aie.objectfifo @d(%s3, {%m3}, 16 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @d_out(%m3, {%s3}, 16 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [@d] -> [@d_out] ([] [])
    aie.objectfifo @e(%s3, {%m3}, 16 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @e_out(%m3, {%s3}, 16 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo.link [@e] -> [@e_out] ([] [])
    aie.objectfifo @f(%s4, {%m3}, 16 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @f_out(%m3, {%s4}, 16 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@f] -> [@f_out] ([] [])
  }
}

// -----

module @joint_reversed {
  aie.device(npu2) {
    %s0 = aie.logical_tile<ShimNOCTile>(0, 0)
    %s1 = aie.logical_tile<ShimNOCTile>(1, 0)
    %s3 = aie.logical_tile<ShimNOCTile>(3, 0)
    %s4 = aie.logical_tile<ShimNOCTile>(4, 0)
    %m0 = aie.logical_tile<MemTile>(0, 1)
    %m1 = aie.logical_tile<MemTile>(1, 1)
    %m2 = aie.logical_tile<MemTile>(2, 1)
    %m3 = aie.logical_tile<MemTile>(3, 1)
    %m4 = aie.logical_tile<MemTile>(4, 1)
    %fill0 = aie.buffer(%m0) {sym_name = "fill0"} : memref<440320xi8>
    %fill1 = aie.buffer(%m1) {sym_name = "fill1"} : memref<493568xi8>
    %fill2 = aie.buffer(%m2) {sym_name = "fill2"} : memref<493568xi8>
    %fill3 = aie.buffer(%m3) {sym_name = "fill3"} : memref<440320xi8>
    %fill4 = aie.buffer(%m4) {sym_name = "fill4"} : memref<524288xi8>
    aie.objectfifo.link [@f] -> [@f_out] ([] [])
    aie.objectfifo @f_out(%m3, {%s4}, 16 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @f(%s4, {%m3}, 16 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@e] -> [@e_out] ([] [])
    aie.objectfifo @e_out(%m3, {%s3}, 16 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @e(%s3, {%m3}, 16 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo.link [@d] -> [@d_out] ([] [])
    aie.objectfifo @d_out(%m3, {%s3}, 16 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @d(%s3, {%m3}, 16 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [@c] -> [@c_out] ([] [])
    aie.objectfifo @c_out(%m0, {%s1}, 16 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @c(%s1, {%m0}, 16 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@b] -> [@b_out] ([] [])
    aie.objectfifo @b_out(%m0, {%s0}, 16 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @b(%s0, {%m0}, 16 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo.link [@a] -> [@a_out] ([] [])
    aie.objectfifo @a_out(%m0, {%s0}, 16 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @a(%s0, {%m0}, 16 : i32) : !aie.objectfifo<memref<256xi32>>
  }
}

// -----

module @cheap_sets_fail {
  aie.device(npu2) {
    %s0 = aie.logical_tile<ShimNOCTile>(0, 0)
    %s1 = aie.logical_tile<ShimNOCTile>(1, 0)
    %s2 = aie.logical_tile<ShimNOCTile>(2, 0)
    %m0 = aie.logical_tile<MemTile>(0, 1)
    %m1 = aie.logical_tile<MemTile>(1, 1)
    %fill0 = aie.buffer(%m0) {sym_name = "fill0"} : memref<440320xi8>
    %fill1 = aie.buffer(%m1) {sym_name = "fill1"} : memref<493184xi8>
    aie.objectfifo @a(%s0, {%m0}, 16 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @a_out(%m0, {%s0}, 16 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [@a] -> [@a_out] ([] [])
    aie.objectfifo @b(%s0, {%m0}, 16 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @b_out(%m0, {%s0}, 16 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo.link [@b] -> [@b_out] ([] [])
    aie.objectfifo @c(%s1, {%m0}, 16 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @c_out(%m0, {%s1}, 16 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@c] -> [@c_out] ([] [])
    aie.objectfifo @t1(%m0, {%s1}, 2 : i32) : !aie.objectfifo<memref<16xi32>> = [dense<1> : memref<16xi32>, dense<2> : memref<16xi32>]
    aie.objectfifo @t2(%m0, {%s2}, 2 : i32) : !aie.objectfifo<memref<16xi32>> = [dense<1> : memref<16xi32>, dense<2> : memref<16xi32>]
    aie.objectfifo @t3(%m0, {%s2}, 2 : i32) : !aie.objectfifo<memref<16xi32>> = [dense<1> : memref<16xi32>, dense<2> : memref<16xi32>]
  }
}

// -----

module @cheap_sets_fail_reversed {
  aie.device(npu2) {
    %s0 = aie.logical_tile<ShimNOCTile>(0, 0)
    %s1 = aie.logical_tile<ShimNOCTile>(1, 0)
    %s2 = aie.logical_tile<ShimNOCTile>(2, 0)
    %m0 = aie.logical_tile<MemTile>(0, 1)
    %m1 = aie.logical_tile<MemTile>(1, 1)
    %fill0 = aie.buffer(%m0) {sym_name = "fill0"} : memref<440320xi8>
    %fill1 = aie.buffer(%m1) {sym_name = "fill1"} : memref<493184xi8>
    aie.objectfifo @t3(%m0, {%s2}, 2 : i32) : !aie.objectfifo<memref<16xi32>> = [dense<1> : memref<16xi32>, dense<2> : memref<16xi32>]
    aie.objectfifo @t2(%m0, {%s2}, 2 : i32) : !aie.objectfifo<memref<16xi32>> = [dense<1> : memref<16xi32>, dense<2> : memref<16xi32>]
    aie.objectfifo @t1(%m0, {%s1}, 2 : i32) : !aie.objectfifo<memref<16xi32>> = [dense<1> : memref<16xi32>, dense<2> : memref<16xi32>]
    aie.objectfifo.link [@c] -> [@c_out] ([] [])
    aie.objectfifo @c_out(%m0, {%s1}, 16 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @c(%s1, {%m0}, 16 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@b] -> [@b_out] ([] [])
    aie.objectfifo @b_out(%m0, {%s0}, 16 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @b(%s0, {%m0}, 16 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo.link [@a] -> [@a_out] ([] [])
    aie.objectfifo @a_out(%m0, {%s0}, 16 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @a(%s0, {%m0}, 16 : i32) : !aie.objectfifo<memref<256xi32>>
  }
}
