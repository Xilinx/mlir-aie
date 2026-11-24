//===- test_sa_walk_search.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t 2> %t.err
// RUN: FileCheck %s --input-file=%t.err
// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie.device(aie-objectFifo-stateful-transform))' %t | FileCheck %s --check-prefix=ALLOC

// (0, 1)'s pools are walked whole by search: of the sets whose walk fits its
// BD ids, the one walking the fewest bytes whose buffers then fit, however the
// fifos are written. Every case here needs a later set than the cheapest.
// Allocation searches the sets; the SA tries one per annealing step, so it
// counts these placements as over capacity, never as fitting.
// CHECK-COUNT-2: warning: SA placer's resource estimate is over capacity

// @s, @x and @f need 75 BD ids. Walking @s and @f fits them and walks the
// fewest bytes, but @f's walked buffer then finds no room; @x alone is next,
// and @s and @f stay split.
// ALLOC-LABEL: @t_sl
// ALLOC-DAG: {sym_name = "x__cons__buf0"} : memref<20480xi32>
// ALLOC-DAG: {sym_name = "s__buf19"} : memref<256xi32>
// ALLOC-LABEL: @t_ls
// ALLOC-DAG: {sym_name = "x__cons__buf0"} : memref<20480xi32>
// ALLOC-DAG: {sym_name = "s__buf19"} : memref<256xi32>

module @t_sl {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %shim1 = aie.logical_tile<ShimNOCTile>(1, 0)
    %m = aie.logical_tile<MemTile>(0, 1)
    %n = aie.logical_tile<MemTile>(1, 1)
    %fill0 = aie.buffer(%m) {sym_name = "fill0"} : memref<437248xi8>
    %fill1 = aie.buffer(%n) {sym_name = "fill1"} : memref<504320xi8>
    aie.objectfifo @s(%m, {%shim1}, 20 : i32) : !aie.objectfifo<memref<256xi32>> = [dense<1> : memref<256xi32>, dense<2> : memref<256xi32>, dense<3> : memref<256xi32>, dense<4> : memref<256xi32>, dense<5> : memref<256xi32>, dense<6> : memref<256xi32>, dense<7> : memref<256xi32>, dense<8> : memref<256xi32>, dense<9> : memref<256xi32>, dense<10> : memref<256xi32>, dense<11> : memref<256xi32>, dense<12> : memref<256xi32>, dense<13> : memref<256xi32>, dense<14> : memref<256xi32>, dense<15> : memref<256xi32>, dense<16> : memref<256xi32>, dense<17> : memref<256xi32>, dense<18> : memref<256xi32>, dense<19> : memref<256xi32>, dense<20> : memref<256xi32>]
    aie.objectfifo @x(%shim, {%m}, 20 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @x_out(%m, {%shim}, 20 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@x] -> [@x_out] ([] [])
    aie.objectfifo @f(%m, {%shim1}, 15 : i32) : !aie.objectfifo<memref<64xi32>> = [dense<1> : memref<64xi32>, dense<2> : memref<64xi32>, dense<3> : memref<64xi32>, dense<4> : memref<64xi32>, dense<5> : memref<64xi32>, dense<6> : memref<64xi32>, dense<7> : memref<64xi32>, dense<8> : memref<64xi32>, dense<9> : memref<64xi32>, dense<10> : memref<64xi32>, dense<11> : memref<64xi32>, dense<12> : memref<64xi32>, dense<13> : memref<64xi32>, dense<14> : memref<64xi32>, dense<15> : memref<64xi32>]
  }
}

// -----

module @t_ls {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %shim1 = aie.logical_tile<ShimNOCTile>(1, 0)
    %m = aie.logical_tile<MemTile>(0, 1)
    %n = aie.logical_tile<MemTile>(1, 1)
    %fill0 = aie.buffer(%m) {sym_name = "fill0"} : memref<437248xi8>
    %fill1 = aie.buffer(%n) {sym_name = "fill1"} : memref<504320xi8>
    aie.objectfifo @x(%shim, {%m}, 20 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @x_out(%m, {%shim}, 20 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@x] -> [@x_out] ([] [])
    aie.objectfifo @s(%m, {%shim1}, 20 : i32) : !aie.objectfifo<memref<256xi32>> = [dense<1> : memref<256xi32>, dense<2> : memref<256xi32>, dense<3> : memref<256xi32>, dense<4> : memref<256xi32>, dense<5> : memref<256xi32>, dense<6> : memref<256xi32>, dense<7> : memref<256xi32>, dense<8> : memref<256xi32>, dense<9> : memref<256xi32>, dense<10> : memref<256xi32>, dense<11> : memref<256xi32>, dense<12> : memref<256xi32>, dense<13> : memref<256xi32>, dense<14> : memref<256xi32>, dense<15> : memref<256xi32>, dense<16> : memref<256xi32>, dense<17> : memref<256xi32>, dense<18> : memref<256xi32>, dense<19> : memref<256xi32>, dense<20> : memref<256xi32>]
    aie.objectfifo @f(%m, {%shim1}, 15 : i32) : !aie.objectfifo<memref<64xi32>> = [dense<1> : memref<64xi32>, dense<2> : memref<64xi32>, dense<3> : memref<64xi32>, dense<4> : memref<64xi32>, dense<5> : memref<64xi32>, dense<6> : memref<64xi32>, dense<7> : memref<64xi32>, dense<8> : memref<64xi32>, dense<9> : memref<64xi32>, dense<10> : memref<64xi32>, dense<11> : memref<64xi32>, dense<12> : memref<64xi32>, dense<13> : memref<64xi32>, dense<14> : memref<64xi32>, dense<15> : memref<64xi32>]
  }
}
