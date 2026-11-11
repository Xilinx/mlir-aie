//===- test_sa_walk_memory_first.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s 2>&1 | FileCheck %s
// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie-place-tiles{placer=sa_placer sa-seed=1},aie.device(aie-objectFifo-stateful-transform))' %s | FileCheck %s --check-prefix=ALLOC

// (0, 1) holds 60 BD ids' worth of ends, so one of @y (one end) and @x (a
// link, two ends), equally deep, is walked whole. The SA and allocation pick
// the same one in either written order: first the pool with smaller objects,
// since @x whole (160 KB) fits on no tile while too few BD ids would only
// walk another; with objects alike, the pool whose ends take more BD ids.
// CHECK-NOT: warning: SA placer's resource estimate is over capacity

// ALLOC-LABEL: @small_first_yx
// ALLOC-DAG: {sym_name = "y__buf0"} : memref<2560xi32>
// ALLOC-DAG: {sym_name = "x__cons__buf19"} : memref<2048xi32>
// ALLOC-LABEL: @small_first_xy
// ALLOC-DAG: {sym_name = "y__buf0"} : memref<2560xi32>
// ALLOC-DAG: {sym_name = "x__cons__buf19"} : memref<2048xi32>
// ALLOC-LABEL: @more_bds_first_yx
// ALLOC-DAG: {sym_name = "x__cons__buf0"} : memref<2560xi32>
// ALLOC-DAG: {sym_name = "y__buf19"} : memref<128xi32>
// ALLOC-LABEL: @more_bds_first_xy
// ALLOC-DAG: {sym_name = "x__cons__buf0"} : memref<2560xi32>
// ALLOC-DAG: {sym_name = "y__buf19"} : memref<128xi32>

module @small_first_yx {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %m = aie.logical_tile<MemTile>(0, 1)
    %n = aie.logical_tile<MemTile>(1, 1)
    %fill0 = aie.buffer(%m) {sym_name = "fill0"} : memref<425984xi8>
    %fill1 = aie.buffer(%n) {sym_name = "fill1"} : memref<425984xi8>
    aie.objectfifo @y(%m, {%shim}, 20 : i32) : !aie.objectfifo<memref<128xi32>> = [dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>]
    aie.objectfifo @x(%shim, {%m}, 20 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo @x_out(%m, {%shim}, 20 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@x] -> [@x_out] ([] [])
  }
}

// -----

module @small_first_xy {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %m = aie.logical_tile<MemTile>(0, 1)
    %n = aie.logical_tile<MemTile>(1, 1)
    %fill0 = aie.buffer(%m) {sym_name = "fill0"} : memref<425984xi8>
    %fill1 = aie.buffer(%n) {sym_name = "fill1"} : memref<425984xi8>
    aie.objectfifo @x(%shim, {%m}, 20 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo @x_out(%m, {%shim}, 20 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@x] -> [@x_out] ([] [])
    aie.objectfifo @y(%m, {%shim}, 20 : i32) : !aie.objectfifo<memref<128xi32>> = [dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>]
  }
}

// -----

module @more_bds_first_yx {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %m = aie.logical_tile<MemTile>(0, 1)
    %n = aie.logical_tile<MemTile>(1, 1)
    %fill0 = aie.buffer(%m) {sym_name = "fill0"} : memref<425984xi8>
    %fill1 = aie.buffer(%n) {sym_name = "fill1"} : memref<425984xi8>
    aie.objectfifo @y(%m, {%shim}, 20 : i32) : !aie.objectfifo<memref<128xi32>> = [dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>]
    aie.objectfifo @x(%shim, {%m}, 20 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo @x_out(%m, {%shim}, 20 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo.link [@x] -> [@x_out] ([] [])
  }
}

// -----

module @more_bds_first_xy {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %m = aie.logical_tile<MemTile>(0, 1)
    %n = aie.logical_tile<MemTile>(1, 1)
    %fill0 = aie.buffer(%m) {sym_name = "fill0"} : memref<425984xi8>
    %fill1 = aie.buffer(%n) {sym_name = "fill1"} : memref<425984xi8>
    aie.objectfifo @x(%shim, {%m}, 20 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo @x_out(%m, {%shim}, 20 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo.link [@x] -> [@x_out] ([] [])
    aie.objectfifo @y(%m, {%shim}, 20 : i32) : !aie.objectfifo<memref<128xi32>> = [dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>, dense<0> : memref<128xi32>]
  }
}
