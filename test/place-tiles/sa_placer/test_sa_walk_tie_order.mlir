//===- test_sa_walk_tie_order.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s 2>&1 | FileCheck %s
// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie.device(aie-place-tiles{placer=sa_placer sa-seed=1},aie-objectFifo-stateful-transform))' %s | FileCheck %s --check-prefix=ALLOC

// @p and @q are equally deep and one walk fits (0, 1)'s BD ids. The SA and
// allocation walk the same one, @p (smaller objects), however the fifos are
// written; @q's objects then fit split across (0, 1) and (1, 1).
// CHECK-NOT: warning: SA placer's resource estimate is over capacity
// ALLOC-LABEL: @p_first
// ALLOC-DAG: aie.buffer(%mem_tile_0_1) {sym_name = "p__cons__buf0"} : memref<4096xi32>
// ALLOC-DAG: aie.buffer(%{{.*}}) {sym_name = "q__cons__buf15"} : memref<1024xi32>
// ALLOC-LABEL: @q_first
// ALLOC-DAG: aie.buffer(%mem_tile_0_1) {sym_name = "p__cons__buf0"} : memref<4096xi32>
// ALLOC-DAG: aie.buffer(%{{.*}}) {sym_name = "q__cons__buf15"} : memref<1024xi32>

module @p_first {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %m = aie.logical_tile<MemTile>(0, 1)
    %n = aie.logical_tile<MemTile>(1, 1)
    %fill0 = aie.buffer(%m) {sym_name = "fill0"} : memref<483328xi8>
    %fill1 = aie.buffer(%n) {sym_name = "fill1"} : memref<483328xi8>
    aie.objectfifo @p(%shim, {%m}, 16 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @p_out(%m, {%shim}, 16 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [@p] -> [@p_out] ([] [])
    aie.objectfifo @q(%shim, {%m}, 16 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @q_out(%m, {%shim}, 16 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@q] -> [@q_out] ([] [])
  }
}

// -----

module @q_first {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %m = aie.logical_tile<MemTile>(0, 1)
    %n = aie.logical_tile<MemTile>(1, 1)
    %fill0 = aie.buffer(%m) {sym_name = "fill0"} : memref<483328xi8>
    %fill1 = aie.buffer(%n) {sym_name = "fill1"} : memref<483328xi8>
    aie.objectfifo @q(%shim, {%m}, 16 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @q_out(%m, {%shim}, 16 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@q] -> [@q_out] ([] [])
    aie.objectfifo @p(%shim, {%m}, 16 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @p_out(%m, {%shim}, 16 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [@p] -> [@p_out] ([] [])
  }
}
