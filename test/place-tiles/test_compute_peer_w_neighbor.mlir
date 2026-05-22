//===- test_compute_peer_w_neighbor.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// One high-fanin reducer (w5) consumes four ObjectFifos from compute peers
// (w1..w4) plus one ObjectFifo from a shim. AIE2 CoreTiles have a budget of
// 2 DMA-in channels, the shim fifo takes one, so three of the four compute-
// peer fifos must come via shared-L1 (needNeighborIn = 3). No tile has 3
// physical N/S compute neighbors, so the placer's forward-look must also
// consider the checkerboard W neighbor that AIE2 isLegalMemAffinity allows;
// otherwise every candidate is rejected with "compute-peer DMA budget
// unsatisfiable" before w5 ever gets a chance.
//
// Captured from
// programming_examples/basic/vector_reduce_max/multi_column_designs/
// row_wise_vector_reduce_max.py with -d npu -i1s 524288 -os 4 --dtype i32,
// where Worker 5 hits this exact 4-peer-in topology.

// RUN: aie-opt --aie-place-tiles %s 2>&1 | FileCheck %s

// CHECK-NOT: error
// CHECK-NOT: compute-peer DMA budget unsatisfiable

module @compute_peer_w_neighbor {
  aie.device(npu1) {
    %w1 = aie.logical_tile<CoreTile>(?, ?)
    %w2 = aie.logical_tile<CoreTile>(?, ?)
    %w3 = aie.logical_tile<CoreTile>(?, ?)
    %w4 = aie.logical_tile<CoreTile>(?, ?)
    %w5 = aie.logical_tile<CoreTile>(?, ?)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in (%shim, {%w5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @from_w1 (%w1, {%w5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @from_w2 (%w2, {%w5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @from_w3 (%w3, {%w5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @from_w4 (%w4, {%w5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%w1) { aie.end }
    aie.core(%w2) { aie.end }
    aie.core(%w3) { aie.end }
    aie.core(%w4) { aie.end }
    aie.core(%w5) { aie.end }
  }
}
