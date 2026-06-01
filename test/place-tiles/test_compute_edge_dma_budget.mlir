//===- test_compute_edge_dma_budget.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Four unpinned compute Workers in a chain-with-merge pattern:
//   W1 (compute) -> {W2, W3} via broadcast
//   W2 (compute) -> W4
//   W3 (compute) -> W4
//   memtile -> W4   (skip1)
//   memtile -> W4   (skip2)
//
// W4 consumes 4 ObjectFifos: 2 from compute peers (W2, W3) and 2 from a
// memtile. The two compute-peer fifos can use shared neighbor memory iff
// the placer puts W4 on a tile with BOTH W2 and W3 as physical neighbors
// (so each peer's L1 can be reached without using a DMA channel); the two
// memtile fifos always need DMA channels. NPU2 compute-tile budget is
// 2 in / 2 out, so W4 only fits when at most two of its incoming fifos
// need a DMA channel. --aie-place-tiles + --aie-objectFifo-stateful-
// transform must produce a placement that satisfies this budget.
//
// Captured from programming_examples/ml/bottleneck/bottleneck.py after
// stripping the residual `tile=AnyMemTile` and three `tile=Tile(0,N)`
// arguments from the IRON Python source.

// RUN: aie-opt --aie-place-tiles --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// CHECK-NOT: error
// CHECK-NOT: DMA channel exceeded

module @compute_edge_dma_budget {
  aie.device(npu2) {
    %w1 = aie.logical_tile<CoreTile>(?, ?)
    %w2 = aie.logical_tile<CoreTile>(?, ?)
    %w3 = aie.logical_tile<CoreTile>(?, ?)
    %w4 = aie.logical_tile<CoreTile>(?, ?)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem  = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @in (%shim, {%w1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @bcast (%w1, {%w2, %w3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @to_w4_a (%w2, {%w4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @to_w4_b (%w3, {%w4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @skip1 (%mem, {%w4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @skip2 (%mem, {%w4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%w1) { aie.end }
    aie.core(%w2) { aie.end }
    aie.core(%w3) { aie.end }
    aie.core(%w4) { aie.end }
  }
}
