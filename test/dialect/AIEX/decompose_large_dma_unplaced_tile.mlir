//===- decompose_large_dma_unplaced_tile.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-decompose-large-dma-bd %s | FileCheck %s

// Decomposition is a shape rewrite, so an unplaced tile is not an error here:
// the pattern declines and runs again once placement substitutes a concrete
// aie.tile. Before the tryGetTileOp guard, resolveTaskAndTile reached
// TileElement::getTileOp() and aborted the process instead.

// CHECK-LABEL: aie.device(npu1)
// CHECK: %[[SHIM:.*]] = aie.logical_tile<ShimNOCTile>(?, ?)
// CHECK: aiex.dma_configure_task(%[[SHIM]], MM2S, 0)
// CHECK: aie.dma_bd

module {
  aie.device(npu1) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)

    aie.runtime_sequence(%arg0: memref<32xi8>) {
      %t = aiex.dma_configure_task(%shim, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<32xi8> offset = 0 len = 32) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}
