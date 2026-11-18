//===- bad_tile_operand.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// The tile operand of an op that lives on a tile is an index, so any index
// value parses there. One that is not an aie.tile or aie.logical_tile is
// rejected rather than dereferenced.

aie.device(npu1) {
  aie.runtime_sequence(%arg0 : memref<64xi32>) {
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{'aiex.dma_configure_task' op tile operand is not an aie.tile or aie.logical_tile}}
    %t = aiex.dma_configure_task(%c0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
      aie.end
    }
  }
}

// -----

// A burst length is checked against the tile's coordinates.
aie.device(npu1) {
  aie.runtime_sequence(%arg0 : memref<64xi32>) {
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{'aiex.dma_configure_task' op tile operand is not an aie.tile or aie.logical_tile}}
    %t = aiex.dma_configure_task(%c0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64) {burst_length = 128 : i32}
      aie.end
    }
  }
}

// -----

// A block argument has no defining op.
aie.device(npu1) {
  aie.runtime_sequence(%arg0 : memref<64xi32>, %tile : index) {
    // expected-error @+1 {{'aiex.dma_configure_task' op tile operand is not an aie.tile or aie.logical_tile}}
    %t = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
      aie.end
    }
  }
}

// -----

// Materializing the chain would build a dma_configure_task on this operand.
aie.device(npu1) {
  aie.bd_chain @chain(%buf : memref<8xi16>) {
    aie.end
  }
  aie.runtime_sequence(%arg0 : memref<8xi16>) {
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{'aiex.dma_start_bd_chain' op tile operand is not an aie.tile or aie.logical_tile}}
    %t = aiex.dma_start_bd_chain @chain(%arg0) : (memref<8xi16>) on (%c0, MM2S, 0)
  }
}

// -----

aie.device(npu1) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{'aie.lock' op tile operand is not an aie.tile or aie.logical_tile}}
  %lock = aie.lock(%c0)
}

// -----

aie.device(npu1) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{'aie.switchbox' op tile operand is not an aie.tile or aie.logical_tile}}
  %sb = aie.switchbox(%c0) {
  }
}
