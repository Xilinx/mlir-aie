// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module @initialized {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    aie.objectfifo.pool @p(%tile) {depth = 2 : i32, initValues = [dense<1> : memref<16xi32>]} : memref<16xi32> {
      aie.objectfifo.segment @whole {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error @+1 {{push_queue does not support initialized objects}}
    aie.objectfifo.dma_endpoint @send(%tile) drains @p {release_strategy = #aie.release_strategy<push_queue>}
  }
}

// -----

module @memtile {
  aie.device(npu2) {
    %tile = aie.tile(0, 1)
    aie.objectfifo.pool @p(%tile) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @whole {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error @+1 {{push_queue requires a sending DMA on a core tile with core-issued task support}}
    aie.objectfifo.dma_endpoint @send(%tile) drains @p {release_strategy = #aie.release_strategy<push_queue>}
  }
}

// -----

module @repeated_packets {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    aie.objectfifo.pool @p(%tile) {depth = 2 : i32, repeatCount = 3 : i32} : memref<16xi32> {
      aie.objectfifo.segment @whole {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error @+1 {{push_queue cannot repeat out-of-order packets}}
    aie.objectfifo.dma_endpoint @send(%tile) drains @p {release_strategy = #aie.release_strategy<push_queue>, out_of_order}
  }
}

// -----

module @repeat_limit {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    aie.objectfifo.pool @p(%tile) {depth = 2 : i32, repeatCount = 257 : i32} : memref<16xi32> {
      aie.objectfifo.segment @whole {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error @+1 {{push_queue repeat_count exceeds the hardware task limit}}
    aie.objectfifo.dma_endpoint @send(%tile) drains @p {release_strategy = #aie.release_strategy<push_queue>}
  }
}

// -----

module @unsupported_target {
  aie.device(xcvc1902) {
    %tile = aie.tile(1, 2)
    aie.objectfifo.pool @p(%tile) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @whole {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error @+1 {{push_queue requires a sending DMA on a core tile with core-issued task support}}
    aie.objectfifo.dma_endpoint @send(%tile) drains @p {release_strategy = #aie.release_strategy<push_queue>}
  }
}
