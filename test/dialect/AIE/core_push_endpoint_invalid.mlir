// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --verify-diagnostics %s -o /dev/null

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    aie.objectfifo.pool @p(%tile) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error @+1 {{core_push requires a filling core endpoint}}
    aie.objectfifo.core_endpoint @end(%tile) drains @p {core_push = #aie.core_push<channel = 0, bd_tags = [0, 1]>}
  }
}

// -----

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    aie.objectfifo.pool @p(%tile) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error @+1 {{core_push requires one BD tag per pool buffer}}
    aie.objectfifo.core_endpoint @end(%tile) fills @p {core_push = #aie.core_push<channel = 0, bd_tags = [0]>}
  }
}

// -----

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    aie.objectfifo.pool @p(%tile) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error @+1 {{core_push requires nonnegative BD tags}}
    aie.objectfifo.core_endpoint @end(%tile) fills @p {core_push = #aie.core_push<channel = 0, bd_tags = [0, -1]>}
  }
}

// -----

// Hand-written core metadata must obey the same initialization restriction
// as push_queue DMA endpoints, even when no DMA endpoint is present.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    aie.objectfifo.pool @p(%tile) {depth = 2 : i32, initValues = [dense<1> : memref<16xi32>]} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error @+1 {{core_push does not support initialized objects}}
    aie.objectfifo.core_endpoint @fill(%tile) fills @p {core_push = #aie.core_push<channel = 0, bd_tags = [7, 8]>}
  }
}
