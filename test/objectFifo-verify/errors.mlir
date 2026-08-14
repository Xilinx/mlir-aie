// RUN: aie-opt --aie-objectfifo-verify --split-input-file --verify-diagnostics %s

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A gap between segments leaves bytes no endpoint is responsible for.
module @gap {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    // expected-error@+1 {{segments must be contiguous, but segment at offset 20 follows 16}}
    aie.objectfifo.pool @p(%tile12) {
      depth = 2 : i32,
      segments = [#aie.objectfifo_segment<offset = 0, size = 16>,
                  #aie.objectfifo_segment<offset = 20, size = 12>]
    } : memref<32xi32>
  }
}

// -----

// Segments that stop short of the element type leave a tail unreachable.
module @short {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    // expected-error@+1 {{segments cover 16 of 32 elements}}
    aie.objectfifo.pool @p(%tile12) {
      depth = 2 : i32, segments = [#aie.objectfifo_segment<offset = 0, size = 16>]
    } : memref<32xi32>
  }
}

// -----

// Nothing reads what the core writes.
module @no_drainer {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    // expected-error@+1 {{segment 0 has no drainer}}
    aie.objectfifo.pool @p(%tile12) {
      depth = 2 : i32, segments = [#aie.objectfifo_segment<offset = 0, size = 16>]
    } : memref<16xi32>
    aie.objectfifo.core_endpoint @c(%tile12) fills @p
  }
}

// -----

// Two writers of one segment race for the same objects.
module @two_fillers {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{segment 0 is filled by more than one endpoint}}
    aie.objectfifo.pool @p(%tile12) {
      depth = 2 : i32, segments = [#aie.objectfifo_segment<offset = 0, size = 16>]
    } : memref<16xi32>
    aie.objectfifo.core_endpoint @c0(%tile12) fills @p
    aie.objectfifo.core_endpoint @c1(%tile13) fills @p
    aie.objectfifo.core_endpoint @c2(%tile13) drains @p
  }
}

// -----

// A DMA endpoint that no flow names carries data nowhere.
module @unconnected {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    aie.objectfifo.pool @p(%tile12) {
      depth = 2 : i32, segments = [#aie.objectfifo_segment<offset = 0, size = 16>]
    } : memref<16xi32>
    aie.objectfifo.core_endpoint @c(%tile12) fills @p
    // expected-error@+1 {{is not connected by any flow}}
    aie.objectfifo.dma_endpoint @d(%tile12) drains of @p
  }
}

// -----

// Releasing more than is acquired underflows the held count as the loop repeats.
