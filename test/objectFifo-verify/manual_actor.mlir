//===- manual_actor.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectfifo-verify %s

// A DMA program the author writes fills the segment, so no endpoint does.
module @manual_fill {
  aie.device(xcve2302) {
    %tile = aie.tile(1, 2)
    aie.objectfifo.pool @pool(%tile) {depth = 1 : i32} : memref<32xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 32 : i32, manualFill}
    }
    aie.objectfifo.core_endpoint @reader(%tile) drains @pool
    %core = aie.core(%tile) {
      %e = aie.objectfifo.acquire @reader (1) : memref<32xi32>
      aie.objectfifo.release @reader (1)
      aie.end
    }
  }
}

// -----

// The same at the other end: the core fills, a hand-written DMA drains.
module @manual_drain {
  aie.device(xcve2302) {
    %tile = aie.tile(1, 2)
    aie.objectfifo.pool @pool(%tile) {depth = 1 : i32} : memref<32xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 32 : i32, manualDrain}
    }
    aie.objectfifo.core_endpoint @writer(%tile) fills @pool
    %core = aie.core(%tile) {
      %e = aie.objectfifo.acquire @writer (1) : memref<32xi32>
      aie.objectfifo.release @writer (1)
      aie.end
    }
  }
}

// -----

// One segment of a pool goes to a hand-written DMA while the other keeps an
// endpoint at both ends.
module @mixed_segments {
  aie.device(xcve2302) {
    %tile = aie.tile(1, 2)
    %shim = aie.tile(1, 0)
    aie.objectfifo.pool @pool(%tile) {depth = 1 : i32} : memref<48xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32, manualFill}
      aie.objectfifo.segment @s1 {offset = 16 : i32, size = 32 : i32}
    }
    aie.objectfifo.dma_endpoint @fill(%tile) fills @pool {segments = [@s1]}
    aie.route_endpoint @src(%shim) DMA {channelIndex = 0 : i32}
    aie.route from @src to [@fill]
    aie.objectfifo.core_endpoint @reader(%tile) drains @pool {segments = [@s0, @s1]}
    %core = aie.core(%tile) {
      %e = aie.objectfifo.acquire @reader (1) : memref<48xi32>
      aie.objectfifo.release @reader (1)
      aie.end
    }
  }
}

// -----

// manualFill claims that end, so an endpoint must not also fill it.
module @manual_fill_and_endpoint {
  aie.device(xcve2302) {
    %tile = aie.tile(1, 2)
    // expected-error@+1 {{segment 0 is filled by more than one endpoint}}
    aie.objectfifo.pool @pool(%tile) {depth = 1 : i32} : memref<32xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 32 : i32, manualFill}
    }
    aie.objectfifo.dma_endpoint @fill(%tile) fills @pool
    aie.objectfifo.core_endpoint @reader(%tile) drains @pool
    %core = aie.core(%tile) {
      %e = aie.objectfifo.acquire @reader (1) : memref<32xi32>
      aie.objectfifo.release @reader (1)
      aie.end
    }
  }
}

// -----

// Without the attribute the segment still needs a filler.
module @no_manual_attribute {
  aie.device(xcve2302) {
    %tile = aie.tile(1, 2)
    // expected-error@+1 {{segment 0 has no filler}}
    aie.objectfifo.pool @pool(%tile) {depth = 1 : i32} : memref<32xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 32 : i32}
    }
    aie.objectfifo.core_endpoint @reader(%tile) drains @pool
    %core = aie.core(%tile) {
      %e = aie.objectfifo.acquire @reader (1) : memref<32xi32>
      aie.objectfifo.release @reader (1)
      aie.end
    }
  }
}
