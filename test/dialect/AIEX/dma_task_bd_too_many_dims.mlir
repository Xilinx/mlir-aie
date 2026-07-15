//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A shim BD on the runtime-sequence path may carry up to 4 dimensions: the 3 ND
// access dimensions (AIETargetModel::getBDMaxDims) plus the leading dimension
// that aiex.shim_dma_single_bd_task hoists into the shim iteration/repeat
// register. A 5-dimension BD is rejected. This is the only verification of the
// BD dimension count on this path -- AIE::DMABDOp::verify skips BDs nested in a
// DMA task op.

// RUN: aie-opt --verify-diagnostics --split-input-file %s

// dma_configure_task (concrete shim tile): 5 dims -> rejected.
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error@+1 {{Cannot give more than 4 dimensions}}
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64 sizes = [1, 1, 1, 1, 1] strides = [0, 0, 0, 0, 1])
        aie.end
      }
    }
  }
}

// -----

// dma_configure_task (concrete shim tile): 4 dims -> accepted (iteration hoist).
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      }
    }
  }
}

// -----

// dma_configure_task_for (tile recovered through the shim DMA allocation
// symbol): 5 dims -> rejected.
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0 (%tile_0_0, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task_for @alloc0 {
        // expected-error@+1 {{Cannot give more than 4 dimensions}}
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64 sizes = [1, 1, 1, 1, 1] strides = [0, 0, 0, 0, 1])
        aie.end
      }
    }
  }
}

// -----

// dma_configure_task_for (resolved shim symbol): 4 dims -> accepted, the common
// case for a shim_dma_single_bd_task tap.
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0 (%tile_0_0, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task_for @alloc0 {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      }
    }
  }
}

// -----

// dma_configure_task_for whose alloc symbol does not resolve: the dimension
// check is deferred (verifies clean) rather than crashing; a later pass that
// substitutes the allocation performs the check on the concrete tile.
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task_for @missing_alloc {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64 sizes = [1, 1, 1, 1, 1] strides = [0, 0, 0, 0, 1])
        aie.end
      }
    }
  }
}
