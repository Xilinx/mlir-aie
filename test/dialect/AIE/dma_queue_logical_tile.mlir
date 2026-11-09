// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles %s | FileCheck %s

// CHECK-LABEL: module @logical_queue
// CHECK: aie.tile(
// CHECK: aie.dma_poll_for_queue_space(S2MM : 0)
// CHECK: aie.dma_push_queue(S2MM : 0) bd_id 0 {out_of_order, repeat_count = 255 : i32}
// CHECK: aie.dma_push_queue(MM2S : 1) bd_id 1
module @logical_queue {
  aie.device(npu2) {
    %tile = aie.logical_tile<CoreTile>(?, ?)
    %core = aie.core(%tile) {
      aie.dma_poll_for_queue_space(S2MM : 0)
      aie.dma_push_queue(S2MM : 0) bd_id 0 {out_of_order, repeat_count = 255 : i32}
      aie.dma_push_queue(MM2S : 1) bd_id 1
      aie.end
    }
  }
}

// -----

module @invalid_repeat_before_placement {
  aie.device(npu2) {
    %tile = aie.logical_tile<CoreTile>(?, ?)
    %core = aie.core(%tile) {
      // expected-error@+1 {{repeat_count must be in [0, 255]}}
      aie.dma_push_queue(MM2S : 0) bd_id 0 {repeat_count = 256 : i32}
      aie.end
    }
  }
}

// -----

module @no_queue_before_placement {
  aie.device(xcvc1902) {
    %tile = aie.logical_tile<CoreTile>(?, ?)
    %core = aie.core(%tile) {
      // expected-error@+1 {{waits on a task queue, which this device has none of}}
      aie.dma_poll_for_queue_space(S2MM : 0)
      aie.end
    }
  }
}

// -----

module @invalid_push_after_placement {
  aie.device(npu2) {
    %tile = aie.logical_tile<CoreTile>(0, 2)
    %core = aie.core(%tile) {
      // expected-error@+1 {{pushes onto MM2S channel 2, but tile (0, 2) has 2}}
      aie.dma_push_queue(MM2S : 2) bd_id 0
      aie.end
    }
  }
}

// -----

module @invalid_poll_after_placement {
  aie.device(npu2) {
    %tile = aie.logical_tile<CoreTile>(0, 2)
    %core = aie.core(%tile) {
      // expected-error@+1 {{waits on S2MM channel 2, but tile (0, 2) has 2}}
      aie.dma_poll_for_queue_space(S2MM : 2)
      aie.end
    }
  }
}
