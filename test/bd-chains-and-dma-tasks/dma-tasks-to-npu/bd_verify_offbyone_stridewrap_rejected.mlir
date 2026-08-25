//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// Rejection-side companion to bd_verify_offbyone_stridewrap.mlir: for each
// of the 5 off-by-one sites in verifyStridesWraps, one value past the true
// limit ("limit+1") and two past it ("limit+2"). Before the fix, limit+1
// slipped past verifyStridesWraps's off-by-one and was instead rejected by
// the compiler-generated 'aiex.npu.writebd' op (with a range stated in
// encoded-value terms, e.g. "[0:63]"), while limit+2 was already rejected
// by verifyStridesWraps at the user's own 'aie.dma_bd' op (with a range
// stated in actual-value terms, e.g. "[1:64]") -- two different ops
// quoting two different-looking ranges for adjacent illegal values of the
// same field. After aligning all 5 sites to the `- 1` form, limit+1 is
// now also caught by verifyStridesWraps, at 'aie.dma_bd', with the same
// message as limit+2.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-dma-tasks-to-npu %s

// ===== D0 stride =====

module @d0stride_limit_plus1 {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4194304xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error@+1 {{'aie.dma_bd' op Stride 0 exceeds the [1:1048576] range.}}
        aie.dma_bd(%arg0 : memref<4194304xi32> offset = 0 len = 2 sizes = [1, 1, 1, 2] strides = [0, 0, 0, 1048577]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

module @d0stride_limit_plus2 {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4194304xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error@+1 {{'aie.dma_bd' op Stride 0 exceeds the [1:1048576] range.}}
        aie.dma_bd(%arg0 : memref<4194304xi32> offset = 0 len = 2 sizes = [1, 1, 1, 2] strides = [0, 0, 0, 1048578]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

// ===== D1 stride =====

module @d1stride_limit_plus1 {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4194304xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error@+1 {{'aie.dma_bd' op Stride 1 exceeds the [1:1048576] range.}}
        aie.dma_bd(%arg0 : memref<4194304xi32> offset = 0 len = 2 sizes = [1, 1, 2, 1] strides = [0, 0, 1048577, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

module @d1stride_limit_plus2 {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4194304xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error@+1 {{'aie.dma_bd' op Stride 1 exceeds the [1:1048576] range.}}
        aie.dma_bd(%arg0 : memref<4194304xi32> offset = 0 len = 2 sizes = [1, 1, 2, 1] strides = [0, 0, 1048578, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

// ===== D2 stride =====

module @d2stride_limit_plus1 {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4194304xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error@+1 {{'aie.dma_bd' op Stride 2 exceeds the [1:1048576] range.}}
        aie.dma_bd(%arg0 : memref<4194304xi32> offset = 0 len = 2000 sizes = [1, 2, 1, 1000] strides = [0, 1048577, 0, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

module @d2stride_limit_plus2 {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4194304xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error@+1 {{'aie.dma_bd' op Stride 2 exceeds the [1:1048576] range.}}
        aie.dma_bd(%arg0 : memref<4194304xi32> offset = 0 len = 2000 sizes = [1, 2, 1, 1000] strides = [0, 1048578, 0, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

// ===== Iteration size =====

module @iteration_size_limit_plus1 {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<200000xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error@+1 {{'aie.dma_bd' op Size 3 exceeds the [1:64] range.}}
        aie.dma_bd(%arg0 : memref<200000xi32> offset = 0 len = 2048 sizes = [65, 1, 1, 2048] strides = [2048, 0, 0, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

module @iteration_size_limit_plus2 {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<200000xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error@+1 {{'aie.dma_bd' op Size 3 exceeds the [1:64] range.}}
        aie.dma_bd(%arg0 : memref<200000xi32> offset = 0 len = 2048 sizes = [66, 1, 1, 2048] strides = [2048, 0, 0, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

// ===== Iteration stride (D3) =====

module @iterstride_limit_plus1 {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4194304xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error@+1 {{'aie.dma_bd' op Stride 3 exceeds the [1:1048576] range.}}
        aie.dma_bd(%arg0 : memref<4194304xi32> offset = 0 len = 1 sizes = [2, 1, 1, 1] strides = [1048577, 0, 0, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

module @iterstride_limit_plus2 {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4194304xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error@+1 {{'aie.dma_bd' op Stride 3 exceeds the [1:1048576] range.}}
        aie.dma_bd(%arg0 : memref<4194304xi32> offset = 0 len = 1 sizes = [2, 1, 1, 1] strides = [1048578, 0, 0, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}
