//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// verifyStridesWraps (lib/Dialect/AIEX/IR/AIEXDialect.cpp) had 5 off-by-one
// sites. The wrap checks (hardwareSizes[0]/[1]) correctly used
// `> (1 << bits) - 1`, but iteration size and all 4 strides used
// `> (1 << bits)` (missing the `- 1`), comparing against *encoded* values
// that are already `actual - 1` biased. This let one extra encoded value
// through per site; downstream NpuWriteBdOp::verify still caught it, so
// nothing illegal ever reached hardware -- this was purely a
// diagnostic-quality defect (see bd_verify_offbyone_stridewrap_rejected.mlir
// for the rejection-side cases this fix changes). These are the boundary
// cases that must stay accepted, at exactly the true (actual-value) limit
// of each field: 1048576 for the four 20-bit shim stride/iteration-stride
// fields, 64 for the 6-bit iteration-size field.

// RUN: aie-opt --split-input-file --aie-dma-tasks-to-npu %s | FileCheck %s

// CHECK-LABEL: @d0stride_limit_accepted
// CHECK: aiex.npu.writebd
// CHECK-SAME: d0_stride = 1048575
module @d0stride_limit_accepted {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4194304xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4194304xi32> offset = 0 len = 2 sizes = [1, 1, 1, 2] strides = [0, 0, 0, 1048576]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

// CHECK-LABEL: @d1stride_limit_accepted
// CHECK: aiex.npu.writebd
// CHECK-SAME: d1_stride = 1048575
module @d1stride_limit_accepted {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4194304xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4194304xi32> offset = 0 len = 2 sizes = [1, 1, 2, 1] strides = [0, 0, 1048576, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

// CHECK-LABEL: @d2stride_limit_accepted
// CHECK: aiex.npu.writebd
// CHECK-SAME: d2_stride = 1048575
module @d2stride_limit_accepted {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4194304xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4194304xi32> offset = 0 len = 2000 sizes = [1, 2, 1, 1000] strides = [0, 1048576, 0, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

// CHECK-LABEL: @iteration_size_limit_accepted
// CHECK: aiex.npu.writebd
// CHECK-SAME: iteration_size = 63
module @iteration_size_limit_accepted {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<200000xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<200000xi32> offset = 0 len = 2048 sizes = [64, 1, 1, 2048] strides = [2048, 0, 0, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

// Iteration stride (D3): the size-1-guarded branch
// `hardwareStrides[3] > (1 << step_bits) && hardwareSizes[3] > 0`.
// Distinct from the iteration *size* check above.

// CHECK-LABEL: @iterstride_limit_accepted
// CHECK: aiex.npu.writebd
// CHECK-SAME: iteration_stride = 1048575
module @iterstride_limit_accepted {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4194304xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4194304xi32> offset = 0 len = 1 sizes = [2, 1, 1, 1] strides = [1048576, 0, 0, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}
