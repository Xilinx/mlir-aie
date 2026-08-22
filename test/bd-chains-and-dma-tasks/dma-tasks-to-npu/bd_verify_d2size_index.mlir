//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// lib/Dialect/AIEX/Transforms/AIEDMATasksToNPU.cpp assigned
//   `d2size = isMemTile(...) ? (*dims)[2].getSize() : 0`
// using the outermost-first original `dims` list, while the neighbouring
// `d0size = sizes[0]` / `d1size = sizes[1]` use the innermost-first array
// produced by getHardwareStridesWraps. This mismatch meant the wrong
// dimension's size landed in d2_size (and which dimension it wrongly
// picked up depended on the dim-list length). d2_size is never packed into
// any BD word on the memtile path today (there is a literal `// TODO:
// D2Size` above the D2 word-packing lines in AIEDmaToNpu.cpp; the real D2
// repeat count is carried entirely by buffer_length, like on shim tiles),
// so this fix corrects the index only -- it deliberately does NOT add a
// width bound for d2_size.

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses --aie-dma-tasks-to-npu %s | FileCheck %s

// A 4-element outermost-first dim list [iter=1, D2=512, D1=5, D0=7]:
// mutually distinct sizes so the misread index is unambiguous.
// d2_size now correctly reads the D2 entry (512), not D1 (5).

// CHECK-LABEL: @pin_d2size_4elem
// CHECK: aiex.npu.writebd
// CHECK-SAME: d0_size = 7
// CHECK-SAME: d1_size = 5
// CHECK-SAME: d2_size = 512
module @pin_d2size_4elem {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) : memref<20000xi32>

    aie.runtime_sequence(%arg0: memref<20000xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%buf : memref<20000xi32> offset = 0 len = 17920 sizes = [1, 512, 5, 7] strides = [0, 35, 7, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

// A 3-element outermost-first dim list [D2=512, D1=5, D0=7] (no explicit
// iteration dim). Before the fix, outermost-first index 2 was D0 (a
// *different* wrong dimension than in the 4-element case above), which is
// why this is an index-convention mixup rather than a simple off-by-one.
// d2_size now correctly reads the D2 entry (512), not D0 (7).

// CHECK-LABEL: @pin_d2size_3elem
// CHECK: aiex.npu.writebd
// CHECK-SAME: d0_size = 7
// CHECK-SAME: d1_size = 5
// CHECK-SAME: d2_size = 512
module @pin_d2size_3elem {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) : memref<20000xi32>

    aie.runtime_sequence(%arg0: memref<20000xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%buf : memref<20000xi32> offset = 0 len = 17920 sizes = [512, 5, 7] strides = [35, 7, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

// Load-bearing: a shim tile has no D2 wrap register -- its repeat count is
// carried entirely by buffer_length, unlike D0/D1 which are genuinely
// 10-bit wrap fields. This guards the d2_size index fix from overreaching:
// on a shim tile, d2_size must stay 0 unconditionally, regardless of what
// value would appear at any dims-list index.

// CHECK-LABEL: @loadbearing_shim_d2_2048
module @loadbearing_shim_d2_2048 {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<16384xi32>) {
      // CHECK: aiex.npu.writebd
      // CHECK-SAME: d0_size = 4
      // CHECK-SAME: d1_size = 1
      // CHECK-SAME: d2_size = 0
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<16384xi32> offset = 0 len = 8192 sizes = [1, 2048, 1, 4] strides = [0, 8, 0, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}
