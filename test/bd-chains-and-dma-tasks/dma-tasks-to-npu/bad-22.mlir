//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics %s

// As bad-20.mlir, but both lists are internally consistent and disagree only in
// rank. Unchecked, AIEDMATasksToNPU's runtime-BD path pads the two mixed lists
// to 4 dimensions in one loop bounded by the sizes rank, indexing the shorter
// strides list and its 4-element destination out of bounds.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<32xi8>) {
      %t1 = "aiex.dma_configure_task"(%tile_0_0) <{channel = 0 : i32, direction = 1 : i32, operandSegmentSizes = array<i32: 1, 0, 0>}> ({
        // expected-error@+1 {{expected the same number of sizes (2) and strides (1)}}
        "aie.dma_bd"(%arg0) <{bd_id = 0 : i32, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, static_len = 32 : i32, static_offset = 4 : i32, static_sizes = array<i64: 8, 2>, static_strides = array<i64: 1>}> : (memref<32xi8>) -> ()
        aie.end
      }) : (index) -> index
    }
  }
}
