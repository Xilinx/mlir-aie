//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// This test ensure that the burst_length property is properly lowered from the DMAConfigureTaskOp to the NPU writebd op.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.runtime_sequence(%arg0: memref<32xi8>) {
      // CHECK: writebd {{.*}} burst_length = 64
      %c4_i32 = arith.constant 4 : i32
      %c16_i32 = arith.constant 16 : i32
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<32xi8> offset = %c4_i32 len = %c16_i32 sizes = [] strides = []) {bd_id = 1 : i32, burst_length = 64 : i32}
          aie.end
      }
    }
  }
}
