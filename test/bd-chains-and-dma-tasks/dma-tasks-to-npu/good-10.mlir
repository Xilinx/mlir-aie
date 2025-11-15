//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 AMD Inc.

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// This test ensure that the burst_length property is properly lowered from the DMAConfigureTaskOp to the NPU writebd op.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.runtime_sequence(%arg0: memref<32xi8>) {
      // CHECK: writebd {{.*}} burst_length = 64
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<32xi8>, 4, 16) {bd_id = 1 : i32, burst_length = 64 : i32}
          aie.end
      }
    }
  }
}
