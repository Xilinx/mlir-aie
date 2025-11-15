// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 AMD Inc.

// RUN: not aie-opt --aie-dma-tasks-to-npu %s 2>&1 | FileCheck %s

// This test ensures that no buffer descriptor can be created with a specified burst length
// outside of a shim tile. Note that this cannot be tested at the dma_bd level since the
// all DMABDOp checks are skipped when inside a dma_configure_task.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(2, 2)

    aie.runtime_sequence(%arg0: memref<8xi16>, %arg1: memref<10xi32>) {
      // CHECK: Burst length is only supported in Shim NOC tiles that are connected to the memory-mapped NOC.
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 7 : i32, burst_length = 256 : i32}
        aie.end
      } {issue_token = true}
    }
  }
}
