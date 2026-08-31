//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// An unranked memref argument occupies a host buffer slot like any other
// memref, so the ranked argument after it is buffer 1.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<*xbf16>, %arg1: memref<64xi32>) {
      // CHECK-DAG: %[[AP0:.*]] = arith.constant 0 : i32
      // CHECK: aiex.npu.address_patch(%[[AP0]] : i32) {addr = 118788 : ui32, arg_idx = 1 : i32}
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg1 : memref<64xi32>) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
    }
  }
}
