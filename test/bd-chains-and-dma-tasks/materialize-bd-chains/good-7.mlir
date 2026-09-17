//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-materialize-bd-chains %s | FileCheck %s

// Materialization inlines every `aiex.dma_start_bd_chain` use into a
// `dma_configure_task`, leaving the `aie.bd_chain` symbol def use-empty. The
// pass must erase the now-dead def rather than leak it into every downstream
// stage. The CHECK-NOT is scoped between the device open and the first
// materialized task, exactly where the leaked def would otherwise print.

// CHECK-LABEL: aie.device
// CHECK-NOT: aie.bd_chain
// CHECK: aiex.dma_configure_task
// CHECK: aiex.dma_start_task
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.bd_chain @dead_after_use(%arg0: memref<8xi16>) {
      aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8)
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<8xi16>) {
      %t1 = aiex.dma_start_bd_chain @dead_after_use(%arg0) : (memref<8xi16>)
                                    on (%tile_0_0, MM2S, 0)
      aiex.dma_await_task(%t1)
    }
  }
}
