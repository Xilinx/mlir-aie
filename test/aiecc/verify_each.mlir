//===- verify_each.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aiecc verifies what each pass pipeline produces; --verify-each verifies after
// every pass instead, and the outputs are the same.

// RUN: rm -rf %t.d && mkdir -p %t.d/once %t.d/each
// RUN: cd %t.d/once && %aiecc --get-npu-insts --get-pdi %s
// RUN: cd %t.d/each && %aiecc --verify-each --get-npu-insts --get-pdi %s
// RUN: cmp %t.d/once/insts_main_seq.bin %t.d/each/insts_main_seq.bin
// RUN: cmp %t.d/once/main.pdi %t.d/each/main.pdi

module {
  aie.device(npu2) @main {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    aie.flow(%shim, DMA : 0, %mem, DMA : 0)
    aie.runtime_sequence @seq(%buf : memref<64xi32>) {
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
    aie.shim_dma_allocation @in (%shim, MM2S, 0)
  }
}
