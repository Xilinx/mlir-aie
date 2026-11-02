// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %aiecc -n --tmpdir %t.default --get-npu-insts --npu-insts-name=%t.default.bin %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %aiecc -n --tmpdir %t.disabled --get-npu-insts --npu-insts-name=%t.disabled.bin --warn-unreleased-transfers=false %s 2>&1 | FileCheck %s --check-prefix=DISABLED --allow-empty
// ENABLED: warning: may exit with 1 DMA transfer(s) neither awaited nor freed
// ENABLED: note: first outstanding transfer starts here
// DISABLED-NOT: warning:

module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    aie.runtime_sequence @leak(%buf : memref<16xi32>) {
      %task = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
    }
  }
}
