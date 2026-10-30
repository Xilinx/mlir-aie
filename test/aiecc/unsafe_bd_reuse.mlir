// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %aiecc -n --tmpdir %t.default --get-npu-insts --npu-insts-name=%t.default.bin %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %aiecc -n --tmpdir %t.disabled --get-npu-insts --npu-insts-name=%t.disabled.bin --warn-unsafe-bd-reuse=false %s 2>&1 | FileCheck %s --check-prefix=DISABLED --allow-empty
//
// ENABLED: warning: {{.*}}reuses buffer descriptor ID 0
// ENABLED: note: released here
// DISABLED-NOT: warning:

module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    aie.runtime_sequence @unsafe(%buf: memref<512xi32>) {
      %first = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%buf : memref<512xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%first)
      aiex.dma_free_task(%first)
      %second = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%buf : memref<512xi32> offset = 256 len = 256)
        aie.end
      }
      aiex.dma_start_task(%second)
    }
  }
}
