// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %c02 = aie.tile(0, 2)
    %c03 = aie.tile(0, 3)
    %s1 = aie.tile(1, 0)
    %c12 = aie.tile(1, 2)
    aie.flow(%s0, DMA : 1, %c02, DMA : 0)
    aie.flow(%s1, DMA : 0, %c12, DMA : 0)
    aie.flow(%s0, DMA : 0, %c03, DMA : 0)
  }
}
