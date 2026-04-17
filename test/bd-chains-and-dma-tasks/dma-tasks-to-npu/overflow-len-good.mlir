//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --aie-dma-tasks-to-npu %s

// This test verifies that a dma_bd with a transfer length that would overflow a
// 32-bit integer when converted to bytes is handled correctly. Without the fix,
// `len * element_size_bytes` would overflow to 0 or a wrong value, causing a
// false "Buffer descriptor length does not match" error.
//
// memref<536870912xi64>: 536870912 elements * 8 bytes = 4294967296 bytes.
// In 32-bit words: 4294967296 / 4 = 1073741824 (= 0x40000000).
// Without overflow fix, the byte count truncates to 0, so len_addr_granularity
// would be 0 instead of 1073741824, triggering a spurious validation error.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<536870912xi64>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<536870912xi64>, 0, 536870912,
                     [<size=1, stride=1>, <size=1, stride=1>, <size=536870912, stride=1>]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}
