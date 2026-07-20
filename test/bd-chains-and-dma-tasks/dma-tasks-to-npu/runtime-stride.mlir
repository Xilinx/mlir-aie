//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// A runtime INNERMOST stride on the dma_task path (int8) is supported (no
// compile-time constant-1 rule), guarded with the unit-stride exemption:
// realizable iff stride == 1 (contiguous) or granule-aligned. Parity with the
// dma_memcpy_nd path.

// CHECK-LABEL: aie.runtime_sequence
// CHECK: aiex.npu.writebd
// CHECK: aiex.npu.assert_bd_divisible(%{{.*}}) {allow_unit, divisor = 4 : i32}

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<4096xi8>, %len: i32, %s: i64) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<4096xi8> offset = 0 len = %len sizes = [1, 8, 16, 4] strides = [4096, 512, 4, %s]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}
