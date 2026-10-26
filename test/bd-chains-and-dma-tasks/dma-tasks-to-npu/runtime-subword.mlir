//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// A runtime length and offset on int8 data: the BD counts 4-byte granules,
// so the length must be whole granules and the patched address 4-byte
// aligned. Static values get the same checks from the dma_bd verifier.

// CHECK-LABEL: aie.runtime_sequence
// CHECK: %[[LEN:.*]] = arith.extui %arg1 : i32 to i64
// CHECK: %[[REM:.*]] = arith.remui %[[LEN]], %c4_i64 : i64
// CHECK: %[[WHOLE:.*]] = arith.cmpi eq, %[[REM]], %{{.*}} : i64
// CHECK: cf.assert %[[WHOLE]], "a runtime DMA length must be a multiple of 4 elements (whole 4-byte granules)"
// CHECK: aiex.npu.blockwrite_values
// CHECK: %[[OFF:.*]] = arith.extui %arg2 : i32 to i64
// CHECK: %[[LOW:.*]] = arith.andi %[[OFF]], %c3_i64 : i64
// CHECK: %[[ALIGNED:.*]] = arith.cmpi eq, %[[LOW]], %{{.*}} : i64
// CHECK: cf.assert %[[ALIGNED]], "a runtime DMA offset is not 4-byte aligned"
// CHECK: aiex.npu.address_patch(%[[OFF]] : i64)

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<4096xi8>, %len: i32, %off: i32) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<4096xi8> offset = %off len = %len sizes = [1, 1, 16, 64] strides = [0, 0, 64, 1]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}
