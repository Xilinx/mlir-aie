//===- dma_to_npu_dynamic_strides.mlir - Dynamic sizes/strides ---*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Tests that the DMA-to-NPU lowering correctly handles dynamic (SSA) sizes
// and strides in npu.dma_memcpy_nd operations. The dynamic path emits a BD
// template blockwrite plus npu.write32 word overrides for the dynamic fields,
// instead of a fully-static blockwrite.

// RUN: aie-opt --split-input-file -aie-dma-to-npu --verify-diagnostics %s | FileCheck %s

// Only the repeat dimension (dim[0], outermost) is dynamic, so d0/d1/d2 sizes
// and strides are all constant and bake into the blockwrite template; only the
// queue-push repeat_count is computed at runtime. This pins the bf16 d0_size
// arithmetic at the byte level: for bf16, d0_size = inSize0 * elemWidth /
// addrGran = 32 * 16 / 32 = 16 (NOT 32 * (16/32) = 0). 16 lands in the high
// 10 bits of BD word[3]: 16 << 20 = 16777216. word[0] is buffer_length =
// d0_size * d1_size * d2_size = 16 * 1 * 32 = 512.
// CHECK-LABEL: module
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<8xi32>
// CHECK-SAME: = dense<[512, 0, 0, 16777216, {{.*}}, {{.*}}, {{.*}}, {{.*}}]>
// CHECK: aiex.npu.blockwrite
// CHECK: aiex.npu.address_patch
// CHECK-SAME: arg_idx = 0 : i32
module  {
  aie.device(npu2) {
    aie.runtime_sequence(%arg0: memref<16384xbf16>, %arg1: i32) {
      // Dynamic repeat count in dim[0]: %arg1 is an SSA value
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c32 = arith.constant 32 : i64
      %dim0 = arith.extui %arg1 : i32 to i64
      aiex.npu.dma_memcpy_nd (%arg0[%c0, %c0, %c0, %c0][%dim0, %c1, %c32, %c32][%c0, %c32, %c32, %c1]) { metadata = @toMem, id = 1 : i64 } : memref<16384xbf16>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @toMem (%tile_0_0, S2MM, 0)
  }
}

// -----

// Dynamic INNER size (dim[3], d0): exercises the runtime d0_size arith chain
// and the word[3] override. The bf16 scaling (sizeBytes = inSize * 16 / 32)
// must be emitted as multiply-then-divide so a sub-word element does not
// truncate to zero. The runtime value is then masked into the 10-bit d0_size
// field (0x3FF << 20). Because the inner size is dynamic, the 10-bit limit
// cannot be checked at compile time and a warning is emitted.
// CHECK-LABEL: module
// d0_size scaling: inSize * elemWidth(16) / addrGran(32), multiply then divide
// CHECK: %[[EW:.*]] = arith.constant 16 : i32
// CHECK: %[[SCALED:.*]] = arith.muli %{{.*}}, %[[EW]] : i32
// CHECK: %[[GRAN:.*]] = arith.constant 32 : i32
// CHECK: %[[D0:.*]] = arith.divui %[[SCALED]], %[[GRAN]] : i32
// word[3] override: d0_size masked to 10 bits and shifted left 20
// CHECK: %[[M:.*]] = arith.constant 1023 : i32
// CHECK: %[[MASKED:.*]] = arith.andi %[[D0]], %[[M]] : i32
// CHECK: %[[SH:.*]] = arith.constant 20 : i32
// CHECK: arith.shli %[[MASKED]], %[[SH]] : i32
// CHECK: aiex.npu.write32
module  {
  aie.device(npu2) {
    aie.runtime_sequence(%arg0: memref<16384xbf16>, %arg1: i32) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c32 = arith.constant 32 : i64
      %dim = arith.extui %arg1 : i32 to i64
      // bf16 with dynamic inner dim (d0)
      // expected-warning@+1 {{dynamic inner size (d0) cannot be checked against the 10-bit hardware limit}}
      aiex.npu.dma_memcpy_nd (%arg0[%c0, %c0, %c0, %c0][%c1, %c1, %c32, %dim][%c0, %c32, %c32, %c1]) { metadata = @inA, id = 2 : i64 } : memref<16384xbf16>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @inA (%tile_0_0, S2MM, 0)
  }
}
