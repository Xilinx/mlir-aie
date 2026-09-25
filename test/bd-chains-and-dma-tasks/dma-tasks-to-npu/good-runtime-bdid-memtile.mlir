//===- good-runtime-bdid-memtile.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-prepare-buffers --aie-assign-buffer-addresses --aie-dma-tasks-to-npu %s | FileCheck %s

// The mem tile counterpart of good-runtime-bdid.mlir. A runtime bd_id makes
// the BD register block address runtime (0xA0000 + bd_id*32 here, against the
// shim's 0x1D000 base), so the whole block is packed into one
// npu.blockwrite_values at that runtime base.
//
// What this test exists to pin is the SECOND consumer of that runtime base.
// The buffer here is a local aie.buffer, so the pointer is written by a
// maskwrite32 into word 1 rather than by an address_patch -- and that
// maskwrite's own register address is equally runtime. Deriving it from the
// same base is what keeps the address landing in the BD that was just
// configured; a constant there would silently patch whichever BD the literal
// happened to name.

// CHECK-LABEL: @runtime_bdid_memtile
// CHECK: %[[POP:.*]] = aiex.dma_bd_pool_pop(0, 1, 0) : i32
// The register block base, 0x1A0000 + bd_id*32, carrying all 8 mem tile words.
// CHECK: %[[MUL:.*]] = arith.muli %[[POP]], %{{.*}} : i32
// CHECK: %[[BASE:.*]] = arith.addi %{{.*}}, %[[MUL]] : i32
// CHECK: aiex.npu.blockwrite_values(%[[BASE]] : i32) values %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i32, i32, i32, i32, i32, i32, i32, i32
// The address word sits at base + 4 (getDmaBdAddressOffset on a mem tile), and
// is derived from the SAME runtime bd_id -- a constant here would patch
// whichever BD the literal named rather than the one just configured.
// CHECK: %[[MUL2:.*]] = arith.muli %[[POP]], %{{.*}} : i32
// CHECK: %[[BASE2:.*]] = arith.addi %{{.*}}, %[[MUL2]] : i32
// CHECK: %[[ADDR:.*]] = arith.addi %[[BASE2]], %{{.*}} : i32
// CHECK: aiex.npu.maskwrite32(%[[ADDR]], %{{.*}}, %{{.*}}) : i32, i32, i32
// CHECK-NOT: aiex.npu.address_patch
// CHECK: aiex.npu.push_queue(0, 1, MM2S : 0) bd_id %[[POP]]

aie.device(npu2) {
  %tile_0_1 = aie.tile(0, 1)
  %buf = aie.buffer(%tile_0_1) : memref<1024xi32>
  aie.runtime_sequence @runtime_bdid_memtile() {
    %bd = aiex.dma_bd_pool_pop(0, 1, 0) : i32
    %t = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
      aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256) bd_id_val %bd : i32
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t)
    aiex.dma_await_task(%t)
    aiex.dma_bd_pool_push(0, 1, 0) bd_id %bd : i32
  }
}
