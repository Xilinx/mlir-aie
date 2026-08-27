//===- reserve_runtime_bd_ids.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A tile's bd_ids are one shared pool. aie-reserve-runtime-bd-ids records the
// bd_id the runtime dma_configure_task pins (0), so the static allocator's
// auto-assignment steps over it instead of reusing the same slot.

// RUN: aie-opt --aie-reserve-runtime-bd-ids --aie-assign-bd-ids %s | FileCheck %s

// CHECK: aie.shim_dma
// CHECK: aie.dma_bd(%{{.*}} : memref<8xi32>) {bd_id = 1 : i32}
aie.device(npu2) {
  %t = aie.tile(0, 0)
  %buf = aie.external_buffer {sym_name = "b"} : memref<8xi32>
  aie.shim_dma(%t) {
    %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
  ^bd:
    aie.dma_bd(%buf : memref<8xi32>)
    aie.next_bd ^end
  ^end:
    aie.end
  }
  aie.runtime_sequence(%a: memref<8xi32>) {
    %task = aiex.dma_configure_task(%t, S2MM, 1) {
      aie.dma_bd(%a : memref<8xi32>) {bd_id = 0 : i32}
      aie.end
    }
  }
}
