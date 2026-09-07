//===- reserve_runtime_bd_ids_nonzero.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The runtime pins a non-zero bd_id (2). The static allocator must reserve that
// exact slot and still fill the lower ids.

// RUN: aie-opt --aie-reserve-runtime-bd-ids --aie-assign-bd-ids %s | FileCheck %s

// CHECK: aie.shim_dma
// CHECK: aie.dma_bd({{.*}}) {bd_id = 0 : i32
// CHECK: aie.dma_bd({{.*}}) {bd_id = 1 : i32
// CHECK: aie.dma_bd({{.*}}) {bd_id = 3 : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  %buf = aie.external_buffer {sym_name = "b"} : memref<8xi32>
  aie.shim_dma(%t) {
    %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
  ^bd0:
    aie.dma_bd(%buf : memref<8xi32>)
    aie.next_bd ^bd1
  ^bd1:
    aie.dma_bd(%buf : memref<8xi32>)
    aie.next_bd ^bd2
  ^bd2:
    aie.dma_bd(%buf : memref<8xi32>)
    aie.next_bd ^end
  ^end:
    aie.end
  }
  aie.runtime_sequence(%a: memref<8xi32>) {
    %task = aiex.dma_configure_task(%t, S2MM, 1) {
      aie.dma_bd(%a : memref<8xi32>) {bd_id = 2 : i32}
      aie.end
    }
  }
}
