//===- test_place_implicit_push_queue.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two fifos pin MM2S 0 of one core, so the split gives both push_queue and
// repeats @a in hardware. Both placers run the split on a copy for the shim
// gates and plan each end from plannedDmaEnd, which must agree with it.

// RUN: aie-opt --aie-place-tiles='placer=sequential_placer' %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=1' %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=2' %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=3' %s | FileCheck %s

// CHECK-LABEL: module @implicit_push_queue
// CHECK: aie.objectfifo @a(%[[SRC:[a-z_0-9]+]],
// CHECK: aie.objectfifo @b(%[[SRC]],
module @implicit_push_queue {
  aie.device(npu2) {
    %src = aie.logical_tile<CoreTile>(?, ?)
    %d1 = aie.logical_tile<CoreTile>(?, ?)
    %d2 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%src, {%d1}, [2, 2]) {repeat_count = 8 : i32, transport = #aie.transport<dma, packet = #aie.packet_info<>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%src, {%d2}, [2, 2]) {transport = #aie.transport<dma, packet = #aie.packet_info<>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    %producer = aie.core(%src) {
      %a0 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      %b0 = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      aie.end
    }
    aie.core(%d1) {
      %a0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
    aie.core(%d2) {
      %b0 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}
