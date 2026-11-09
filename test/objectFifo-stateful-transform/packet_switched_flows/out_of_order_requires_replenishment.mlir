// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectFifo-stateful-transform --split-input-file --verify-diagnostics %s | FileCheck %s

module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_4 = aie.tile(0, 4)
    // expected-error@+1 {{unbounded out-of-order receiving channel requires a local core consumer release}}
    aie.objectfifo @a(%tile_0_2, {%mem_tile_0_1}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%mem_tile_0_1, {%tile_0_4}, 2 : i32) {transport = #aie.transport<dma>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a] -> [@b]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %0 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}


// -----

// CHECK-LABEL: module @bounded_memtile
// CHECK-NOT: aie.dma_push_queue
// CHECK: aie.dma_start(S2MM, {{.*}}repeat_count = 1) {out_of_order}
module @bounded_memtile {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_4 = aie.tile(0, 4)
    aie.objectfifo @a(%tile_0_2, {%mem_tile_0_1}, 2 : i32) {iter_count = 1 : i32, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%mem_tile_0_1, {%tile_0_4}, 2 : i32) {iter_count = 1 : i32, transport = #aie.transport<dma>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a] -> [@b]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      %1 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %0 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      %1 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}
