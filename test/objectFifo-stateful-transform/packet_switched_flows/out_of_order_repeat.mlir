// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate --aie-objectfifo-insert-core-queue-pushes --aie-objectfifo-lower-dmas --aie-assign-bd-ids %s | FileCheck %s

// Repeating a source buffer still advances the destination buffer on each send.
// CHECK: aie.dma_bd(%a__buf0{{.*}}out_of_order_id = 0 : i32}
// CHECK: aie.dma_bd(%a__buf0{{.*}}out_of_order_id = 1 : i32}
// CHECK: aie.dma_bd(%a__buf0{{.*}}out_of_order_id = 0 : i32}
// CHECK: aie.dma_bd(%a__buf1{{.*}}out_of_order_id = 1 : i32}
// CHECK: aie.dma_bd(%a__buf1{{.*}}out_of_order_id = 0 : i32}
// CHECK: aie.dma_bd(%a__buf1{{.*}}out_of_order_id = 1 : i32}
module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)
    aie.objectfifo @a(%tile_0_2, {%tile_0_4}, 2 : i32) {repeat_count = 3 : i32, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      %1 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      %2 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
  }
}
