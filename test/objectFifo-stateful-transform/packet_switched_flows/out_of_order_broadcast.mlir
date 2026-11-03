// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate --aie-objectfifo-insert-core-queue-pushes --aie-objectfifo-lower-dmas --aie-assign-bd-ids %s | FileCheck %s

// The second destination also receives another FIFO, so independent ID
// allocation would give the broadcast different IDs on its two destinations.
// CHECK-DAG: aie.dma_bd(%a__buf0{{.*}}out_of_order_id = [[PING:[0-9]+]] : i32}
// CHECK-DAG: aie.dma_bd(%a__buf1{{.*}}out_of_order_id = [[PONG:[0-9]+]] : i32}
// CHECK-DAG: aie.dma_bd(%a__cons0__buf0{{.*}}bd_id = [[PING]] : i32
// CHECK-DAG: aie.dma_bd(%a__cons0__buf1{{.*}}bd_id = [[PONG]] : i32
// CHECK-DAG: aie.dma_bd(%a__cons1__buf0{{.*}}bd_id = [[PING]] : i32
// CHECK-DAG: aie.dma_bd(%a__cons1__buf1{{.*}}bd_id = [[PONG]] : i32
module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    aie.objectfifo @b(%tile_1_2, {%tile_1_4}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a(%tile_0_2, {%tile_0_4, %tile_1_4}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>, #aie.end_port<DMA : 0>], transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      aie.end
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %0 = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
    %core_1_4 = aie.core(%tile_1_4) {
      %0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      %1 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}
