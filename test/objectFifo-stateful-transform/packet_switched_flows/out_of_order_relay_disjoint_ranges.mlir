// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-assign-bd-ids %s | FileCheck %s

// Odd MemTile channels use IDs 24-47; core receiving IDs start at zero.
// CHECK-DAG: aie.dma_bd(%a__cons__buf0{{.*}}out_of_order_id = 24 : i32
// CHECK-DAG: aie.dma_bd(%a__cons__buf1{{.*}}out_of_order_id = 25 : i32
// CHECK-DAG: aie.dma_bd(%b__cons__buf0{{.*}}{bd_id = 24 : i32
// CHECK-DAG: aie.dma_bd(%b__cons__buf1{{.*}}{bd_id = 25 : i32
// CHECK-DAG: aie.dma_bd(%b__cons__buf0{{.*}}out_of_order_id = 0 : i32
// CHECK-DAG: aie.dma_bd(%b__cons__buf1{{.*}}out_of_order_id = 1 : i32
// CHECK-DAG: aie.dma_bd(%c__cons__buf0{{.*}}{bd_id = 0 : i32
// CHECK-DAG: aie.dma_bd(%c__cons__buf1{{.*}}{bd_id = 1 : i32
// Each relay hop selects the next pool independently of its own receiving IDs.
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %m0 = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @a(%shim, {%m0}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 1 : i32, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%m0, {%m1}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 1>], iter_count = 1 : i32, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%m1, {%core}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 1 : i32, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a] -> [@b]([] [])
    aie.objectfifo.link [@b] -> [@c]([] [])
    %consumer = aie.core(%core) {
      %first = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @c(Consume, 1)
      %second = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @c(Consume, 1)
      aie.end
    }
  }
}
