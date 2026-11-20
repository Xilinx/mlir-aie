//===- out_of_order_join_test.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Two cores' objects joined on a third core share its one S2MM channel out of
// order. Each receiving descriptor fills one segment of a joined buffer, and
// each segment's receiver takes its own run of tags, so a packet names the
// segment of the buffer it lands in. Freeing a joined object frees a
// descriptor per segment, so the core counts two objects a release toward
// replenishing the channel.

// RUN: aie-opt --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate --aie-objectfifo-insert-core-queue-pushes --aie-objectfifo-lower-cores --aie-objectfifo-lower-dmas --aie-assign-lock-ids --aie-assign-buffer-addresses --aie-assign-bd-ids %s | FileCheck %s

// CHECK-LABEL: @join_shared_s2mm
// CHECK:       %core_2_5 = aie.core(%tile_2_5) {
// CHECK:         aie.use_lock(%j__conslock0, AcquireGreaterEqual
// CHECK:         aie.use_lock(%j__conslock1, AcquireGreaterEqual
// CHECK:         aie.use_lock(%j__prodlock1, Release
// CHECK:         %[[TWO:.*]] = arith.constant 2 : i32
// CHECK:         arith.subi %{{.*}}, %[[TWO]]
// CHECK:         aie.dma_push_queue(S2MM : 0)
// @a's sender names the first segment's descriptors.
// CHECK:       aie.dma_bd(%a__buf0{{.*}}out_of_order_id = [[A0:[0-9]+]] : i32}
// CHECK:       aie.dma_bd(%a__buf1{{.*}}out_of_order_id = [[A1:[0-9]+]] : i32}
// One out-of-order channel takes both segments of each buffer.
// CHECK:       aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 255) {out_of_order}
// CHECK:       aie.dma_bd(%j__buf0 : memref<32xi32> offset = 0 len = 16) {bd_id = [[A0]] : i32
// CHECK:       aie.dma_bd(%j__buf1 : memref<32xi32> offset = 0 len = 16) {bd_id = [[A1]] : i32
// CHECK:       aie.dma_bd(%j__buf0 : memref<32xi32> offset = 16 len = 16) {bd_id = [[B0:[0-9]+]] : i32
// CHECK:       aie.dma_bd(%j__buf1 : memref<32xi32> offset = 16 len = 16) {bd_id = [[B1:[0-9]+]] : i32
// CHECK-NOT:   aie.dma_start(S2MM
// @b's sender names the second segment's.
// CHECK:       aie.dma_bd(%b__buf0{{.*}}out_of_order_id = [[B0]] : i32}
// CHECK:       aie.dma_bd(%b__buf1{{.*}}out_of_order_id = [[B1]] : i32}
module @join_shared_s2mm {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_5 = aie.tile(2, 5)
    aie.objectfifo @a(%tile_0_2, {%tile_2_5}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%tile_1_3, {%tile_2_5}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @j(%tile_2_5, {%tile_2_5}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.allocate @j(%tile_2_5)
    aie.objectfifo.link [@a, @b] -> [@j]([0, 16] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %1 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      aie.end
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %1 = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      aie.end
    }
    %core_2_5 = aie.core(%tile_2_5) {
      %1 = aie.objectfifo.acquire @j(Consume, 1) : memref<32xi32>
      aie.objectfifo.release @j(Consume, 1)
      aie.end
    }
  }
}

