//===- out_of_order_shared_channel_test.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Out-of-order fifos share one S2MM channel. A channel whose fifos take turns
// on a BD chain holds their order and so serves one fifo; an out-of-order one
// holds none, because each object names the descriptor that takes it.
//
// The fifos take disjoint runs of bd ids, and each sender names the ids its own
// receiver holds. Both fifos here receive on channel 1 of tile (0, 5).

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate --aie-objectfifo-lower-cores --aie-objectfifo-lower-dmas --aie-assign-lock-ids --aie-assign-buffer-addresses --aie-assign-bd-ids %s | FileCheck %s

// CHECK-LABEL: @shared_s2mm
// @a receives into the first run of ids, and its sender names them.
// CHECK:       aie.dma_bd(%a_buff_0{{.*}}out_of_order_id = 0 : i32}
// CHECK:       aie.dma_bd(%a_buff_1{{.*}}out_of_order_id = 1 : i32}
// CHECK:       aie.dma_start(S2MM, 1, {{.*}}) {out_of_order}
// CHECK:       aie.dma_bd(%a_cons_buff_0{{.*}}bd_id = 0 : i32
// CHECK:       aie.dma_bd(%a_cons_buff_1{{.*}}bd_id = 1 : i32
// @b shares the channel and takes the next run.
// CHECK:       aie.dma_start(S2MM, 1, {{.*}}) {out_of_order}
// CHECK:       aie.dma_bd(%b_cons_buff_0{{.*}}bd_id = 2 : i32
// CHECK:       aie.dma_bd(%b_cons_buff_1{{.*}}bd_id = 3 : i32
// CHECK:       aie.dma_bd(%b_buff_0{{.*}}out_of_order_id = 2 : i32}
// CHECK:       aie.dma_bd(%b_buff_1{{.*}}out_of_order_id = 3 : i32}
module @shared_s2mm {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @a(%tile_0_2, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>>, out_of_order, iter_count = 1 : i32, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%tile_0_3, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 4>>, out_of_order, iter_count = 1 : i32, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %1 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %1 = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      aie.end
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %1 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      %2 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}
