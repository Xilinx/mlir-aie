//===- out_of_order_two_channels_test.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Out-of-order fifos receiving on two channels of one tile. An out-of-order
// channel takes the buffer its packet names by bd id, and the hardware checks
// no further, so an id names the same descriptor whichever channel it arrives
// on. The ids therefore have to be free across the tile, not just the channel.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate --aie-objectfifo-lower-cores --aie-objectfifo-lower-dmas --aie-assign-lock-ids --aie-assign-buffer-addresses --aie-assign-bd-ids %s | FileCheck %s

// CHECK-LABEL: @two_channels
// @a receives on channel 0.
// CHECK:       aie.dma_bd(%a_buff_0{{.*}}out_of_order_id = 0 : i32}
// CHECK:       aie.dma_bd(%a_buff_1{{.*}}out_of_order_id = 1 : i32}
// CHECK:       aie.dma_bd(%a_cons_buff_0{{.*}}bd_id = 0 : i32
// CHECK:       aie.dma_bd(%a_cons_buff_1{{.*}}bd_id = 1 : i32
// @b receives on channel 1 of the same tile, and takes the next run of ids.
// CHECK:       aie.dma_bd(%b_cons_buff_0{{.*}}bd_id = 2 : i32
// CHECK:       aie.dma_bd(%b_cons_buff_1{{.*}}bd_id = 3 : i32
// CHECK:       aie.dma_bd(%b_buff_0{{.*}}out_of_order_id = 2 : i32}
// CHECK:       aie.dma_bd(%b_buff_1{{.*}}out_of_order_id = 3 : i32}
module @two_channels {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @a(%tile_0_2, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>, out_of_order = true>, iter_count = 1 : i32, cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%tile_0_3, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 4>, out_of_order = true>, iter_count = 1 : i32, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
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
