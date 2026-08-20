//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// A runtime DMA task BD that stamps an out_of_order_id must carry that id into
// the emitted npu.writebd. It used to be dropped to 0, sending every packet to
// slot 0. The contrast BD, with no out_of_order_id, must still emit 0.

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<8xi16>, %arg1: memref<8xi16>) {
      // CHECK: aiex.npu.writebd {{.*}}bd_id = 3 : i32{{.*}}out_of_order_id = 5 : i32
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8) {bd_id = 3 : i32, out_of_order_id = 5 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.end
      }
      aiex.dma_start_task(%t1)

      // CHECK: aiex.npu.writebd {{.*}}bd_id = 4 : i32{{.*}}out_of_order_id = 0 : i32
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg1 : memref<8xi16> offset = 0 len = 8) {bd_id = 4 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.end
      }
      aiex.dma_start_task(%t2)
    }
  }
}
