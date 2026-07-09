//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-materialize-bd-chains %s | FileCheck %s

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aie.shim_dma_allocation @alloc0 (%tile_0_0, MM2S, 0)

    aie.bd_chain @simple_chain(%arg0: memref<8xi16>, %arg1: memref<12xi16>) {
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c12_i32 = arith.constant 12 : i32
            aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32) {packet = #aie.packet_info<pkt_type = 1, pkt_id = 1>}
            aie.next_bd ^bd1
        ^bd1:
            aie.dma_bd(%arg1 : memref<12xi16> offset = %c0_i32 len = %c12_i32) {packet = #aie.packet_info<pkt_type = 1, pkt_id = 2>}
            aie.end
    }

    aie.runtime_sequence(%arg0: memref<8xi16>, %arg1: memref<12xi16>) {
      %t1 = aiex.dma_start_bd_chain_for @simple_chain(%arg0, %arg1) : (memref<8xi16>, memref<12xi16>)
                                        for @alloc0
      // CHECK: %[[task1:.+]] = aiex.dma_configure_task(%{{.*}}tile_0_0, MM2S, 0) {
      // CHECK:   aie.dma_bd(%arg0 : memref<8xi16> offset = {{.*}} len = {{.*}}) {packet = #aie.packet_info<pkt_type = 1, pkt_id = 1>}
      // CHECK:   aie.next_bd ^bb1
      // CHECK: ^bb1:
      // CHECK:   aie.dma_bd(%arg1 : memref<12xi16> offset = {{.*}} len = {{.*}}) {packet = #aie.packet_info<pkt_type = 1, pkt_id = 2>}
      // CHECK:   aie.end
      // CHECK: }
      // CHECK: aiex.dma_start_task(%[[task1]])
      aiex.dma_await_task(%t1)
    }
  }
}
