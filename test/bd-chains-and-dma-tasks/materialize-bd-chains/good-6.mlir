//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --aie-materialize-bd-chains %s | FileCheck %s

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aie.shim_dma_allocation @alloc0 (MM2S, 0, 0)

    aie.bd_chain @simple_chain(%arg0: memref<8xi16>, %arg1: memref<12xi16>) {
            aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {packet = #aie.packet_info<pkt_type = 1, pkt_id = 1>}
            aie.next_bd ^bd1
        ^bd1:
            aie.dma_bd(%arg1 : memref<12xi16>, 0, 12) {packet = #aie.packet_info<pkt_type = 1, pkt_id = 2>}
            aie.end
    }

    aiex.runtime_sequence(%arg0: memref<8xi16>, %arg1: memref<12xi16>) {
      %t1 = aiex.dma_start_bd_chain_for @simple_chain(%arg0, %arg1) : (memref<8xi16>, memref<12xi16>)
                                        for @alloc0
      // CHECK: %[[task1:.+]] = aiex.dma_configure_task(%{{.*}}tile_0_0, MM2S, 0) {
      // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {packet = #aie.packet_info<pkt_type = 1, pkt_id = 1>}
      // CHECK:   aie.next_bd ^bb1
      // CHECK: ^bb1:
      // CHECK:   aie.dma_bd(%arg1 : memref<12xi16>, 0, 12) {packet = #aie.packet_info<pkt_type = 1, pkt_id = 2>}
      // CHECK:   aie.end
      // CHECK: }
      // CHECK: aiex.dma_start_task(%[[task1]])
      aiex.dma_await_task(%t1)
    }
  }
}
