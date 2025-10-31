//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 AMD Inc.

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aiex.runtime_sequence(%arg0: memref<32xi8>, %arg1: memref<32xi8>) {
      // CHECK: writebd {{.*}} packet_id = 2 : i32, packet_type = 1 : i32
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0, <pkt_type = 1, pkt_id = 1>) {
          aie.dma_bd(%arg0 : memref<32xi8>, 4, 16) {bd_id = 0 : i32, packet = #aie.packet_info<pkt_type = 1, pkt_id = 2>}
          aie.end
      }
    }
  }
}
