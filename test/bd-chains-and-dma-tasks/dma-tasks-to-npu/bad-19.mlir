// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 AMD Inc.

// RUN: not aie-opt --aie-dma-tasks-to-npu %s 2>&1 | FileCheck %s

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aiex.runtime_sequence(%arg0: memref<8xi16>) {
      // CHECK: Packet Type exceeds the maximum supported by 3 bits.
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 7 : i32, packet = #aie.packet_info<pkt_type = 8, pkt_id = 2>}
        aie.end
      } {issue_token = true}
    }
  }
}
