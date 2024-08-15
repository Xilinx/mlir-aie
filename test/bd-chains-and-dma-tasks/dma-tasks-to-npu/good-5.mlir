//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// REQUIRES: ryzen_ai
//
// RUN: aie-opt --aie-assign-buffer-addresses --aie-dma-tasks-to-npu %s | FileCheck %s

// This test ensures that a chained buffer descriptor configuration in the runtime
// sequence gets lowered to the correct NPU instruction sequence register write
// instructions to set all BDs addresses to their correct value when combined with
// the automatic buffer address allocation.

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    // CHECK: %{{.*}} = aie.buffer(%tile_0_1) {address = [[ADDR1:[0-9]+]] {{.*}}}
    %buf0 = aie.buffer(%tile_0_1) : memref<32xi8> 
    // CHECK: %{{.*}} = aie.buffer(%tile_0_1) {address = [[ADDR2:[0-9]+]] {{.*}}}
    %buf1 = aie.buffer(%tile_0_1) : memref<32xi8> 

    aiex.runtime_sequence(%arg0: memref<32xi8>) {
      %t1 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
          // CHECK: aiex.npu.write32 {address = 1167364 : ui32, value = [[ADDR1]] : ui32}
          aie.dma_bd(%buf0 : memref<32xi8>, 4, 16) {bd_id = 0 : i32}
          aie.next_bd ^bd2
        ^bd2:
          // CHECK: aiex.npu.write32 {address = 1167396 : ui32, value = [[ADDR2]] : ui32}
          aie.dma_bd(%buf1 : memref<32xi8>, 4, 16) {bd_id = 1 : i32}
          aie.end
      }
    }
  }
}

