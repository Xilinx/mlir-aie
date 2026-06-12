//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --aie-assign-buffer-addresses --aie-dma-tasks-to-npu %s | FileCheck %s

// This test ensures that a chained buffer descriptor configuration in the runtime
// sequence gets lowered to the correct NPU instruction sequence register write
// instructions to set all BDs addresses to their correct value when combined with
// the automatic buffer address allocation.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    // CHECK: %{{.*}} = aie.buffer(%{{.*}}tile_0_1) {address = [[ADDR1:[0-9]+]] {{.*}}}
    %buf0 = aie.buffer(%tile_0_1) : memref<32xi8>
    // CHECK: %{{.*}} = aie.buffer(%{{.*}}tile_0_1) {address = [[ADDR2:[0-9]+]] {{.*}}}
    %buf1 = aie.buffer(%tile_0_1) : memref<32xi8>

    aie.runtime_sequence(%arg0: memref<32xi8>) {
      %t1 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
          // CHECK-DAG: %[[MMA:.+]] = arith.constant 1703940 : i32
          // CHECK-DAG: %[[MMV:.+]] = arith.constant 1 : i32
          // CHECK-DAG: %[[MMM:.+]] = arith.constant 524287 : i32
          // CHECK: aiex.npu.maskwrite32(%[[MMA]], %[[MMV]], %[[MMM]])
          aie.dma_bd(%buf0 : memref<32xi8>, 4, 16) {bd_id = 0 : i32}
          aie.next_bd ^bd2
        ^bd2:
          // CHECK-DAG: %[[MA0:.+]] = arith.constant 1703972 : i32
          // CHECK-DAG: %[[MV0:.+]] = arith.constant 16385 : i32
          // CHECK-DAG: %[[MM0:.+]] = arith.constant 524287 : i32
          // CHECK: aiex.npu.maskwrite32(%[[MA0]], %[[MV0]], %[[MM0]])
          aie.dma_bd(%buf1 : memref<32xi8>, 4, 16) {bd_id = 1 : i32}
          aie.end
      }
    }
  }
}
