//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-materialize-bd-chains %s | FileCheck %s
// XFAIL: *
// Lock references remain symbols when the chain is materialized.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.lock @lock_0(%tile_0_0, 0)
    aie.lock @lock_1(%tile_0_0, 1)
    aie.lock @lock_2(%tile_0_0, 2)

    aie.bd_chain @simple_chain(%buf: memref<8xi16>) {
            %c1_ul1 = arith.constant 1 : i32
            aie.use_lock(@lock_0, "Acquire", %c1_ul1)
            aie.dma_bd(%buf : memref<8xi16> offset = 0 len = 8)
            %c1_ul2 = arith.constant 1 : i32
            aie.use_lock(@lock_1, "Release", %c1_ul2)
            aie.next_bd ^bd1
        ^bd1:
            %c1_ul3 = arith.constant 1 : i32
            aie.use_lock(@lock_1, "Acquire", %c1_ul3)
            aie.dma_bd(%buf : memref<8xi16> offset = 0 len = 8)
            %c1_ul4 = arith.constant 1 : i32
            aie.use_lock(@lock_2, "Release", %c1_ul4)
            aie.next_bd ^bd2
        ^bd2:
            %c1_ul5 = arith.constant 1 : i32
            aie.use_lock(@lock_2, "Acquire", %c1_ul5)
            aie.dma_bd(%buf : memref<8xi16> offset = 0 len = 8)
            %c1_ul6 = arith.constant 1 : i32
            aie.use_lock(@lock_0, "Release", %c1_ul6)
            aie.end
    }

    aie.runtime_sequence(%buf: memref<8xi16>) {
      %t1 = aiex.dma_start_bd_chain @simple_chain(%buf) : (memref<8xi16>)
                                    on (%tile_0_0, MM2S, 0)
      // CHECK: %[[task1:.+]] = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      // CHECK:   aie.use_lock(@lock_0, "Acquire", %{{.*}})
      // CHECK:   aie.dma_bd(%buf : memref<8xi16> offset = {{.*}} len = {{.*}})
      // CHECK:   aie.use_lock(@lock_1, "Release", %{{.*}})
      // CHECK:   aie.next_bd ^bb1
      // CHECK: ^bb1:
      // CHECK:   aie.use_lock(@lock_1, "Acquire", %{{.*}})
      // CHECK:   aie.dma_bd(%buf : memref<12xi16> offset = {{.*}} len = {{.*}})
      // CHECK:   aie.use_lock(@lock_2, "Release", %{{.*}})
      // CHECK:   aie.next_bd ^bb2
      // CHECK: ^bb2:
      // CHECK:   aie.use_lock(@lock_2, "Acquire", %{{.*}})
      // CHECK:   aie.dma_bd(%buf : memref<8xi16> offset = {{.*}} len = {{.*}})
      // CHECK:   aie.use_lock(@lock_1, "Release", %{{.*}})
      // CHECK:   aie.end
      // CHECK: }
      // CHECK: aiex.dma_start_task(%[[task1]])
      %t2 = aiex.dma_start_bd_chain @simple_chain(%buf) : (memref<8xi16>)
                                    on (%tile_0_0, MM2S, 1)
      // CHECK: %[[task2:.+]] = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      // CHECK:   aie.use_lock(@lock_0, "Acquire", %{{.*}})
      // CHECK:   aie.dma_bd(%buf : memref<8xi16> offset = {{.*}} len = {{.*}})
      // CHECK:   aie.use_lock(@lock_0, "Release", %{{.*}})
      // CHECK:   aie.next_bd ^bb1
      // CHECK: ^bb1:
      // CHECK:   aie.use_lock(@lock_0, "Acquire", %{{.*}})
      // CHECK:   aie.dma_bd(%buf : memref<12xi16> offset = {{.*}} len = {{.*}})
      // CHECK:   aie.use_lock(@lock_0, "Release", %{{.*}})
      // CHECK:   aie.next_bd ^bb2
      // CHECK: ^bb2:
      // CHECK:   aie.use_lock(@lock_0, "Acquire", %{{.*}})
      // CHECK:   aie.dma_bd(%buf : memref<8xi16> offset = {{.*}} len = {{.*}})
      // CHECK:   aie.use_lock(@lock_0, "Release", %{{.*}})
      // CHECK:   aie.end
      // CHECK: }
      // CHECK: aiex.dma_start_task(%[[task2]])
      %t3 = aiex.dma_start_bd_chain @simple_chain(%buf) : (memref<8xi16>)
                                    on (%tile_0_0, S2MM, 0)
      // CHECK: %[[task3:.+]] = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      // CHECK:   aie.use_lock(@lock_2, "Acquire", %{{.*}})
      // CHECK:   aie.dma_bd(%buf : memref<8xi16> offset = {{.*}} len = {{.*}})
      // CHECK:   aie.use_lock(@lock_1, "Release", %{{.*}})
      // CHECK:   aie.next_bd ^bb1
      // CHECK: ^bb1:
      // CHECK:   aie.use_lock(@lock_1, "Acquire", %{{.*}})
      // CHECK:   aie.dma_bd(%buf : memref<12xi16> offset = {{.*}} len = {{.*}})
      // CHECK:   aie.use_lock(@lock_0, "Release", %{{.*}})
      // CHECK:   aie.next_bd ^bb2
      // CHECK: ^bb2:
      // CHECK:   aie.use_lock(@lock_0, "Acquire", %{{.*}})
      // CHECK:   aie.dma_bd(%buf : memref<8xi16> offset = {{.*}} len = {{.*}})
      // CHECK:   aie.use_lock(@lock_2, "Release", %{{.*}})
      // CHECK:   aie.end
      // CHECK: }
      // CHECK: aiex.dma_start_task(%[[task3]])
      aiex.dma_await_task(%t1)
    }
  }
}
