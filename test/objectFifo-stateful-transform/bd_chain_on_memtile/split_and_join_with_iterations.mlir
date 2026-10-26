//===- bd_chain_on_memtile/split_and_join_with_iterations.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK:     %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4, repeat_count = 7)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%out_prod_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%out_buff_0 : memref<1024xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%out_cons_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(%out_prod_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%out_buff_1 : memref<1024xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%out_cons_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:
// CHECK:       aie.end
// CHECK:     ^bb4:
// CHECK:       %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb8, repeat_count = 7)
// CHECK:     ^bb5:
// CHECK:       aie.use_lock(%out_prod_lock_1, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%out_buff_0 : memref<1024xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%out_cons_lock_1, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb6
// CHECK:     ^bb6:
// CHECK:       aie.use_lock(%out_prod_lock_1, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%out_buff_1 : memref<1024xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%out_cons_lock_1, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb7
// CHECK:     ^bb7:
// CHECK:       aie.end
// CHECK:     ^bb8:
// CHECK:       %2 = aie.dma_start(S2MM, 2, ^bb9, ^bb13)
// CHECK:     ^bb9:
// CHECK:       aie.use_lock(%in_cons_prod_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%in_cons_buff_0 : memref<1024xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%in_cons_cons_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb10
// CHECK:     ^bb10:
// CHECK:       aie.use_lock(%in_cons_prod_lock_1, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%in_cons_buff_0 : memref<1024xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%in_cons_cons_lock_1, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb11
// CHECK:     ^bb11:
// CHECK:       aie.use_lock(%in_cons_prod_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%in_cons_buff_1 : memref<1024xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%in_cons_cons_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb12
// CHECK:     ^bb12:
// CHECK:       aie.use_lock(%in_cons_prod_lock_1, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%in_cons_buff_1 : memref<1024xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%in_cons_cons_lock_1, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb9
// CHECK:     ^bb13:
// CHECK:       %3 = aie.dma_start(MM2S, 0, ^bb14, ^bb17, repeat_count = 7)
// CHECK:     ^bb14:
// CHECK:       aie.use_lock(%in_cons_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%in_cons_buff_0 : memref<1024xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%in_cons_prod_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb15
// CHECK:     ^bb15:
// CHECK:       aie.use_lock(%in_cons_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%in_cons_buff_1 : memref<1024xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%in_cons_prod_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb16
// CHECK:     ^bb16:
// CHECK:       aie.end
// CHECK:     ^bb17:
// CHECK:       %4 = aie.dma_start(MM2S, 1, ^bb18, ^bb21, repeat_count = 7)
// CHECK:     ^bb18:
// CHECK:       aie.use_lock(%in_cons_cons_lock_1, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%in_cons_buff_0 : memref<1024xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%in_cons_prod_lock_1, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb19
// CHECK:     ^bb19:
// CHECK:       aie.use_lock(%in_cons_cons_lock_1, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%in_cons_buff_1 : memref<1024xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%in_cons_prod_lock_1, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb20
// CHECK:     ^bb20:
// CHECK:       aie.end
// CHECK:     ^bb21:
// CHECK:       %5 = aie.dma_start(MM2S, 2, ^bb22, ^bb26)
// CHECK:     ^bb22:
// CHECK:       aie.use_lock(%out_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%out_buff_0 : memref<1024xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%out_prod_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb23
// CHECK:     ^bb23:
// CHECK:       aie.use_lock(%out_cons_lock_1, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%out_buff_0 : memref<1024xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%out_prod_lock_1, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb24
// CHECK:     ^bb24:
// CHECK:       aie.use_lock(%out_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%out_buff_1 : memref<1024xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%out_prod_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb25
// CHECK:     ^bb25:
// CHECK:       aie.use_lock(%out_cons_lock_1, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%out_buff_1 : memref<1024xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%out_prod_lock_1, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb22
// CHECK:     ^bb26:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_0_2 = aie.mem(%tile_0_2) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4, repeat_count = 7)
// CHECK:     ^bb1:  // pred: ^bb0
// CHECK:       aie.use_lock(%split_0_cons_prod_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%split_0_cons_buff_0 : memref<512xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%split_0_cons_cons_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%split_0_cons_prod_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%split_0_cons_buff_1 : memref<512xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%split_0_cons_cons_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.end
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 0, ^bb5, ^bb8, repeat_count = 7)
// CHECK:     ^bb5:  // pred: ^bb4
// CHECK:       aie.use_lock(%join_0_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%join_0_buff_0 : memref<512xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%join_0_prod_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb6
// CHECK:     ^bb6:  // pred: ^bb5
// CHECK:       aie.use_lock(%join_0_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%join_0_buff_1 : memref<512xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%join_0_prod_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb7
// CHECK:     ^bb7:  // pred: ^bb6
// CHECK:       aie.end
// CHECK:     ^bb8:  // pred: ^bb4
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_0_3 = aie.mem(%tile_0_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4, repeat_count = 7)
// CHECK:     ^bb1:  // pred: ^bb0
// CHECK:       aie.use_lock(%split_1_cons_prod_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%split_1_cons_buff_0 : memref<512xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%split_1_cons_cons_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%split_1_cons_prod_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%split_1_cons_buff_1 : memref<512xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%split_1_cons_cons_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.end
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 0, ^bb5, ^bb8, repeat_count = 7)
// CHECK:     ^bb5:  // pred: ^bb4
// CHECK:       aie.use_lock(%join_1_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%join_1_buff_0 : memref<512xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%join_1_prod_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb6
// CHECK:     ^bb6:  // pred: ^bb5
// CHECK:       aie.use_lock(%join_1_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%join_1_buff_1 : memref<512xui8> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%join_1_prod_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb7
// CHECK:     ^bb7:  // pred: ^bb6
// CHECK:       aie.end
// CHECK:     ^bb8:  // pred: ^bb4
// CHECK:       aie.end
// CHECK:     }

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    aie.objectfifo @in(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xui8>>
    aie.objectfifo @split_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) {iter_count = 8 : i32} : !aie.objectfifo<memref<512xui8>>
    aie.objectfifo @split_1(%mem_tile_0_1, {%tile_0_3}, 2 : i32) {iter_count = 8 : i32} : !aie.objectfifo<memref<512xui8>>
    aie.objectfifo.link [@in] -> [@split_0, @split_1]([] [0, 512])
    aie.objectfifo @join_0(%tile_0_2, {%mem_tile_0_1}, 2 : i32) {iter_count = 8 : i32} : !aie.objectfifo<memref<512xui8>>
    aie.objectfifo @join_1(%tile_0_3, {%mem_tile_0_1}, 2 : i32) {iter_count = 8 : i32} : !aie.objectfifo<memref<512xui8>>
    aie.objectfifo @out(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xui8>>
    aie.objectfifo.link [@join_0, @join_1] -> [@out]([0, 512] [])
  }
}
