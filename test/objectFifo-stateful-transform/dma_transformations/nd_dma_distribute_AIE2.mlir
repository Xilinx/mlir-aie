//===- nd_dma_distribute_AIE2.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s

// Currently not supported: link should also support lengths in addition to offsets, currently length cannot be correctly inferred.

// CHECK: module @ndDMAObjFifoAIE2 {
// CHECK:   aie.device(xcve2302) {
// CHECK-DAG:     %[[tile_1_0:.*]] = aie.tile(1, 0)
// CHECK-DAG:     %[[tile_1_1:.*]] = aie.tile(1, 1)
// CHECK-DAG:     %[[tile_2_2:.*]] = aie.tile(2, 2)
// CHECK-DAG:     %[[tile_2_3:.*]] = aie.tile(2, 3)
// CHECK-DAG:     aie.buffer @[[VAL_0:of2_cons_buff_0]](%{{.*}}tile_2_3) : memref<128xi32>
// CHECK-DAG:     aie.buffer @[[VAL_1:of2_cons_buff_1]](%{{.*}}tile_2_3) : memref<128xi32>
// CHECK-DAG:     aie.lock @[[VAL_2:of2_cons_prod_lock_0]](%{{.*}}tile_2_3) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_3:of2_cons_cons_lock_0]](%{{.*}}tile_2_3) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[VAL_4:of1_cons_buff_0]](%{{.*}}tile_2_2) : memref<128xi32>
// CHECK-DAG:     aie.buffer @[[VAL_5:of1_cons_buff_1]](%{{.*}}tile_2_2) : memref<128xi32>
// CHECK-DAG:     aie.lock @[[VAL_6:of1_cons_prod_lock_0]](%{{.*}}tile_2_2) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_7:of1_cons_cons_lock_0]](%{{.*}}tile_2_2) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[VAL_8:of0_cons_buff_0]](%{{.*}}tile_1_1) : memref<256xi32>
// CHECK-DAG:     aie.buffer @[[VAL_9:of0_cons_buff_1]](%{{.*}}tile_1_1) : memref<256xi32>
// CHECK-DAG:     aie.lock @[[VAL_10:of0_cons_prod_lock_0]](%{{.*}}tile_1_1) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_11:of0_cons_cons_lock_0]](%{{.*}}tile_1_1) {init = 0 : i32}
// CHECK-DAG:     aie.lock @[[VAL_12:of0_cons_prod_lock_1]](%{{.*}}tile_1_1) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_13:of0_cons_cons_lock_1]](%{{.*}}tile_1_1) {init = 0 : i32}
// CHECK-DAG:     aie.lock @[[VAL_14:of0_prod_lock_0]](%{{.*}}tile_1_0) {init = 1 : i32}
// CHECK-DAG:     aie.lock @[[VAL_15:of0_cons_lock_0]](%{{.*}}tile_1_0) {init = 0 : i32}
// CHECK-DAG:     aie.flow(%[[tile_1_0:.*]], DMA : 0, %[[tile_1_1:.*]], DMA : 0)
// CHECK-DAG:     aie.flow(%[[tile_1_1:.*]], DMA : 0, %[[tile_2_2:.*]], DMA : 0)
// CHECK-DAG:     aie.flow(%[[tile_1_1:.*]], DMA : 1, %[[tile_2_3:.*]], DMA : 0)
// CHECK-DAG:     aie.shim_dma_allocation @of0_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
// CHECK:     %memtile_dma_1_1 = aie.memtile_dma(%{{.*}}tile_1_1) {
// CHECK:       %[[CHECK_SSA_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:       aie.use_lock(@[[VAL_10]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_8_ACCESS_0:.*]] = aie.get_buffer @[[VAL_8]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[VAL_8_ACCESS_0]] : memref<256xi32> offset = {{.*}} len = {{.*}} sizes = {{.*}} strides = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_11]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[VAL_12]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_8_ACCESS_1:.*]] = aie.get_buffer @[[VAL_8]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[VAL_8_ACCESS_1]] : memref<256xi32> offset = {{.*}} len = {{.*}} sizes = {{.*}} strides = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_13]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(@[[VAL_10]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_9_ACCESS_2:.*]] = aie.get_buffer @[[VAL_9]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[VAL_9_ACCESS_2]] : memref<256xi32> offset = {{.*}} len = {{.*}} sizes = {{.*}} strides = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_11]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(@[[VAL_12]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_9_ACCESS_3:.*]] = aie.get_buffer @[[VAL_9]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[VAL_9_ACCESS_3]] : memref<256xi32> offset = {{.*}} len = {{.*}} sizes = {{.*}} strides = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_13]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb5:  // pred: ^bb0
// CHECK:       %[[CHECK_SSA_1:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
// CHECK:     ^bb6:  // 2 preds: ^bb5, ^bb7
// CHECK:       aie.use_lock(@[[VAL_11]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_8_ACCESS_4:.*]] = aie.get_buffer @[[VAL_8]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[VAL_8_ACCESS_4]] : memref<256xi32> offset = {{.*}} len = {{.*}} sizes = {{.*}} strides = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_10]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb7
// CHECK:     ^bb7:  // pred: ^bb6
// CHECK:       aie.use_lock(@[[VAL_11]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_9_ACCESS_5:.*]] = aie.get_buffer @[[VAL_9]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[VAL_9_ACCESS_5]] : memref<256xi32> offset = {{.*}} len = {{.*}} sizes = {{.*}} strides = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_10]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb6
// CHECK:     ^bb8:  // pred: ^bb5
// CHECK:       %[[CHECK_SSA_2:.*]] = aie.dma_start(MM2S, 1, ^bb9, ^bb11)
// CHECK:     ^bb9:  // 2 preds: ^bb8, ^bb10
// CHECK:       aie.use_lock(@[[VAL_13]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_8_ACCESS_6:.*]] = aie.get_buffer @[[VAL_8]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[VAL_8_ACCESS_6]] : memref<256xi32> offset = {{.*}} len = {{.*}} sizes = {{.*}} strides = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_12]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb10
// CHECK:     ^bb10:  // pred: ^bb9
// CHECK:       aie.use_lock(@[[VAL_13]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_9_ACCESS_7:.*]] = aie.get_buffer @[[VAL_9]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[VAL_9_ACCESS_7]] : memref<256xi32> offset = {{.*}} len = {{.*}} sizes = {{.*}} strides = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_12]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb9
// CHECK:     ^bb11:  // pred: ^bb8
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_2 = aie.mem(%[[tile_2_2:.*]]) {
// CHECK:       %[[CHECK_SSA_3:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(@[[VAL_6]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_4_ACCESS_8:.*]] = aie.get_buffer @[[VAL_4]] : memref<128xi32>
// CHECK:       aie.dma_bd(%[[VAL_4_ACCESS_8]] : memref<128xi32> offset = {{.*}} len = {{.*}} sizes = {{.*}} strides = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_7]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[VAL_6]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_5_ACCESS_9:.*]] = aie.get_buffer @[[VAL_5]] : memref<128xi32>
// CHECK:       aie.dma_bd(%[[VAL_5_ACCESS_9]] : memref<128xi32> offset = {{.*}} len = {{.*}} sizes = {{.*}} strides = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_7]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_3 = aie.mem(%[[tile_2_3:.*]]) {
// CHECK:       %[[CHECK_SSA_4:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(@[[VAL_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_0_ACCESS_10:.*]] = aie.get_buffer @[[VAL_0]] : memref<128xi32>
// CHECK:       aie.dma_bd(%[[VAL_0_ACCESS_10]] : memref<128xi32> offset = {{.*}} len = {{.*}} sizes = {{.*}} strides = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_3]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[VAL_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_1_ACCESS_11:.*]] = aie.get_buffer @[[VAL_1]] : memref<128xi32>
// CHECK:       aie.dma_bd(%[[VAL_1_ACCESS_11]] : memref<128xi32> offset = {{.*}} len = {{.*}} sizes = {{.*}} strides = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_3]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @ndDMAObjFifoAIE2 {
 aie.device(xcve2302) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile10, {%tile11},
                         2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.objectfifo @of1 (%tile11 dimensionsToStream [<size = 2, stride = 64>,
                                           <size = 2, stride = 4>,
                                           <size = 8, stride = 8>,
                                           <size = 4, stride = 1>],
                        {%tile22}, 2 : i32) : !aie.objectfifo<memref<128xi32>>

    aie.objectfifo @of2 (%tile11 dimensionsToStream [<size = 2, stride = 64>,
                                           <size = 2, stride = 4>,
                                           <size = 8, stride = 8>,
                                           <size = 4, stride = 1>],
                        {%tile23}, 2 : i32) : !aie.objectfifo<memref<128xi32>>
   aie.objectfifo.link [ @of0 ] -> [ @of1, @of2 ] ([][0, 128])
 }
}
