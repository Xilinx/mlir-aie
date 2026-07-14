//===- consumer_stream_AIE2.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=false" %s | FileCheck %s

// CHECK: module @consumer_stream_AIE2 {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:     %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK:     %[[VAL_2:.*]] = aie.tile(3, 3)
// CHECK:     %of_consumer_stream_buff_0 = aie.buffer(%tile_1_2) {sym_name = "of_consumer_stream_buff_0"} : memref<16xi32>
// CHECK:     %of_consumer_stream_buff_1 = aie.buffer(%tile_1_2) {sym_name = "of_consumer_stream_buff_1"} : memref<16xi32>
// CHECK:     %of_consumer_stream_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "of_consumer_stream_prod_lock_0"}
// CHECK:     %of_consumer_stream_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "of_consumer_stream_cons_lock_0"}
// CHECK:     aie.flow(%tile_1_2, DMA : 0, %tile_3_3, Core : 0)
// CHECK:     %mem_1_2 = aie.mem(%tile_1_2) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%of_consumer_stream_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%of_consumer_stream_buff_0 : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%of_consumer_stream_prod_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%of_consumer_stream_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%of_consumer_stream_buff_1 : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%of_consumer_stream_prod_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @consumer_stream_AIE2 {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of_consumer_stream (%tile12, {%tile33}, 2 : i32) {aie_stream = 1 : i32, aie_stream_port = 0 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}
