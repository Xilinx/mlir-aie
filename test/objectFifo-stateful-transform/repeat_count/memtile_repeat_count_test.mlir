//===- memtile_repeat_count_test.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @repeatCount {
// CHECK:   aie.device(npu1) {
// CHECK:     memref.global "public" @of1_cons : memref<16xi32>
// CHECK:     memref.global "public" @of1 : memref<16xi32>
// CHECK:     %{{.*}}tile_1_1 = aie.tile(1, 1)
// CHECK:     %{{.*}}tile_1_3 = aie.tile(1, 3)
// CHECK:     %of1_cons_buff_0 = aie.buffer(%{{.*}}tile_1_3) {sym_name = "of1_cons_buff_0"} : memref<16xi32> 
// CHECK:     %of1_cons_prod_lock = aie.lock(%{{.*}}tile_1_3, 0) {init = 1 : i32, sym_name = "of1_cons_prod_lock"}
// CHECK:     %of1_cons_cons_lock = aie.lock(%{{.*}}tile_1_3, 1) {init = 0 : i32, sym_name = "of1_cons_cons_lock"}
// CHECK:     %of1_buff_0 = aie.buffer(%{{.*}}tile_1_1) {sym_name = "of1_buff_0"} : memref<16xi32> 
// CHECK:     %of1_prod_lock = aie.lock(%{{.*}}tile_1_1, 0) {init = 3 : i32, sym_name = "of1_prod_lock"}
// CHECK:     %of1_cons_lock = aie.lock(%{{.*}}tile_1_1, 1) {init = 0 : i32, sym_name = "of1_cons_lock"}
// CHECK:     aie.flow(%{{.*}}tile_1_1, DMA : 0, %{{.*}}tile_1_3, DMA : 0)
// CHECK:     %memtile_dma_1_1 = aie.memtile_dma(%{{.*}}tile_1_1) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2, repeat_count = 2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(%of1_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of1_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of1_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_1_3 = aie.mem(%{{.*}}tile_1_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(%of1_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of1_cons_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of1_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @repeatCount {
 aie.device(npu1) {
    %tile11 = aie.tile(1, 1)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of1 (%tile11, {%tile13}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
 }
}
