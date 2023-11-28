//===- nd_dma_distribute_AIE2.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s

// CHECK: module @ndDMAObjFifoAIE2 {
// CHECK:   AIE.device(xcve2302) {
// CHECK:     memref.global "public" @of2_cons : memref<128xi32>
// CHECK:     memref.global "public" @of2 : memref<128xi32>
// CHECK:     memref.global "public" @of1_cons : memref<128xi32>
// CHECK:     memref.global "public" @of1 : memref<128xi32>
// CHECK:     memref.global "public" @of0_cons : memref<256xi32>
// CHECK:     memref.global "public" @of0 : memref<256xi32>
// CHECK:     %[[tile_1_0:.*]] = AIE.tile(1, 0)
// CHECK:     %[[tile_1_1:.*]] = AIE.tile(1, 1)
// CHECK:     %[[tile_2_2:.*]] = AIE.tile(2, 2)
// CHECK:     %[[tile_2_3:.*]] = AIE.tile(2, 3)
// CHECK:     %[[of2_cons_buf_0:.*]] = AIE.buffer(%[[tile_2_3:.*]]) {sym_name = "of2_cons_buff_0"} : memref<128xi32>
// CHECK:     %[[of2_cons_buf_1:.*]] = AIE.buffer(%[[tile_2_3:.*]]) {sym_name = "of2_cons_buff_1"} : memref<128xi32>
// CHECK:     %[[of2_cons_prod_lock:.*]] = AIE.lock(%[[tile_2_3:.*]], 0) {init = 2 : i32, sym_name = "of2_cons_prod_lock"}
// CHECK:     %[[of2_cons_cons_lock:.*]] = AIE.lock(%[[tile_2_3:.*]], 1) {init = 0 : i32, sym_name = "of2_cons_cons_lock"}
// CHECK:     %[[of1_cons_buf_0:.*]] = AIE.buffer(%[[tile_2_2:.*]]) {sym_name = "of1_cons_buff_0"} : memref<128xi32>
// CHECK:     %[[of1_cons_buf_1:.*]] = AIE.buffer(%[[tile_2_2:.*]]) {sym_name = "of1_cons_buff_1"} : memref<128xi32>
// CHECK:     %[[of1_cons_prod_lock:.*]] = AIE.lock(%[[tile_2_2:.*]], 0) {init = 2 : i32, sym_name = "of1_cons_prod_lock"}
// CHECK:     %[[of1_cons_cons_lock:.*]] = AIE.lock(%[[tile_2_2:.*]], 1) {init = 0 : i32, sym_name = "of1_cons_cons_lock"}
// CHECK:     %[[of0_cons_buf_0:.*]] = AIE.buffer(%[[tile_1_1:.*]]) {sym_name = "of0_cons_buff_0"} : memref<256xi32>
// CHECK:     %[[of0_cons_buf_1:.*]] = AIE.buffer(%[[tile_1_1:.*]]) {sym_name = "of0_cons_buff_1"} : memref<256xi32>
// CHECK:     %[[of0_cons_prod_lock:.*]] = AIE.lock(%[[tile_1_1:.*]], 0) {init = 4 : i32, sym_name = "of0_cons_prod_lock"}
// CHECK:     %[[of0_cons_cons_lock:.*]] = AIE.lock(%[[tile_1_1:.*]], 1) {init = 0 : i32, sym_name = "of0_cons_cons_lock"}
// CHECK:     %[[of0_prod_lock:.*]] = AIE.lock(%[[tile_1_0:.*]], 0) {init = 0 : i32, sym_name = "of0_prod_lock"}
// CHECK:     %[[of0_cons_lock:.*]] = AIE.lock(%[[tile_1_0:.*]], 1) {init = 0 : i32, sym_name = "of0_cons_lock"}
// CHECK:     AIE.flow(%[[tile_1_0:.*]], DMA : 0, %[[tile_1_1:.*]], DMA : 0)
// CHECK:     AIE.flow(%[[tile_1_1:.*]], DMA : 0, %[[tile_2_2:.*]], DMA : 0)
// CHECK:     AIE.flow(%[[tile_1_1:.*]], DMA : 1, %[[tile_2_3:.*]], DMA : 0)
// CHECK:     AIE.shimDMAAllocation @of0(MM2S, 0, 1)
// CHECK:     %18 = AIE.memTileDMA(%[[tile_1_1:.*]]) {
// CHECK:       %21 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%[[of0_cons_prod_lock:.*]], AcquireGreaterEqual, 2)
// CHECK:       AIE.dmaBd(<%[[of0_cons_buf_0:.*]] : memref<256xi32>, 0, 256>, 0)
// CHECK:       AIE.useLock(%[[of0_cons_cons_lock:.*]], Release, 2)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%[[of0_cons_prod_lock:.*]], AcquireGreaterEqual, 2)
// CHECK:       AIE.dmaBd(<%[[of0_cons_buf_1:.*]] : memref<256xi32>, 0, 256>, 0)
// CHECK:       AIE.useLock(%[[of0_cons_cons_lock:.*]], Release, 2)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       %22 = AIE.dmaStart(MM2S, 0, ^bb4, ^bb6)
// CHECK:     ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:       AIE.useLock(%[[of0_cons_cons_lock:.*]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%[[of0_cons_buf_0:.*]] : memref<256xi32>, 0, 128>, 0, [<4, 64>, <2, 4>, <8, 8>, <4, 1>])
// CHECK:       AIE.useLock(%[[of0_cons_prod_lock:.*]], Release, 1)
// CHECK:       AIE.nextBd ^bb5
// CHECK:     ^bb5:  // pred: ^bb4
// CHECK:       AIE.useLock(%[[of0_cons_cons_lock:.*]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%[[of0_cons_buf_1:.*]] : memref<256xi32>, 0, 128>, 0, [<4, 64>, <2, 4>, <8, 8>, <4, 1>])
// CHECK:       AIE.useLock(%[[of0_cons_prod_lock:.*]], Release, 1)
// CHECK:       AIE.nextBd ^bb4
// CHECK:     ^bb6:  // pred: ^bb3
// CHECK:       %23 = AIE.dmaStart(MM2S, 1, ^bb7, ^bb9)
// CHECK:     ^bb7:  // 2 preds: ^bb6, ^bb8
// CHECK:       AIE.useLock(%[[of0_cons_cons_lock:.*]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%[[of0_cons_buf_0:.*]] : memref<256xi32>, 512, 128>, 0, [<4, 64>, <2, 4>, <8, 8>, <4, 1>])
// CHECK:       AIE.useLock(%[[of0_cons_prod_lock:.*]], Release, 1)
// CHECK:       AIE.nextBd ^bb8
// CHECK:     ^bb8:  // pred: ^bb7
// CHECK:       AIE.useLock(%[[of0_cons_cons_lock:.*]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%[[of0_cons_buf_1:.*]] : memref<256xi32>, 512, 128>, 0, [<4, 64>, <2, 4>, <8, 8>, <4, 1>])
// CHECK:       AIE.useLock(%[[of0_cons_prod_lock:.*]], Release, 1)
// CHECK:       AIE.nextBd ^bb7
// CHECK:     ^bb9:  // pred: ^bb6
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %19 = AIE.mem(%[[tile_2_2:.*]]) {
// CHECK:       %21 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%[[of1_cons_prod_lock:.*]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%[[of1_cons_buf_0:.*]] : memref<128xi32>, 0, 128>, 0)
// CHECK:       AIE.useLock(%[[of1_cons_cons_lock:.*]], Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%[[of1_cons_prod_lock:.*]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%[[of1_cons_buf_1:.*]] : memref<128xi32>, 0, 128>, 0)
// CHECK:       AIE.useLock(%[[of1_cons_cons_lock:.*]], Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %20 = AIE.mem(%[[tile_2_3:.*]]) {
// CHECK:       %21 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%[[of2_cons_prod_lock:.*]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%[[of2_cons_buf_0:.*]] : memref<128xi32>, 0, 128>, 0)
// CHECK:       AIE.useLock(%[[of2_cons_cons_lock:.*]], Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%[[of2_cons_prod_lock:.*]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%[[of2_cons_buf_1:.*]] : memref<128xi32>, 0, 128>, 0)
// CHECK:       AIE.useLock(%[[of2_cons_cons_lock:.*]], Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @ndDMAObjFifoAIE2 {
 AIE.device(xcve2302) {
    %tile10 = AIE.tile(1, 0)
    %tile11 = AIE.tile(1, 1)
    %tile22 = AIE.tile(2, 2)
    %tile23 = AIE.tile(2, 3)

    AIE.objectfifo @of0 (%tile10, {%tile11},
                         2 : i32) : !AIE.objectfifo<memref<256xi32>>

    AIE.objectfifo @of1 (%tile11 toStream [< 4,64>,
                                           < 2, 4>, 
                                           < 8, 8>, 
                                           < 4, 1>],
                        {%tile22}, 2 : i32) : !AIE.objectfifo<memref<128xi32>>

    AIE.objectfifo @of2 (%tile11 toStream [< 4,64>,
                                           < 2, 4>, 
                                           < 8, 8>, 
                                           < 4, 1>],
                        {%tile23}, 2 : i32) : !AIE.objectfifo<memref<128xi32>>
   // expected-error@+1 {{'AIE.objectfifo.link' op currently does not support objectFifos with dimensionsFromStreamPerConsumer.}}
   AIE.objectfifo.link [ @of0 ] -> [ @of1, @of2 ] ()
 }
}
