//===- stream_to_link_AIE2.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_stream_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_stream_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "of_stream_cons_prod_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of_stream_cons_cons_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.tile(3, 3)
// CHECK:           aie.flow(%[[VAL_6]], Core : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           aie.shim_dma_allocation @of_out_shim_alloc(%[[VAL_0]], S2MM, 0)
// CHECK:           %[[VAL_7:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_8:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_9:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_8]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_8]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_8]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_8]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_10:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_8]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_8]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_8]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_8]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @stream_to_link_AIE2 {
 aie.device(xcve2302) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of_stream (%tile33, {%tile11}, 2 : i32) {aie_stream = 0 : i32, aie_stream_port = 0 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out (%tile11, {%tile10}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@of_stream] -> [@of_out] ([] [])
  }
}
