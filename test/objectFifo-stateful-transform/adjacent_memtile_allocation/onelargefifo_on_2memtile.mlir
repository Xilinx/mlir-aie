//===- onelargefifo_on_2memtile.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1) {
// CHECK:           %[[VAL_2:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_3:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "out1_buff_0"} : memref<8xi32>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "out1_buff_1"} : memref<8xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_3]]) {init = 2 : i32, sym_name = "out1_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_3]]) {init = 0 : i32, sym_name = "out1_cons_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "out1_cons_buff_0"} : memref<8xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "out1_cons_buff_1"} : memref<8xi32>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "out1_cons_prod_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "out1_cons_cons_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "in1_cons_buff_0"} : memref<8xi32>
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "in1_cons_buff_1"} : memref<8xi32>
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_3]]) {init = 2 : i32, sym_name = "in1_cons_prod_lock_0"}
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_3]]) {init = 0 : i32, sym_name = "in1_cons_cons_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "in0_cons_buff_0"} : memref<96000xi32>
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "in0_cons_buff_1"} : memref<96000xi32>
// CHECK:           %[[VAL_18:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "in0_cons_prod_lock_0"}
// CHECK:           %[[VAL_19:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "in0_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 1, %[[VAL_0]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_1]], DMA : 1)
// CHECK:           aie.shim_dma_allocation @in0_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           aie.shim_dma_allocation @out0_shim_alloc(%[[VAL_0]], S2MM, 0)
// CHECK:           %[[VAL_20:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_21:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_22:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_18]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<96000xi32> offset = 0 len = 96000)
// CHECK:             aie.use_lock(%[[VAL_19]], Release, %[[VAL_21]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_18]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<96000xi32> offset = 0 len = 96000)
// CHECK:             aie.use_lock(%[[VAL_19]], Release, %[[VAL_21]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_23:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<96000xi32> offset = 0 len = 96000)
// CHECK:             aie.use_lock(%[[VAL_18]], Release, %[[VAL_21]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<96000xi32> offset = 0 len = 96000)
// CHECK:             aie.use_lock(%[[VAL_18]], Release, %[[VAL_21]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             %[[VAL_24:.*]] = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<8xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_21]])
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<8xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_21]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb9:
// CHECK:             %[[VAL_25:.*]] = aie.dma_start(S2MM, 1, ^bb10, ^bb12)
// CHECK:           ^bb10:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<8xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_21]])
// CHECK:             aie.next_bd ^bb11
// CHECK:           ^bb11:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<8xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_21]])
// CHECK:             aie.next_bd ^bb10
// CHECK:           ^bb12:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_27:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_28:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_27]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<8xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_27]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_27]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<8xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_27]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_29:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_27]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<8xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_27]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_27]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<8xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_27]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @in0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<96000xi32>>
    aie.objectfifo @in1(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@in0] -> [@in1]([] [])
    aie.objectfifo @out0(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @out1(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@out1] -> [@out0]([] [])
  }
}
