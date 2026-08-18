//===- onelargefifo_on_2memtile.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "in0_cons_buff_1"} : memref<96000xi32>
// CHECK:           %[[VAL_2:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_3:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "in0_cons_buff_0"} : memref<96000xi32>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_3]]) {init = 2 : i32, sym_name = "in0_cons_prod_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_3]]) {init = 0 : i32, sym_name = "in0_cons_cons_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_3]]) {init = 2 : i32, sym_name = "in0_cons_prod_lock_1"}
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_3]]) {init = 0 : i32, sym_name = "in0_cons_cons_lock_1"}
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "out1_cons_buff_0"} : memref<8xi32>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "out1_cons_buff_1"} : memref<8xi32>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_3]]) {init = 2 : i32, sym_name = "out1_cons_prod_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_3]]) {init = 0 : i32, sym_name = "out1_cons_cons_lock_0"}
// CHECK:           %[[VAL_13:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_13]]) {sym_name = "in1_cons_buff_0"} : memref<8xi32>
// CHECK:           %[[VAL_15:.*]] = aie.buffer(%[[VAL_13]]) {sym_name = "in1_cons_buff_1"} : memref<8xi32>
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_13]]) {init = 2 : i32, sym_name = "in1_cons_prod_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.lock(%[[VAL_13]]) {init = 0 : i32, sym_name = "in1_cons_cons_lock_0"}
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_13]]) {sym_name = "out1_buff_0"} : memref<8xi32>
// CHECK:           %[[VAL_19:.*]] = aie.buffer(%[[VAL_13]]) {sym_name = "out1_buff_1"} : memref<8xi32>
// CHECK:           %[[VAL_20:.*]] = aie.lock(%[[VAL_13]]) {init = 2 : i32, sym_name = "out1_prod_lock_0"}
// CHECK:           %[[VAL_21:.*]] = aie.lock(%[[VAL_13]]) {init = 0 : i32, sym_name = "out1_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_2]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_13]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_3]], DMA : 1, %[[VAL_2]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_13]], DMA : 0, %[[VAL_3]], DMA : 1)
// CHECK:           aie.shim_dma_allocation @in0_shim_alloc(%[[VAL_2]], MM2S, 0)
// CHECK:           aie.shim_dma_allocation @out0_shim_alloc(%[[VAL_2]], S2MM, 0)
// CHECK:           %[[VAL_22:.*]] = aie.memtile_dma(%[[VAL_3]]) {
// CHECK:             %[[VAL_23:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_24:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<96000xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<96000xi32> offset = 8 len = 95992)
// CHECK:             aie.use_lock(%[[VAL_8]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<96000xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<96000xi32> offset = 8 len = 95992)
// CHECK:             aie.use_lock(%[[VAL_8]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             %[[VAL_25:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<96000xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<96000xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb8:
// CHECK:             %[[VAL_26:.*]] = aie.dma_start(MM2S, 1, ^bb9, ^bb11)
// CHECK:           ^bb9:
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<8xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb10
// CHECK:           ^bb10:
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<8xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb9
// CHECK:           ^bb11:
// CHECK:             %[[VAL_27:.*]] = aie.dma_start(S2MM, 1, ^bb12, ^bb14)
// CHECK:           ^bb12:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<8xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb13
// CHECK:           ^bb13:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<8xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb12
// CHECK:           ^bb14:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = aie.mem(%[[VAL_13]]) {
// CHECK:             %[[VAL_29:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_30:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<8xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_29]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_15]] : memref<8xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_29]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_31:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<8xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_20]], Release, %[[VAL_29]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_19]] : memref<8xi32> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_20]], Release, %[[VAL_29]])
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
