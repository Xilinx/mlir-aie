//===- repeat_count_test.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_2:.*]] = aie.lock(%[[VAL_0]]) {init = 1 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "of0_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_4]]) {init = 1 : i32, sym_name = "of0_cons_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_4]]) {init = 0 : i32, sym_name = "of0_cons_cons_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "of1_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_4]]) {init = 3 : i32, sym_name = "of1_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_4]]) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_11]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_11]]) {init = 1 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_11]]) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_4]], DMA : 0, %[[VAL_11]], DMA : 0)
// CHECK:           %[[VAL_15:.*]] = aie.core(%[[VAL_4]]) {
// CHECK:             %[[VAL_16:.*]] = arith.constant 12 : index
// CHECK:             %[[VAL_17:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_18:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_20:.*]] = arith.constant 3 : i32
// CHECK:             scf.for %[[VAL_21:.*]] = %[[VAL_18]] to %[[VAL_16]] step %[[VAL_17]] {
// CHECK:               aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_19]])
// CHECK:               aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_20]])
// CHECK:               aie.use_lock(%[[VAL_10]], Release, %[[VAL_20]])
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[VAL_19]])
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_23:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_24:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_2]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_26:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_27:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_28:.*]] = aie.dma_start(MM2S, 0, ^bb3, ^bb4, repeat_count = 2)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = aie.mem(%[[VAL_11]]) {
// CHECK:             %[[VAL_30:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_31:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %[[VAL_30]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_30]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @repeatCount {
 aie.device(npu1) {
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of0 (%tile11, {%tile12}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of1 (%tile12, {%tile13}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>

    %core33 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %height = arith.constant 12 : index

      scf.for %indexInHeight = %c0 to %height step %c1 {
         %subview0_obj0 = aie.objectfifo.acquire @of0(Consume) : memref<16xi32>
         %subview1_obj0 = aie.objectfifo.acquire @of1(Produce) : memref<16xi32>
         aie.objectfifo.release @of1(Produce) [1]
         aie.objectfifo.release @of0(Consume) [1]
      }

      aie.end
   }
 }
}
