//===- same_depth_objectfifos_test.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// Both fifos have depth [2, 2]. The consumer slides a 2-element window
// (acquire 2, release 1), so the runtime lowering peels the first iteration
// (index 0), threads the output/input buffer indices and held counts through a
// 4-way iter_args loop, then peels the final iteration.

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK:           func.func @add_10_i32(%[[VAL_0:.*]]: memref<10xi32>, %[[VAL_1:.*]]: memref<10xi32>, %[[VAL_2:.*]]: memref<10xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_3:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_4:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "output_fifo_buff_0"} : memref<10xi32>
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "output_fifo_buff_1"} : memref<10xi32>
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_4]]) {init = 2 : i32, sym_name = "output_fifo_prod_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_4]]) {init = 0 : i32, sym_name = "output_fifo_cons_lock_0"}
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "input_fifo_cons_buff_0"} : memref<10xi32>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "input_fifo_cons_buff_1"} : memref<10xi32>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_4]]) {init = 2 : i32, sym_name = "input_fifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_4]]) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_4]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           %[[VAL_13:.*]] = aie.core(%[[VAL_4]]) {
// CHECK:             %[[VAL_14:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_15:.*]] = arith.constant 9 : index
// CHECK:             %[[VAL_16:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_17:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_18:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_19:.*]] = arith.constant 2 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_14]])
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_14]])
// CHECK:             func.call @add_10_i32(%[[VAL_9]], %[[VAL_9]], %[[VAL_5]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:             aie.use_lock(%[[VAL_8]], Release, %[[VAL_14]])
// CHECK:             %[[VAL_20:.*]]:2 = scf.for %[[VAL_21:.*]] = %[[VAL_17]] to %[[VAL_15]] step %[[VAL_16]] iter_args(%[[VAL_22:.*]] = %[[VAL_14]], %[[VAL_23:.*]] = %[[VAL_18]]) -> (i32, i32) {
// CHECK:               aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_14]])
// CHECK:               %[[VAL_24:.*]] = arith.index_cast %[[VAL_22]] : i32 to index
// CHECK:               %[[VAL_25:.*]] = scf.index_switch %[[VAL_24]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_5]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_6]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_5]] : memref<10xi32>
// CHECK:               }
// CHECK:               aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_14]])
// CHECK:               %[[VAL_26:.*]] = arith.index_cast %[[VAL_23]] : i32 to index
// CHECK:               %[[VAL_27:.*]] = scf.index_switch %[[VAL_26]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_9]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_10]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_9]] : memref<10xi32>
// CHECK:               }
// CHECK:               %[[VAL_28:.*]] = arith.index_cast %[[VAL_23]] : i32 to index
// CHECK:               %[[VAL_29:.*]] = scf.index_switch %[[VAL_28]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_10]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_9]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_10]] : memref<10xi32>
// CHECK:               }
// CHECK:               func.call @add_10_i32(%[[VAL_27]], %[[VAL_29]], %[[VAL_25]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_11]], Release, %[[VAL_14]])
// CHECK:               %[[VAL_30:.*]] = arith.addi %[[VAL_23]], %[[VAL_14]] : i32
// CHECK:               %[[VAL_31:.*]] = arith.cmpi sge, %[[VAL_30]], %[[VAL_19]] : i32
// CHECK:               %[[VAL_32:.*]] = arith.select %[[VAL_31]], %[[VAL_18]], %[[VAL_30]] : i32
// CHECK:               aie.use_lock(%[[VAL_8]], Release, %[[VAL_14]])
// CHECK:               %[[VAL_33:.*]] = arith.addi %[[VAL_22]], %[[VAL_14]] : i32
// CHECK:               %[[VAL_34:.*]] = arith.cmpi sge, %[[VAL_33]], %[[VAL_19]] : i32
// CHECK:               %[[VAL_35:.*]] = arith.select %[[VAL_34]], %[[VAL_18]], %[[VAL_33]] : i32
// CHECK:               scf.yield %[[VAL_35]], %[[VAL_32]] : i32, i32
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_14]])
// CHECK:             %[[VAL_36:.*]] = arith.index_cast %[[VAL_37:.*]]#0 : i32 to index
// CHECK:             %[[VAL_38:.*]] = scf.index_switch %[[VAL_36]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_5]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_6]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_5]] : memref<10xi32>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_14]])
// CHECK:             %[[VAL_39:.*]] = arith.index_cast %[[VAL_37]]#1 : i32 to index
// CHECK:             %[[VAL_40:.*]] = scf.index_switch %[[VAL_39]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_9]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_10]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_9]] : memref<10xi32>
// CHECK:             }
// CHECK:             %[[VAL_41:.*]] = arith.index_cast %[[VAL_37]]#1 : i32 to index
// CHECK:             %[[VAL_42:.*]] = scf.index_switch %[[VAL_41]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_10]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_9]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_10]] : memref<10xi32>
// CHECK:             }
// CHECK:             func.call @add_10_i32(%[[VAL_40]], %[[VAL_42]], %[[VAL_38]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_19]])
// CHECK:             aie.use_lock(%[[VAL_8]], Release, %[[VAL_14]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @input_fifo_shim_alloc(%[[VAL_3]], MM2S, 0)
// CHECK:           aie.shim_dma_allocation @output_fifo_shim_alloc(%[[VAL_3]], S2MM, 0)
// CHECK:           %[[VAL_43:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_44:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_45:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_44]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_44]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_44]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_44]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_46:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_44]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_44]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_44]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_44]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module {
  aie.device(npu1_1col) {
    func.func @add_10_i32(%line_in1: memref<10xi32>, %line_in2: memref<10xi32>, %line_out: memref<10xi32>) -> () {
        return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @input_fifo(%tile_0_0, {%tile_0_2}, [2, 2]) : !aie.objectfifo<memref<10xi32>>
    aie.objectfifo @output_fifo(%tile_0_2, {%tile_0_0}, [2, 2]) : !aie.objectfifo<memref<10xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 9 : index

      %0 = aie.objectfifo.acquire @output_fifo(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
      %2 = aie.objectfifo.acquire @input_fifo(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
      %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
      func.call @add_10_i32(%3, %3, %1) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
      aie.objectfifo.release @output_fifo(Produce, 1)

      scf.for %arg0 = %c0 to %c8 step %c1 {
        %4 = aie.objectfifo.acquire @output_fifo(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        %6 = aie.objectfifo.acquire @input_fifo(Consume, 2) : !aie.objectfifosubview<memref<10xi32>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        %8 = aie.objectfifo.subview.access %6[1] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        func.call @add_10_i32(%7, %8, %5) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo(Consume, 1)
        aie.objectfifo.release @output_fifo(Produce, 1)
      }

      %9 = aie.objectfifo.acquire @output_fifo(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
      %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
      %11 = aie.objectfifo.acquire @input_fifo(Consume, 2) : !aie.objectfifosubview<memref<10xi32>>
      %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
      %13 = aie.objectfifo.subview.access %11[1] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
      func.call @add_10_i32(%12, %13, %10) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
      aie.objectfifo.release @input_fifo(Consume, 2)
      aie.objectfifo.release @output_fifo(Produce, 1)

      aie.end
    }
  }
}
