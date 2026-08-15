//===- pass_flag_test.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// `default-dynamic=true` sets the lowering for cores that do not carry an
// explicit `dynamic_objfifo_lowering` attribute. core_0_2 sets that attribute to
// false and so is statically unrolled (loop step 2, buffers bound directly),
// while core_0_4 follows the dynamic default and keeps a rolled loop carrying
// the output/input buffer indices as iter_args.

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK:           func.func @passthrough_10_i32(%[[VAL_0:.*]]: memref<10xi32>, %[[VAL_1:.*]]: memref<10xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_2:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_3:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "input_fifo_cons_buff_0"} : memref<10xi32>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "input_fifo_cons_buff_1"} : memref<10xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_3]]) {init = 2 : i32, sym_name = "input_fifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_3]]) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "output_fifo_buff_0"} : memref<10xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "output_fifo_buff_1"} : memref<10xi32>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_3]]) {init = 2 : i32, sym_name = "output_fifo_prod_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_3]]) {init = 0 : i32, sym_name = "output_fifo_cons_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.tile(0, 4)
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_12]]) {sym_name = "input_fifo2_cons_buff_0"} : memref<10xi32>
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_12]]) {sym_name = "input_fifo2_cons_buff_1"} : memref<10xi32>
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_12]]) {init = 2 : i32, sym_name = "input_fifo2_cons_prod_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_12]]) {init = 0 : i32, sym_name = "input_fifo2_cons_cons_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_12]]) {sym_name = "output_fifo2_buff_0"} : memref<10xi32>
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_12]]) {sym_name = "output_fifo2_buff_1"} : memref<10xi32>
// CHECK:           %[[VAL_19:.*]] = aie.lock(%[[VAL_12]]) {init = 2 : i32, sym_name = "output_fifo2_prod_lock_0"}
// CHECK:           %[[VAL_20:.*]] = aie.lock(%[[VAL_12]]) {init = 0 : i32, sym_name = "output_fifo2_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_2]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_2]], DMA : 1, %[[VAL_12]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_12]], DMA : 0, %[[VAL_2]], DMA : 1)
// CHECK:           %[[VAL_21:.*]] = aie.core(%[[VAL_3]]) {
// CHECK:             %[[VAL_22:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_23:.*]] = arith.constant 10 : index
// CHECK:             %[[VAL_24:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_25:.*]] = arith.constant 1 : i32
// CHECK:             scf.for %[[VAL_26:.*]] = %[[VAL_24]] to %[[VAL_23]] step %[[VAL_22]] {
// CHECK:               aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_25]])
// CHECK:               aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_25]])
// CHECK:               func.call @passthrough_10_i32(%[[VAL_4]], %[[VAL_8]]) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[VAL_25]])
// CHECK:               aie.use_lock(%[[VAL_11]], Release, %[[VAL_25]])
// CHECK:               aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_25]])
// CHECK:               aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_25]])
// CHECK:               func.call @passthrough_10_i32(%[[VAL_5]], %[[VAL_9]]) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[VAL_25]])
// CHECK:               aie.use_lock(%[[VAL_11]], Release, %[[VAL_25]])
// CHECK:             }
// CHECK:             aie.end
// CHECK:           } {dynamic_objfifo_lowering = false}
// CHECK:           %[[VAL_27:.*]] = aie.core(%[[VAL_12]]) {
// CHECK:             %[[VAL_28:.*]] = arith.constant 10 : index
// CHECK:             %[[VAL_29:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_30:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_31:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_32:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_33:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_34:.*]]:2 = scf.for %[[VAL_35:.*]] = %[[VAL_30]] to %[[VAL_28]] step %[[VAL_29]] iter_args(%[[VAL_36:.*]] = %[[VAL_31]], %[[VAL_37:.*]] = %[[VAL_31]]) -> (i32, i32) {
// CHECK:               aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, %[[VAL_32]])
// CHECK:               %[[VAL_38:.*]] = arith.index_cast %[[VAL_36]] : i32 to index
// CHECK:               %[[VAL_39:.*]] = scf.index_switch %[[VAL_38]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_17]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_18]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_17]] : memref<10xi32>
// CHECK:               }
// CHECK:               aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_32]])
// CHECK:               %[[VAL_40:.*]] = arith.index_cast %[[VAL_37]] : i32 to index
// CHECK:               %[[VAL_41:.*]] = scf.index_switch %[[VAL_40]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_13]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_14]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_13]] : memref<10xi32>
// CHECK:               }
// CHECK:               func.call @passthrough_10_i32(%[[VAL_41]], %[[VAL_39]]) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_15]], Release, %[[VAL_32]])
// CHECK:               %[[VAL_42:.*]] = arith.addi %[[VAL_37]], %[[VAL_32]] : i32
// CHECK:               %[[VAL_43:.*]] = arith.cmpi sge, %[[VAL_42]], %[[VAL_33]] : i32
// CHECK:               %[[VAL_44:.*]] = arith.select %[[VAL_43]], %[[VAL_31]], %[[VAL_42]] : i32
// CHECK:               aie.use_lock(%[[VAL_20]], Release, %[[VAL_32]])
// CHECK:               %[[VAL_45:.*]] = arith.addi %[[VAL_36]], %[[VAL_32]] : i32
// CHECK:               %[[VAL_46:.*]] = arith.cmpi sge, %[[VAL_45]], %[[VAL_33]] : i32
// CHECK:               %[[VAL_47:.*]] = arith.select %[[VAL_46]], %[[VAL_31]], %[[VAL_45]] : i32
// CHECK:               scf.yield %[[VAL_47]], %[[VAL_44]] : i32, i32
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @input_fifo_shim_alloc(%[[VAL_2]], MM2S, 0)
// CHECK:           aie.shim_dma_allocation @output_fifo_shim_alloc(%[[VAL_2]], S2MM, 0)
// CHECK:           aie.shim_dma_allocation @input_fifo2_shim_alloc(%[[VAL_2]], MM2S, 1)
// CHECK:           aie.shim_dma_allocation @output_fifo2_shim_alloc(%[[VAL_2]], S2MM, 1)
// CHECK:           %[[VAL_48:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_49:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_50:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_49]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_49]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_49]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_49]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_51:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_49]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_49]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_49]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_49]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_52:.*]] = aie.mem(%[[VAL_12]]) {
// CHECK:             %[[VAL_53:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_54:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_53]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_53]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_53]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_53]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_55:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, %[[VAL_53]])
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[VAL_19]], Release, %[[VAL_53]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, %[[VAL_53]])
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[VAL_19]], Release, %[[VAL_53]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module {
  aie.device(npu1_1col) {
    func.func @passthrough_10_i32(%line_in: memref<10xi32>, %line_out: memref<10xi32>) -> () {
        return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)
    aie.objectfifo @input_fifo(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<10xi32>>
    aie.objectfifo @output_fifo(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<10xi32>>

    aie.objectfifo @input_fifo2(%tile_0_0, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<10xi32>>
    aie.objectfifo @output_fifo2(%tile_0_4, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<10xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index

      scf.for %arg0 = %c0 to %c10 step %c1 {
        %1 = aie.objectfifo.acquire @output_fifo(Produce) : memref<10xi32>
        %3 = aie.objectfifo.acquire @input_fifo(Consume) : memref<10xi32>
        func.call @passthrough_10_i32(%3, %1) : (memref<10xi32>, memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo(Consume) [1]
        aie.objectfifo.release @output_fifo(Produce) [1]
      }

      aie.end
    } {dynamic_objfifo_lowering = false}

    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index

      scf.for %arg0 = %c0 to %c10 step %c1 {
        %1 = aie.objectfifo.acquire @output_fifo2(Produce) : memref<10xi32>
        %3 = aie.objectfifo.acquire @input_fifo2(Consume) : memref<10xi32>
        func.call @passthrough_10_i32(%3, %1) : (memref<10xi32>, memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo2(Consume) [1]
        aie.objectfifo.release @output_fifo2(Produce) [1]
      }

      aie.end
    }
  }
}
