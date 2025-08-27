//===- core_flag_test.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK:     %[[VAL_2:.*]] = aie.buffer(%{{.*}}tile_0_4) {sym_name = "output_fifo2_buff_0"} : memref<10xi32> 
// CHECK:     %[[VAL_3:.*]] = aie.buffer(%{{.*}}tile_0_4) {sym_name = "output_fifo2_buff_1"} : memref<10xi32> 
// CHECK:     %[[VAL_4:.*]] = aie.lock(%{{.*}}tile_0_4, 2) {init = 2 : i32, sym_name = "output_fifo2_prod_lock_0"}
// CHECK:     %[[VAL_5:.*]] = aie.lock(%{{.*}}tile_0_4, 3) {init = 0 : i32, sym_name = "output_fifo2_cons_lock_0"}
// CHECK:     %[[VAL_6:.*]] = aie.buffer(%{{.*}}tile_0_4) {sym_name = "input_fifo2_cons_buff_0"} : memref<10xi32> 
// CHECK:     %[[VAL_7:.*]] = aie.buffer(%{{.*}}tile_0_4) {sym_name = "input_fifo2_cons_buff_1"} : memref<10xi32> 
// CHECK:     %[[VAL_8:.*]] = aie.lock(%{{.*}}tile_0_4, 0) {init = 2 : i32, sym_name = "input_fifo2_cons_prod_lock_0"}
// CHECK:     %[[VAL_9:.*]] = aie.lock(%{{.*}}tile_0_4, 1) {init = 0 : i32, sym_name = "input_fifo2_cons_cons_lock_0"}
// CHECK:     %[[VAL_14:.*]] = aie.buffer(%{{.*}}tile_0_2) {sym_name = "output_fifo_buff_0"} : memref<10xi32> 
// CHECK:     %[[VAL_15:.*]] = aie.buffer(%{{.*}}tile_0_2) {sym_name = "output_fifo_buff_1"} : memref<10xi32> 
// CHECK:     %[[VAL_16:.*]] = aie.lock(%{{.*}}tile_0_2, 2) {init = 2 : i32, sym_name = "output_fifo_prod_lock_0"}
// CHECK:     %[[VAL_17:.*]] = aie.lock(%{{.*}}tile_0_2, 3) {init = 0 : i32, sym_name = "output_fifo_cons_lock_0"}
// CHECK:     %[[VAL_18:.*]] = aie.buffer(%{{.*}}tile_0_2) {sym_name = "input_fifo_cons_buff_0"} : memref<10xi32> 
// CHECK:     %[[VAL_19:.*]] = aie.buffer(%{{.*}}tile_0_2) {sym_name = "input_fifo_cons_buff_1"} : memref<10xi32> 
// CHECK:     %[[VAL_20:.*]] = aie.lock(%{{.*}}tile_0_2, 0) {init = 2 : i32, sym_name = "input_fifo_cons_prod_lock_0"}
// CHECK:     %[[VAL_21:.*]] = aie.lock(%{{.*}}tile_0_2, 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock_0"}
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_2, DMA : 0)
// CHECK:     aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_0_0, DMA : 0)
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 1, %{{.*}}tile_0_4, DMA : 0)
// CHECK:     aie.flow(%{{.*}}tile_0_4, DMA : 0, %{{.*}}tile_0_0, DMA : 1)
// CHECK:     %buffer_0_2 = aie.buffer(%{{.*}}tile_0_2) : memref<2xi32>  
// CHECK:     %core_0_2 = aie.core(%{{.*}}tile_0_2) {
// CHECK:       %c0_i32 = arith.constant 0 : i32
// CHECK:       %c0 = arith.constant 0 : index
// CHECK:       %c2_i32 = arith.constant 2 : i32
// CHECK:       memref.store %c0_i32, %buffer_0_2[%c0] : memref<2xi32>
// CHECK:       %c1 = arith.constant 1 : index
// CHECK:       %c2_i32_0 = arith.constant 2 : i32
// CHECK:       memref.store %c0_i32, %buffer_0_2[%c1] : memref<2xi32>
// CHECK:       %c0_1 = arith.constant 0 : index
// CHECK:       %c1_2 = arith.constant 1 : index
// CHECK:       %c10 = arith.constant 10 : index
// CHECK:       scf.for %arg0 = %c0_1 to %c10 step %c1_2 {
// CHECK:         aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, 1)
// CHECK:         %0 = memref.load %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         %1 = arith.index_cast %0 : i32 to index
// CHECK:         %2 = scf.index_switch %1 -> memref<10xi32> 
// CHECK:         case 0 {
// CHECK:           scf.yield %[[VAL_14]] : memref<10xi32>
// CHECK:         }
// CHECK:         case 1 {
// CHECK:           scf.yield %[[VAL_15]] : memref<10xi32>
// CHECK:         }
// CHECK:         default {
// CHECK:           scf.yield %[[VAL_14]] : memref<10xi32>
// CHECK:         }
// CHECK:         aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, 1)
// CHECK:         %3 = memref.load %buffer_0_2[%c1] : memref<2xi32>
// CHECK:         %4 = arith.index_cast %3 : i32 to index
// CHECK:         %5 = scf.index_switch %4 -> memref<10xi32> 
// CHECK:         case 0 {
// CHECK:           scf.yield %[[VAL_18]] : memref<10xi32>
// CHECK:         }
// CHECK:         case 1 {
// CHECK:           scf.yield %[[VAL_19]] : memref<10xi32>
// CHECK:         }
// CHECK:         default {
// CHECK:           scf.yield %[[VAL_18]] : memref<10xi32>
// CHECK:         }
// CHECK:         func.call @passthrough_10_i32(%5, %2) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:         aie.use_lock(%[[VAL_20]], Release, 1)
// CHECK:         %6 = memref.load %buffer_0_2[%c1] : memref<2xi32>
// CHECK:         %c1_i32 = arith.constant 1 : i32
// CHECK:         %7 = arith.addi %6, %c1_i32 : i32
// CHECK:         %8 = arith.cmpi sge, %7, %c2_i32_0 : i32 
// CHECK:         %9 = arith.subi %7, %c2_i32_0 : i32 
// CHECK:         %10 = arith.select %8, %9, %7 : i32
// CHECK:         memref.store %10, %buffer_0_2[%c1] : memref<2xi32>
// CHECK:         aie.use_lock(%[[VAL_17]], Release, 1)
// CHECK:         %11 = memref.load %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         %c1_i32_3 = arith.constant 1 : i32
// CHECK:         %12 = arith.addi %11, %c1_i32_3 : i32
// CHECK:         %13 = arith.cmpi sge, %12, %c2_i32 : i32
// CHECK:         %14 = arith.subi %12, %c2_i32 : i32 
// CHECK:         %15 = arith.select %13, %14, %12 : i32
// CHECK:         memref.store %15, %buffer_0_2[%c0] : memref<2xi32>
// CHECK:       }
// CHECK:       aie.end
// CHECK:     } {dynamic_objfifo_lowering = true}
// CHECK:     %core_0_4 = aie.core(%{{.*}}tile_0_4) {
// CHECK:       %c0 = arith.constant 0 : index
// CHECK:       %c1 = arith.constant 1 : index
// CHECK:       %c10 = arith.constant 10 : index
// CHECK:       %c2 = arith.constant 2 : index
// CHECK:       scf.for %arg0 = %c0 to %c10 step %c2 {
// CHECK:         aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:         aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:         func.call @passthrough_10_i32(%[[VAL_6]], %[[VAL_2]]) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:         aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:         aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:         aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:         aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:         func.call @passthrough_10_i32(%[[VAL_7]], %[[VAL_3]]) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:         aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:         aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:       }
// CHECK:       aie.end
// CHECK:     }
// CHECK:     aie.shim_dma_allocation @input_fifo(MM2S, 0, 0)

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
        %0 = aie.objectfifo.acquire @output_fifo(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        %2 = aie.objectfifo.acquire @input_fifo(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        func.call @passthrough_10_i32(%3, %1) : (memref<10xi32>, memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo(Consume, 1)
        aie.objectfifo.release @output_fifo(Produce, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}

    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index

      scf.for %arg0 = %c0 to %c10 step %c1 {
        %0 = aie.objectfifo.acquire @output_fifo2(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        %2 = aie.objectfifo.acquire @input_fifo2(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        func.call @passthrough_10_i32(%3, %1) : (memref<10xi32>, memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo2(Consume, 1)
        aie.objectfifo.release @output_fifo2(Produce, 1)
      }

      aie.end
    }
  }
}
