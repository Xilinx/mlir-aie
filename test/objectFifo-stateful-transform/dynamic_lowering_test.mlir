//===- dynamic_lowering_test.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform=dynamic-objFifos %s | FileCheck %s

// CHECK:   aie.device(npu1_1col) {
// CHECK:       memref.global "public" @output_fifo_cons : memref<10xi32>
// CHECK:       memref.global "public" @output_fifo : memref<10xi32>
// CHECK:       memref.global "public" @input_fifo_cons : memref<10xi32>
// CHECK:       memref.global "public" @input_fifo : memref<10xi32>
// CHECK:       func.func @add_10_i32(%arg0: memref<10xi32>, %arg1: memref<10xi32>, %arg2: memref<10xi32>) {
// CHECK:         return
// CHECK:       }
// CHECK:       %tile_0_0 = aie.tile(0, 0)
// CHECK:       %tile_0_2 = aie.tile(0, 2)
// CHECK:       %output_fifo_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 1 : i32, sym_name = "output_fifo_cons_prod_lock"}
// CHECK:       %output_fifo_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "output_fifo_cons_cons_lock"}
// CHECK:       %output_fifo_buff_0 = aie.buffer(%tile_0_2) {sym_name = "output_fifo_buff_0"} : memref<10xi32> 
// CHECK:       %output_fifo_buff_1 = aie.buffer(%tile_0_2) {sym_name = "output_fifo_buff_1"} : memref<10xi32> 
// CHECK:       %output_fifo_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "output_fifo_prod_lock"}
// CHECK:       %output_fifo_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "output_fifo_cons_lock"}
// CHECK:       %input_fifo_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "input_fifo_cons_buff_0"} : memref<10xi32> 
// CHECK:       %input_fifo_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "input_fifo_cons_buff_1"} : memref<10xi32> 
// CHECK:       %input_fifo_cons_buff_2 = aie.buffer(%tile_0_2) {sym_name = "input_fifo_cons_buff_2"} : memref<10xi32> 
// CHECK:       %input_fifo_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 3 : i32, sym_name = "input_fifo_cons_prod_lock"}
// CHECK:       %input_fifo_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock"}
// CHECK:       %input_fifo_prod_lock = aie.lock(%tile_0_0, 0) {init = 1 : i32, sym_name = "input_fifo_prod_lock"}
// CHECK:       %input_fifo_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "input_fifo_cons_lock"}
// CHECK:       aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
// CHECK:       aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)
// CHECK:       %buffer_0_2 = aie.buffer(%tile_0_2) : memref<2xi32> 
// CHECK:       %core_0_2 = aie.core(%tile_0_2) {
// CHECK:         %c0_i32 = arith.constant 0 : i32
// CHECK:         %c0 = arith.constant 0 : index
// CHECK:         %c2_i32 = arith.constant 2 : i32
// CHECK:         memref.store %c0_i32, %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         %c1 = arith.constant 1 : index
// CHECK:         %c3_i32 = arith.constant 3 : i32
// CHECK:         memref.store %c0_i32, %buffer_0_2[%c1] : memref<2xi32>
// CHECK:         %c0_0 = arith.constant 0 : index
// CHECK:         %c1_1 = arith.constant 1 : index
// CHECK:         %c9 = arith.constant 9 : index
// CHECK:         aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)
// CHECK:         %0 = memref.load %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         %1 = arith.index_cast %0 : i32 to index
// CHECK:         %2 = scf.index_switch %1 -> memref<10xi32> 
// CHECK:         case 0 {
// CHECK:           scf.yield %output_fifo_buff_0 : memref<10xi32>
// CHECK:         }
// CHECK:         case 1 {
// CHECK:           scf.yield %output_fifo_buff_1 : memref<10xi32>
// CHECK:         }
// CHECK:         default {
// CHECK:           scf.yield %output_fifo_buff_0 : memref<10xi32>
// CHECK:         }
// CHECK:         aie.use_lock(%input_fifo_cons_cons_lock, AcquireGreaterEqual, 1)
// CHECK:         %3 = memref.load %buffer_0_2[%c1] : memref<2xi32>
// CHECK:         %4 = arith.index_cast %3 : i32 to index
// CHECK:         %5 = scf.index_switch %4 -> memref<10xi32> 
// CHECK:         case 0 {
// CHECK:           scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
// CHECK:         }
// CHECK:         case 1 {
// CHECK:           scf.yield %input_fifo_cons_buff_1 : memref<10xi32>
// CHECK:         }
// CHECK:         case 2 {
// CHECK:           scf.yield %input_fifo_cons_buff_2 : memref<10xi32>
// CHECK:         }
// CHECK:         default {
// CHECK:           scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
// CHECK:         }
// CHECK:         func.call @add_10_i32(%5, %5, %2) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:         aie.use_lock(%output_fifo_cons_lock, Release, 1)
// CHECK:         %6 = memref.load %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         %c1_i32 = arith.constant 1 : i32
// CHECK:         %7 = arith.addi %6, %c1_i32 : i32
// CHECK:         %8 = arith.remsi %7, %c2_i32 : i32
// CHECK:         memref.store %8, %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         scf.for %arg0 = %c0_0 to %c9 step %c1_1 {
// CHECK:           aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)
// CHECK:           %24 = memref.load %buffer_0_2[%c0] : memref<2xi32>
// CHECK:           %25 = arith.index_cast %24 : i32 to index
// CHECK:           %26 = scf.index_switch %25 -> memref<10xi32> 
// CHECK:           case 0 {
// CHECK:             scf.yield %output_fifo_buff_0 : memref<10xi32>
// CHECK:           }
// CHECK:           case 1 {
// CHECK:             scf.yield %output_fifo_buff_1 : memref<10xi32>
// CHECK:           }
// CHECK:           default {
// CHECK:             scf.yield %output_fifo_buff_0 : memref<10xi32>
// CHECK:           }
// CHECK:           aie.use_lock(%input_fifo_cons_cons_lock, AcquireGreaterEqual, 1)
// CHECK:           %27 = memref.load %buffer_0_2[%c1] : memref<2xi32>
// CHECK:           %28 = arith.index_cast %27 : i32 to index
// CHECK:           %29 = scf.index_switch %28 -> memref<10xi32> 
// CHECK:           case 0 {
// CHECK:             scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
// CHECK:           }
// CHECK:           case 1 {
// CHECK:             scf.yield %input_fifo_cons_buff_1 : memref<10xi32>
// CHECK:           }
// CHECK:           case 2 {
// CHECK:             scf.yield %input_fifo_cons_buff_2 : memref<10xi32>
// CHECK:           }
// CHECK:           default {
// CHECK:             scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
// CHECK:           }
// CHECK:           %30 = memref.load %buffer_0_2[%c1] : memref<2xi32>
// CHECK:           %31 = arith.index_cast %30 : i32 to index
// CHECK:           %32 = scf.index_switch %31 -> memref<10xi32> 
// CHECK:           case 0 {
// CHECK:             scf.yield %input_fifo_cons_buff_1 : memref<10xi32>
// CHECK:           }
// CHECK:           case 1 {
// CHECK:             scf.yield %input_fifo_cons_buff_2 : memref<10xi32>
// CHECK:           }
// CHECK:           case 2 {
// CHECK:             scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
// CHECK:           }
// CHECK:           default {
// CHECK:             scf.yield %input_fifo_cons_buff_1 : memref<10xi32>
// CHECK:           }
// CHECK:           func.call @add_10_i32(%29, %32, %26) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:           aie.use_lock(%input_fifo_cons_prod_lock, Release, 1)
// CHECK:           %33 = memref.load %buffer_0_2[%c1] : memref<2xi32>
// CHECK:           %c1_i32_4 = arith.constant 1 : i32
// CHECK:           %34 = arith.addi %33, %c1_i32_4 : i32
// CHECK:           %35 = arith.remsi %34, %c3_i32 : i32
// CHECK:           memref.store %35, %buffer_0_2[%c1] : memref<2xi32>
// CHECK:           aie.use_lock(%output_fifo_cons_lock, Release, 1)
// CHECK:           %36 = memref.load %buffer_0_2[%c0] : memref<2xi32>
// CHECK:           %c1_i32_5 = arith.constant 1 : i32
// CHECK:           %37 = arith.addi %36, %c1_i32_5 : i32
// CHECK:           %38 = arith.remsi %37, %c2_i32 : i32
// CHECK:           memref.store %38, %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         }
// CHECK:         aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)
// CHECK:         %9 = memref.load %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         %10 = arith.index_cast %9 : i32 to index
// CHECK:         %11 = scf.index_switch %10 -> memref<10xi32> 
// CHECK:         case 0 {
// CHECK:           scf.yield %output_fifo_buff_0 : memref<10xi32>
// CHECK:         }
// CHECK:         case 1 {
// CHECK:           scf.yield %output_fifo_buff_1 : memref<10xi32>
// CHECK:         }
// CHECK:         default {
// CHECK:           scf.yield %output_fifo_buff_0 : memref<10xi32>
// CHECK:         }
// CHECK:         aie.use_lock(%input_fifo_cons_cons_lock, AcquireGreaterEqual, 1)
// CHECK:         %12 = memref.load %buffer_0_2[%c1] : memref<2xi32>
// CHECK:         %13 = arith.index_cast %12 : i32 to index
// CHECK:         %14 = scf.index_switch %13 -> memref<10xi32> 
// CHECK:         case 0 {
// CHECK:           scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
// CHECK:         }
// CHECK:         case 1 {
// CHECK:           scf.yield %input_fifo_cons_buff_1 : memref<10xi32>
// CHECK:         }
// CHECK:         case 2 {
// CHECK:           scf.yield %input_fifo_cons_buff_2 : memref<10xi32>
// CHECK:         }
// CHECK:         default {
// CHECK:           scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
// CHECK:         }
// CHECK:         %15 = memref.load %buffer_0_2[%c1] : memref<2xi32>
// CHECK:         %16 = arith.index_cast %15 : i32 to index
// CHECK:         %17 = scf.index_switch %16 -> memref<10xi32> 
// CHECK:         case 0 {
// CHECK:           scf.yield %input_fifo_cons_buff_1 : memref<10xi32>
// CHECK:         }
// CHECK:         case 1 {
// CHECK:           scf.yield %input_fifo_cons_buff_2 : memref<10xi32>
// CHECK:         }
// CHECK:         case 2 {
// CHECK:           scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
// CHECK:         }
// CHECK:         default {
// CHECK:           scf.yield %input_fifo_cons_buff_1 : memref<10xi32>
// CHECK:         }
// CHECK:         func.call @add_10_i32(%14, %17, %11) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:         aie.use_lock(%input_fifo_cons_prod_lock, Release, 2)
// CHECK:         %18 = memref.load %buffer_0_2[%c1] : memref<2xi32>
// CHECK:         %c2_i32_2 = arith.constant 2 : i32
// CHECK:         %19 = arith.addi %18, %c2_i32_2 : i32
// CHECK:         %20 = arith.remsi %19, %c3_i32 : i32
// CHECK:         memref.store %20, %buffer_0_2[%c1] : memref<2xi32>
// CHECK:         aie.use_lock(%output_fifo_cons_lock, Release, 1)
// CHECK:         %21 = memref.load %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         %c1_i32_3 = arith.constant 1 : i32
// CHECK:         %22 = arith.addi %21, %c1_i32_3 : i32
// CHECK:         %23 = arith.remsi %22, %c2_i32 : i32
// CHECK:         memref.store %23, %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         aie.end
// CHECK:       }
// CHECK:       aie.shim_dma_allocation @input_fifo(MM2S, 0, 0)


module {
  aie.device(npu1_1col) {
    func.func @add_10_i32(%line_in1: memref<10xi32>, %line_in2: memref<10xi32>, %line_out: memref<10xi32>) -> () {
        return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @input_fifo(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<10xi32>>
    aie.objectfifo @output_fifo(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<10xi32>>

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
