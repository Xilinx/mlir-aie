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

// CHECK:       %core_0_2 = aie.core(%tile_0_2) {
// CHECK:         %c0 = arith.constant 0 : index
// CHECK:         %c0_1 = arith.constant 0 : index
// CHECK:         %c2 = arith.constant 2 : index
// CHECK:         memref.store %c0, %buffer_0_2[%c0_1] : memref<2xindex>
// CHECK:         %c1 = arith.constant 1 : index
// CHECK:         %c3 = arith.constant 3 : index
// CHECK:         memref.store %c0, %buffer_0_2[%c1] : memref<2xindex>
// CHECK:         %c0_2 = arith.constant 0 : index
// CHECK:         %c1_3 = arith.constant 1 : index
// CHECK:         %c9 = arith.constant 9 : index
// CHECK:         aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)
// CHECK:         %0 = memref.load %buffer_0_2[%c0_1] : memref<2xindex>
// CHECK:         %1 = scf.index_switch %0 -> memref<10xi32> 
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
// CHECK:         %2 = memref.load %buffer_0_2[%c1] : memref<2xindex>
// CHECK:         %3 = scf.index_switch %2 -> memref<10xi32> 
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
// CHECK:         func.call @add_10_i32(%3, %3, %1) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:         aie.use_lock(%output_fifo_cons_lock, Release, 1)
// CHECK:         %4 = memref.load %buffer_0_2[%c0_1] : memref<2xindex>
// CHECK:         %c1_4 = arith.constant 1 : index
// CHECK:         %5 = arith.addi %4, %c1_4 : index
// CHECK:         %6 = arith.remsi %5, %c2 : index
// CHECK:         memref.store %6, %buffer_0_2[%c0_1] : memref<2xindex>
// CHECK:         scf.for %arg0 = %c0_2 to %c9 step %c1_3 {
// CHECK:           aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)
// CHECK:           %19 = memref.load %buffer_0_2[%c0_1] : memref<2xindex>
// CHECK:           %20 = scf.index_switch %19 -> memref<10xi32> 
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
// CHECK:           %21 = memref.load %buffer_0_2[%c1] : memref<2xindex>
// CHECK:           %22 = scf.index_switch %21 -> memref<10xi32> 
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
// CHECK:           %23 = memref.load %buffer_0_2[%c1] : memref<2xindex>
// CHECK:           %24 = scf.index_switch %23 -> memref<10xi32> 
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
// CHECK:           func.call @add_10_i32(%22, %24, %20) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:           aie.use_lock(%input_fifo_cons_prod_lock, Release, 1)
// CHECK:           %25 = memref.load %buffer_0_2[%c1] : memref<2xindex>
// CHECK:           %c1_7 = arith.constant 1 : index
// CHECK:           %26 = arith.addi %25, %c1_7 : index
// CHECK:           %27 = arith.remsi %26, %c3 : index
// CHECK:           memref.store %27, %buffer_0_2[%c1] : memref<2xindex>
// CHECK:           aie.use_lock(%output_fifo_cons_lock, Release, 1)
// CHECK:           %28 = memref.load %buffer_0_2[%c0_1] : memref<2xindex>
// CHECK:           %c1_8 = arith.constant 1 : index
// CHECK:           %29 = arith.addi %28, %c1_8 : index
// CHECK:           %30 = arith.remsi %29, %c2 : index
// CHECK:           memref.store %30, %buffer_0_2[%c0_1] : memref<2xindex>
// CHECK:         }
// CHECK:         aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)
// CHECK:         %7 = memref.load %buffer_0_2[%c0_1] : memref<2xindex>
// CHECK:         %8 = scf.index_switch %7 -> memref<10xi32> 
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
// CHECK:         %9 = memref.load %buffer_0_2[%c1] : memref<2xindex>
// CHECK:         %10 = scf.index_switch %9 -> memref<10xi32> 
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
// CHECK:         %11 = memref.load %buffer_0_2[%c1] : memref<2xindex>
// CHECK:         %12 = scf.index_switch %11 -> memref<10xi32> 
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
// CHECK:         func.call @add_10_i32(%10, %12, %8) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:         aie.use_lock(%input_fifo_cons_prod_lock, Release, 2)
// CHECK:         %13 = memref.load %buffer_0_2[%c1] : memref<2xindex>
// CHECK:         %c2_5 = arith.constant 2 : index
// CHECK:         %14 = arith.addi %13, %c2_5 : index
// CHECK:         %15 = arith.remsi %14, %c3 : index
// CHECK:         memref.store %15, %buffer_0_2[%c1] : memref<2xindex>
// CHECK:         aie.use_lock(%output_fifo_cons_lock, Release, 1)
// CHECK:         %16 = memref.load %buffer_0_2[%c0_1] : memref<2xindex>
// CHECK:         %c1_6 = arith.constant 1 : index
// CHECK:         %17 = arith.addi %16, %c1_6 : index
// CHECK:         %18 = arith.remsi %17, %c2 : index
// CHECK:         memref.store %18, %buffer_0_2[%c0_1] : memref<2xindex>
// CHECK:         aie.end
// CHECK:       }

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
