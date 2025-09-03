//===- same_depth_objectfifos_test.mlir -------------------------*- MLIR -*-===//
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
// CHECK:       %{{.*}}tile_0_0 = aie.tile(0, 0)
// CHECK:       %{{.*}}tile_0_2 = aie.tile(0, 2)
// CHECK:       %[[VAL_2:.*]] = aie.buffer(%{{.*}}tile_0_2) {sym_name = "output_fifo_buff_0"} : memref<10xi32> 
// CHECK:       %[[VAL_3:.*]] = aie.buffer(%{{.*}}tile_0_2) {sym_name = "output_fifo_buff_1"} : memref<10xi32> 
// CHECK:       %[[VAL_4:.*]] = aie.lock(%{{.*}}tile_0_2, 2) {init = 2 : i32, sym_name = "output_fifo_prod_lock_0"}
// CHECK:       %[[VAL_5:.*]] = aie.lock(%{{.*}}tile_0_2, 3) {init = 0 : i32, sym_name = "output_fifo_cons_lock_0"}
// CHECK:       %[[VAL_6:.*]] = aie.buffer(%{{.*}}tile_0_2) {sym_name = "input_fifo_cons_buff_0"} : memref<10xi32> 
// CHECK:       %[[VAL_7:.*]] = aie.buffer(%{{.*}}tile_0_2) {sym_name = "input_fifo_cons_buff_1"} : memref<10xi32> 
// CHECK:       %[[VAL_8:.*]] = aie.lock(%{{.*}}tile_0_2, 0) {init = 2 : i32, sym_name = "input_fifo_cons_prod_lock_0"}
// CHECK:       %[[VAL_9:.*]] = aie.lock(%{{.*}}tile_0_2, 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock_0"}
// CHECK:       aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_2, DMA : 0)
// CHECK:       aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_0_0, DMA : 0)
// CHECK:       %buffer_0_2 = aie.buffer(%{{.*}}tile_0_2) : memref<2xi32> 
// CHECK:       %core_0_2 = aie.core(%{{.*}}tile_0_2) {
// CHECK:         %c0_i32 = arith.constant 0 : i32
// CHECK:         %c0 = arith.constant 0 : index
// CHECK:         %c2_i32 = arith.constant 2 : i32
// CHECK:         memref.store %c0_i32, %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         %c1 = arith.constant 1 : index
// CHECK:         %c2_i32_0 = arith.constant 2 : i32
// CHECK:         memref.store %c0_i32, %buffer_0_2[%c1] : memref<2xi32>
// CHECK:         %c0_1 = arith.constant 0 : index
// CHECK:         %c1_2 = arith.constant 1 : index
// CHECK:         %c9 = arith.constant 9 : index
// CHECK:         aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:         %0 = memref.load %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         %1 = arith.index_cast %0 : i32 to index
// CHECK:         %2 = scf.index_switch %1 -> memref<10xi32> 
// CHECK:         case 0 {
// CHECK:           scf.yield %[[VAL_2]] : memref<10xi32>
// CHECK:         }
// CHECK:         case 1 {
// CHECK:           scf.yield %[[VAL_3]] : memref<10xi32>
// CHECK:         }
// CHECK:         default {
// CHECK:           scf.yield %[[VAL_2]] : memref<10xi32>
// CHECK:         }
// CHECK:         aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:         %3 = memref.load %buffer_0_2[%c1] : memref<2xi32>
// CHECK:         %4 = arith.index_cast %3 : i32 to index
// CHECK:         %5 = scf.index_switch %4 -> memref<10xi32> 
// CHECK:         case 0 {
// CHECK:           scf.yield %[[VAL_6]] : memref<10xi32>
// CHECK:         }
// CHECK:         case 1 {
// CHECK:           scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:         }
// CHECK:         default {
// CHECK:           scf.yield %[[VAL_6]] : memref<10xi32>
// CHECK:         }
// CHECK:         func.call @add_10_i32(%5, %5, %2) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:         aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:         %6 = memref.load %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         %c1_i32 = arith.constant 1 : i32
// CHECK:         %7 = arith.addi %6, %c1_i32 : i32
// CHECK:         %8 = arith.cmpi sge, %7, %c2_i32 : i32
// CHECK:         %9 = arith.subi %7, %c2_i32 : i32
// CHECK:         %10 = arith.select %8, %9, %7 : i32
// CHECK:         memref.store %10, %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         scf.for %arg0 = %c0_1 to %c9 step %c1_2 {
// CHECK:           aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:           %30 = memref.load %buffer_0_2[%c0] : memref<2xi32>
// CHECK:           %31 = arith.index_cast %30 : i32 to index
// CHECK:           %32 = scf.index_switch %31 -> memref<10xi32> 
// CHECK:           case 0 {
// CHECK:             scf.yield %[[VAL_2]] : memref<10xi32>
// CHECK:           }
// CHECK:           case 1 {
// CHECK:             scf.yield %[[VAL_3]] : memref<10xi32>
// CHECK:           }
// CHECK:           default {
// CHECK:             scf.yield %[[VAL_2]] : memref<10xi32>
// CHECK:           }
// CHECK:           aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:           %33 = memref.load %buffer_0_2[%c1] : memref<2xi32>
// CHECK:           %34 = arith.index_cast %33 : i32 to index
// CHECK:           %35 = scf.index_switch %34 -> memref<10xi32> 
// CHECK:           case 0 {
// CHECK:             scf.yield %[[VAL_6]] : memref<10xi32>
// CHECK:           }
// CHECK:           case 1 {
// CHECK:             scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:           }
// CHECK:           default {
// CHECK:             scf.yield %[[VAL_6]] : memref<10xi32>
// CHECK:           }
// CHECK:           %36 = memref.load %buffer_0_2[%c1] : memref<2xi32>
// CHECK:           %37 = arith.index_cast %36 : i32 to index
// CHECK:           %38 = scf.index_switch %37 -> memref<10xi32> 
// CHECK:           case 0 {
// CHECK:             scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:           }
// CHECK:           case 1 {
// CHECK:             scf.yield %[[VAL_6]] : memref<10xi32>
// CHECK:           }
// CHECK:           default {
// CHECK:             scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:           }
// CHECK:           func.call @add_10_i32(%35, %38, %32) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:           aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:           %39 = memref.load %buffer_0_2[%c1] : memref<2xi32>
// CHECK:           %c1_i32_5 = arith.constant 1 : i32
// CHECK:           %40 = arith.addi %39, %c1_i32_5 : i32
// CHECK:           %41 = arith.cmpi sge, %40, %c2_i32_0 : i32
// CHECK:           %42 = arith.subi %40, %c2_i32_0 : i32
// CHECK:           %43 = arith.select %41, %42, %40 : i32
// CHECK:           memref.store %43, %buffer_0_2[%c1] : memref<2xi32>
// CHECK:           aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:           %44 = memref.load %buffer_0_2[%c0] : memref<2xi32>
// CHECK:           %c1_i32_6 = arith.constant 1 : i32
// CHECK:           %45 = arith.addi %44, %c1_i32_6 : i32
// CHECK:           %46 = arith.cmpi sge, %45, %c2_i32 : i32
// CHECK:           %47 = arith.subi %45, %c2_i32 : i32
// CHECK:           %48 = arith.select %46, %47, %45 : i32
// CHECK:           memref.store %48, %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         }
// CHECK:         aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:         %11 = memref.load %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         %12 = arith.index_cast %11 : i32 to index
// CHECK:         %13 = scf.index_switch %12 -> memref<10xi32> 
// CHECK:         case 0 {
// CHECK:           scf.yield %[[VAL_2]] : memref<10xi32>
// CHECK:         }
// CHECK:         case 1 {
// CHECK:           scf.yield %[[VAL_3]] : memref<10xi32>
// CHECK:         }
// CHECK:         default {
// CHECK:           scf.yield %[[VAL_2]] : memref<10xi32>
// CHECK:         }
// CHECK:         aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:         %14 = memref.load %buffer_0_2[%c1] : memref<2xi32>
// CHECK:         %15 = arith.index_cast %14 : i32 to index
// CHECK:         %16 = scf.index_switch %15 -> memref<10xi32> 
// CHECK:         case 0 {
// CHECK:           scf.yield %[[VAL_6]] : memref<10xi32>
// CHECK:         }
// CHECK:         case 1 {
// CHECK:           scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:         }
// CHECK:         default {
// CHECK:           scf.yield %[[VAL_6]] : memref<10xi32>
// CHECK:         }
// CHECK:         %17 = memref.load %buffer_0_2[%c1] : memref<2xi32>
// CHECK:         %18 = arith.index_cast %17 : i32 to index
// CHECK:         %19 = scf.index_switch %18 -> memref<10xi32> 
// CHECK:         case 0 {
// CHECK:           scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:         }
// CHECK:         case 1 {
// CHECK:           scf.yield %[[VAL_6]] : memref<10xi32>
// CHECK:         }
// CHECK:        default {
// CHECK:           scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:         }
// CHECK:         func.call @add_10_i32(%16, %19, %13) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:         aie.use_lock(%[[VAL_8]], Release, 2)
// CHECK:         %20 = memref.load %buffer_0_2[%c1] : memref<2xi32>
// CHECK:         %c2_i32_3 = arith.constant 2 : i32
// CHECK:         %21 = arith.addi %20, %c2_i32_3 : i32
// CHECK:         %22 = arith.cmpi sge, %21, %c2_i32_0 : i32
// CHECK:         %23 = arith.subi %21, %c2_i32_0 : i32
// CHECK:         %24 = arith.select %22, %23, %21 : i32
// CHECK:         memref.store %24, %buffer_0_2[%c1] : memref<2xi32>
// CHECK:         aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:         %25 = memref.load %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         %c1_i32_4 = arith.constant 1 : i32
// CHECK:         %26 = arith.addi %25, %c1_i32_4 : i32
// CHECK:         %27 = arith.cmpi sge, %26, %c2_i32 : i32
// CHECK:         %28 = arith.subi %26, %c2_i32 : i32
// CHECK:         %29 = arith.select %27, %28, %26 : i32
// CHECK:         memref.store %29, %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         aie.end
// CHECK:       }

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
