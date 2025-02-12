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
// CHECK:       %output_fifo_cons_prod_lock = aie.lock(%{{.*}}tile_0_0, 2) {init = 1 : i32, sym_name = "output_fifo_cons_prod_lock"}
// CHECK:       %output_fifo_cons_cons_lock = aie.lock(%{{.*}}tile_0_0, 3) {init = 0 : i32, sym_name = "output_fifo_cons_cons_lock"}
// CHECK:       %output_fifo_buff_0 = aie.buffer(%{{.*}}tile_0_2) {sym_name = "output_fifo_buff_0"} : memref<10xi32> 
// CHECK:       %output_fifo_buff_1 = aie.buffer(%{{.*}}tile_0_2) {sym_name = "output_fifo_buff_1"} : memref<10xi32> 
// CHECK:       %output_fifo_prod_lock = aie.lock(%{{.*}}tile_0_2, 2) {init = 2 : i32, sym_name = "output_fifo_prod_lock"}
// CHECK:       %output_fifo_cons_lock = aie.lock(%{{.*}}tile_0_2, 3) {init = 0 : i32, sym_name = "output_fifo_cons_lock"}
// CHECK:       %input_fifo_cons_buff_0 = aie.buffer(%{{.*}}tile_0_2) {sym_name = "input_fifo_cons_buff_0"} : memref<10xi32> 
// CHECK:       %input_fifo_cons_buff_1 = aie.buffer(%{{.*}}tile_0_2) {sym_name = "input_fifo_cons_buff_1"} : memref<10xi32> 
// CHECK:       %input_fifo_cons_prod_lock = aie.lock(%{{.*}}tile_0_2, 0) {init = 2 : i32, sym_name = "input_fifo_cons_prod_lock"}
// CHECK:       %input_fifo_cons_cons_lock = aie.lock(%{{.*}}tile_0_2, 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock"}
// CHECK:       %input_fifo_prod_lock = aie.lock(%{{.*}}tile_0_0, 0) {init = 1 : i32, sym_name = "input_fifo_prod_lock"}
// CHECK:       %input_fifo_cons_lock = aie.lock(%{{.*}}tile_0_0, 1) {init = 0 : i32, sym_name = "input_fifo_cons_lock"}
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
// CHECK:         default {
// CHECK:           scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
// CHECK:         }
// CHECK:         func.call @add_10_i32(%5, %5, %2) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:         aie.use_lock(%output_fifo_cons_lock, Release, 1)
// CHECK:         %6 = memref.load %buffer_0_2[%c0] : memref<2xi32>
// CHECK:         %c1_i32 = arith.constant 1 : i32
// CHECK:         %7 = arith.addi %6, %c1_i32 : i32
// CHECK:         %8 = arith.cmpi sge, %7, %c2_i32 : i32
// CHECK:         %9 = arith.subi %7, %c2_i32 : i32
// CHECK:         %10 = arith.select %8, %9, %7 : i32
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

      aie.end
    }
  }
}
