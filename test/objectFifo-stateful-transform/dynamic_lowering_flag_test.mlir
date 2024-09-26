//===- dynamic_lowering_flag_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK:     %core_0_2 = aie.core(%tile_0_2) {
// CHECK:        %c0 = arith.constant 0 : index
// CHECK:        %c0_1 = arith.constant 0 : index
// CHECK:        %c2 = arith.constant 2 : index
// CHECK:        memref.store %c0, %buffer_0_2_0[%c0_1] : memref<2xindex>
// CHECK:        %c1 = arith.constant 1 : index
// CHECK:        %c2_2 = arith.constant 2 : index
// CHECK:        memref.store %c0, %buffer_0_2_0[%c1] : memref<2xindex>
// CHECK:        %c0_3 = arith.constant 0 : index
// CHECK:        %c0_4 = arith.constant 0 : index
// CHECK:        %c2_5 = arith.constant 2 : index
// CHECK:        memref.store %c0_3, %buffer_0_2[%c0_4] : memref<2xindex>
// CHECK:        %c1_6 = arith.constant 1 : index
// CHECK:        %c2_7 = arith.constant 2 : index
// CHECK:        memref.store %c0_3, %buffer_0_2[%c1_6] : memref<2xindex>
// CHECK:        %c0_8 = arith.constant 0 : index
// CHECK:        %c1_9 = arith.constant 1 : index
// CHECK:        %c10 = arith.constant 10 : index
// CHECK:        scf.for %arg0 = %c0_8 to %c10 step %c1_9 {
// CHECK:          aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)
// CHECK:          %0 = memref.load %buffer_0_2_0[%c0_1] : memref<2xindex>
// CHECK:          %1 = scf.index_switch %0 -> memref<10xi32> 
// CHECK:          case 0 {
// CHECK:            scf.yield %output_fifo_buff_0 : memref<10xi32>
// CHECK:          }
// CHECK:          case 1 {
// CHECK:            scf.yield %output_fifo_buff_1 : memref<10xi32>
// CHECK:          }
// CHECK:          default {
// CHECK:            scf.yield %output_fifo_buff_0 : memref<10xi32>
// CHECK:          }
// CHECK:          %2 = memref.load %buffer_0_2[%c0_4] : memref<2xindex>
// CHECK:          %3 = scf.index_switch %2 -> memref<10xi32> 
// CHECK:          case 0 {
// CHECK:            scf.yield %output_fifo_buff_0 : memref<10xi32>
// CHECK:          }
// CHECK:          case 1 {
// CHECK:            scf.yield %output_fifo_buff_1 : memref<10xi32>
// CHECK:          }
// CHECK:          default {
// CHECK:            scf.yield %output_fifo_buff_0 : memref<10xi32>
// CHECK:          }
// CHECK:          aie.use_lock(%input_fifo_cons_cons_lock, AcquireGreaterEqual, 1)
// CHECK:          %4 = memref.load %buffer_0_2_0[%c1] : memref<2xindex>
// CHECK:          %5 = scf.index_switch %4 -> memref<10xi32> 
// CHECK:          case 0 {
// CHECK:            scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
// CHECK:          }
// CHECK:          case 1 {
// CHECK:            scf.yield %input_fifo_cons_buff_1 : memref<10xi32>
// CHECK:          }
// CHECK:          default {
// CHECK:            scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
// CHECK:          }
// CHECK:          %6 = memref.load %buffer_0_2[%c1_6] : memref<2xindex>
// CHECK:          %7 = scf.index_switch %6 -> memref<10xi32> 
// CHECK:          case 0 {
// CHECK:            scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
// CHECK:          }
// CHECK:          case 1 {
// CHECK:            scf.yield %input_fifo_cons_buff_1 : memref<10xi32>
// CHECK:          }
// CHECK:          default {
// CHECK:            scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
// CHECK:          }
// CHECK:          func.call @passthrough_10_i32(%7, %3) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:          aie.use_lock(%input_fifo_cons_prod_lock, Release, 1)
// CHECK:          %8 = memref.load %buffer_0_2_0[%c1] : memref<2xindex>
// CHECK:          %c1_10 = arith.constant 1 : index
// CHECK:          %9 = arith.addi %8, %c1_10 : index
// CHECK:          %10 = arith.remsi %9, %c2_2 : index
// CHECK:          memref.store %10, %buffer_0_2_0[%c1] : memref<2xindex>
// CHECK:          %11 = memref.load %buffer_0_2[%c1_6] : memref<2xindex>
// CHECK:          %c1_11 = arith.constant 1 : index
// CHECK:          %12 = arith.addi %11, %c1_11 : index
// CHECK:          %13 = arith.remsi %12, %c2_7 : index
// CHECK:          memref.store %13, %buffer_0_2[%c1_6] : memref<2xindex>
// CHECK:          aie.use_lock(%output_fifo_cons_lock, Release, 1)
// CHECK:          %14 = memref.load %buffer_0_2_0[%c0_1] : memref<2xindex>
// CHECK:          %c1_12 = arith.constant 1 : index
// CHECK:          %15 = arith.addi %14, %c1_12 : index
// CHECK:          %16 = arith.remsi %15, %c2 : index
// CHECK:          memref.store %16, %buffer_0_2_0[%c0_1] : memref<2xindex>
// CHECK:          %17 = memref.load %buffer_0_2[%c0_4] : memref<2xindex>
// CHECK:          %c1_13 = arith.constant 1 : index
// CHECK:          %18 = arith.addi %17, %c1_13 : index
// CHECK:          %19 = arith.remsi %18, %c2_5 : index
// CHECK:          memref.store %19, %buffer_0_2[%c0_4] : memref<2xindex>
// CHECK:        }
// CHECK:        aie.end
// CHECK:      } {dynamic_objfifo_lowering = true}
// CHECK:      aie.shim_dma_allocation @input_fifo(MM2S, 0, 0)
// CHECK:      %core_0_4 = aie.core(%tile_0_4) {
// CHECK:        %c0 = arith.constant 0 : index
// CHECK:        %c1 = arith.constant 1 : index
// CHECK:        %c10 = arith.constant 10 : index
// CHECK:        %c2 = arith.constant 2 : index
// CHECK:        scf.for %arg0 = %c0 to %c10 step %c2 {
// CHECK:          aie.use_lock(%output_fifo2_prod_lock, AcquireGreaterEqual, 1)
// CHECK:          aie.use_lock(%input_fifo2_cons_cons_lock, AcquireGreaterEqual, 1)
// CHECK:          func.call @passthrough_10_i32(%input_fifo2_cons_buff_0, %output_fifo2_buff_0) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:          aie.use_lock(%input_fifo2_cons_prod_lock, Release, 1)
// CHECK:          aie.use_lock(%output_fifo2_cons_lock, Release, 1)
// CHECK:          aie.use_lock(%output_fifo2_prod_lock, AcquireGreaterEqual, 1)
// CHECK:          aie.use_lock(%input_fifo2_cons_cons_lock, AcquireGreaterEqual, 1)
// CHECK:          func.call @passthrough_10_i32(%input_fifo2_cons_buff_1, %output_fifo2_buff_1) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:          aie.use_lock(%input_fifo2_cons_prod_lock, Release, 1)
// CHECK:          aie.use_lock(%output_fifo2_cons_lock, Release, 1)
// CHECK:        }
// CHECK:        aie.end
// CHECK:      }

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
