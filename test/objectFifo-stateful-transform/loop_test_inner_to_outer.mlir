//===- loop_test_inner_to_outer.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   module {
// CHECK:             aie.device(npu1_1col) {
// CHECK:               memref.global "public" @of_2 : memref<16xi32>
// CHECK:               memref.global "public" @of_1 : memref<16xi32>
// CHECK:               %tile_0_2 = aie.tile(0, 2)
// CHECK:               %tile_0_3 = aie.tile(0, 3)
// CHECK:               %of_2_buff_0 = aie.buffer(%tile_0_2) {sym_name = "of_2_buff_0"} : memref<16xi32> 
// CHECK:               %of_2_prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "of_2_prod_lock"}
// CHECK:               %of_2_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "of_2_cons_lock"}
// CHECK:               %of_1_buff_0 = aie.buffer(%tile_0_3) {sym_name = "of_1_buff_0"} : memref<16xi32> 
// CHECK:               %of_1_prod_lock = aie.lock(%tile_0_3, 0) {init = 1 : i32, sym_name = "of_1_prod_lock"}
// CHECK:               %of_1_cons_lock = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "of_1_cons_lock"}
// CHECK:               func.func @some_work(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: index, %arg3: index) {
// CHECK:                 return
// CHECK:               }
// CHECK:               %core_0_2 = aie.core(%tile_0_2) {
// CHECK:                 %c0 = arith.constant 0 : index
// CHECK:                 %c1 = arith.constant 1 : index
// CHECK:                 %c12 = arith.constant 12 : index
// CHECK:                 %c13 = arith.constant 13 : index
// CHECK:                 scf.for %arg0 = %c0 to %c12 step %c1 {
// CHECK:                   scf.for %arg1 = %c0 to %c13 step %c1 {
// CHECK:                     aie.use_lock(%of_1_cons_lock, AcquireGreaterEqual, 1)
// CHECK:                     scf.for %arg2 = %c0 to %c13 step %c1 {
// CHECK:                       aie.use_lock(%of_2_prod_lock, AcquireGreaterEqual, 1)
// CHECK:                       func.call @some_work(%of_1_buff_0, %of_2_buff_0, %arg0, %arg1) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                       aie.use_lock(%of_2_cons_lock, Release, 1)
// CHECK:                     }
// CHECK:                     aie.use_lock(%of_1_prod_lock, Release, 1)
// CHECK:                   }
// CHECK:                 }
// CHECK:                 scf.for %arg0 = %c0 to %c13 step %c1 {
// CHECK:                   aie.use_lock(%of_1_cons_lock, AcquireGreaterEqual, 1)
// CHECK:                   scf.for %arg1 = %c0 to %c13 step %c1 {
// CHECK:                     aie.use_lock(%of_2_prod_lock, AcquireGreaterEqual, 1)
// CHECK:                     func.call @some_work(%of_1_buff_0, %of_2_buff_0, %c0, %arg0) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                     aie.use_lock(%of_2_cons_lock, Release, 1)
// CHECK:                   }
// CHECK:                   aie.use_lock(%of_1_prod_lock, Release, 1)
// CHECK:                 }
// CHECK:                 aie.end
// CHECK:               }
// CHECK:             }
// CHECK:           }

module {
  aie.device(npu1_1col) {
    %tile12 = aie.tile(0, 2)
    %tile13 = aie.tile(0, 3)
    aie.objectfifo @of_1 (%tile13, {%tile12}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_2 (%tile12, {%tile13}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    func.func @some_work(%line_inA:memref<16xi32>, %line_inB:memref<16xi32>, %indexH:index, %indexW:index) -> () {
      return
    }
    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12 = arith.constant 12 : index
      %c13 = arith.constant 13 : index

      scf.for %indexInHeight = %c0 to %c12 step %c1 {

        scf.for %indexInWidth = %c0 to %c13 step %c1 {
          %subviewIn = aie.objectfifo.acquire @of_1 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>

          scf.for %i = %c0 to %c13 step %c1 {
            %subviewOut = aie.objectfifo.acquire @of_2 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elemIn, %elemOut, %indexInHeight, %indexInWidth) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
            aie.objectfifo.release @of_2 (Produce, 1)
          }

          aie.objectfifo.release @of_1 (Consume, 1)
        }
      }

      scf.for %indexInWidth = %c0 to %c13 step %c1 {
        %subviewIn1 = aie.objectfifo.acquire @of_1 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>

        scf.for %i = %c0 to %c13 step %c1 {
          %subviewOut1 = aie.objectfifo.acquire @of_2 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
          %elemIn = aie.objectfifo.subview.access %subviewIn1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
          %elemOut = aie.objectfifo.subview.access %subviewOut1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
          func.call @some_work(%elemIn, %elemOut, %c0, %indexInWidth) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
          aie.objectfifo.release @of_2 (Produce, 1)
        }

        aie.objectfifo.release @of_1 (Consume, 1)
      }

      aie.end
    }
  }
}
