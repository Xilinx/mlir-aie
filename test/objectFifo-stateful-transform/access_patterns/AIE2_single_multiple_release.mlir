//===- AIE2_single_multiple_release.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK:      %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:      %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK:      %[[VAL_2:.*]] = aie.tile(0, 3)
// CHECK:      %[[VAL_3:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of2_cons_buff_0"} : memref<16xi32> 
// CHECK:      %[[VAL_4:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of2_cons_buff_1"} : memref<16xi32> 
// CHECK:      %[[VAL_5:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of2_cons_buff_2"} : memref<16xi32> 
// CHECK:      %[[VAL_6:.*]] = aie.lock(%[[VAL_2]], 0) {init = 3 : i32, sym_name = "of2_cons_prod_lock_0"}
// CHECK:      %[[VAL_7:.*]] = aie.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "of2_cons_cons_lock_0"}
// CHECK:      %[[VAL_10:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_cons_buff_0"} : memref<16xi32> 
// CHECK:      %[[VAL_11:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_cons_buff_1"} : memref<16xi32> 
// CHECK:      %[[VAL_12:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_cons_buff_2"} : memref<16xi32> 
// CHECK:      %[[VAL_13:.*]] = aie.lock(%[[VAL_1]], 0) {init = 3 : i32, sym_name = "of_cons_prod_lock_0"}
// CHECK:      %[[VAL_14:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "of_cons_cons_lock_0"}
// CHECK:      aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:      aie.flow(%[[VAL_0]], DMA : 1, %[[VAL_2]], DMA : 0)
// CHECK:      func.func @some_work(%arg0: memref<16xi32>) {
// CHECK:        return
// CHECK:      }
// CHECK:      %[[VAL_15:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:        aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, 2)
// CHECK:        func.call @some_work(%[[VAL_10]]) : (memref<16xi32>) -> ()
// CHECK:        func.call @some_work(%[[VAL_11]]) : (memref<16xi32>) -> ()
// CHECK:        aie.use_lock(%[[VAL_13]], Release, 2)
// CHECK:        aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, 1)
// CHECK:        func.call @some_work(%[[VAL_12]]) : (memref<16xi32>) -> ()
// CHECK:        aie.use_lock(%[[VAL_13]], Release, 1)
// CHECK:        aie.end
// CHECK:      }
// CHECK:      %[[VAL_16:.*]] = aie.core(%[[VAL_2]]) {
// CHECK:        aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 2)
// CHECK:        func.call @some_work(%[[VAL_3]]) : (memref<16xi32>) -> ()
// CHECK:        func.call @some_work(%[[VAL_4]]) : (memref<16xi32>) -> ()
// CHECK:        aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:        aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:        aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:        func.call @some_work(%[[VAL_5]]) : (memref<16xi32>) -> ()
// CHECK:        aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:        aie.end
// CHECK:      }

module @single_multiple_release {
    aie.device(npu1_1col) {
        %tile00 = aie.tile(0, 0)
        %tile02 = aie.tile(0, 2)
        %tile03 = aie.tile(0, 3)

        aie.objectfifo @of (%tile00, {%tile02}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of2 (%tile00, {%tile03}, 3 : i32) : !aie.objectfifo<memref<16xi32>>

        func.func @some_work(%line_in:memref<16xi32>) -> () {
            return
        }

        %core12 = aie.core(%tile02) {
            %0 = aie.objectfifo.acquire @of(Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %2 = aie.objectfifo.subview.access %0[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%1) : (memref<16xi32>) -> ()
            func.call @some_work(%2) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of(Consume, 2)
            %3 = aie.objectfifo.acquire @of(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
            %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%4) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of(Consume, 1)
            aie.end
        }

        %core13 = aie.core(%tile03) {
            %0 = aie.objectfifo.acquire @of2(Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %2 = aie.objectfifo.subview.access %0[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%1) : (memref<16xi32>) -> ()
            func.call @some_work(%2) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of2(Consume, 1)
            aie.objectfifo.release @of2(Consume, 1)
            %3 = aie.objectfifo.acquire @of2(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
            %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%4) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of2(Consume, 1)
            aie.end
        }
    }
}
