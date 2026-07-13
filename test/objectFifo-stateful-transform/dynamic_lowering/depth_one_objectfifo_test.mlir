//===- depth_one_objectfifo_test.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=false" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK:           func.func @passthrough_10_i32(%[[VAL_0:.*]]: memref<10xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_3:.*]] = aie.tile(0, 4)
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "input_fifo_cons_buff_0"} : memref<10xi32>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_2]], 0) {init = 1 : i32, sym_name = "input_fifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32, sym_name = "input_fifo_prod_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "input_fifo_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_2]]) : memref<1xi32>
// CHECK:           %[[VAL_10:.*]] = aie.core(%[[VAL_2]]) {
// CHECK:             %[[VAL_11:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_12:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_13:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_14:.*]] = arith.constant 1 : i32
// CHECK:             memref.store %[[VAL_12]], %[[VAL_9]]{{\[}}%[[VAL_13]]] : memref<1xi32>
// CHECK:             %[[VAL_15:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_16:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_17:.*]] = arith.constant 10 : index
// CHECK:             %[[VAL_18:.*]] = scf.for %[[VAL_19:.*]] = %[[VAL_15]] to %[[VAL_17]] step %[[VAL_16]] iter_args(%[[VAL_20:.*]] = %[[VAL_11]]) -> (i32) {
// CHECK:               %[[VAL_21:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_22:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_23:.*]] = arith.subi %[[VAL_21]], %[[VAL_20]] : i32
// CHECK:               %[[VAL_24:.*]] = arith.maxsi %[[VAL_23]], %[[VAL_22]] : i32
// CHECK:               aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:               %[[VAL_25:.*]] = arith.addi %[[VAL_20]], %[[VAL_24]] : i32
// CHECK:               func.call @passthrough_10_i32(%[[VAL_4]]) : (memref<10xi32>) -> ()
// CHECK:               %[[VAL_26:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_5]], Release, %[[VAL_26]])
// CHECK:               %[[VAL_27:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_28:.*]] = arith.subi %[[VAL_25]], %[[VAL_27]] : i32
// CHECK:               %[[VAL_29:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_13]]] : memref<1xi32>
// CHECK:               %[[VAL_30:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_31:.*]] = arith.addi %[[VAL_29]], %[[VAL_30]] : i32
// CHECK:               %[[VAL_32:.*]] = arith.cmpi sge, %[[VAL_31]], %[[VAL_14]] : i32
// CHECK:               %[[VAL_33:.*]] = arith.subi %[[VAL_31]], %[[VAL_14]] : i32
// CHECK:               %[[VAL_34:.*]] = arith.select %[[VAL_32]], %[[VAL_33]], %[[VAL_31]] : i32
// CHECK:               memref.store %[[VAL_34]], %[[VAL_9]]{{\[}}%[[VAL_13]]] : memref<1xi32>
// CHECK:               scf.yield %[[VAL_28]] : i32
// CHECK:             }
// CHECK:             aie.end
// CHECK:           } {dynamic_objfifo_lowering = true}
// CHECK:           aie.shim_dma_allocation @input_fifo_shim_alloc(%[[VAL_1]], MM2S, 0)
// CHECK:           %[[VAL_35:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_36:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_37:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_37]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_38:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_38]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module {
  aie.device(npu1_1col) {
    func.func @passthrough_10_i32(%line_in: memref<10xi32>) -> () {
        return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)
    aie.objectfifo @input_fifo(%tile_0_0, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<10xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index

      scf.for %arg0 = %c0 to %c10 step %c1 {
        %0 = aie.objectfifo.acquire @input_fifo(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        func.call @passthrough_10_i32(%1) : (memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo(Consume, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
