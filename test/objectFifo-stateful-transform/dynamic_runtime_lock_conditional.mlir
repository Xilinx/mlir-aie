//===- dynamic_runtime_lock_conditional.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An acquire in the loop body with a release nested in an `scf.if`. The
// runtime held-counter bookkeeping handles conditional acquire/release with
// no static analysis: the held decrement and the buffer-index update are
// emitted inside the `scf.if`, so they only run when the branch is taken.
// The next iteration's acquire computes its delta from the current held
// value, so it acquires nothing extra when the release did not fire.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK-LABEL: module {
// CHECK:         aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifo_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifo_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifo_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifo_cons_buff_3"} : memref<8xi8>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]], 0) {init = 4 : i32, sym_name = "fifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "fifo_cons_cons_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_0]], 0) {init = 3 : i32, sym_name = "fifo_prod_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_1]]) : memref<1xi32>
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_1]]) : memref<1xi32>
// CHECK:           %[[VAL_15:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_16:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_17:.*]] = arith.constant 0 : index
// CHECK:             memref.store %[[VAL_16]], %[[VAL_14]]{{\[}}%[[VAL_17]]] : memref<1xi32>
// CHECK:             %[[VAL_18:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_19:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_20:.*]] = arith.constant 4 : i32
// CHECK:             memref.store %[[VAL_18]], %[[VAL_13]]{{\[}}%[[VAL_19]]] : memref<1xi32>
// CHECK:             %[[VAL_21:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_22:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_23:.*]] = arith.constant 14 : index
// CHECK:             %[[VAL_24:.*]] = arith.constant true
// CHECK:             scf.for %[[VAL_25:.*]] = %[[VAL_21]] to %[[VAL_23]] step %[[VAL_22]] {
// CHECK:               %[[VAL_26:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_17]]] : memref<1xi32>
// CHECK:               %[[VAL_27:.*]] = arith.constant 3 : i32
// CHECK:               %[[VAL_28:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_29:.*]] = arith.subi %[[VAL_27]], %[[VAL_26]] : i32
// CHECK:               %[[VAL_30:.*]] = arith.maxsi %[[VAL_29]], %[[VAL_28]] : i32
// CHECK:               aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_30]])
// CHECK:               %[[VAL_31:.*]] = arith.addi %[[VAL_26]], %[[VAL_30]] : i32
// CHECK:               memref.store %[[VAL_31]], %[[VAL_14]]{{\[}}%[[VAL_17]]] : memref<1xi32>
// CHECK:               scf.if %[[VAL_24]] {
// CHECK:                 %[[VAL_32:.*]] = arith.constant 1 : i32
// CHECK:                 aie.use_lock(%[[VAL_6]], Release, %[[VAL_32]])
// CHECK:                 %[[VAL_33:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_17]]] : memref<1xi32>
// CHECK:                 %[[VAL_34:.*]] = arith.constant 1 : i32
// CHECK:                 %[[VAL_35:.*]] = arith.subi %[[VAL_33]], %[[VAL_34]] : i32
// CHECK:                 memref.store %[[VAL_35]], %[[VAL_14]]{{\[}}%[[VAL_17]]] : memref<1xi32>
// CHECK:                 %[[VAL_36:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_19]]] : memref<1xi32>
// CHECK:                 %[[VAL_37:.*]] = arith.constant 1 : i32
// CHECK:                 %[[VAL_38:.*]] = arith.addi %[[VAL_36]], %[[VAL_37]] : i32
// CHECK:                 %[[VAL_39:.*]] = arith.cmpi sge, %[[VAL_38]], %[[VAL_20]] : i32
// CHECK:                 %[[VAL_40:.*]] = arith.subi %[[VAL_38]], %[[VAL_20]] : i32
// CHECK:                 %[[VAL_41:.*]] = arith.select %[[VAL_39]], %[[VAL_40]], %[[VAL_38]] : i32
// CHECK:                 memref.store %[[VAL_41]], %[[VAL_13]]{{\[}}%[[VAL_19]]] : memref<1xi32>
// CHECK:               }
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_42:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_43:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_44:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_44]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_45:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_45]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_46:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_46]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_47:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_47]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_48:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_48]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_49:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_49]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_50:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_51:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_52:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_52]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_53:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_53]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_54:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_54]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_55:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_55]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_56:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_56]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_57:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_57]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             %[[VAL_58:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_58]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_59:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_59]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
// CHECK:       }

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %true = arith.constant true
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %x = aie.objectfifo.acquire @fifo(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        scf.if %true {
          aie.objectfifo.release @fifo(Consume, 1)
        }
      }
      aie.end
    }
  }
}
