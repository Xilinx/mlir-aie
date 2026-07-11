//===- dynamic_runtime_lock_basic.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Dynamic objectFifo lowering emits runtime lock bookkeeping. Each acquire
// tracks how many elements are already held in a per-fifo i32 counter and
// acquires only the delta at runtime via a value-carrying
// `AcquireGreaterEqual`. Each release decrements the same counter. This
// replaces the old static cyclostatic analysis / loop peeling.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifo_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifo_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_1]], 0) {init = 2 : i32, sym_name = "fifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "fifo_cons_cons_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_3"} : memref<8xi8>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_0]], 0) {init = 4 : i32, sym_name = "fifo_prod_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_1]]) : memref<1xi32>
// CHECK:           %[[VAL_13:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_14:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_15:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_17:.*]] = arith.constant 2 : i32
// CHECK:             memref.store %[[VAL_15]], %[[VAL_12]]{{\[}}%[[VAL_16]]] : memref<1xi32>
// CHECK:             %[[VAL_18:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_20:.*]] = arith.constant 14 : index
// CHECK:             %[[VAL_21:.*]] = scf.for %[[VAL_22:.*]] = %[[VAL_18]] to %[[VAL_20]] step %[[VAL_19]] iter_args(%[[VAL_23:.*]] = %[[VAL_14]]) -> (i32) {
// CHECK:               %[[VAL_24:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_25:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_26:.*]] = arith.subi %[[VAL_24]], %[[VAL_23]] : i32
// CHECK:               %[[VAL_27:.*]] = arith.maxsi %[[VAL_26]], %[[VAL_25]] : i32
// CHECK:               aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_27]])
// CHECK:               %[[VAL_28:.*]] = arith.addi %[[VAL_23]], %[[VAL_27]] : i32
// CHECK:               %[[VAL_29:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_16]]] : memref<1xi32>
// CHECK:               %[[VAL_30:.*]] = arith.index_cast %[[VAL_29]] : i32 to index
// CHECK:               %[[VAL_31:.*]] = scf.index_switch %[[VAL_30]] -> memref<8xi8>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_2]] : memref<8xi8>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_3]] : memref<8xi8>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_2]] : memref<8xi8>
// CHECK:               }
// CHECK:               %[[VAL_32:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_4]], Release, %[[VAL_32]])
// CHECK:               %[[VAL_33:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_34:.*]] = arith.subi %[[VAL_28]], %[[VAL_33]] : i32
// CHECK:               %[[VAL_35:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_16]]] : memref<1xi32>
// CHECK:               %[[VAL_36:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_37:.*]] = arith.addi %[[VAL_35]], %[[VAL_36]] : i32
// CHECK:               %[[VAL_38:.*]] = arith.cmpi sge, %[[VAL_37]], %[[VAL_17]] : i32
// CHECK:               %[[VAL_39:.*]] = arith.subi %[[VAL_37]], %[[VAL_17]] : i32
// CHECK:               %[[VAL_40:.*]] = arith.select %[[VAL_38]], %[[VAL_39]], %[[VAL_37]] : i32
// CHECK:               memref.store %[[VAL_40]], %[[VAL_12]]{{\[}}%[[VAL_16]]] : memref<1xi32>
// CHECK:               scf.yield %[[VAL_34]] : i32
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_41:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_42:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_43:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_43]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_44:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_44]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_45:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_45]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_46:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_46]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_47:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_47]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_48:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_48]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             %[[VAL_49:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_49]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_50:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_50]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_51:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_52:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_53:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_53]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_54:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_54]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_55:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_55]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_56:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_56]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 4 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %a = aie.objectfifo.acquire @fifo(Consume, 1) : !aie.objectfifosubview<memref<8xi8>>
        %e = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<8xi8>> -> memref<8xi8>
        aie.objectfifo.release @fifo(Consume, 1)
      }
      aie.end
    }
  }
}
