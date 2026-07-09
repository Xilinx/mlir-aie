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

// CHECK-LABEL: module {
// CHECK:         aie.device(npu2) {
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
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_1]]) : memref<1xi32>
// CHECK:           %[[VAL_14:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_15:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK:             memref.store %[[VAL_15]], %[[VAL_13]]{{\[}}%[[VAL_16]]] : memref<1xi32>
// CHECK:             %[[VAL_17:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_18:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_19:.*]] = arith.constant 2 : i32
// CHECK:             memref.store %[[VAL_17]], %[[VAL_12]]{{\[}}%[[VAL_18]]] : memref<1xi32>
// CHECK:             %[[VAL_20:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_21:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_22:.*]] = arith.constant 14 : index
// CHECK:             scf.for %[[VAL_23:.*]] = %[[VAL_20]] to %[[VAL_22]] step %[[VAL_21]] {
// CHECK:               %[[VAL_24:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_16]]] : memref<1xi32>
// CHECK:               %[[VAL_25:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_26:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_27:.*]] = arith.subi %[[VAL_25]], %[[VAL_24]] : i32
// CHECK:               %[[VAL_28:.*]] = arith.maxsi %[[VAL_27]], %[[VAL_26]] : i32
// CHECK:               aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_28]])
// CHECK:               %[[VAL_29:.*]] = arith.addi %[[VAL_24]], %[[VAL_28]] : i32
// CHECK:               memref.store %[[VAL_29]], %[[VAL_13]]{{\[}}%[[VAL_16]]] : memref<1xi32>
// CHECK:               %[[VAL_30:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_18]]] : memref<1xi32>
// CHECK:               %[[VAL_31:.*]] = arith.index_cast %[[VAL_30]] : i32 to index
// CHECK:               %[[VAL_32:.*]] = scf.index_switch %[[VAL_31]] -> memref<8xi8>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_2]] : memref<8xi8>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_3]] : memref<8xi8>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_2]] : memref<8xi8>
// CHECK:               }
// CHECK:               %[[VAL_33:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_4]], Release, %[[VAL_33]])
// CHECK:               %[[VAL_34:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_16]]] : memref<1xi32>
// CHECK:               %[[VAL_35:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_36:.*]] = arith.subi %[[VAL_34]], %[[VAL_35]] : i32
// CHECK:               memref.store %[[VAL_36]], %[[VAL_13]]{{\[}}%[[VAL_16]]] : memref<1xi32>
// CHECK:               %[[VAL_37:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_18]]] : memref<1xi32>
// CHECK:               %[[VAL_38:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_39:.*]] = arith.addi %[[VAL_37]], %[[VAL_38]] : i32
// CHECK:               %[[VAL_40:.*]] = arith.cmpi sge, %[[VAL_39]], %[[VAL_19]] : i32
// CHECK:               %[[VAL_41:.*]] = arith.subi %[[VAL_39]], %[[VAL_19]] : i32
// CHECK:               %[[VAL_42:.*]] = arith.select %[[VAL_40]], %[[VAL_41]], %[[VAL_39]] : i32
// CHECK:               memref.store %[[VAL_42]], %[[VAL_12]]{{\[}}%[[VAL_18]]] : memref<1xi32>
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_43:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_44:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_45:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_45]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_46:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_46]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_47:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_47]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_48:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_48]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_49:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_49]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_50:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_50]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             %[[VAL_51:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_51]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_52:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_52]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_53:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_54:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_55:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_55]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_56:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_56]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_57:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_57]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_58:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_58]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
// CHECK:       }

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
