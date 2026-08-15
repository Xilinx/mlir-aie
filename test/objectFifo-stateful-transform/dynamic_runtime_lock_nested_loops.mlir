//===- dynamic_runtime_lock_nested_loops.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Acquires/releases at both the outer and inner loop levels on two fifos.
// The runtime held counters are shared across the loop nest, so the inner
// loop's per-iteration acquire delta and the outer loop's acquire delta are
// each computed from the current held value regardless of loop depth.

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_W_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_W_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_W_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]]) {init = 3 : i32, sym_name = "inOF_W_prod_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "inOF_W_cons_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_X_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_X_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_X_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_0]]) {init = 3 : i32, sym_name = "inOF_X_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "inOF_X_cons_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_11]]) {sym_name = "inOF_W_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_11]]) {sym_name = "inOF_W_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_11]]) {sym_name = "inOF_W_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_11]]) {init = 3 : i32, sym_name = "inOF_W_cons_prod_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_11]]) {init = 0 : i32, sym_name = "inOF_W_cons_cons_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_11]]) {sym_name = "inOF_X_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_11]]) {sym_name = "inOF_X_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_19:.*]] = aie.buffer(%[[VAL_11]]) {sym_name = "inOF_X_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_20:.*]] = aie.buffer(%[[VAL_11]]) {sym_name = "inOF_X_cons_buff_3"} : memref<8xi8>
// CHECK:           %[[VAL_21:.*]] = aie.lock(%[[VAL_11]]) {init = 4 : i32, sym_name = "inOF_X_cons_prod_lock_0"}
// CHECK:           %[[VAL_22:.*]] = aie.lock(%[[VAL_11]]) {init = 0 : i32, sym_name = "inOF_X_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_11]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 1, %[[VAL_11]], DMA : 1)
// CHECK:           %[[VAL_23:.*]] = aie.core(%[[VAL_11]]) {
// CHECK:             %[[VAL_24:.*]] = arith.constant 14 : index
// CHECK:             %[[VAL_25:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_26:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_27:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_28:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_29:.*]] = arith.constant 3 : i32
// CHECK:             %[[VAL_30:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_31:.*]] = arith.constant 4 : i32
// CHECK:             %[[VAL_32:.*]] = arith.constant -2 : i32
// CHECK:             %[[VAL_33:.*]]:4 = scf.for %[[VAL_34:.*]] = %[[VAL_26]] to %[[VAL_24]] step %[[VAL_25]] iter_args(%[[VAL_35:.*]] = %[[VAL_27]], %[[VAL_36:.*]] = %[[VAL_27]], %[[VAL_37:.*]] = %[[VAL_27]], %[[VAL_38:.*]] = %[[VAL_27]]) -> (i32, i32, i32, i32) {
// CHECK:               %[[VAL_39:.*]] = arith.subi %[[VAL_28]], %[[VAL_37]] : i32
// CHECK:               %[[VAL_40:.*]] = arith.maxsi %[[VAL_39]], %[[VAL_27]] : i32
// CHECK:               aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_40]])
// CHECK:               %[[VAL_41:.*]] = arith.addi %[[VAL_37]], %[[VAL_40]] : i32
// CHECK:               %[[VAL_42:.*]]:2 = scf.for %[[VAL_43:.*]] = %[[VAL_26]] to %[[VAL_24]] step %[[VAL_25]] iter_args(%[[VAL_44:.*]] = %[[VAL_36]], %[[VAL_45:.*]] = %[[VAL_38]]) -> (i32, i32) {
// CHECK:                 %[[VAL_46:.*]] = arith.subi %[[VAL_29]], %[[VAL_45]] : i32
// CHECK:                 %[[VAL_47:.*]] = arith.maxsi %[[VAL_46]], %[[VAL_27]] : i32
// CHECK:                 aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, %[[VAL_47]])
// CHECK:                 %[[VAL_48:.*]] = arith.addi %[[VAL_45]], %[[VAL_47]] : i32
// CHECK:                 aie.use_lock(%[[VAL_21]], Release, %[[VAL_30]])
// CHECK:                 %[[VAL_49:.*]] = arith.subi %[[VAL_48]], %[[VAL_30]] : i32
// CHECK:                 %[[VAL_50:.*]] = arith.addi %[[VAL_44]], %[[VAL_30]] : i32
// CHECK:                 %[[VAL_51:.*]] = arith.cmpi sge, %[[VAL_50]], %[[VAL_31]] : i32
// CHECK:                 %[[VAL_52:.*]] = arith.select %[[VAL_51]], %[[VAL_27]], %[[VAL_50]] : i32
// CHECK:                 scf.yield %[[VAL_52]], %[[VAL_49]] : i32, i32
// CHECK:               }
// CHECK:               aie.use_lock(%[[VAL_21]], Release, %[[VAL_28]])
// CHECK:               %[[VAL_53:.*]] = arith.subi %[[VAL_54:.*]]#1, %[[VAL_28]] : i32
// CHECK:               %[[VAL_55:.*]] = arith.addi %[[VAL_54]]#0, %[[VAL_28]] : i32
// CHECK:               %[[VAL_56:.*]] = arith.cmpi sge, %[[VAL_55]], %[[VAL_31]] : i32
// CHECK:               %[[VAL_57:.*]] = arith.addi %[[VAL_54]]#0, %[[VAL_32]] : i32
// CHECK:               %[[VAL_58:.*]] = arith.select %[[VAL_56]], %[[VAL_57]], %[[VAL_55]] : i32
// CHECK:               aie.use_lock(%[[VAL_15]], Release, %[[VAL_30]])
// CHECK:               %[[VAL_59:.*]] = arith.subi %[[VAL_41]], %[[VAL_30]] : i32
// CHECK:               %[[VAL_60:.*]] = arith.addi %[[VAL_35]], %[[VAL_30]] : i32
// CHECK:               %[[VAL_61:.*]] = arith.cmpi sge, %[[VAL_60]], %[[VAL_29]] : i32
// CHECK:               %[[VAL_62:.*]] = arith.select %[[VAL_61]], %[[VAL_27]], %[[VAL_60]] : i32
// CHECK:               scf.yield %[[VAL_62]], %[[VAL_58]], %[[VAL_59]], %[[VAL_53]] : i32, i32, i32, i32
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_30]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_63:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_64:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_65:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_64]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_64]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_64]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_64]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_64]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_64]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %[[VAL_66:.*]] = aie.dma_start(MM2S, 1, ^bb5, ^bb8)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_64]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_64]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_64]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_64]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_64]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_64]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb8:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_67:.*]] = aie.mem(%[[VAL_11]]) {
// CHECK:             %[[VAL_68:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_69:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_68]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_68]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_68]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_68]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_68]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_68]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %[[VAL_70:.*]] = aie.dma_start(S2MM, 1, ^bb5, ^bb9)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_68]])
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_22]], Release, %[[VAL_68]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_68]])
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_22]], Release, %[[VAL_68]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_68]])
// CHECK:             aie.dma_bd(%[[VAL_19]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_22]], Release, %[[VAL_68]])
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_68]])
// CHECK:             aie.dma_bd(%[[VAL_20]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_22]], Release, %[[VAL_68]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb9:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @inOF_W(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>
    aie.objectfifo @inOF_X(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %w_obj0, %w_obj1 = aie.objectfifo.acquire @inOF_W(Consume) : memref<8xi8>, memref<8xi8>
        scf.for %arg1 = %c0 to %c14 step %c1 {
          %x_obj0, %x_obj1, %x_obj2 = aie.objectfifo.acquire @inOF_X(Consume) : memref<8xi8>, memref<8xi8>, memref<8xi8>
          aie.objectfifo.release @inOF_X(Consume) [1]
        }
        aie.objectfifo.release @inOF_X(Consume) [2]
        aie.objectfifo.release @inOF_W(Consume) [1]
      }
      aie.objectfifo.release @inOF_W(Consume) [1]
      aie.end
    }
  }
}
