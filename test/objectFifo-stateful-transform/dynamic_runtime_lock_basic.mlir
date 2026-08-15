//===- dynamic_runtime_lock_basic.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Dynamic objectFifo lowering emits runtime lock bookkeeping. Each acquire
// tracks how many elements are already held in a per-fifo i32 counter and
// acquires only the delta at runtime via a value-carrying
// `AcquireGreaterEqual`. Each release decrements the same counter. The consumer
// below slides a two-element window (acquire 2, release 1), so the delta is 2 on
// the first iteration and 1 on every later one and stays a runtime value.

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_3"} : memref<8xi8>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]]) {init = 4 : i32, sym_name = "fifo_prod_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_7]]) {sym_name = "fifo_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_7]]) {sym_name = "fifo_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_7]]) {sym_name = "fifo_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_7]]) {init = 3 : i32, sym_name = "fifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_7]]) {init = 0 : i32, sym_name = "fifo_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_7]], DMA : 0)
// CHECK:           %[[VAL_13:.*]] = aie.core(%[[VAL_7]]) {
// CHECK:             %[[VAL_14:.*]] = arith.constant 14 : index
// CHECK:             %[[VAL_15:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_17:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_18:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_20:.*]] = arith.constant 3 : i32
// CHECK:             %[[VAL_21:.*]]:2 = scf.for %[[VAL_22:.*]] = %[[VAL_16]] to %[[VAL_14]] step %[[VAL_15]] iter_args(%[[VAL_23:.*]] = %[[VAL_17]], %[[VAL_24:.*]] = %[[VAL_17]]) -> (i32, i32) {
// CHECK:               %[[VAL_25:.*]] = arith.subi %[[VAL_18]], %[[VAL_24]] : i32
// CHECK:               %[[VAL_26:.*]] = arith.maxsi %[[VAL_25]], %[[VAL_17]] : i32
// CHECK:               aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_24]], %[[VAL_26]] : i32
// CHECK:               aie.use_lock(%[[VAL_11]], Release, %[[VAL_19]])
// CHECK:               %[[VAL_28:.*]] = arith.subi %[[VAL_27]], %[[VAL_19]] : i32
// CHECK:               %[[VAL_29:.*]] = arith.addi %[[VAL_23]], %[[VAL_19]] : i32
// CHECK:               %[[VAL_30:.*]] = arith.cmpi sge, %[[VAL_29]], %[[VAL_20]] : i32
// CHECK:               %[[VAL_31:.*]] = arith.select %[[VAL_30]], %[[VAL_17]], %[[VAL_29]] : i32
// CHECK:               scf.yield %[[VAL_31]], %[[VAL_28]] : i32, i32
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_32:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_33:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_34:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = aie.mem(%[[VAL_7]]) {
// CHECK:             %[[VAL_36:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_37:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_36]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_36]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_36]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_36]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_36]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_36]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
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
        %e0, %e1 = aie.objectfifo.acquire @fifo(Consume) : memref<8xi8>, memref<8xi8>
        aie.objectfifo.release @fifo(Consume) [1]
      }
      aie.end
    }
  }
}
