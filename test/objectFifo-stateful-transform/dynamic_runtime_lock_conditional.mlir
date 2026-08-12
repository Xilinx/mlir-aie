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

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="dynamic-objFifos=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
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
// CHECK:           %[[VAL_14:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[INIT:.*]] = arith.constant 0 : i32
// CHECK:             %[[LB:.*]] = arith.constant 0 : index
// CHECK:             %[[STEP:.*]] = arith.constant 1 : index
// CHECK:             %[[UB:.*]] = arith.constant 14 : index
// CHECK:             %[[TRUE:.*]] = arith.constant true
// CHECK:             %{{.*}}:2 = scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[IDX:.*]] = %[[INIT]], %[[HELD:.*]] = %[[INIT]]) -> (i32, i32) {
// CHECK:               %[[THREE:.*]] = arith.constant 3 : i32
// CHECK:               %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:               %[[SUB:.*]] = arith.subi %[[THREE]], %[[HELD]] : i32
// CHECK:               %[[DELTA:.*]] = arith.maxsi %[[SUB]], %[[ZERO]] : i32
// CHECK:               aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[DELTA]])
// CHECK:               %[[NEWHELD:.*]] = arith.addi %[[HELD]], %[[DELTA]] : i32
// CHECK:               %{{.*}}:2 = scf.if %[[TRUE]] -> (i32, i32) {
// CHECK:                 %[[ONE:.*]] = arith.constant 1 : i32
// CHECK:                 aie.use_lock(%[[VAL_6]], Release, %[[ONE]])
// CHECK:                 %[[RELHELD:.*]] = arith.subi %[[NEWHELD]], %[[ONE]] : i32
// CHECK:                 %[[FOUR:.*]] = arith.constant 4 : i32
// CHECK:                 %[[ONE2:.*]] = arith.constant 1 : i32
// CHECK:                 %[[NEXTIDX:.*]] = arith.addi %[[IDX]], %[[ONE2]] : i32
// CHECK:                 %[[WRAP:.*]] = arith.cmpi sge, %[[NEXTIDX]], %[[FOUR]] : i32
// CHECK:                 %[[WRAPPED:.*]] = arith.subi %[[NEXTIDX]], %[[FOUR]] : i32
// CHECK:                 %[[SEL:.*]] = arith.select %[[WRAP]], %[[WRAPPED]], %[[NEXTIDX]] : i32
// CHECK:                 scf.yield %[[SEL]], %[[RELHELD]] : i32, i32
// CHECK:               } else {
// CHECK:                 scf.yield %[[IDX]], %[[NEWHELD]] : i32, i32
// CHECK:               }
// CHECK:               scf.yield %{{.*}}#0, %{{.*}}#1 : i32, i32
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_41:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_42:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_43:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_43]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_44:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_44]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_45:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_45]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_46:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_46]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_47:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_47]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_48:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_48]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_49:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_50:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_51:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_51]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_52:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_52]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_53:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_53]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_54:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_54]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_55:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_55]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_56:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_56]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             %[[VAL_57:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_57]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_58:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_58]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

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
