//===- dynamic_runtime_lock_scf_while.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Acquire/release inside an scf.while body. The runtime held-counter approach
// needs no loop peeling or trip-count reasoning: the counter lives across
// iterations, so the while body lowers directly with a value-carrying
// `AcquireGreaterEqual`. A post-loop release decrements the counter.

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
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "buf"} : memref<1xindex>
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           %[[VAL_15:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[INIT:.*]] = arith.constant 0 : i32
// CHECK:             %[[LB:.*]] = arith.constant 0 : index
// CHECK:             %[[STEP:.*]] = arith.constant 1 : index
// CHECK:             %[[UB:.*]] = arith.constant 14 : index
// The while loop carries the user index plus the objectFifo buffer index and
// held count as iter_args (mem2reg-promoted from bookkeeping allocas).
// CHECK:             %[[LOOP:.*]]:3 = scf.while (%[[LI:.*]] = %[[LB]], %[[IDX:.*]] = %[[INIT]], %[[HELD:.*]] = %[[INIT]]) : (index, i32, i32) -> (index, i32, i32) {
// CHECK:               %[[CMP:.*]] = arith.cmpi slt, %[[LI]], %[[UB]] : index
// CHECK:               scf.condition(%[[CMP]]) %[[LI]], %[[IDX]], %[[HELD]] : index, i32, i32
// CHECK:             } do {
// CHECK:             ^bb0(%[[BLI:.*]]: index, %[[BIDX:.*]]: i32, %[[BHELD:.*]]: i32):
// CHECK:               %[[THREE:.*]] = arith.constant 3 : i32
// CHECK:               %[[SUB:.*]] = arith.subi %[[THREE]], %[[BHELD]] : i32
// CHECK:               %[[DELTA:.*]] = arith.maxsi %[[SUB]], %{{.*}} : i32
// CHECK:               aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[DELTA]])
// CHECK:               %[[NHELD:.*]] = arith.addi %[[BHELD]], %[[DELTA]] : i32
// CHECK:               %[[ONE:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[ONE]])
// CHECK:               %[[RHELD:.*]] = arith.subi %[[NHELD]], %[[ONE]] : i32
// CHECK:               %[[FOUR:.*]] = arith.constant 4 : i32
// CHECK:               %[[NIDX:.*]] = arith.addi %[[BIDX]], %{{.*}} : i32
// CHECK:               %[[W:.*]] = arith.cmpi sge, %[[NIDX]], %[[FOUR]] : i32
// CHECK:               %[[WW:.*]] = arith.subi %[[NIDX]], %[[FOUR]] : i32
// CHECK:               %[[SEL:.*]] = arith.select %[[W]], %[[WW]], %[[NIDX]] : i32
// CHECK:               %[[NLI:.*]] = arith.addi %[[BLI]], %[[STEP]] : index
// CHECK:               scf.yield %[[NLI]], %[[SEL]], %[[RHELD]] : index, i32, i32
// CHECK:             }
// The final loop index is written to the user buffer; then drain held objects.
// CHECK:             memref.store %[[LOOP]]#0, %[[VAL_13]]{{\[}}%[[LB]]] : memref<1xindex>
// CHECK:             %[[DT:.*]] = arith.constant 2 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[DT]])
// CHECK:             %{{.*}} = arith.subi %[[LOOP]]#2, %[[DT]] : i32
// CHECK:             %{{.*}} = arith.addi %[[LOOP]]#1, %{{.*}} : i32
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_55:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_56:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_56]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_57:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_57]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_58:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_58]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_59:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_59]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_60:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_60]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_61:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_61]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_62:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_63:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_64:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_64]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_65:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_65]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_66:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_66]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_67:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_67]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_68:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_68]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_69:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_69]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             %[[VAL_70:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_70]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_71:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_71]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) {sym_name = "buf"} : memref<1xindex>

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %r = scf.while (%arg0 = %c0) : (index) -> index {
        %cond = arith.cmpi slt, %arg0, %c14 : index
        scf.condition(%cond) %arg0 : index
      } do {
      ^bb0(%arg1: index):
        %x = aie.objectfifo.acquire @fifo(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifo(Consume, 1)
        %next = arith.addi %arg1, %c1 : index
        scf.yield %next : index
      }
      memref.store %r, %buf[%c0] : memref<1xindex>
      aie.objectfifo.release @fifo(Consume, 2)
      aie.end
    }
  }
}
