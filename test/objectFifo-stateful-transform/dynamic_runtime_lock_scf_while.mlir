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

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[MT:.*]] = aie.tile(0, 1)
// CHECK:           %[[T2:.*]] = aie.tile(0, 2)
// CHECK:           %[[CB0:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifo_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[CB1:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifo_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[CB2:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifo_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[CB3:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifo_cons_buff_3"} : memref<8xi8>
// CHECK:           %[[CPROD:.*]] = aie.lock(%[[T2]], 0) {init = 4 : i32, sym_name = "fifo_cons_prod_lock_0"}
// CHECK:           %[[CCONS:.*]] = aie.lock(%[[T2]], 1) {init = 0 : i32, sym_name = "fifo_cons_cons_lock_0"}
// CHECK:           %[[B0:.*]] = aie.buffer(%[[MT]]) {sym_name = "fifo_buff_0"} : memref<8xi8>
// CHECK:           %[[B1:.*]] = aie.buffer(%[[MT]]) {sym_name = "fifo_buff_1"} : memref<8xi8>
// CHECK:           %[[B2:.*]] = aie.buffer(%[[MT]]) {sym_name = "fifo_buff_2"} : memref<8xi8>
// CHECK:           %[[PROD:.*]] = aie.lock(%[[MT]], 0) {init = 3 : i32, sym_name = "fifo_prod_lock_0"}
// CHECK:           %[[CONS:.*]] = aie.lock(%[[MT]], 1) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
// CHECK:           %[[BUF:.*]] = aie.buffer(%[[T2]]) {sym_name = "buf"} : memref<1xindex>
// CHECK:           aie.flow(%[[MT]], DMA : 0, %[[T2]], DMA : 0)
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// CHECK:             %[[C14:.*]] = arith.constant 14 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C0I:.*]] = arith.constant 0 : i32
// CHECK:             %[[C3I:.*]] = arith.constant 3 : i32
// CHECK:             %[[C1I:.*]] = arith.constant 1 : i32
// CHECK:             %[[C4I:.*]] = arith.constant 4 : i32
// CHECK:             %[[C2I:.*]] = arith.constant 2 : i32
// CHECK:             %[[W:.*]]:3 = scf.while (%[[A0:.*]] = %[[C0]], %[[A1:.*]] = %[[C0I]], %[[A2:.*]] = %[[C0I]]) : (index, i32, i32) -> (index, i32, i32) {
// CHECK:               %[[CND:.*]] = arith.cmpi slt, %[[A0]], %[[C14]] : index
// CHECK:               scf.condition(%[[CND]]) %[[A0]], %[[A1]], %[[A2]] : index, i32, i32
// CHECK:             } do {
// CHECK:             ^bb0(%[[BA0:.*]]: index, %[[BA1:.*]]: i32, %[[BA2:.*]]: i32):
// CHECK:               %[[SUB:.*]] = arith.subi %[[C3I]], %[[BA2]] : i32
// CHECK:               %[[DELTA:.*]] = arith.maxsi %[[SUB]], %[[C0I]] : i32
// CHECK:               aie.use_lock(%[[CCONS]], AcquireGreaterEqual, %[[DELTA]])
// CHECK:               %[[NH:.*]] = arith.addi %[[BA2]], %[[DELTA]] : i32
// CHECK:               aie.use_lock(%[[CPROD]], Release, %[[C1I]])
// CHECK:               %[[RH:.*]] = arith.subi %[[NH]], %[[C1I]] : i32
// CHECK:               %[[NX:.*]] = arith.addi %[[BA1]], %[[C1I]] : i32
// CHECK:               %[[CMP:.*]] = arith.cmpi sge, %[[NX]], %[[C4I]] : i32
// CHECK:               %[[SEL:.*]] = arith.select %[[CMP]], %[[C0I]], %[[NX]] : i32
// CHECK:               %[[INC:.*]] = arith.addi %[[BA0]], %[[C1]] : index
// CHECK:               scf.yield %[[INC]], %[[SEL]], %[[RH]] : index, i32, i32
// CHECK:             }
// CHECK:             memref.store %[[W]]#0, %[[BUF]]{{\[}}%[[C0]]] : memref<1xindex>
// CHECK:             aie.use_lock(%[[CPROD]], Release, %[[C2I]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.memtile_dma(%[[MT]]) {
// CHECK:             %[[M1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[B0]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[B1]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[B2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.mem(%[[T2]]) {
// CHECK:             %[[N1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[CB0]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[CCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[CB1]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[CCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[CB2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[CCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[CB3]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[CCONS]], Release, %[[N1]])
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
