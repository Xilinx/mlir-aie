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

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK-DAG:           %[[MT:.*]] = aie.tile(0, 1)
// CHECK-DAG:           %[[T2:.*]] = aie.tile(0, 2)
// CHECK-DAG:           aie.buffer @[[CB0:fifo_cons_buff_0]](%[[T2]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[CB1:fifo_cons_buff_1]](%[[T2]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[CB2:fifo_cons_buff_2]](%[[T2]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[CB3:fifo_cons_buff_3]](%[[T2]]) : memref<8xi8>
// CHECK-DAG:           aie.lock @[[CPROD:fifo_cons_prod_lock_0]](%[[T2]]) {init = 4 : i32}
// CHECK-DAG:           aie.lock @[[CCONS:fifo_cons_cons_lock_0]](%[[T2]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[B0:fifo_buff_0]](%[[MT]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[B1:fifo_buff_1]](%[[MT]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[B2:fifo_buff_2]](%[[MT]]) : memref<8xi8>
// CHECK-DAG:           aie.lock @[[PROD:fifo_prod_lock_0]](%[[MT]]) {init = 3 : i32}
// CHECK-DAG:           aie.lock @[[CONS:fifo_cons_lock_0]](%[[MT]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[BUF:buf]](%[[T2]]) : memref<1xindex>
// CHECK-DAG:           aie.flow(%[[MT]], DMA : 0, %[[T2]], DMA : 0)
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// CHECK-DAG:             %[[C14:.*]] = arith.constant 14 : index
// CHECK-DAG:             %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:             %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[C0I:.*]] = arith.constant 0 : i32
// CHECK-DAG:             %[[C3I:.*]] = arith.constant 3 : i32
// CHECK-DAG:             %[[C1I:.*]] = arith.constant 1 : i32
// CHECK-DAG:             %[[C2I:.*]] = arith.constant 2 : i32
// CHECK:             %[[W:.*]]:2 = scf.while (%[[A0:.*]] = %[[C0]], %[[A1:.*]] = %[[C0I]]) : (index, i32) -> (index, i32) {
// CHECK:               %[[CND:.*]] = arith.cmpi slt, %[[A0]], %[[C14]] : index
// CHECK:               scf.condition(%[[CND]]) %[[A0]], %[[A1]] : index, i32
// CHECK:             } do {
// CHECK:             ^bb0(%[[BA0:.*]]: index, %[[BA1:.*]]: i32):
// CHECK:               %[[SUB:.*]] = arith.subi %[[C3I]], %[[BA1]] : i32
// CHECK:               %[[DELTA:.*]] = arith.maxsi %[[SUB]], %[[C0I]] : i32
// CHECK:               aie.use_lock(@[[CCONS]], AcquireGreaterEqual, %[[DELTA]])
// CHECK:               %[[NH:.*]] = arith.addi %[[BA1]], %[[DELTA]] : i32
// CHECK:               aie.use_lock(@[[CPROD]], Release, %[[C1I]])
// CHECK:               %[[RH:.*]] = arith.subi %[[NH]], %[[C1I]] : i32
// CHECK:               %[[INC:.*]] = arith.addi %[[BA0]], %[[C1]] : index
// CHECK:               scf.yield %[[INC]], %[[RH]] : index, i32
// CHECK:             }
// CHECK:             memref.store %[[W]]#0, %{{[^ ,)]+}}{{\[}}%[[C0]]] : memref<1xindex>
// CHECK:             aie.use_lock(@[[CPROD]], Release, %[[C2I]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.memtile_dma(%[[MT]]) {
// CHECK:             %[[M1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(@[[CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             %[[B0_ACCESS_0:.*]] = aie.get_buffer @[[B0]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[B0_ACCESS_0]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(@[[CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             %[[B1_ACCESS_1:.*]] = aie.get_buffer @[[B1]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[B1_ACCESS_1]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(@[[CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             %[[B2_ACCESS_2:.*]] = aie.get_buffer @[[B2]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[B2_ACCESS_2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.mem(%[[T2]]) {
// CHECK:             %[[N1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(@[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[CB0_ACCESS_3:.*]] = aie.get_buffer @[[CB0]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[CB0_ACCESS_3]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[CCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(@[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[CB1_ACCESS_4:.*]] = aie.get_buffer @[[CB1]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[CB1_ACCESS_4]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[CCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(@[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[CB2_ACCESS_5:.*]] = aie.get_buffer @[[CB2]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[CB2_ACCESS_5]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[CCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(@[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[CB3_ACCESS_6:.*]] = aie.get_buffer @[[CB3]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[CB3_ACCESS_6]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[CCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.buffer @buf(%tile_0_2) : memref<1xindex>

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
        %x_obj0, %x_obj1, %x_obj2 = aie.objectfifo.acquire @fifo(Consume, 3) : memref<8xi8>, memref<8xi8>, memref<8xi8>
        aie.objectfifo.release @fifo(Consume, 1)
        %next = arith.addi %arg1, %c1 : index
        scf.yield %next : index
      }
      %buf = aie.get_buffer @buf : memref<1xindex>
      memref.store %r, %buf[%c0] : memref<1xindex>
      aie.objectfifo.release @fifo(Consume, 2)
      aie.end
    }
  }
}
