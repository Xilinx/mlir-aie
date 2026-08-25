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
// CHECK:           %[[MT:.*]] = aie.tile(0, 1)
// CHECK:           %[[T2:.*]] = aie.tile(0, 2)
// CHECK:           %[[CB0:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifo_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[CB1:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifo_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[CB2:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifo_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[CPROD:.*]] = aie.lock(%[[T2]]) {init = 3 : i32, sym_name = "fifo_cons_prod_lock_0"}
// CHECK:           %[[CCONS:.*]] = aie.lock(%[[T2]]) {init = 0 : i32, sym_name = "fifo_cons_cons_lock_0"}
// CHECK:           %[[B0:.*]] = aie.buffer(%[[MT]]) {sym_name = "fifo_buff_0"} : memref<8xi8>
// CHECK:           %[[B1:.*]] = aie.buffer(%[[MT]]) {sym_name = "fifo_buff_1"} : memref<8xi8>
// CHECK:           %[[B2:.*]] = aie.buffer(%[[MT]]) {sym_name = "fifo_buff_2"} : memref<8xi8>
// CHECK:           %[[B3:.*]] = aie.buffer(%[[MT]]) {sym_name = "fifo_buff_3"} : memref<8xi8>
// CHECK:           %[[PROD:.*]] = aie.lock(%[[MT]]) {init = 4 : i32, sym_name = "fifo_prod_lock_0"}
// CHECK:           %[[CONS:.*]] = aie.lock(%[[MT]]) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
// CHECK:           aie.flow(%[[MT]], DMA : 0, %[[T2]], DMA : 0)
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// CHECK:             %[[C14:.*]] = arith.constant 14 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[INIT:.*]] = arith.constant 0 : i32
// CHECK:             %[[TWO:.*]] = arith.constant 2 : i32
// CHECK:             %[[ONE:.*]] = arith.constant 1 : i32
// CHECK:             %[[THREE:.*]] = arith.constant 3 : i32
// CHECK:             %{{.*}}:2 = scf.for %{{.*}} = %[[C0]] to %[[C14]] step %[[C1]] iter_args(%[[IDX:.*]] = %[[INIT]], %[[HELD:.*]] = %[[INIT]]) -> (i32, i32) {
// CHECK:               %[[DELTA:.*]] = arith.subi %[[TWO]], %[[HELD]] : i32
// CHECK:               %[[ACQ:.*]] = arith.maxsi %[[DELTA]], %[[INIT]] : i32
// CHECK:               aie.use_lock(%[[CCONS]], AcquireGreaterEqual, %[[ACQ]])
// CHECK:               %[[HELDACQ:.*]] = arith.addi %[[HELD]], %[[ACQ]] : i32
// CHECK:               aie.use_lock(%[[CPROD]], Release, %[[ONE]])
// CHECK:               %[[HELDREL:.*]] = arith.subi %[[HELDACQ]], %[[ONE]] : i32
// CHECK:               %[[NX:.*]] = arith.addi %[[IDX]], %[[ONE]] : i32
// CHECK:               %[[CMP:.*]] = arith.cmpi sge, %[[NX]], %[[THREE]] : i32
// CHECK:               %[[SEL:.*]] = arith.select %[[CMP]], %[[INIT]], %[[NX]] : i32
// CHECK:               scf.yield %[[SEL]], %[[HELDREL]] : i32, i32
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.memtile_dma(%[[MT]]) {
// CHECK:             %[[M1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb1, ^bb5)
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
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[B3]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.mem(%[[T2]]) {
// CHECK:             %[[N1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
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
        %a = aie.objectfifo.acquire @fifo(Consume, 2) : !aie.objectfifosubview<memref<8xi8>>
        %e0 = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<8xi8>> -> memref<8xi8>
        %e1 = aie.objectfifo.subview.access %a[1] : !aie.objectfifosubview<memref<8xi8>> -> memref<8xi8>
        aie.objectfifo.release @fifo(Consume, 1)
      }
      aie.end
    }
  }
}
