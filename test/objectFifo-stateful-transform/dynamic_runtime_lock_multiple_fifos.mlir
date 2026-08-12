//===- dynamic_runtime_lock_multiple_fifos.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two objectFifos acquired/released with different counts in the same loop
// body. Each fifo gets its own runtime held counter and its own
// value-carrying `AcquireGreaterEqual`; the trailing post-loop releases
// decrement the respective counters.

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="dynamic-objFifos=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifoY_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifoY_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifoY_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]], 2) {init = 3 : i32, sym_name = "fifoY_cons_prod_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]], 3) {init = 0 : i32, sym_name = "fifoY_cons_cons_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifoY_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifoY_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_0]], 2) {init = 2 : i32, sym_name = "fifoY_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_0]], 3) {init = 0 : i32, sym_name = "fifoY_cons_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifoX_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifoX_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifoX_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifoX_cons_buff_3"} : memref<8xi8>
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_1]], 0) {init = 4 : i32, sym_name = "fifoX_cons_prod_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "fifoX_cons_cons_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifoX_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifoX_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_19:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifoX_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_20:.*]] = aie.lock(%[[VAL_0]], 0) {init = 3 : i32, sym_name = "fifoX_prod_lock_0"}
// CHECK:           %[[VAL_21:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "fifoX_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 1, %[[VAL_1]], DMA : 1)
// CHECK:           %[[VAL_23:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[INIT:.*]] = arith.constant 0 : i32
// CHECK:             %[[LB:.*]] = arith.constant 0 : index
// CHECK:             %[[STEP:.*]] = arith.constant 1 : index
// CHECK:             %[[UB:.*]] = arith.constant 14 : index
// Two objectFifos (X, Y): each carries a rotating buffer index and a held
// count as loop iter_args (idxX, idxY, heldX, heldY); no bookkeeping memrefs.
// CHECK:             %[[LOOP:.*]]:4 = scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[IDXX:.*]] = %[[INIT]], %[[IDXY:.*]] = %[[INIT]], %[[HELDX:.*]] = %[[INIT]], %[[HELDY:.*]] = %[[INIT]]) -> (i32, i32, i32, i32) {
// CHECK:               %[[TX:.*]] = arith.constant 3 : i32
// CHECK:               %[[SX:.*]] = arith.subi %[[TX]], %[[HELDX]] : i32
// CHECK:               %[[DX:.*]] = arith.maxsi %[[SX]], %{{.*}} : i32
// CHECK:               aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[DX]])
// CHECK:               %[[NHX:.*]] = arith.addi %[[HELDX]], %[[DX]] : i32
// CHECK:               %[[TY:.*]] = arith.constant 2 : i32
// CHECK:               %[[SY:.*]] = arith.subi %[[TY]], %[[HELDY]] : i32
// CHECK:               %[[DY:.*]] = arith.maxsi %[[SY]], %{{.*}} : i32
// CHECK:               aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[DY]])
// CHECK:               %[[NHY:.*]] = arith.addi %[[HELDY]], %[[DY]] : i32
// CHECK:               %[[ONEX:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_15]], Release, %[[ONEX]])
// CHECK:               %[[RHX:.*]] = arith.subi %[[NHX]], %[[ONEX]] : i32
// CHECK:               %[[FOURX:.*]] = arith.constant 4 : i32
// CHECK:               %[[NIX:.*]] = arith.addi %[[IDXX]], %{{.*}} : i32
// CHECK:               %[[WX:.*]] = arith.cmpi sge, %[[NIX]], %[[FOURX]] : i32
// CHECK:               %[[WWX:.*]] = arith.subi %[[NIX]], %[[FOURX]] : i32
// CHECK:               %[[SELX:.*]] = arith.select %[[WX]], %[[WWX]], %[[NIX]] : i32
// CHECK:               %[[ONEY:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_5]], Release, %[[ONEY]])
// CHECK:               %[[RHY:.*]] = arith.subi %[[NHY]], %[[ONEY]] : i32
// CHECK:               %[[THREEY:.*]] = arith.constant 3 : i32
// CHECK:               %[[NIY:.*]] = arith.addi %[[IDXY]], %{{.*}} : i32
// CHECK:               %[[WY:.*]] = arith.cmpi sge, %[[NIY]], %[[THREEY]] : i32
// CHECK:               %[[WWY:.*]] = arith.subi %[[NIY]], %[[THREEY]] : i32
// CHECK:               %[[SELY:.*]] = arith.select %[[WY]], %[[WWY]], %[[NIY]] : i32
// CHECK:               scf.yield %[[SELX]], %[[SELY]], %[[RHX]], %[[RHY]] : i32, i32, i32, i32
// CHECK:             }
// Drain the objects still held once the loop finishes.
// CHECK:             %[[DTX:.*]] = arith.constant 2 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[DTX]])
// CHECK:             %{{.*}} = arith.subi %[[LOOP]]#2, %[[DTX]] : i32
// CHECK:             %{{.*}} = arith.addi %[[LOOP]]#0, %{{.*}} : i32
// CHECK:             %[[DTY:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[DTY]])
// CHECK:             %{{.*}} = arith.subi %[[LOOP]]#3, %[[DTY]] : i32
// CHECK:             %{{.*}} = arith.addi %[[LOOP]]#1, %{{.*}} : i32
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_85:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_86:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_87:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_87]])
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_88:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_20]], Release, %[[VAL_88]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_89:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_89]])
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_90:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_20]], Release, %[[VAL_90]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_91:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_91]])
// CHECK:             aie.dma_bd(%[[VAL_19]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_92:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_20]], Release, %[[VAL_92]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %[[VAL_93:.*]] = aie.dma_start(MM2S, 1, ^bb5, ^bb7)
// CHECK:           ^bb5:
// CHECK:             %[[VAL_94:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_94]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_95:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_95]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             %[[VAL_96:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_96]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_97:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_97]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_98:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_99:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_100:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_100]])
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_101:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_101]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_102:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_102]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_103:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_103]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_104:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_104]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_105:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_105]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             %[[VAL_106:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_106]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_107:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_107]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             %[[VAL_108:.*]] = aie.dma_start(S2MM, 1, ^bb6, ^bb9)
// CHECK:           ^bb6:
// CHECK:             %[[VAL_109:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_109]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_110:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_110]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             %[[VAL_111:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_111]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_112:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_112]])
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             %[[VAL_113:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_113]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_114:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_114]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb9:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifoX(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>
    aie.objectfifo @fifoY(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %x = aie.objectfifo.acquire @fifoX(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        %y = aie.objectfifo.acquire @fifoY(Consume, 2) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifoX(Consume, 1)
        aie.objectfifo.release @fifoY(Consume, 1)
      }
      aie.objectfifo.release @fifoX(Consume, 2)
      aie.objectfifo.release @fifoY(Consume, 1)
      aie.end
    }
  }
}
