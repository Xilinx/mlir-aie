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

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="dynamic-objFifos=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inOF_X_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inOF_X_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inOF_X_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inOF_X_cons_buff_3"} : memref<8xi8>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]], 2) {init = 4 : i32, sym_name = "inOF_X_cons_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_1]], 3) {init = 0 : i32, sym_name = "inOF_X_cons_cons_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_X_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_X_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_X_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_0]], 2) {init = 3 : i32, sym_name = "inOF_X_prod_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_0]], 3) {init = 0 : i32, sym_name = "inOF_X_cons_lock_0"}
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inOF_W_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inOF_W_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_15:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inOF_W_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_1]], 0) {init = 3 : i32, sym_name = "inOF_W_cons_prod_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "inOF_W_cons_cons_lock_0"}
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_W_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_19:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_W_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_20:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_W_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_21:.*]] = aie.lock(%[[VAL_0]], 0) {init = 3 : i32, sym_name = "inOF_W_prod_lock_0"}
// CHECK:           %[[VAL_22:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "inOF_W_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 1, %[[VAL_1]], DMA : 1)
// CHECK:           %[[VAL_24:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[INIT:.*]] = arith.constant 0 : i32
// CHECK:             %[[LB:.*]] = arith.constant 0 : index
// CHECK:             %[[STEP:.*]] = arith.constant 1 : index
// CHECK:             %[[UB:.*]] = arith.constant 14 : index
// Outer loop carries idxW, idxX, heldW, heldX as iter_args; the inner loop
// threads idxX/heldX as its own iter_args - no bookkeeping memrefs.
// CHECK:             %[[OUT:.*]]:4 = scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[IDXW:.*]] = %[[INIT]], %[[IDXX:.*]] = %[[INIT]], %[[HELDW:.*]] = %[[INIT]], %[[HELDX:.*]] = %[[INIT]]) -> (i32, i32, i32, i32) {
// CHECK:               %[[TW:.*]] = arith.constant 2 : i32
// CHECK:               %[[SW:.*]] = arith.subi %[[TW]], %[[HELDW]] : i32
// CHECK:               %[[DW:.*]] = arith.maxsi %[[SW]], %{{.*}} : i32
// CHECK:               aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, %[[DW]])
// CHECK:               %[[NHW:.*]] = arith.addi %[[HELDW]], %[[DW]] : i32
// CHECK:               %[[IN:.*]]:2 = scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[IIDXX:.*]] = %[[IDXX]], %[[IHELDX:.*]] = %[[HELDX]]) -> (i32, i32) {
// CHECK:                 %[[TX:.*]] = arith.constant 3 : i32
// CHECK:                 %[[SX:.*]] = arith.subi %[[TX]], %[[IHELDX]] : i32
// CHECK:                 %[[DX:.*]] = arith.maxsi %[[SX]], %{{.*}} : i32
// CHECK:                 aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[DX]])
// CHECK:                 %[[NHX:.*]] = arith.addi %[[IHELDX]], %[[DX]] : i32
// CHECK:                 %[[ONEX:.*]] = arith.constant 1 : i32
// CHECK:                 aie.use_lock(%[[VAL_6]], Release, %[[ONEX]])
// CHECK:                 %[[RHX:.*]] = arith.subi %[[NHX]], %[[ONEX]] : i32
// CHECK:                 %[[FX:.*]] = arith.constant 4 : i32
// CHECK:                 %[[NIX:.*]] = arith.addi %[[IIDXX]], %{{.*}} : i32
// CHECK:                 %[[WX:.*]] = arith.cmpi sge, %[[NIX]], %[[FX]] : i32
// CHECK:                 %[[WWX:.*]] = arith.subi %[[NIX]], %[[FX]] : i32
// CHECK:                 %[[SELX:.*]] = arith.select %[[WX]], %[[WWX]], %[[NIX]] : i32
// CHECK:                 scf.yield %[[SELX]], %[[RHX]] : i32, i32
// CHECK:               }
// CHECK:               %[[DTX:.*]] = arith.constant 2 : i32
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[DTX]])
// CHECK:               %[[ORHX:.*]] = arith.subi %[[IN]]#1, %[[DTX]] : i32
// CHECK:               %[[FX2:.*]] = arith.constant 4 : i32
// CHECK:               %[[ONIX:.*]] = arith.addi %[[IN]]#0, %{{.*}} : i32
// CHECK:               %[[OWX:.*]] = arith.cmpi sge, %[[ONIX]], %[[FX2]] : i32
// CHECK:               %[[OWWX:.*]] = arith.subi %[[ONIX]], %[[FX2]] : i32
// CHECK:               %[[OSELX:.*]] = arith.select %[[OWX]], %[[OWWX]], %[[ONIX]] : i32
// CHECK:               %[[ONEW:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_16]], Release, %[[ONEW]])
// CHECK:               %[[RHW:.*]] = arith.subi %[[NHW]], %[[ONEW]] : i32
// CHECK:               %[[TW3:.*]] = arith.constant 3 : i32
// CHECK:               %[[NIW:.*]] = arith.addi %[[IDXW]], %{{.*}} : i32
// CHECK:               %[[WW:.*]] = arith.cmpi sge, %[[NIW]], %[[TW3]] : i32
// CHECK:               %[[WWW:.*]] = arith.subi %[[NIW]], %[[TW3]] : i32
// CHECK:               %[[SELW:.*]] = arith.select %[[WW]], %[[WWW]], %[[NIW]] : i32
// CHECK:               scf.yield %[[SELW]], %[[OSELX]], %[[RHW]], %[[ORHX]] : i32, i32, i32, i32
// CHECK:             }
// Drain the objects still held by the outer objectFifo after the loop.
// CHECK:             %[[FDTW:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[FDTW]])
// CHECK:             %{{.*}} = arith.subi %[[OUT]]#2, %[[FDTW]] : i32
// CHECK:             %{{.*}} = arith.addi %[[OUT]]#0, %{{.*}} : i32
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_89:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_90:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_91:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, %[[VAL_91]])
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_92:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_21]], Release, %[[VAL_92]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_93:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, %[[VAL_93]])
// CHECK:             aie.dma_bd(%[[VAL_19]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_94:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_21]], Release, %[[VAL_94]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_95:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, %[[VAL_95]])
// CHECK:             aie.dma_bd(%[[VAL_20]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_96:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_21]], Release, %[[VAL_96]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %[[VAL_97:.*]] = aie.dma_start(MM2S, 1, ^bb5, ^bb8)
// CHECK:           ^bb5:
// CHECK:             %[[VAL_98:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_98]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_99:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_99]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             %[[VAL_100:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_100]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_101:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_101]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             %[[VAL_102:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_102]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_103:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_103]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb8:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_104:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_105:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_106:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_106]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_107:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_107]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_108:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_108]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_109:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_109]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_110:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_110]])
// CHECK:             aie.dma_bd(%[[VAL_15]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_111:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_111]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %[[VAL_112:.*]] = aie.dma_start(S2MM, 1, ^bb5, ^bb9)
// CHECK:           ^bb5:
// CHECK:             %[[VAL_113:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_113]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_114:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_114]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             %[[VAL_115:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_115]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_116:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_116]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             %[[VAL_117:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_117]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_118:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_118]])
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             %[[VAL_119:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_119]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_120:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_120]])
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
        %w = aie.objectfifo.acquire @inOF_W(Consume, 2) : !aie.objectfifosubview<memref<8xi8>>
        scf.for %arg1 = %c0 to %c14 step %c1 {
          %x = aie.objectfifo.acquire @inOF_X(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
          aie.objectfifo.release @inOF_X(Consume, 1)
        }
        aie.objectfifo.release @inOF_X(Consume, 2)
        aie.objectfifo.release @inOF_W(Consume, 1)
      }
      aie.objectfifo.release @inOF_W(Consume, 1)
      aie.end
    }
  }
}
