//===- global_generation_test.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL: module {
// CHECK:   aie.device(npu1) {
// CHECK-DAG:     memref.global "public" @out1_cons : memref<8xi32>
// CHECK-DAG:     memref.global "public" @out1 : memref<8xi32>
// CHECK-DAG:     memref.global "public" @out0_cons : memref<64000xi32>
// CHECK-DAG:     memref.global "public" @out0 : memref<64000xi32>
// CHECK-DAG:     memref.global "public" @in1_cons : memref<8xi32>
// CHECK-DAG:     memref.global "public" @in1 : memref<8xi32>
// CHECK-DAG:     memref.global "public" @in0_cons : memref<64000xi32>
// CHECK-DAG:     memref.global "public" @in0 : memref<64000xi32>
// CHECK-DAG:     %[[SHIM_NOC_TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK-DAG:     %[[MEM_TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK-DAG:     %[[MEM_TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK-DAG:     %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK-DAG:     %[[OUT1_BUFF_0:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "out1_buff_0"} : memref<8xi32>
// CHECK-DAG:     %[[OUT1_BUFF_1:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "out1_buff_1"} : memref<8xi32>
// CHECK-DAG:     %[[OUT1_PROD_LOCK:.*]] = aie.lock(%[[TILE_0_2]], 2) {init = 2 : i32, sym_name = "out1_prod_lock_0"}
// CHECK-DAG:     %[[OUT1_CONS_LOCK:.*]] = aie.lock(%[[TILE_0_2]], 3) {init = 0 : i32, sym_name = "out1_cons_lock_0"}
// CHECK-DAG:     %[[OUT0_CONS_PROD_LOCK:.*]] = aie.lock(%[[SHIM_NOC_TILE_0_0]], 2) {init = 1 : i32, sym_name = "out0_cons_prod_lock_0"}
// CHECK-DAG:     %[[OUT0_CONS_CONS_LOCK:.*]] = aie.lock(%[[SHIM_NOC_TILE_0_0]], 3) {init = 0 : i32, sym_name = "out0_cons_cons_lock_0"}
// CHECK-DAG:     %[[OUT0_BUFF_0:.*]] = aie.buffer(%[[MEM_TILE_1_1]]) {sym_name = "out0_buff_0"} : memref<64000xi32>
// CHECK-DAG:     %[[OUT0_BUFF_1:.*]] = aie.buffer(%[[MEM_TILE_1_1]]) {sym_name = "out0_buff_1"} : memref<64000xi32>
// CHECK-DAG:     %[[OUT0_PROD_LOCK:.*]] = aie.lock(%[[MEM_TILE_0_1]], 2) {init = 2 : i32, sym_name = "out0_prod_lock_0"}
// CHECK-DAG:     %[[OUT0_CONS_LOCK:.*]] = aie.lock(%[[MEM_TILE_0_1]], 3) {init = 0 : i32, sym_name = "out0_cons_lock_0"}
// CHECK-DAG:     %[[IN1_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "in1_cons_buff_0"} : memref<8xi32>
// CHECK-DAG:     %[[IN1_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "in1_cons_buff_1"} : memref<8xi32>
// CHECK-DAG:     %[[IN1_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_0_2]], 0) {init = 2 : i32, sym_name = "in1_cons_prod_lock_0"}
// CHECK-DAG:     %[[IN1_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_0_2]], 1) {init = 0 : i32, sym_name = "in1_cons_cons_lock_0"}
// CHECK-DAG:     %[[IN0_CONS_BUFF_0:.*]] = aie.buffer(%[[MEM_TILE_0_1]]) {sym_name = "in0_cons_buff_0"} : memref<64000xi32>
// CHECK-DAG:     %[[IN0_CONS_BUFF_1:.*]] = aie.buffer(%[[MEM_TILE_0_1]]) {sym_name = "in0_cons_buff_1"} : memref<64000xi32>
// CHECK-DAG:     %[[IN0_CONS_PROD_LOCK:.*]] = aie.lock(%[[MEM_TILE_0_1]], 0) {init = 2 : i32, sym_name = "in0_cons_prod_lock_0"}
// CHECK-DAG:     %[[IN0_CONS_CONS_LOCK:.*]] = aie.lock(%[[MEM_TILE_0_1]], 1) {init = 0 : i32, sym_name = "in0_cons_cons_lock_0"}
// CHECK-DAG:     %[[IN0_PROD_LOCK:.*]] = aie.lock(%[[SHIM_NOC_TILE_0_0]], 0) {init = 1 : i32, sym_name = "in0_prod_lock_0"}
// CHECK-DAG:     %[[IN0_CONS_LOCK:.*]] = aie.lock(%[[SHIM_NOC_TILE_0_0]], 1) {init = 0 : i32, sym_name = "in0_cons_lock_0"}
// CHECK:     aie.flow(%[[SHIM_NOC_TILE_0_0]], DMA : 0, %[[MEM_TILE_0_1]], DMA : 1)
// CHECK:     aie.flow(%[[MEM_TILE_0_1]], DMA : 1, %[[TILE_0_2]], DMA : 0)
// CHECK:     aie.flow(%[[MEM_TILE_0_1]], DMA : 0, %[[SHIM_NOC_TILE_0_0]], DMA : 0)
// CHECK:     aie.flow(%[[TILE_0_2]], DMA : 0, %[[MEM_TILE_0_1]], DMA : 0)
// CHECK:     %[[CORE_0_2:.*]] = aie.core(%[[TILE_0_2]]) {
// CHECK:       scf.for %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:         aie.use_lock(%[[IN1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:         aie.use_lock(%[[OUT1_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:         scf.for %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:           %{{.*}} = memref.load %[[IN1_CONS_BUFF_0]][%{{.*}}] : memref<8xi32>
// CHECK:           memref.store %{{.*}}, %[[OUT1_BUFF_0]][%{{.*}}] : memref<8xi32>
// CHECK:         }
// CHECK:         aie.use_lock(%[[IN1_CONS_PROD_LOCK]], Release, 1)
// CHECK:         aie.use_lock(%[[OUT1_CONS_LOCK]], Release, 1)
// CHECK:         aie.use_lock(%[[IN1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:         aie.use_lock(%[[OUT1_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:         scf.for %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:           %{{.*}} = memref.load %[[IN1_CONS_BUFF_1]][%{{.*}}] : memref<8xi32>
// CHECK:           memref.store %{{.*}}, %[[OUT1_BUFF_1]][%{{.*}}] : memref<8xi32>
// CHECK:         }
// CHECK:         aie.use_lock(%[[IN1_CONS_PROD_LOCK]], Release, 1)
// CHECK:         aie.use_lock(%[[OUT1_CONS_LOCK]], Release, 1)
// CHECK:       }
// CHECK:       aie.end
// CHECK:     }
// CHECK:     aiex.runtime_sequence
// CHECK:     aie.shim_dma_allocation @in0(MM2S, 0, 0)
// CHECK:     %{{.*}} = aie.memtile_dma(%[[MEM_TILE_0_1]]) {
// CHECK:       aie.dma_start(S2MM, 1, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[IN0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[IN0_CONS_BUFF_0]] : memref<64000xi32>, 0, 64000)
// CHECK:       aie.use_lock(%[[IN0_CONS_CONS_LOCK]], Release, 1)
// CHECK:     ^bb3:
// CHECK:       aie.dma_start(MM2S, 1, ^bb4, ^bb6)
// CHECK:     ^bb4:
// CHECK:       aie.use_lock(%[[IN0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[IN0_CONS_BUFF_0]] : memref<64000xi32>, 0, 64000)
// CHECK:       aie.use_lock(%[[IN0_CONS_PROD_LOCK]], Release, 1)
// CHECK:     ^bb6:
// CHECK:       aie.dma_start(MM2S, 0, ^bb7, ^bb9)
// CHECK:     ^bb7:
// CHECK:       aie.use_lock(%[[OUT0_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OUT0_BUFF_0]] : memref<64000xi32>, 0, 64000)
// CHECK:       aie.use_lock(%[[OUT0_PROD_LOCK]], Release, 1)
// CHECK:     ^bb9:
// CHECK:       aie.dma_start(S2MM, 0, ^bb10, ^bb12)
// CHECK:     ^bb10:
// CHECK:       aie.use_lock(%[[OUT0_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OUT0_BUFF_0]] : memref<64000xi32>, 0, 64000)
// CHECK:       aie.use_lock(%[[OUT0_CONS_LOCK]], Release, 1)
// CHECK:     }
// CHECK:     %{{.*}} = aie.mem(%[[TILE_0_2]]) {
// CHECK:       aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[IN1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[IN1_CONS_BUFF_0]] : memref<8xi32>, 0, 8)
// CHECK:       aie.use_lock(%[[IN1_CONS_CONS_LOCK]], Release, 1)
// CHECK:     ^bb3:
// CHECK:       aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:     ^bb4:
// CHECK:       aie.use_lock(%[[OUT1_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OUT1_BUFF_0]] : memref<8xi32>, 0, 8)
// CHECK:       aie.use_lock(%[[OUT1_PROD_LOCK]], Release, 1)
// CHECK:     }
// CHECK:     aie.shim_dma_allocation @out0(S2MM, 0, 0)
// CHECK:   }
// CHECK: }


module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @in0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64000xi32>> 
    aie.objectfifo @in1(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8xi32>> 
    aie.objectfifo.link [@in0] -> [@in1]([] [])
    aie.objectfifo @out0(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64000xi32>> 
    aie.objectfifo @out1(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<8xi32>> 
    aie.objectfifo.link [@out1] -> [@out0]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c8000 = arith.constant 8000 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c8000 step %c1 {
        %0 = aie.objectfifo.acquire @in1(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %2 = aie.objectfifo.acquire @out1(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %c0_0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c8 step %c1_1 {
          %4 = memref.load %1[%arg1] : memref<8xi32>
          %c1_i32 = arith.constant 1 : i32
          %5 = arith.addi %4, %c1_i32 : i32
          memref.store %5, %3[%arg1] : memref<8xi32>
        }
        aie.objectfifo.release @in1(Consume, 1)
        aie.objectfifo.release @out1(Produce, 1)
      }
      aie.end
    }
    aiex.runtime_sequence @sequence(%arg0: memref<64000xi32>, %arg1: memref<64000xi32>, %arg2: memref<64000xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64000][0, 0, 0, 1]) {id = 1 : i64, metadata = @in0} : memref<64000xi32>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 1, 64000][0, 0, 0, 1]) {id = 0 : i64, metadata = @out0} : memref<64000xi32>
      aiex.npu.dma_wait {symbol = @out0}
    }
  }
}
