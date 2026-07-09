//===- lower_set_lock.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-set-lock %s | FileCheck %s

module @test_simple_lock_set {
  aie.device(npu2) {
    %tile22 = aie.tile(2, 2)
    %lock22_0 = aie.lock(%tile22, 0)  {init = 1 : i32}
    %lock22_15 = aie.lock(%tile22, 15)  {init = 0 : i32}

    %memtile11 = aie.tile(1, 1)
    %lock11_3 = aie.lock(%memtile11, 3)  {init = 1 : i32}
    %lock11_56 = aie.lock(%memtile11, 56)  {init = 1 : i32}

    %shimtile00 = aie.tile(0, 0)
    %lock00_5 = aie.lock(%shimtile00, 5)  {init = 1 : i32}

    %core22 = aie.core(%tile22) {
        aie.use_lock(%lock22_0, "Acquire", 0)
        aie.use_lock(%lock22_15, "Acquire", 1)
        // Do operations
        aie.end
    }

    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in0 (%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @out0 (%tile_0_0, MM2S, 0)

    aie.runtime_sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
        // Set some runtime parameters before starting execution

        // 126976 = 0x0001F000 (lock 0 address in a compute tile's local address space)
        // CHECK-DAG: %[[V0:.*]] = arith.constant 0 : i32
        // CHECK-DAG: %[[A0:.*]] = arith.constant 126976 : i32
        // CHECK: aiex.npu.write32(%[[A0]], %[[V0]]) {column = 2 : i32, row = 2 : i32} : i32, i32
        aiex.set_lock(%lock22_0, 0)
        // 127216 = 0X0001F0F0
        // CHECK-DAG: %[[V1:.*]] = arith.constant 1 : i32
        // CHECK-DAG: %[[A1:.*]] = arith.constant 127216 : i32
        // CHECK: aiex.npu.write32(%[[A1]], %[[V1]]) {column = 2 : i32, row = 2 : i32} : i32, i32
        aiex.set_lock(%lock22_15, 1)
        // 786480 = 0x000C0030
        // CHECK-DAG: %[[V2:.*]] = arith.constant 0 : i32
        // CHECK-DAG: %[[A2:.*]] = arith.constant 786480 : i32
        // CHECK: aiex.npu.write32(%[[A2]], %[[V2]]) {column = 1 : i32, row = 1 : i32} : i32, i32
        aiex.set_lock(%lock11_3, 0)
        // 787328 = 0x000C0380
        // CHECK-DAG: %[[V3:.*]] = arith.constant 0 : i32
        // CHECK-DAG: %[[A3:.*]] = arith.constant 787328 : i32
        // CHECK: aiex.npu.write32(%[[A3]], %[[V3]]) {column = 1 : i32, row = 1 : i32} : i32, i32
        aiex.set_lock(%lock11_56, 0)
        // 82000 = 0x00014050
        // CHECK-DAG: %[[V4:.*]] = arith.constant 0 : i32
        // CHECK-DAG: %[[A4:.*]] = arith.constant 82000 : i32
        // CHECK: aiex.npu.write32(%[[A4]], %[[V4]]) {column = 0 : i32, row = 0 : i32} : i32, i32
        aiex.set_lock(%lock00_5, 0)
    }
  }
}
