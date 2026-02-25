//===- lower_set_lock.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
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
        // CHECK: aiex.npu.write32 {address = 126976 : ui32, column = 2 : i32, row = 2 : i32, value = 0 : ui32}
        aiex.set_lock(%lock22_0, 0)
        // 127216 = 0X0001F0F0
        // CHECK: aiex.npu.write32 {address = 127216 : ui32, column = 2 : i32, row = 2 : i32, value = 1 : ui32}
        aiex.set_lock(%lock22_15, 1)
        // 786480 = 0x000C0030
        // CHECK: aiex.npu.write32 {address = 786480 : ui32, column = 1 : i32, row = 1 : i32, value = 0 : ui32}
        aiex.set_lock(%lock11_3, 0)
        // 787328 = 0x000C0380
        // CHECK: aiex.npu.write32 {address = 787328 : ui32, column = 1 : i32, row = 1 : i32, value = 0 : ui32}
        aiex.set_lock(%lock11_56, 0)
        // 82000 = 0x00014050
        // CHECK: aiex.npu.write32 {address = 82000 : ui32, column = 0 : i32, row = 0 : i32, value = 0 : ui32}
        aiex.set_lock(%lock00_5, 0)
    }
  }
}
