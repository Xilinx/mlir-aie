//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This test verifies that lock SSA values can be passed to precompiled C
// kernels as index arguments. The C kernel uses acquire_equal()/release()
// intrinsics directly with the localized lock IDs.

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    %in_buff = aie.buffer(%tile_0_2) {sym_name = "in_buff"} : memref<1024xi32>
    %out_buff = aie.buffer(%tile_0_2) {sym_name = "out_buff"} : memref<1024xi32>

    // Input locks: DMA writes data, core reads data
    %in_prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "in_prod_lock"}
    %in_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in_cons_lock"}

    // Output locks: core writes data, DMA reads data
    %out_prod_lock = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "out_prod_lock"}
    %out_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "out_cons_lock"}

    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)

    // External C function: receives buffers and lock IDs for both input/output
    func.func private @scale_with_locks(%in: memref<1024xi32>,
                                         %out: memref<1024xi32>,
                                         %in_cons_lk: index,
                                         %in_prod_lk: index,
                                         %out_prod_lk: index,
                                         %out_cons_lk: index) -> ()

    %core_0_2 = aie.core(%tile_0_2) {
      // Pass all lock IDs to C kernel
      func.call @scale_with_locks(%in_buff, %out_buff,
                                   %in_cons_lock, %in_prod_lock,
                                   %out_prod_lock, %out_cons_lock)
        : (memref<1024xi32>, memref<1024xi32>, index, index, index, index) -> ()
      aie.end
    } { link_with = "kernel.o" }

    aie.runtime_sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
      // Configure shim DMA to send data to core tile
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      }

      // Configure core DMA to receive from shim into in_buff
      %t1 = aiex.dma_configure_task(%tile_0_2, S2MM, 0) {
        aie.use_lock(%in_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%in_buff : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.use_lock(%in_cons_lock, Release, 1)
        aie.end
      }

      // Start input path
      aiex.dma_start_task(%t0)
      aiex.dma_start_task(%t1)

      // Configure core DMA to send out_buff to shim
      %t2 = aiex.dma_configure_task(%tile_0_2, MM2S, 0) {
        aie.use_lock(%out_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%out_buff : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.use_lock(%out_prod_lock, Release, 1)
        aie.end
      }

      // Configure shim DMA to receive output from core
      %t3 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%arg1 : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.end
      } {issue_token = true}

      // Start output path
      aiex.dma_start_task(%t2)
      aiex.dma_start_task(%t3)
      aiex.dma_await_task(%t3)
    }
  }
}
