//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A control packet reconfiguration of a memory tile relay. Column 0's shim feeds the
// memory tile (0, 1), whose MM2S channel sends the data on to column 1's shim.
// The channel waits on a lock only its S2MM descriptor releases, so
// aie-check-core-start-gating proves it gated and its queue push stays with the
// memory tile's other writes, while column 1's switches are configured beside
// it. Every word must make the round trip.

module {
  aie.device(npu2) @main {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out(%tile_1_0, S2MM, 0)

    aie.runtime_sequence @sequence(%arg0: memref<256xi32>, %arg1: memref<256xi32>) {
      aiex.npu.load_pdi {device_ref = @cfg}
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, metadata = @out, issue_token = true} : memref<256xi32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, metadata = @in} : memref<256xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }

  aie.device(npu2) @cfg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_1_0 = aie.tile(1, 0)

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_1_0, DMA : 0)

    %buff = aie.buffer(%tile_0_1) {sym_name = "buff"} : memref<256xi32>
    %prod = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "prod"}
    %cons = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "cons"}

    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^in, ^out)
    ^in:
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%prod, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buff : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%cons, Release, %c1_i32)
      aie.next_bd ^in
    ^out:
      %1 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buff : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%prod, Release, %c1)
      aie.next_bd ^send
    ^end:
      aie.end
    }
  }
}
