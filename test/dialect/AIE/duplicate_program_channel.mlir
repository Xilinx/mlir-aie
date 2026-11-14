//===- duplicate_program_channel.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DeviceOp::verify rejects two DMA programs on one tile starting the same
// channel and direction: the channel runs one chain. Tiles of one position
// count as one; different channels, or directions, are fine.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu1_1col) {
  %mem = aie.tile(0, 1)
  %buf_a = aie.buffer(%mem) {sym_name = "buf_a"} : memref<16xi32>
  %buf_b = aie.buffer(%mem) {sym_name = "buf_b"} : memref<16xi32>
  %dma_a = aie.memtile_dma(%mem) {
    // expected-note@+1 {{the other start is here}}
    %s = aie.dma_start(MM2S, 0, ^b, ^end)
  ^b:
    aie.dma_bd(%buf_a : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.next_bd ^b
  ^end:
    aie.end
  }
  %dma_b = aie.memtile_dma(%mem) {
    // expected-error@+1 {{'aie.dma_start' op MM2S channel 0 is started by another DMA program on this tile too}}
    %s = aie.dma_start(MM2S, 0, ^b, ^end)
  ^b:
    aie.dma_bd(%buf_b : memref<16xi32> offset = 0 len = 16) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.next_bd ^b
  ^end:
    aie.end
  }
}

// -----

aie.device(npu1_1col) {
  %mem = aie.tile(0, 1)
  %mem_pinned = aie.logical_tile<MemTile>(0, 1)
  %buf_a = aie.buffer(%mem) {sym_name = "buf_a"} : memref<16xi32>
  %buf_b = aie.buffer(%mem_pinned) {sym_name = "buf_b"} : memref<16xi32>
  %dma_a = aie.memtile_dma(%mem) {
    // expected-note@+1 {{the other start is here}}
    %s = aie.dma_start(S2MM, 1, ^b, ^end)
  ^b:
    aie.dma_bd(%buf_a : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.next_bd ^b
  ^end:
    aie.end
  }
  %dma_b = aie.memtile_dma(%mem_pinned) {
    // expected-error@+1 {{'aie.dma_start' op S2MM channel 1 is started by another DMA program on this tile too}}
    %s = aie.dma_start(S2MM, 1, ^b, ^end)
  ^b:
    aie.dma_bd(%buf_b : memref<16xi32> offset = 0 len = 16) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.next_bd ^b
  ^end:
    aie.end
  }
}

// -----

aie.device(npu1_1col) {
  %mem = aie.tile(0, 1)
  %buf_a = aie.buffer(%mem) {sym_name = "buf_a"} : memref<16xi32>
  %buf_b = aie.buffer(%mem) {sym_name = "buf_b"} : memref<16xi32>
  %dma_a = aie.memtile_dma(%mem) {
    %s = aie.dma_start(MM2S, 0, ^b, ^end)
  ^b:
    aie.dma_bd(%buf_a : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32, next_bd_id = 0 : i32}
    aie.next_bd ^b
  ^end:
    aie.end
  }
  %dma_b = aie.memtile_dma(%mem) {
    %s = aie.dma_start(MM2S, 1, ^b, ^next)
  ^b:
    aie.dma_bd(%buf_b : memref<16xi32> offset = 0 len = 16) {bd_id = 1 : i32, next_bd_id = 1 : i32}
    aie.next_bd ^b
  ^next:
    %t = aie.dma_start(S2MM, 0, ^c, ^end)
  ^c:
    aie.dma_bd(%buf_b : memref<16xi32> offset = 0 len = 16) {bd_id = 2 : i32, next_bd_id = 2 : i32}
    aie.next_bd ^c
  ^end:
    aie.end
  }
}
