//===- dma_start_channel_roundtrip.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --mlir-print-op-generic %s | FileCheck %s --check-prefix=GENERIC
// RUN: aie-opt --mlir-print-op-generic %s | aie-opt | FileCheck %s

// GENERIC: "aie.dma_start"(){{.*}}<{channel = 1 : i32, channel_dir = 0 : i32
// GENERIC: "aie.dma_start"(){{.*}}<{channel = @out, channel_dir = 1 : i32

// CHECK: = aie.dma_start(S2MM, 1, ^bb1, ^bb2)
// CHECK: = aie.dma_start(MM2S, @out, ^bb3, ^bb4, repeat_count = 2)
aie.device(npu2) {
  %mt = aie.tile(0, 1)
  %b = aie.buffer(%mt) {sym_name = "b"} : memref<64xi32>
  aie.route_endpoint @out(%mt) DMA
  aie.memtile_dma(%mt) {
    aie.dma_start(S2MM, 1, ^bd0, ^s1)
  ^bd0:
    aie.dma_bd(%b : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd0
  ^s1:
    aie.dma_start(MM2S, @out, ^bd1, ^end, repeat_count = 2)
  ^bd1:
    aie.dma_bd(%b : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd1
  ^end:
    aie.end
  }
}
