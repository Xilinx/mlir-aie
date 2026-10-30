// RUN: aie-opt --aie-objectfifo-lower-dmas --mlir-print-debuginfo %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A tile's DMA program holds the descriptors of every endpoint on that tile, so
// it is located at all of them, not only the endpoint that created it. Each
// descriptor chain keeps its own endpoint's location.

module @shared_program {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)

    %b0 = aie.buffer(%tile12) {sym_name = "b0"} : memref<16xi32>
    %free = aie.lock(%tile12) {init = 1 : i32, sym_name = "free"}
    %full = aie.lock(%tile12) {init = 0 : i32, sym_name = "full"}
    aie.objectfifo.pool @out_pool(%tile12) {
      depth = 1 : i32, buffers = [@b0]
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {consumeLock = @full, offset = 0 : i32, produceLock = @free, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @out_dma(%tile12) drains @out_pool {
      channelIndex = 0 : i32
    } loc("of_out"("design.py":30:2))

    %c0 = aie.buffer(%tile12) {sym_name = "c0"} : memref<16xi32>
    %cfree = aie.lock(%tile12) {init = 1 : i32, sym_name = "cfree"}
    %cfull = aie.lock(%tile12) {init = 0 : i32, sym_name = "cfull"}
    aie.objectfifo.pool @in_pool(%tile12) {
      depth = 1 : i32, buffers = [@c0]
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {consumeLock = @cfull, offset = 0 : i32, produceLock = @cfree, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @in_dma(%tile12) fills @in_pool {
      channelIndex = 0 : i32
    } loc("of_in"("design.py":40:2))
  }
}

// CHECK:     aie.dma_start(MM2S, 0, {{.*}}) loc(#[[OUT:loc[0-9]*]])
// CHECK:     aie.dma_bd(%b0 : {{.*}}) loc(#[[OUT]])
// CHECK:     aie.dma_start(S2MM, 0, {{.*}}) loc(#[[IN:loc[0-9]*]])
// CHECK:     aie.dma_bd(%c0 : {{.*}}) loc(#[[IN]])
// CHECK:   } loc(#[[MEM:loc[0-9]*]])
// CHECK-DAG: #[[OUT]] = loc("of_out"(
// CHECK-DAG: #[[IN]] = loc("of_in"(
// CHECK-DAG: #[[MEM]] = loc(fused[#[[OUT]], #[[IN]]])
