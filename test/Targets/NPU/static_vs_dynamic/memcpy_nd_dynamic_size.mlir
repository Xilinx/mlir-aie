//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Static-vs-dynamic equivalence for a genuinely runtime DMA *size* on the
// dma_memcpy_nd path -- the case that actually exercises the dynamic BD-word
// encoder (DmaToNpuPattern::lowerDynamic), unlike memcpy_nd.mlir where only the
// rtp value is runtime and the DMA sizes stay constant.
//
// A runtime size does NOT produce a byte-identical stream: the static path
// bakes the size into one blockwrite, while the dynamic path emits a
// zero-template blockwrite + per-word write32 overrides. They program the SAME
// final BD registers, so the comparator replays both streams into a register
// map and compares that (compiled with -DDYN_STRUCTURAL) rather than diffing
// bytes. golden-vs-static stays byte-exact.
//
// @nd_static bakes the outer (d2) size N=4; @nd_dynamic takes it as %n. The
// transfer is non-contiguous (d1 stride 512 != inner product 32), so it stays
// ND mode and the d1/d2 size/stride words become write32 overrides.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: rm -rf %t.d && mkdir -p %t.d

// RUN: aie-opt --aie-dma-to-npu %s -o %t.d/lowered.mlir

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false \
// RUN:   -aie-sequence-name=nd_static %t.d/lowered.mlir > %t.d/golden.hex

// RUN: aie-translate --aie-npu-to-cpp %t.d/lowered.mlir > %t.d/gen.h

// RUN: %host_clang -std=c++17 -DDYN_STRUCTURAL -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen.h"' \
// RUN:   -DSTATIC_FN=generate_txn_main_nd_static \
// RUN:   -DDYN_FN=generate_txn_main_nd_dynamic -DARGVAL=4 \
// RUN:   %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/cmp.exe
// RUN: %t.d/cmp.exe %t.d/golden.hex

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_in (%tile_0_0, MM2S, 0)

    aie.runtime_sequence @nd_static(%in: memref<8192xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 4, 8, 32][4096, 512, 32, 1]) {id = 0 : i64, metadata = @of_in} : memref<8192xi32>
    }

    aie.runtime_sequence @nd_dynamic(%in: memref<8192xi32>, %n: i64) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, %n, 8, 32][4096, 512, 32, 1]) {id = 0 : i64, metadata = @of_in} : memref<8192xi32>
    }
  }
}
