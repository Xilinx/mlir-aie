//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// npu1 counterpart of memcpy_nd.mlir. Device info is baked into the TXN header
// words, so the static ≡ dynamic equivalence is checked per device generation;
// this file covers npu1 while memcpy_nd.mlir covers npu2. See that file and
// README.md for how the three-way comparison works.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: aie-opt --aie-dma-to-npu %s -o %t.d/lowered.mlir
// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false \
// RUN:   -aie-sequence-name=memcpy_static %t.d/lowered.mlir > %t.d/golden.hex
// RUN: aie-translate --aie-npu-to-cpp %t.d/lowered.mlir > %t.d/gen.h
// RUN: %host_clang -std=c++17 -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen.h"' \
// RUN:   -DSTATIC_FN=generate_txn_main_memcpy_static \
// RUN:   -DDYN_FN=generate_txn_main_memcpy_dynamic -DARGVAL=4096 \
// RUN:   %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/cmp.exe
// RUN: %t.d/cmp.exe %t.d/golden.hex

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %rtp = aie.buffer(%tile_0_2) {sym_name = "rtp", address = 49152 : i32} : memref<16xi32>
    aie.shim_dma_allocation @of_in  (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @of_out (%tile_0_0, S2MM, 0)

    aie.runtime_sequence @memcpy_static(%in: memref<8192xi32>, %out: memref<8192xi32>) {
      %c4096 = arith.constant 4096 : i32
      %c4097 = arith.constant 4097 : i32
      %mwaddr = arith.constant 196612 : i32
      %mwmask = arith.constant 255 : i32
      aiex.npu.rtp_write(@rtp, 0, %c4096) : i32
      aiex.npu.rtp_write(@rtp, 4, %c4097) : i32
      aiex.npu.maskwrite32(%mwaddr, %c4096, %mwmask) : i32, i32, i32
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][2, 4, 8, 64][2048, 512, 64, 1]) {id = 1 : i64, metadata = @of_out} : memref<8192xi32>
      aiex.npu.dma_memcpy_nd(%in [0, 0, 0, 0][1, 8, 16, 32][4096, 512, 32, 1]) {id = 0 : i64, metadata = @of_in} : memref<8192xi32>
    }

    aie.runtime_sequence @memcpy_dynamic(%in: memref<8192xi32>, %out: memref<8192xi32>, %n: i32) {
      %c1 = arith.constant 1 : i32
      %np1 = arith.addi %n, %c1 : i32
      %mwaddr = arith.constant 196612 : i32
      %mwmask = arith.constant 255 : i32
      aiex.npu.rtp_write(@rtp, 0, %n) : i32
      aiex.npu.rtp_write(@rtp, 4, %np1) : i32
      aiex.npu.maskwrite32(%mwaddr, %n, %mwmask) : i32, i32, i32
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][2, 4, 8, 64][2048, 512, 64, 1]) {id = 1 : i64, metadata = @of_out} : memref<8192xi32>
      aiex.npu.dma_memcpy_nd(%in [0, 0, 0, 0][1, 8, 16, 32][4096, 512, 32, 1]) {id = 0 : i64, metadata = @of_in} : memref<8192xi32>
    }
  }
}
