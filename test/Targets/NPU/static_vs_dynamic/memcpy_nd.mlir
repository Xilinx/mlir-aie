//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Static-vs-dynamic TXN equivalence for the aiex.npu.dma_memcpy_nd path.
//
// Two runtime sequences describe the same transfer: @memcpy_static bakes the
// rtp value in as a constant; @memcpy_dynamic takes it as a runtime %n arg.
// The harness proves three word streams are byte-identical:
//   - the production binary emitter on @memcpy_static (the golden),
//   - generate_txn_main_memcpy_static() (EmitC path, same constants),
//   - generate_txn_main_memcpy_dynamic(4096) (runtime arg == the baked constant).
//
// Each sequence carries two distinct 4-D DMA patterns so more than one BD
// blockwrite / address_patch encoding is exercised. rtp_write is the only
// runtime-driven field: it is the most a runtime argument can affect without
// forcing the BD onto the per-register write32 path, which would make the
// static and dynamic streams differ structurally rather than in value.
//
// To add another size, see README.md in this directory.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: rm -rf %t.d && mkdir -p %t.d

// Lower the dma_memcpy_nd ops to terminal npu ops.
// RUN: aie-opt --aie-dma-to-npu %s -o %t.d/lowered.mlir

// Golden word stream from the production binary emitter.
// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false \
// RUN:   -aie-sequence-name=memcpy_static %t.d/lowered.mlir > %t.d/golden.hex

// One generated header holds both generate_txn_main_memcpy_static/_dynamic.
// RUN: aie-translate --aie-npu-to-cpp %t.d/lowered.mlir > %t.d/gen.h

// Host-compile and run the three-way comparator.
// RUN: %host_clang -std=c++17 -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen.h"' \
// RUN:   -DSTATIC_FN=generate_txn_main_memcpy_static \
// RUN:   -DDYN_FN=generate_txn_main_memcpy_dynamic -DARGVAL=4096 \
// RUN:   %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/cmp.exe
// RUN: %t.d/cmp.exe %t.d/golden.hex

// Second size: a distinct static sequence bakes N=8192, and the same dynamic
// sequence is invoked with 8192. This exercises the add-a-size recipe and
// guards the runtime-argument path at more than one value.
// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false \
// RUN:   -aie-sequence-name=memcpy_static_8192 %t.d/lowered.mlir > %t.d/golden_8192.hex
// RUN: %host_clang -std=c++17 -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen.h"' \
// RUN:   -DSTATIC_FN=generate_txn_main_memcpy_static_8192 \
// RUN:   -DDYN_FN=generate_txn_main_memcpy_dynamic -DARGVAL=8192 \
// RUN:   %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/cmp_8192.exe
// RUN: %t.d/cmp_8192.exe %t.d/golden_8192.hex

module {
  aie.device(npu2) {
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

    // Second size for the add-a-size recipe: identical body to @memcpy_static
    // but bakes N=8192. The dynamic sequence above is invoked with 8192 to
    // match it.
    aie.runtime_sequence @memcpy_static_8192(%in: memref<8192xi32>, %out: memref<8192xi32>) {
      %c8192 = arith.constant 8192 : i32
      %c8193 = arith.constant 8193 : i32
      %mwaddr = arith.constant 196612 : i32
      %mwmask = arith.constant 255 : i32
      aiex.npu.rtp_write(@rtp, 0, %c8192) : i32
      aiex.npu.rtp_write(@rtp, 4, %c8193) : i32
      aiex.npu.maskwrite32(%mwaddr, %c8192, %mwmask) : i32, i32, i32
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][2, 4, 8, 64][2048, 512, 64, 1]) {id = 1 : i64, metadata = @of_out} : memref<8192xi32>
      aiex.npu.dma_memcpy_nd(%in [0, 0, 0, 0][1, 8, 16, 32][4096, 512, 32, 1]) {id = 0 : i64, metadata = @of_in} : memref<8192xi32>
    }
  }
}
