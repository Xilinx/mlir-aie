//===- cpp_static_vs_dynamic_txn.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// End-to-end equivalence check: the TXN word stream produced from a static
// `aie.runtime_sequence` (this file, hardcoded N=4096) must be bit-identical
// to the stream produced from a runtime-parameterized version
// (Inputs/static_vs_dynamic_txn/passthrough_dynamic.mlir) when the dynamic
// version is invoked with N=4096.
//
// All DMA descriptor fields are kept as constants on both sides so the BD
// pass folds them into `blockwrite` ops; only `aiex.npu.rtp_write`
// operations are wired to the runtime %n parameter on the dynamic side.
// This is the most a runtime parameter can affect on the dynamic side
// without breaking blockwrite folding of the BDs and producing a
// structurally different (per-register `write32`) TXN stream.
//
// Two 4-D DMA patterns and two `rtp_write`s (one direct, one derived) are
// included to exercise more TXN op encodings and word territory than a
// single flat transfer would.
//
// Why a host-compiled comparison rather than a pure FileCheck?  aie-translate
// has no flag to substitute SSA i32 arguments of a runtime_sequence at
// translate time, so a `--aie-npu-to-binary` round-trip on the dynamic MLIR
// cannot resolve %n to a constant.  The comparison therefore happens in
// compiled C++.  Each generated header defines a `generate_txn_sequence`
// symbol, so the two headers are compiled in separate translation units
// (`gen_static.cpp`, `gen_dynamic.cpp`) and exposed under unique wrapper
// names that the harness `compare_main.cpp` calls.
//
// To add a new size N to this test:
//   1. Drop a new static MLIR (e.g. passthrough_static_NEW.mlir) into
//      Inputs/static_vs_dynamic_txn/, identical to this file but with the
//      `arith.constant N : i32` value updated.
//   2. Add three RUN lines: one to generate the static header, one to
//      compile gen_static.cpp with -DSTATIC_HEADER and -DSTATIC_NAME, one
//      to link & invoke compare_main.cpp with -Dstatic_txn=...  passing N.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: rm -rf %t.d && mkdir -p %t.d

// Generate the dynamic TXN header.
// RUN: aiecc --no-xchesscc --no-xbridge --aie-generate-txn-cpp \
// RUN:   --txn-cpp-name=%t.d/dynamic_txn.h --no-compile --no-link \
// RUN:   %S/Inputs/static_vs_dynamic_txn/passthrough_dynamic.mlir

// Generate the static TXN header for N=4096 (this file).
// RUN: aiecc --no-xchesscc --no-xbridge --aie-generate-txn-cpp \
// RUN:   --txn-cpp-name=%t.d/static_txn_4096.h --no-compile --no-link %s

// Compile harness wrappers.  gen_static.cpp uses -D macros so the same
// source file can be reused for additional N values.
// RUN: clang++ -std=c++17 -O0 -I%t.d -I%S/../../include \
// RUN:   -DSTATIC_HEADER='"static_txn_4096.h"' -DSTATIC_NAME=static_txn_4096 \
// RUN:   -c %S/Inputs/static_vs_dynamic_txn/gen_static.cpp \
// RUN:   -o %t.d/gen_static_4096.o
// RUN: clang++ -std=c++17 -O0 -I%t.d -I%S/../../include \
// RUN:   -c %S/Inputs/static_vs_dynamic_txn/gen_dynamic.cpp \
// RUN:   -o %t.d/gen_dynamic.o

// compare_main.cpp's prototypes use the canonical name `static_txn`; the
// -D rename selects which generated wrapper resolves it for this run.

// --- N=4096 comparison ---
// RUN: clang++ -std=c++17 -O0 -Dstatic_txn=static_txn_4096 \
// RUN:   %S/Inputs/static_vs_dynamic_txn/compare_main.cpp \
// RUN:   %t.d/gen_static_4096.o %t.d/gen_dynamic.o \
// RUN:   -o %t.compare_4096.exe
// RUN: %t.compare_4096.exe 4096

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>

    %rtp = aie.buffer(%tile_0_2) {sym_name = "rtp"} : memref<16xi32>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c1_i32 = arith.constant 1 : i32

      %subview_in = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<64xi32>>
      %elem_in = aie.objectfifo.subview.access %subview_in[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>

      %subview_out = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>
      %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>

      scf.for %i = %c0 to %c64 step %c1 {
        %val = memref.load %elem_in[%i] : memref<64xi32>
        %result = arith.addi %val, %c1_i32 : i32
        memref.store %result, %elem_out[%i] : memref<64xi32>
      }

      aie.objectfifo.release @of_in(Consume, 1)
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } {link_with = ""}

    aie.runtime_sequence(%in : memref<8192xi32>, %out : memref<8192xi32>) {
      // Static N=4096; second rtp_write value is the derived %n+1 = 4097.
      %c4096_i32 = arith.constant 4096 : i32
      %c4097_i32 = arith.constant 4097 : i32
      aiex.npu.rtp_write(@rtp, 0 : ui32, %c4096_i32) : i32
      aiex.npu.rtp_write(@rtp, 4 : ui32, %c4097_i32) : i32

      aiex.npu.write32 {address = 196612 : ui32, value = 42 : ui32}

      %c0   = arith.constant    0 : i64
      %c1   = arith.constant    1 : i64
      %c2   = arith.constant    2 : i64
      %c4   = arith.constant    4 : i64
      %c8   = arith.constant    8 : i64
      %c16  = arith.constant   16 : i64
      %c32  = arith.constant   32 : i64
      %c64  = arith.constant   64 : i64
      %c128 = arith.constant  128 : i64
      %c256 = arith.constant  256 : i64
      %c512 = arith.constant  512 : i64
      %c2048 = arith.constant 2048 : i64
      %c4096 = arith.constant 4096 : i64

      // First 4-D pattern: sizes=[2,4,8,64], strides=[2048,512,64,1].
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0]
                                 [%c2,%c4,%c8,%c64]
                                 [%c2048,%c512,%c64,%c1])
        {metadata = @of_out, id = 1 : i64} : memref<8192xi32>

      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0]
                                [%c2,%c4,%c8,%c64]
                                [%c2048,%c512,%c64,%c1])
        {metadata = @of_in, id = 0 : i64, issue_token = true} : memref<8192xi32>

      aiex.npu.dma_wait {symbol = @of_out}

      // Second 4-D pattern with different sizes/strides.
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0]
                                 [%c1,%c8,%c16,%c32]
                                 [%c4096,%c512,%c32,%c1])
        {metadata = @of_out, id = 1 : i64} : memref<8192xi32>

      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0]
                                [%c1,%c8,%c16,%c32]
                                [%c4096,%c512,%c32,%c1])
        {metadata = @of_in, id = 0 : i64, issue_token = true} : memref<8192xi32>

      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
