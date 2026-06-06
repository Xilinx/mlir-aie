//===- unknown_args_rejected.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verify that aiecc rejects unknown command-line options that appear before
// the `--` host-passthrough separator. See issue #2989.

// REQUIRES: peano

// An unknown long option is rejected.
// RUN: not aiecc --garbage %s 2>&1 | FileCheck %s --check-prefix=GARBAGE
// GARBAGE: {{[Uu]nknown command line argument.*--garbage}}

// A typo'd `--aie-*` option is rejected (the original report from #2989).
// RUN: not aiecc --aie-genrate-npu-insts %s 2>&1 | FileCheck %s --check-prefix=TYPO
// TYPO: {{[Uu]nknown command line argument.*--aie-genrate-npu-insts}}

// Anything after `--` is forwarded to host compilation and not validated by
// aiecc. With --no-compile-host the passthrough args are not consumed at all,
// which is fine.
// RUN: aiecc --no-xchesscc --no-xbridge --no-compile-host -n %s -- --garbage 2>&1 | FileCheck %s --check-prefix=PASSTHROUGH
// PASSTHROUGH-NOT: {{[Uu]nknown command line argument}}

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    }

    aie.runtime_sequence(%in : memref<16xi32>, %out : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_in, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
