//===- bad_buffer.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s
// CHECK:error: 'aie.tile' op Buffer '"c"' must be defined directly under the device scope. Currently it is nested inside a core tile.

module @test {
  aie.device(xcvc1902) {
    %0 = aie.tile(3, 3)
    %b1 = aie.buffer(%0) { sym_name = "a" } : memref<16xi8>
    %3 = aie.tile(4, 4)
    %4 = aie.buffer(%3) : memref<500xi32>
    aie.core(%0) {
     %b2 = aie.buffer(%0) { sym_name = "c" } : memref<16xi16>
     %1 = aie.buffer(%0) { sym_name = "b" } : memref<3x3xi16> = dense<[[0, 4096, 0], [4096, -16384, 4096], [0, 4096, 0]]>
      aie.end
    }
    aie.core(%3) {
      aie.end
    }
  }
}

