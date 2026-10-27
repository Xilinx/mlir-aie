//===- bad_buffer.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Buffer symbols cannot be nested inside a core. Verification rejects this
// malformed design before aie-prepare-buffers or core compilation can run.
// aie-assign-buffer-addresses assigns the addresses.
// RUN: not aie-opt --aie-prepare-buffers %s 2>&1 | FileCheck %s
// CHECK: error: 'aie.buffer' op symbol's parent must have the SymbolTable trait
// CHECK-NEXT: aie.buffer @c(

module @test {
  aie.device(xcvc1902) {
    %0 = aie.tile(3, 3)
    aie.buffer @a(%0) : memref<16xi8>
    %3 = aie.tile(4, 4)
    aie.buffer @buffer_4(%3) : memref<500xi32>
    aie.core(%0) {
      aie.buffer @c(%0) : memref<16xi16>
      aie.buffer @b(%0) : memref<3x3xi16> = dense<[[0, 4096, 0], [4096, -16384, 4096], [0, 4096, 0]]>
      aie.end
    }
    aie.core(%3) {
      aie.end
    }
  }
}
