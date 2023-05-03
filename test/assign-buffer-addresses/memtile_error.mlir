//===- memtile_error.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-assign-buffer-addresses %s |& FileCheck %s
// CHECK:   error: 'AIE.tile' op allocated buffers exceeded available memory

module @test {
 AIE.device(xcve2302) {
  %0 = AIE.tile(3, 1)
  %b1 = AIE.buffer(%0) { sym_name = "a" } : memref<132000xi32>
  AIE.mem(%0) {
    AIE.end
  }
 }
}
