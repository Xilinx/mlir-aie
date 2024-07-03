//===- else_condition_check.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses : Fails
// 
// malloc(): unaligned tcache chunk detected
// PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
// Stack dump:
// 0.      Program arguments: aie-opt --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses else_condition_check.mlir
// malloc(): unaligned tcache chunk detected
// Aborted (core dumped)


// RUN: aie-opt --aie-objectFifo-stateful-transform else_condition_check.mlir | aie-opt --aie-assign-buffer-addresses : Passes

module @test {
 aie.device(xcvc1902) {
  %tile12 = aie.tile(1, 2)
  %1 = aie.buffer(%tile12) { sym_name = "a" } : memref<4096xi32>  //16384 bytes
  %b1 = aie.buffer(%tile12) { sym_name = "b" } : memref<16xi16> // 32 bytes
  %tile13 = aie.tile(1, 3)
  aie.objectfifo @act_3_4(%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
 }
}
