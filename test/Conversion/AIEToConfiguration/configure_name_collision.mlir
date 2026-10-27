// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --convert-aie-to-transaction --split-input-file %s | FileCheck %s

// There is no input runtime sequence. Its generated name must avoid the buffer
// symbol, and the post-pass symbol-table verifier must accept both declarations.
// CHECK-LABEL: aie.device(npu1_1col) @buffer_collision
// CHECK-DAG: aie.buffer @configure(
// CHECK-DAG: aie.runtime_sequence @configure0()
// CHECK-DAG: aiex.npu.blockwrite(
aie.device(npu1_1col) @buffer_collision {
  %tile = aie.tile(0, 2)
  aie.buffer @configure(%tile) {address = 1024 : i32} : memref<4xi32> = dense<[1, 2, 3, 4]>
}

// -----

// A lock's name is just as real a symbol-table entry.
// CHECK-LABEL: aie.device(npu1_1col) @lock_collision
// CHECK-DAG: aie.lock @configure(
// CHECK-DAG: aie.runtime_sequence @configure0()
aie.device(npu1_1col) @lock_collision {
  %tile = aie.tile(0, 2)
  aie.lock @configure(%tile, 0) {init = 1 : i32}
}
