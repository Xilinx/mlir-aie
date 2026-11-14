//===- objectfifo_link_nested_ref.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A link names fifos by flat symbols; a nested reference names none.

aie.device(npu2) {
  %t00 = aie.tile(0, 0)
  %t01 = aie.tile(0, 1)
  %t02 = aie.tile(0, 2)
  aie.objectfifo @in(%t00, {%t01}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @out(%t01, {%t02}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  // expected-error@+1 {{'aie.objectfifo.link' op names no objectfifo @in::@x}}
  aie.objectfifo.link [@in::@x] -> [@out]([] [])
}

// -----

// An allocate's verifier, which runs first, looks for the links naming its
// fifo, and so meets the nested reference too.
aie.device(npu2_1col) {
  %t00 = aie.tile(0, 0)
  %t02 = aie.tile(0, 2)
  %t03 = aie.tile(0, 3)
  aie.objectfifo @in(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @out(%t02, {%t03}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo.allocate @out (%t02)
  // expected-error@+1 {{'aie.objectfifo.link' op names no objectfifo @in::@x}}
  aie.objectfifo.link [@in::@x] -> [@out]([] [])
}
