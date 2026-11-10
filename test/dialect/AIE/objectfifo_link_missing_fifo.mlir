//===- objectfifo_link_missing_fifo.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A link naming a fifo that does not exist is reported, also when an
// allocate's verifier, which runs first, looks for the links naming its fifo.

aie.device(npu2_1col) {
  %s = aie.tile(0, 0)
  %a = aie.tile(0, 2)
  %l = aie.tile(0, 3)
  aie.objectfifo @f(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @f2(%s, {%l}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo.allocate @f (%l)
  // expected-error@+1 {{'aie.objectfifo.link' op names no objectfifo @missing}}
  aie.objectfifo.link [@f, @f2] -> [@missing] ([0, 8] [])
}

// -----

aie.device(npu2_1col) {
  %s = aie.tile(0, 0)
  %a = aie.tile(0, 2)
  %l = aie.tile(0, 3)
  aie.objectfifo @f(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @f2(%s, {%l}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  // expected-error@+1 {{'aie.objectfifo.link' op names no objectfifo @missing}}
  aie.objectfifo.link [@f, @f2] -> [@missing] ([0, 8] [])
}
