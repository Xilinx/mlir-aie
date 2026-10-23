//===- channel_turns_bad.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.objectfifo.pool @a(%t) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.pool @b(%t) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{takes no turn of its own in `turns`}}
    aie.objectfifo.dma_endpoint @a_in(%t) fills @a {turns = [@b_in]}
    aie.objectfifo.dma_endpoint @b_in(%t) fills @b {turns = [@b_in]}
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.objectfifo.pool @a(%t) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{turns names @nowhere, which is no objectfifo.dma_endpoint}}
    aie.objectfifo.dma_endpoint @a_in(%t) fills @a {turns = [@a_in, @nowhere]}
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %u = aie.tile(0, 3)
    aie.objectfifo.pool @a(%t) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.pool @b(%u) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{takes turns with "b_in", which is not on this tile and channel with the same turns}}
    aie.objectfifo.dma_endpoint @a_in(%t) fills @a {turns = [@a_in, @b_in]}
    aie.objectfifo.dma_endpoint @b_in(%u) fills @b {turns = [@a_in, @b_in]}
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.objectfifo.pool @a(%t) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{taking turns on a channel, it cannot also dispatch, carry a header, stop after iterCount or repeat}}
    aie.objectfifo.dma_endpoint @a_in(%t) fills @a {turns = [@a_in], iterCount = 4 : i32}
  }
}
