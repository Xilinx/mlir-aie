//===- test_sa_spill_largest_first.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The placer's spill estimate and aie-objectfifo-allocate place mem tile
// buffers in one order, largest buffer first, so they agree on whether these
// fit. @deep's 59 objects need more BD ids than a channel has, so it is walked
// whole: one 302080-byte buffer that fits only at home, ahead of @big's
// objects. With 64 bytes less room at home, neither fits it.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL: module @fits
// CHECK: %[[M0:.*]] = aie.tile(0, 1)
// CHECK: aie.buffer(%[[M0]]) {sym_name = "deep__cons__buf0"} : memref<302080xi8>
module @fits {
  aie.device(npu2_2col) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %m0 = aie.logical_tile<MemTile>(0, 1)
    %m1 = aie.logical_tile<MemTile>(1, 1)
    %r0 = aie.buffer(%m0) : memref<122176xi8>
    %r1 = aie.buffer(%m1) : memref<324288xi8>
    aie.objectfifo @big(%s0, {%m0}, [2 : i32, 2 : i32]) : !aie.objectfifo<memref<100032xi8>>
    aie.objectfifo @big_out(%m0, {%s0}, 2 : i32) : !aie.objectfifo<memref<100032xi8>>
    aie.objectfifo.link [@big] -> [@big_out] ([] [])
    aie.objectfifo @small_a(%s1, {%m0}, 9 : i32) : !aie.objectfifo<memref<3072xi8>>
    aie.objectfifo @small_b(%s1, {%m0}, 9 : i32) : !aie.objectfifo<memref<3072xi8>>
    aie.objectfifo @small(%m0, {%s1}, 9 : i32) : !aie.objectfifo<memref<6144xi8>>
    aie.objectfifo.link [@small_a, @small_b] -> [@small] ([0, 3072] [])
    aie.objectfifo @deep(%s0, {%m0}, [2 : i32, 59 : i32]) : !aie.objectfifo<memref<5120xi8>>
    aie.objectfifo @deep_out(%m0, {%s1}, 2 : i32) : !aie.objectfifo<memref<5120xi8>>
    aie.objectfifo.link [@deep] -> [@deep_out] ([] [])
  }
}

// -----

module @does_not_fit {
  // expected-warning @below {{SA placer's resource estimate is over capacity}}
  aie.device(npu2_2col) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    // expected-note@+1 {{memory spillover penalty=4560 on (0, 1)}}
    %m0 = aie.logical_tile<MemTile>(0, 1)
    %m1 = aie.logical_tile<MemTile>(1, 1)
    %r0 = aie.buffer(%m0) : memref<122240xi8>
    %r1 = aie.buffer(%m1) : memref<324288xi8>
    aie.objectfifo @big(%s0, {%m0}, [2 : i32, 2 : i32]) : !aie.objectfifo<memref<100032xi8>>
    aie.objectfifo @big_out(%m0, {%s0}, 2 : i32) : !aie.objectfifo<memref<100032xi8>>
    // expected-error @below {{could not place buffers in accessible memory with available capacity}}
    aie.objectfifo.link [@big] -> [@big_out] ([] [])
    aie.objectfifo @small_a(%s1, {%m0}, 9 : i32) : !aie.objectfifo<memref<3072xi8>>
    aie.objectfifo @small_b(%s1, {%m0}, 9 : i32) : !aie.objectfifo<memref<3072xi8>>
    aie.objectfifo @small(%m0, {%s1}, 9 : i32) : !aie.objectfifo<memref<6144xi8>>
    aie.objectfifo.link [@small_a, @small_b] -> [@small] ([0, 3072] [])
    aie.objectfifo @deep(%s0, {%m0}, [2 : i32, 59 : i32]) : !aie.objectfifo<memref<5120xi8>>
    aie.objectfifo @deep_out(%m0, {%s1}, 2 : i32) : !aie.objectfifo<memref<5120xi8>>
    aie.objectfifo.link [@deep] -> [@deep_out] ([] [])
  }
}
