//===- discontiguous_segments_rejected.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core sees one memref, so the segments an endpoint selects have to be a
// single run of the object.

// RUN: aie-opt --verify-diagnostics -split-input-file %s

module {
  aie.device(xcve2302) {
    %t = aie.tile(1, 2)
    aie.objectfifo.pool @p(%t) {depth = 2 : i32,
        segments = [#aie.objectfifo_segment<offset = 0, size = 16>,
                    #aie.objectfifo_segment<offset = 16, size = 16>,
                    #aie.objectfifo_segment<offset = 32, size = 16>]} : memref<48xi32>
    // expected-error@+1 {{a core endpoint's segments must be contiguous}}
    aie.objectfifo.core_endpoint @ends(%t) fills @p {segments = array<i32: 0, 2>}
  }
}

// -----

// A run of segments in the middle of an object is fine.

module {
  aie.device(xcve2302) {
    %t = aie.tile(1, 2)
    aie.objectfifo.pool @p(%t) {depth = 2 : i32,
        segments = [#aie.objectfifo_segment<offset = 0, size = 16>,
                    #aie.objectfifo_segment<offset = 16, size = 16>,
                    #aie.objectfifo_segment<offset = 32, size = 16>]} : memref<48xi32>
    aie.objectfifo.core_endpoint @middle(%t) fills @p {segments = array<i32: 1, 2>}
  }
}
