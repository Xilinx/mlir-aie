//===- partial_segments_rejected.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core is handed whole buffers, so an endpoint covering only part of an
// object would need a memref.subview at the segment offset that the lowering
// does not emit yet. Rejected rather than silently handing over the whole
// buffer.

// RUN: aie-opt --verify-diagnostics -split-input-file %s

module {
  aie.device(xcve2302) {
    %t = aie.tile(1, 2)
    aie.objectfifo.pool @p(%t) {depth = 2 : i32,
        segments = [#aie.objectfifo_segment<offset = 0, size = 16>,
                    #aie.objectfifo_segment<offset = 16, size = 16>]} : memref<32xi32>
    // expected-error@+1 {{a core endpoint must cover its pool's whole object}}
    aie.objectfifo.core_endpoint @half(%t) fills @p {segments = array<i32: 0>}
  }
}

// -----

// Selecting every segment is the ordinary case and stays legal.

module {
  aie.device(xcve2302) {
    %t = aie.tile(1, 2)
    aie.objectfifo.pool @p(%t) {depth = 2 : i32,
        segments = [#aie.objectfifo_segment<offset = 0, size = 16>,
                    #aie.objectfifo_segment<offset = 16, size = 16>]} : memref<32xi32>
    aie.objectfifo.core_endpoint @whole(%t) fills @p {segments = array<i32: 0, 1>}
  }
}
