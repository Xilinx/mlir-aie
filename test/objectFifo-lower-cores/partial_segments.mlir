//===- partial_segments.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core endpoint covering one segment of a shared object reaches it through a
// memref.subview, which is what lets a core take one side of a join.

// RUN: aie-opt --aie-objectfifo-lower-cores %s | FileCheck %s

module {
  aie.device(xcve2302) {
    %t = aie.tile(1, 2)
    %b0 = aie.buffer(%t) {sym_name = "b0"} : memref<32xi32>
    %b1 = aie.buffer(%t) {sym_name = "b1"} : memref<32xi32>
    %pl = aie.lock(%t) {init = 2 : i32, sym_name = "pl"}
    %cl = aie.lock(%t) {init = 0 : i32, sym_name = "cl"}
    aie.objectfifo.pool @p(%t) {depth = 2 : i32, buffers = [@b0, @b1],
        segments = [#aie.objectfifo_segment<offset = 0, size = 16, produceLock = @pl, consumeLock = @cl>,
                    #aie.objectfifo_segment<offset = 16, size = 16, produceLock = @pl, consumeLock = @cl>]} : memref<32xi32>
    aie.objectfifo.core_endpoint @half(%t) fills @p {segments = array<i32: 1>}
    %c = aie.core(%t) {
      %e = aie.objectfifo.acquire @half : memref<16xi32, strided<[1], offset: 16>>
      aie.objectfifo.release @half [1]
      aie.end
    }
  }
}

// CHECK-LABEL: aie.core
// CHECK:   %[[V0:.*]] = memref.subview %b0[16] [16] [1] : memref<32xi32> to memref<16xi32, strided<[1], offset: 16>>
// CHECK:   %[[V1:.*]] = memref.subview %b1[16] [16] [1] : memref<32xi32> to memref<16xi32, strided<[1], offset: 16>>
// CHECK:   scf.index_switch
// CHECK:     scf.yield %[[V0]]
// CHECK:     scf.yield %[[V1]]
