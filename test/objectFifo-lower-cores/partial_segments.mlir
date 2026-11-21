//===- partial_segments.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core endpoint covering a segment past the shared object's first element
// reaches it through a buffer aliasing each pool buffer at the segment's offset
// (alias_of, alias_offset), a plain memref, which is what lets a core take one
// side of a join.

// RUN: aie-opt --aie-objectfifo-lower-cores %s | FileCheck %s

module {
  aie.device(xcve2302) {
    %t = aie.tile(1, 2)
    %b0 = aie.buffer(%t) {sym_name = "b0"} : memref<32xi32>
    %b1 = aie.buffer(%t) {sym_name = "b1"} : memref<32xi32>
    %pl = aie.lock(%t) {init = 2 : i32, sym_name = "pl"}
    %cl = aie.lock(%t) {init = 0 : i32, sym_name = "cl"}
    aie.objectfifo.pool @p(%t) {depth = 2 : i32, buffers = [@b0, @b1]} : memref<32xi32> {
      aie.objectfifo.segment @s0 {consumeLock = @cl, offset = 0 : i32, produceLock = @pl, size = 16 : i32}
      aie.objectfifo.segment @s1 {consumeLock = @cl, offset = 16 : i32, produceLock = @pl, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @half(%t) fills @p {segments = [@s1]}
    %c = aie.core(%t) {
      %e = aie.objectfifo.acquire @half (1) : memref<16xi32>
      %i = arith.constant 0 : index
      %v = arith.constant 1 : i32
      memref.store %v, %e[%i] : memref<16xi32>
      aie.objectfifo.release @half (1)
      aie.end
    }
  }
}

// CHECK:       %[[A0:.*]] = aie.buffer(%{{.*}}) {alias_of = @b0, alias_offset = 64 : i32, sym_name = "b0__at16"} : memref<16xi32>
// CHECK:       %[[A1:.*]] = aie.buffer(%{{.*}}) {alias_of = @b1, alias_offset = 64 : i32, sym_name = "b1__at16"} : memref<16xi32>
// CHECK-LABEL: aie.core
// CHECK:   %[[FRONT:.*]] = memref.alloca() {aie.objectfifo.object_slot} : memref<memref<16xi32>>
// CHECK:   memref.store %[[A0]], %[[FRONT]][]
// CHECK:   %[[BACK:.*]] = memref.alloca() {aie.objectfifo.object_slot}
// CHECK:   memref.store %[[A1]], %[[BACK]][]
// CHECK:   %[[OBJ:.*]] = memref.load %[[FRONT]][]
// CHECK:   memref.store %{{.*}}, %[[OBJ]][%{{.*}}]
