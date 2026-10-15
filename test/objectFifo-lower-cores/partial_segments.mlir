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
    aie.objectfifo.pool @p(%t) {depth = 2 : i32, buffers = [@b0, @b1]} : memref<32xi32> {
      aie.objectfifo.segment @s0 {consumeLock = @cl, offset = 0 : i32, produceLock = @pl, size = 16 : i32}
      aie.objectfifo.segment @s1 {consumeLock = @cl, offset = 16 : i32, produceLock = @pl, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @half(%t) fills @p {segments = [@s1]}
    %c = aie.core(%t) {
      %e = aie.objectfifo.acquire @half (1) : memref<16xi32, strided<[1], offset: 16>>
      %i = arith.constant 0 : index
      %v = arith.constant 1 : i32
      memref.store %v, %e[%i] : memref<16xi32, strided<[1], offset: 16>>
      aie.objectfifo.release @half (1)
      aie.end
    }
  }
}

// CHECK-LABEL: aie.core
// CHECK:   %[[V0:.*]] = memref.subview %b0[16] [16] [1] : memref<32xi32> to memref<16xi32, strided<[1], offset: 16>>
// CHECK:   %[[V1:.*]] = memref.subview %b1[16] [16] [1] : memref<32xi32> to memref<16xi32, strided<[1], offset: 16>>
// CHECK:   %[[FRONT:.*]] = memref.alloca() {aie.objectfifo.object_slot} : memref<memref<16xi32, strided<[1], offset: 16>>>
// CHECK:   memref.store %[[V0]], %[[FRONT]][]
// CHECK:   %[[BACK:.*]] = memref.alloca() {aie.objectfifo.object_slot}
// CHECK:   memref.store %[[V1]], %[[BACK]][]
// CHECK:   %[[OBJ:.*]] = memref.load %[[FRONT]][]
// CHECK:   memref.store %{{.*}}, %[[OBJ]][%{{.*}}]
