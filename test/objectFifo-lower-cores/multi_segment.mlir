// RUN: aie-opt --aie-objectfifo-lower-cores %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A core working several segments of one pool takes a lock per segment, all
// with the same delta, before the object is handed over.

module @multi_segment {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)

    %b0 = aie.buffer(%tile12) {sym_name = "b0"} : memref<48xi32>
    %f0 = aie.lock(%tile12) {init = 1 : i32, sym_name = "f0"}
    %u0 = aie.lock(%tile12) {init = 0 : i32, sym_name = "u0"}
    %f1 = aie.lock(%tile12) {init = 1 : i32, sym_name = "f1"}
    %u1 = aie.lock(%tile12) {init = 0 : i32, sym_name = "u1"}

    aie.objectfifo.pool @pool(%tile12) {
      depth = 1 : i32, buffers = [@b0],
      segments = [#aie.objectfifo_segment<offset = 0, size = 16,
                                          produceLock = @f0, consumeLock = @u0>,
                  #aie.objectfifo_segment<offset = 16, size = 32,
                                          produceLock = @f1, consumeLock = @u1>]
    } : memref<48xi32>
    aie.objectfifo.core_endpoint @reader(%tile12) drains @pool {segments = array<i32: 0, 1>}

    %core = aie.core(%tile12) {
      %e = aie.objectfifo.acquire @reader (1) : memref<48xi32>
      aie.objectfifo.release @reader (1)
      aie.end
    }
  }
}

// CHECK-LABEL: @multi_segment
// CHECK:   aie.core(%{{.*}}) {
// CHECK:     %[[DELTA:.*]] = arith.maxsi
// CHECK:     aie.use_lock(%u0, AcquireGreaterEqual, %[[DELTA]])
// CHECK:     aie.use_lock(%u1, AcquireGreaterEqual, %[[DELTA]])
// CHECK:     aie.use_lock(%f0, Release, %[[COUNT:.*]])
// CHECK:     aie.use_lock(%f1, Release, %[[COUNT]])
