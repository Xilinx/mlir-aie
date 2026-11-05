// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// @deep's pool is walked whole only once its descriptors run short, and then
// its objects are one 302080-byte buffer that fits only at home, before @big's
// objects take the room. Buffers are placed largest buffer first, so the order
// is chosen again after the walk.
// CHECK-LABEL: module @walked_first
// CHECK-DAG: %[[M0:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[M1:.*]] = aie.tile(1, 1)
// CHECK-DAG: aie.buffer(%[[M0]]) {sym_name = "deep__cons__buf0"} : memref<302080xi8>
// CHECK-DAG: aie.buffer(%[[M0]]) {sym_name = "big__cons__buf0"}
// CHECK-DAG: aie.buffer(%[[M1]]) {sym_name = "big__cons__buf1"}
module @walked_first {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %s2 = aie.tile(2, 0)
    %m0 = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    %r0 = aie.buffer(%m0) : memref<122176xi8>
    %r1 = aie.buffer(%m1) : memref<324288xi8>
    aie.objectfifo @big(%s0, {%m0}, [2 : i32, 2 : i32]) : !aie.objectfifo<memref<100032xi8>>
    aie.objectfifo @big_out(%m0, {%s0}, 2 : i32) : !aie.objectfifo<memref<100032xi8>>
    aie.objectfifo.link [@big] -> [@big_out] ([] [])
    aie.objectfifo @small_a(%s1, {%m0}, 9 : i32) : !aie.objectfifo<memref<3072xi8>>
    aie.objectfifo @small_b(%s1, {%m0}, 9 : i32) : !aie.objectfifo<memref<3072xi8>>
    aie.objectfifo @small(%m0, {%s1}, 9 : i32) : !aie.objectfifo<memref<6144xi8>>
    aie.objectfifo.link [@small_a, @small_b] -> [@small] ([0, 3072] [])
    aie.objectfifo @deep(%s2, {%m0}, [2 : i32, 59 : i32]) : !aie.objectfifo<memref<5120xi8>>
    aie.objectfifo @deep_out(%m0, {%s2}, 2 : i32) : !aie.objectfifo<memref<5120xi8>>
    aie.objectfifo.link [@deep] -> [@deep_out] ([] [])
  }
}
