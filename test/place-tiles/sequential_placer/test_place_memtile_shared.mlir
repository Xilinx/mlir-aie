// RUN: aie-opt --split-input-file --aie-place-tiles --aie-objectfifo-split --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles --aie-objectFifo-stateful-transform="skip-verify=true" %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Both endpoints share one 300-KiB allocation, not two endpoint allocations.
module @shared {
  aie.device(npu2) {
    %a = aie.logical_tile<MemTile>(0, 1)
    %b = aie.logical_tile<MemTile>(1, 1)
    %reserved = aie.buffer(%a) : memref<217088xi8>
    aie.objectfifo @shared(%a, {%b}, 2 : i32) : !aie.objectfifo<memref<153600xi8>>
  }
}
// CHECK-LABEL: module @shared
// CHECK: %[[HOME:.*]] = aie.tile(0, 1)
// CHECK-DAG: aie.buffer(%[[HOME]]) {sym_name = "shared_buff_0"}
// CHECK-DAG: aie.buffer(%[[HOME]]) {sym_name = "shared_buff_1"}
// CHECK: aie.objectfifo.pool @shared_pool(%[[HOME]]) {buffers = [@shared_buff_0, @shared_buff_1]
// CHECK-NOT: aie.objectfifo.pool

// -----

// A delegate relocates that same storage. Neither endpoint gets another copy.
module @delegated {
  aie.device(npu2) {
    %endpoint = aie.logical_tile<MemTile>(0, 1)
    %delegate = aie.tile(1, 1)
    %reserved = aie.buffer(%delegate) : memref<217088xi8>
    aie.objectfifo @delegated(%endpoint, {%endpoint}, 2 : i32) : !aie.objectfifo<memref<153600xi8>>
    aie.objectfifo.allocate @delegated(%delegate)
  }
}
// CHECK-LABEL: module @delegated
// CHECK: %[[DELEGATE:.*]] = aie.tile(1, 1)
// CHECK-DAG: aie.buffer(%[[DELEGATE]]) {sym_name = "delegated_buff_0"}
// CHECK-DAG: aie.buffer(%[[DELEGATE]]) {sym_name = "delegated_buff_1"}
// CHECK: aie.objectfifo.pool @delegated_pool(%[[DELEGATE]]) {buffers = [@delegated_buff_0, @delegated_buff_1]
// CHECK-NOT: aie.objectfifo.pool
