// RUN: aie-opt --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --aie-objectfifo-allocate --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --aie-objectfifo-allocate --aie-place-tiles --aie-objectfifo-lower-dmas --aie-assign-lock-ids --aie-assign-buffer-addresses %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A fully resolved logical endpoint can trigger the same locality repair as
// its physical alias. Keep the small pool local to its pinned channel while
// spilling part of the large pool, including when reserving the repair again.
module {
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    %alias = aie.logical_tile<MemTile>(0, 1)
    %reserved = aie.buffer(%home) {sym_name = "reserved"} : memref<196608xi8>
    aie.objectfifo.pool @large(%home) {depth = 2 : i32} : memref<147456xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 147456 : i32}
    }
    aie.objectfifo.pool @small(%home) {depth = 2 : i32} : memref<32768xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 32768 : i32}
    }
    aie.objectfifo.dma_endpoint @large_dma(%home) fills @large
    aie.objectfifo.dma_endpoint @small_dma(%alias) fills @small {channelIndex = 5 : i32}
  }
}
// CHECK-DAG: %[[HOME:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[ALIAS:.*]] = aie.logical_tile<MemTile>(0, 1)
// CHECK-DAG: %[[NEIGHBOR:.*]] = aie.tile(1, 1)
// CHECK-DAG: aie.buffer(%[[ALIAS]]) {sym_name = "small_buff_0"}
// CHECK-DAG: aie.buffer(%[[ALIAS]]) {sym_name = "small_buff_1"}
// CHECK-DAG: aie.lock(%[[ALIAS]]) {{.*}}sym_name = "small_prod_lock_0"
// CHECK-DAG: aie.lock(%[[ALIAS]]) {{.*}}sym_name = "small_cons_lock_0"
// CHECK-DAG: aie.buffer(%[[HOME]]) {sym_name = "large_buff_0"}
// CHECK-DAG: aie.buffer(%[[NEIGHBOR]]) {sym_name = "large_buff_1"}
// CHECK: @large_dma(%[[HOME]]) fills @large {channelIndex = 0 : i32}
// CHECK: @small_dma(%[[ALIAS]]) fills @small {channelIndex = 5 : i32}
