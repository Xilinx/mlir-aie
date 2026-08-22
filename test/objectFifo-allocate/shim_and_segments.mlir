// RUN: aie-opt --aie-objectfifo-allocate %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A shim end the runtime drives holds no objects, so it gets no buffers and no
// locks -- only the allocation record the runtime sequence resolves.

module @shim_and_segments {
  aie.device(npu1) {
    %shim = aie.tile(0, 0)
    %memtile = aie.tile(0, 1)

    aie.objectfifo.dangling_endpoint @in_shim(%shim) DMA {fifoName = "in"}

    // Two participants writing one object need a lock pair each.
    aie.objectfifo.pool @in_pool(%memtile) {
      depth = 2 : i32,
      segments = [#aie.objectfifo_segment<offset = 0, size = 16>,
                  #aie.objectfifo_segment<offset = 16, size = 32>]
    } : memref<48xi32>
    aie.objectfifo.dma_endpoint @in_dma(%memtile) fills @in_pool {segments = array<i32: 0, 1>}

    aie.objectfifo.flow from @in_shim to [@in_dma]
  }
}

// CHECK-LABEL: @shim_and_segments
// CHECK-DAG:   %[[SHIM:.*]] = aie.tile(0, 0)
// CHECK-DAG:   %[[MT:.*]] = aie.tile(0, 1)
// CHECK-NOT: aie.buffer(%[[SHIM]])
// CHECK-NOT: aie.lock(%[[SHIM]])
// CHECK:   aie.buffer(%[[MT]]) {sym_name = "in_buff_0"} : memref<48xi32>
// CHECK:   aie.buffer(%[[MT]]) {sym_name = "in_buff_1"} : memref<48xi32>
// CHECK:   aie.lock(%[[MT]]) {init = 2 : i32, sym_name = "in_prod_lock_0"}
// CHECK:   aie.lock(%[[MT]]) {init = 0 : i32, sym_name = "in_cons_lock_0"}
// CHECK:   aie.lock(%[[MT]]) {init = 2 : i32, sym_name = "in_prod_lock_1"}
// CHECK:   aie.lock(%[[MT]]) {init = 0 : i32, sym_name = "in_cons_lock_1"}
// CHECK:   aie.objectfifo.dangling_endpoint @in_shim(%[[SHIM]]) DMA {channelIndex = 0 : i32, fifoName = "in"}
// CHECK:   aie.flow(%[[SHIM]], DMA : 0, %[[MT]], DMA : 0)
// CHECK:   aie.shim_dma_allocation @in_shim_alloc(%[[SHIM]], MM2S, 0)
