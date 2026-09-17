// RUN: aie-opt --split-input-file --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --aie-objectfifo-lower-dmas --aie-assign-lock-ids --aie-assign-buffer-addresses %s -o /dev/null
// RUN: sed 's/ fills / drains /g' %s | aie-opt --split-input-file --aie-objectfifo-allocate | FileCheck %s
// RUN: sed 's/ fills / drains /g' %s | aie-opt --split-input-file --aie-objectfifo-allocate --aie-objectfifo-lower-dmas --aie-assign-lock-ids --aie-assign-buffer-addresses -o /dev/null
// RUN: aie-opt --split-input-file --aie-objectfifo-allocate %s -o %t
// RUN: aie-opt --split-input-file --aie-objectfifo-allocate %t -o %t2
// RUN: diff %t %t2

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// There are 128 KiB free at home and 192 KiB in the neighbor. Size-first
// puts one 96-KiB object on each, then one 64-KiB object in the neighbor.
// Neither tile can hold the final object. This fails before channel repair.
// Demand-first puts both 64-KiB objects at home and both 96-KiB objects in
// the neighbor, fitting exactly. The same planner must accept this fallback.
//
// Remote users of the low-demand pool must not inflate its home demand.
// An unrelated home exercises the k-way merge without reordering its slots
// into the other tile's demand list.
module @packing_failure {
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    %neighbor = aie.tile(1, 1)
    %other = aie.tile(2, 1)
    %reserved0 = aie.buffer(%home) {sym_name = "reserved0"} : memref<393216xi8>
    %reserved1 = aie.buffer(%neighbor) {sym_name = "reserved1"} : memref<327680xi8>
    aie.objectfifo.pool @low(%home) {depth = 2 : i32} : memref<98304xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 98304 : i32}
    }
    aie.objectfifo.dma_endpoint @low_dma(%home) fills @low
    aie.objectfifo.dma_endpoint @remote0(%neighbor) fills @low
    aie.objectfifo.dma_endpoint @remote1(%neighbor) fills @low
    aie.objectfifo.dma_endpoint @remote2(%neighbor) fills @low

    aie.objectfifo.pool @high(%home) {depth = 2 : i32} : memref<65536xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 65536 : i32}
    }
    aie.objectfifo.dma_endpoint @high0(%home) fills @high
    aie.objectfifo.dma_endpoint @high1(%home) fills @high

    aie.objectfifo.pool @unrelated(%other) {depth = 1 : i32} : memref<131072xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 131072 : i32}
    }
  }
}
// CHECK-LABEL: module @packing_failure
// CHECK-DAG: %[[HOME:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[NEIGHBOR:.*]] = aie.tile(1, 1)
// CHECK-DAG: %[[OTHER:.*]] = aie.tile(2, 1)
// CHECK-DAG: aie.buffer(%[[HOME]]) {sym_name = "high_buff_0"}
// CHECK-DAG: aie.buffer(%[[HOME]]) {sym_name = "high_buff_1"}
// CHECK-DAG: aie.buffer(%[[NEIGHBOR]]) {sym_name = "low_buff_0"}
// CHECK-DAG: aie.buffer(%[[NEIGHBOR]]) {sym_name = "low_buff_1"}
// CHECK-DAG: aie.buffer(%[[OTHER]]) {sym_name = "unrelated_buff_0"}
// CHECK: @low_dma(%[[HOME]]) {{fills|drains}} @low {channelIndex = 0 : i32}
// CHECK: @high0(%[[HOME]]) {{fills|drains}} @high {channelIndex = 1 : i32}
// CHECK: @high1(%[[HOME]]) {{fills|drains}} @high {channelIndex = 2 : i32}

// -----

// Both orders would succeed, but give different placements. Preserve the
// successful size-first result rather than applying demand ordering eagerly.
module @preserve_size_first {
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    %neighbor = aie.tile(1, 1)
    %reserved = aie.buffer(%home) {sym_name = "reserved"} : memref<358400xi8>
    aie.objectfifo.pool @low(%home) {depth = 2 : i32} : memref<102400xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 102400 : i32}
    }
    aie.objectfifo.dma_endpoint @low_dma(%home) fills @low
    aie.objectfifo.pool @high(%home) {depth = 2 : i32} : memref<40960xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 40960 : i32}
    }
    aie.objectfifo.dma_endpoint @high0(%home) fills @high
    aie.objectfifo.dma_endpoint @high1(%home) fills @high
  }
}
// CHECK-LABEL: module @preserve_size_first
// CHECK-DAG: %[[HOME:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[NEIGHBOR:.*]] = aie.tile(1, 1)
// CHECK-DAG: aie.buffer(%[[HOME]]) {sym_name = "low_buff_0"}
// CHECK-DAG: aie.buffer(%[[HOME]]) {sym_name = "high_buff_0"}
// CHECK-DAG: aie.buffer(%[[NEIGHBOR]]) {sym_name = "low_buff_1"}
// CHECK-DAG: aie.buffer(%[[NEIGHBOR]]) {sym_name = "high_buff_1"}
