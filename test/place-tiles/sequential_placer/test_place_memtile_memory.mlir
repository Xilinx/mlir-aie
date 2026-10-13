//===- test_place_memtile_memory.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles --aie-objectfifo-split --aie-objectfifo-allocate %s | FileCheck %s --check-prefix=ALLOC
// RUN: aie-opt --aie-place-tiles --aie-objectFifo-stateful-transform="skip-verify=true" --aie-assign-lock-ids --aie-assign-buffer-addresses %s -o /dev/null
// RUN: sed -e 's/196608xi8/32768xi8/' -e 's/147456xi8/65536xi8/' %s > %t
// RUN: aie-opt --aie-place-tiles --aie-objectfifo-split --aie-objectfifo-allocate %t | FileCheck %s --check-prefix=BOUNDARY
// RUN: aie-opt --aie-place-tiles --aie-objectFifo-stateful-transform="skip-verify=true" --aie-assign-lock-ids --aie-assign-buffer-addresses %t -o /dev/null

// A reduction of #3720. Six input channels fit on one MemTile, but merging
// these pools there makes both a forward and the four-way join spill. Their
// five inputs cannot all use the four adjacent-memory channels.
// Keep automatic merging and spilling. Allocation must reserve the small
// four-way join locally rather than consume all local memory with forwards.
// b0 uses nine-byte BFP elements and a per-endpoint depth: neither the shim's
// depth of one nor the forward's repeat_count determines its MemTile storage.
// This resource-only test omits core programs, hence skip-verify above.
// At the boundary (32 KiB reserved + 288 + 128 + 64 KiB), all three pools
// must still merge: all six input channels are usable with local buffers.

// CHECK: %[[M0:.*]] = aie.tile(0, 1)
// CHECK: aie.objectfifo @b0_in(%{{.*}}, {%[[M0]]}
// CHECK: aie.objectfifo @b1_in(%{{.*}}, {%[[M0]]}
// CHECK: aie.objectfifo @join0(%{{.*}}, {%[[M0]]}
// ALLOC-DAG: %[[HOME:.*]] = aie.tile(0, 1)
// ALLOC-DAG: %[[NEIGHBOR:.*]] = aie.tile(1, 1)
// ALLOC-DAG: aie.buffer(%[[HOME]]) {sym_name = "join_out_buff_0"}
// ALLOC-DAG: aie.buffer(%[[HOME]]) {sym_name = "join_out_buff_1"}
// ALLOC-DAG: aie.buffer(%[[HOME]]) {sym_name = "b0_in_cons_buff_0"}
// ALLOC-DAG: aie.buffer(%[[NEIGHBOR]]) {sym_name = "b0_in_cons_buff_1"}
// ALLOC-DAG: aie.buffer(%[[NEIGHBOR]]) {sym_name = "b1_in_cons_buff_0"}
// ALLOC-DAG: aie.buffer(%[[NEIGHBOR]]) {sym_name = "b1_in_cons_buff_1"}
// BOUNDARY: %[[M:.*]] = aie.tile(0, 1)
// BOUNDARY-DAG: aie.buffer(%[[M]]) {sym_name = "b0_in_cons_buff_0"}
// BOUNDARY-DAG: aie.buffer(%[[M]]) {sym_name = "b0_in_cons_buff_1"}
// BOUNDARY-DAG: aie.buffer(%[[M]]) {sym_name = "b1_in_cons_buff_0"}
// BOUNDARY-DAG: aie.buffer(%[[M]]) {sym_name = "b1_in_cons_buff_1"}
// BOUNDARY-DAG: aie.buffer(%[[M]]) {sym_name = "join_out_buff_0"}
// BOUNDARY-DAG: aie.buffer(%[[M]]) {sym_name = "join_out_buff_1"}
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %home = aie.tile(0, 1)
    %core0 = aie.tile(0, 2)
    %core1 = aie.tile(0, 3)
    %core2 = aie.tile(0, 4)
    %core3 = aie.tile(0, 5)
    %b0 = aie.logical_tile<MemTile>(?, ?)
    %b1 = aie.logical_tile<MemTile>(?, ?)
    %join = aie.logical_tile<MemTile>(?, ?)
    %reserved = aie.buffer(%home) : memref<196608xi8>

    aie.objectfifo @b0_in(%shim, {%b0}, [1, 2]) : !aie.objectfifo<memref<16384x!aiex.bfp<"v8bfp16ebs8">>>
    aie.objectfifo @b0_out(%b0, {%core0}, 2 : i32) {repeat_count = 4 : i32} : !aie.objectfifo<memref<64x!aiex.bfp<"v8bfp16ebs8">>>
    aie.objectfifo.link [@b0_in] -> [@b0_out]([] [])
    aie.objectfifo @b1_in(%shim, {%b1}, 2 : i32) : !aie.objectfifo<memref<147456xi8>>
    aie.objectfifo @b1_out(%b1, {%core1}, 2 : i32) : !aie.objectfifo<memref<64xi8>>
    aie.objectfifo.link [@b1_in] -> [@b1_out]([] [])

    aie.objectfifo @join0(%core0, {%join}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>
    aie.objectfifo @join1(%core1, {%join}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>
    aie.objectfifo @join2(%core2, {%join}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>
    aie.objectfifo @join3(%core3, {%join}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>
    aie.objectfifo @join_out(%join, {%shim}, 2 : i32) : !aie.objectfifo<memref<32768xi8>>
    aie.objectfifo.link [@join0, @join1, @join2, @join3] -> [@join_out]([0, 8192, 16384, 24576] [])
  }
}
