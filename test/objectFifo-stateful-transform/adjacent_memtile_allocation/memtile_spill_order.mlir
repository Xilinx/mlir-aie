//===- memtile_spill_order.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// Three MemTile ObjectFifos where IR-order allocation fails but
// size-sorted (large-first) allocation succeeds.
//
// @block at col 3 (depth 1, 524000B): fills col 3 so @large can't spill right.
// @small at col 1 (depth 2, 260000B): fits locally (520000 < 524288).
// @large at col 2 (depth 2, 400000B): must spill one buffer.
//
// Without large-first sort (IR order: block, small, large):
//   @small fills col 1 (4288B remaining), @large can't spill left or right → FAIL.
//
// With large-first sort (block 524000, large 400000, small 260000):
//   @large spills to col 1 (empty), @small spills to col 0 → SUCCESS.

// large_cons_buff_0 on col 2 (home), large_cons_buff_1 spills to col 1
// CHECK-DAG: aie.buffer(%mem_tile_2_1) {sym_name = "large_cons_buff_0"}
// CHECK-DAG: aie.buffer(%mem_tile_1_1) {sym_name = "large_cons_buff_1"}
// small buffers both spill to col 0
// CHECK-DAG: aie.buffer(%mem_tile_0_1) {sym_name = "small_cons_buff_0"}
// CHECK-DAG: aie.buffer(%mem_tile_0_1) {sym_name = "small_cons_buff_1"}
// block buffer on col 3 (home)
// CHECK-DAG: aie.buffer(%mem_tile_3_1) {sym_name = "block_cons_buff_0"}

module {
  aie.device(npu2) {
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %shim2 = aie.tile(2, 0)
    %shim3 = aie.tile(3, 0)
    %mem1 = aie.tile(1, 1)
    %mem2 = aie.tile(2, 1)
    %mem3 = aie.tile(3, 1)
    %core1 = aie.tile(1, 2)
    %core2 = aie.tile(2, 2)
    %core3 = aie.tile(3, 2)

    // IR order: block first, then small, then large.
    aie.objectfifo @block(%shim3, {%mem3}, 1 : i32) : !aie.objectfifo<memref<524000xi8>>

    aie.objectfifo @small(%shim1, {%mem1}, 2 : i32) : !aie.objectfifo<memref<260000xi8>>

    aie.objectfifo @large(%shim2, {%mem2}, 2 : i32) : !aie.objectfifo<memref<400000xi8>>

    aie.objectfifo @small_out(%mem1, {%core1}, 2 : i32) : !aie.objectfifo<memref<260000xi8>>
    aie.objectfifo.link [@small] -> [@small_out]([] [])

    aie.objectfifo @large_out(%mem2, {%core2}, 2 : i32) : !aie.objectfifo<memref<400000xi8>>
    aie.objectfifo.link [@large] -> [@large_out]([] [])

    aie.objectfifo @block_out(%mem3, {%core3}, 1 : i32) : !aie.objectfifo<memref<524000xi8>>
    aie.objectfifo.link [@block] -> [@block_out]([] [])
  }
}
