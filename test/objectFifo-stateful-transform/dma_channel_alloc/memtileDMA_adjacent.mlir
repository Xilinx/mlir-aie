//===- memtileDMA_adjacent.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// Two forwards and a four-way join share column 0:
// six S2MM channels, but only four of them can access adjacent memory.
// The reservation and the two forwards fill local memory (128+256+128 KiB),
// so the join's 64 KiB spills to column 1. Its four inputs must get channels
// 0..3 even though the two local inputs occur first in the IR.
// Neither access-pattern dimensions nor repeat_count are needed to trigger
// the restricted-channel allocation that made the placement in #3720 fail.

// CHECK-DAG: %[[M0:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[M1:.*]] = aie.tile(1, 1)
// CHECK-DAG: %[[S0:.*]] = aie.tile(0, 0)
// CHECK-DAG: %[[S1:.*]] = aie.tile(1, 0)
// CHECK-DAG: %[[S2:.*]] = aie.tile(2, 0)
// CHECK-DAG: aie.buffer(%[[M1]]) {sym_name = "join_out_buff_0"}
// CHECK-DAG: aie.buffer(%[[M1]]) {sym_name = "join_out_buff_1"}
// CHECK-DAG: aie.flow(%[[S0]], DMA : 0, %[[M0]], DMA : 4)
// CHECK-DAG: aie.flow(%[[S0]], DMA : 1, %[[M0]], DMA : 5)
// CHECK-DAG: aie.flow(%[[S1]], DMA : 0, %[[M0]], DMA : 0)
// CHECK-DAG: aie.flow(%[[S1]], DMA : 1, %[[M0]], DMA : 1)
// CHECK-DAG: aie.flow(%[[S2]], DMA : 0, %[[M0]], DMA : 2)
// CHECK-DAG: aie.flow(%[[S2]], DMA : 1, %[[M0]], DMA : 3)
module {
  aie.device(npu2) {
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %shim2 = aie.tile(2, 0)
    %home = aie.tile(0, 1)
    %reserved = aie.buffer(%home) : memref<131072xi8>

    aie.objectfifo @a_in(%shim0, {%home}, 2 : i32) : !aie.objectfifo<memref<131072xi8>>
    aie.objectfifo @a_out(%home, {%shim0}, 2 : i32) : !aie.objectfifo<memref<131072xi8>>
    aie.objectfifo.link [@a_in] -> [@a_out]([] [])
    aie.objectfifo @b_in(%shim0, {%home}, 2 : i32) : !aie.objectfifo<memref<65536xi8>>
    aie.objectfifo @b_out(%home, {%shim1}, 2 : i32) : !aie.objectfifo<memref<65536xi8>>
    aie.objectfifo.link [@b_in] -> [@b_out]([] [])

    aie.objectfifo @join0(%shim1, {%home}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>
    aie.objectfifo @join1(%shim1, {%home}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>
    aie.objectfifo @join2(%shim2, {%home}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>
    aie.objectfifo @join3(%shim2, {%home}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>
    aie.objectfifo @join_out(%home, {%shim2}, 2 : i32) : !aie.objectfifo<memref<32768xi8>>
    aie.objectfifo.link [@join0, @join1, @join2, @join3] -> [@join_out]([0, 8192, 16384, 24576] [])
  }
}
