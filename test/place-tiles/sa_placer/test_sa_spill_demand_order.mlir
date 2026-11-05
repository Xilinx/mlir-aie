//===- test_sa_spill_demand_order.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// When its size-first order cannot place a placement's mem tile buffers,
// aie-objectfifo-allocate tries its demand-first order (demandFirstOrder), and
// so does the placer's estimate. (0, 1) has 131072 bytes free and (1, 1)
// 196608: only @high's two 65536-byte objects at home and @low's two
// 98304-byte objects next door fit. Size-first places @low first, and the
// spill choices of @fill's nine objects on (3, 1), which come between, use up
// the search's retries before it revisits @low. Demand-first puts @high, with
// two ends filling it, first at home.

// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// CHECK-NOT: warning: SA placer's resource estimate is over capacity
// CHECK: %[[HOME:.*]] = aie.tile(0, 1)
// CHECK-DAG: aie.buffer(%[[HOME]]) {sym_name = "high__buf0"}
// CHECK-DAG: aie.buffer(%[[HOME]]) {sym_name = "high__buf1"}
module {
  aie.device(npu2) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %m0 = aie.logical_tile<MemTile>(0, 1)
    %m1 = aie.logical_tile<MemTile>(1, 1)
    %m3 = aie.logical_tile<MemTile>(3, 1)
    %r0 = aie.buffer(%m0) : memref<393216xi8>
    %r1 = aie.buffer(%m1) : memref<327680xi8>
    aie.objectfifo @low_in(%s0, {%m0}, 2 : i32) : !aie.objectfifo<memref<98304xi8>>
    aie.objectfifo @low(%m0, {%s0}, 2 : i32) : !aie.objectfifo<memref<98304xi8>>
    aie.objectfifo.link [@low_in] -> [@low] ([] [])
    aie.objectfifo @high_a(%s1, {%m0}, 2 : i32) : !aie.objectfifo<memref<32768xi8>>
    aie.objectfifo @high_b(%s1, {%m0}, 2 : i32) : !aie.objectfifo<memref<32768xi8>>
    aie.objectfifo @high(%m0, {%s1}, 2 : i32) : !aie.objectfifo<memref<65536xi8>>
    aie.objectfifo.link [@high_a, @high_b] -> [@high] ([0, 32768] [])
    aie.objectfifo @fill_in(%s2, {%m3}, 9 : i32) : !aie.objectfifo<memref<80000xi8>>
    aie.objectfifo @fill(%m3, {%s2}, 9 : i32) : !aie.objectfifo<memref<80000xi8>>
    aie.objectfifo.link [@fill_in] -> [@fill] ([] [])
  }
}
