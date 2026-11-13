//===- test_sa_allocate_aie1_reach.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// On AIE1 the memory affinity rule alone lets a row-1 core reach the shim
// tile below it, but split keeps a fifo's objects off a shim. The SA moves the
// overflowing core's fifos only to a neighbour split accepts as their
// delegate: the core above.

// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-objectfifo-split %s | FileCheck %s

// CHECK-LABEL: @aie1_relocate
// CHECK-DAG: %[[UP:.*]] = aie.tile(0, 2)
// CHECK-DAG: aie.objectfifo.pool @intra0__pool(%[[UP]])
// CHECK-DAG: aie.objectfifo.pool @intra1__pool(%[[UP]])
// CHECK-DAG: aie.objectfifo.pool @intra2__pool(%[[UP]])
module @aie1_relocate {
  aie.device(xcvc1902) {
    %coreA = aie.logical_tile<CoreTile>(0, 1)
    %wts = aie.buffer(%coreA) {sym_name = "weights"} : memref<6144xi32>
    aie.objectfifo @intra0(%coreA, {%coreA}, 2 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<2048xi8>>
    aie.objectfifo @intra1(%coreA, {%coreA}, 2 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<2048xi8>>
    aie.objectfifo @intra2(%coreA, {%coreA}, 2 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<2048xi8>>
    aie.core(%coreA) { aie.end }
  }
}
