//===- test_sa_allocate.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verify SA placer generates objectfifo.allocate when a core tile's intratile
// ObjectFifos overflow 64KB local memory and must spill to a neighbor's
// shared memory.

// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=42' %s | FileCheck %s

// Core A has 50KB of static weights + 24KB intratile fifos = 74KB > 64KB.
// SA should generate allocate ops redirecting intratile fifos to a neighbor.
//
// CHECK-LABEL: @intratile_spill
// CHECK: aie.objectfifo.allocate
// CHECK-NOT: aie.logical_tile
module @intratile_spill {
  aie.device(npu2) {
    %coreA = aie.logical_tile<CoreTile>(?, ?)
    %coreB = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @data(%coreA, {%coreB}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    // 50KB static weights
    %wts = aie.buffer(%coreA) {sym_name = "weights"} : memref<12800xi32>

    // Intratile fifos (3 x 8KB = 24KB, total 74KB > 64KB)
    aie.objectfifo @intra0(%coreA, {%coreA}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<2048xi8>>
    aie.objectfifo @intra1(%coreA, {%coreA}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<2048xi8>>
    aie.objectfifo @intra2(%coreA, {%coreA}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<2048xi8>>

    aie.core(%coreA) { aie.end }
    aie.core(%coreB) { aie.end }
  }
}
