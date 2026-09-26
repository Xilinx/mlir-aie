//===- test_sa_allocate.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Verify SA placer generates objectfifo.allocate when a core tile's intratile
// ObjectFifos overflow 64KB local memory, and does not generate allocate
// when memory fits.

// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=42' %s | FileCheck %s

// Core A has 50KB of static weights + 18KB intratile fifos (3 x 2048B x 3 depth)
// = 68KB > 64KB. SA should generate allocate ops redirecting intratile fifos
// to a neighbor.
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

    // Intratile fifos (3 x 6KB = 18KB, total 68KB > 64KB)
    aie.objectfifo @intra0(%coreA, {%coreA}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<2048xi8>>
    aie.objectfifo @intra1(%coreA, {%coreA}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<2048xi8>>
    aie.objectfifo @intra2(%coreA, {%coreA}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<2048xi8>>

    aie.core(%coreA) { aie.end }
    aie.core(%coreB) { aie.end }
  }
}

// -----

// Intratile fifos within capacity: no allocate should be generated
// CHECK-LABEL: @no_allocate_within_capacity
// CHECK-NOT: aie.objectfifo.allocate
// CHECK-NOT: aie.logical_tile
module @no_allocate_within_capacity {
  aie.device(npu2) {
    %coreA = aie.logical_tile<CoreTile>(?, ?)
    %coreB = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @data(%coreA, {%coreB}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    // Small intratile fifos: 2 x (512B x 3 depth) = 3KB, well within 64KB
    aie.objectfifo @intra0(%coreA, {%coreA}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<512xi8>>
    aie.objectfifo @intra1(%coreA, {%coreA}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<512xi8>>

    aie.core(%coreA) { aie.end }
    aie.core(%coreB) { aie.end }
  }
}

// -----

// The same overflowing intratile fifos, already allocated to coreB by the
// user: SA charges them to coreB and adds no allocate of its own.
// CHECK-LABEL: @user_allocate_kept
// CHECK: %[[B:.*]] = aie.tile(2, 4)
// CHECK-NOT: aie.objectfifo.allocate
// CHECK: aie.objectfifo @intra0
// CHECK-NEXT: aie.objectfifo.allocate @intra0(%[[B]])
// CHECK-NEXT: aie.objectfifo @intra1
// CHECK-NEXT: aie.objectfifo.allocate @intra1(%[[B]])
// CHECK-NEXT: aie.objectfifo @intra2
// CHECK-NEXT: aie.objectfifo.allocate @intra2(%[[B]])
// CHECK-NOT: aie.objectfifo.allocate
module @user_allocate_kept {
  aie.device(npu2) {
    %coreA = aie.logical_tile<CoreTile>(2, 3)
    %coreB = aie.logical_tile<CoreTile>(2, 4)
    %coreC = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @data(%coreA, {%coreC}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    %wts = aie.buffer(%coreA) {sym_name = "weights"} : memref<12800xi32>

    aie.objectfifo @intra0(%coreA, {%coreA}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<2048xi8>>
    aie.objectfifo.allocate @intra0(%coreB)
    aie.objectfifo @intra1(%coreA, {%coreA}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<2048xi8>>
    aie.objectfifo.allocate @intra1(%coreB)
    aie.objectfifo @intra2(%coreA, {%coreA}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<2048xi8>>
    aie.objectfifo.allocate @intra2(%coreB)

    aie.core(%coreA) { aie.end }
    aie.core(%coreB) { aie.end }
    aie.core(%coreC) { aie.end }
  }
}
