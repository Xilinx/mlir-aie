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

// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=42' %s | FileCheck %s

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

// -----

// Intratile ObjectFifos on core0 overflow memory (3 x depth3 x 8KB = 72KB > 64KB)
// with inter-core and L3 connections
//
// CHECK-LABEL: @intratile_overflow_3x8kb
// CHECK: aie.objectfifo.allocate
// CHECK-NOT: aie.logical_tile
module @intratile_overflow_3x8kb {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem = aie.logical_tile<MemTile>(?, ?)
    %core0 = aie.logical_tile<CoreTile>(?, ?)
    %core1 = aie.logical_tile<CoreTile>(?, ?)

    // Input/output through mem tile
    aie.objectfifo @input(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @to_core(%mem, {%core0}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [@input] -> [@to_core]([] [0])

    // Inter-core connection
    aie.objectfifo @core_to_core(%core0, {%core1}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    // Intratile ObjectFifos on core0 that overflow memory (3 x depth3 x 8KB = 72KB > 64KB)
    aie.objectfifo @intra0(%core0, {%core0}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<8192xi8>>
    aie.objectfifo @intra1(%core0, {%core0}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<8192xi8>>
    aie.objectfifo @intra2(%core0, {%core0}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<8192xi8>>

    // Output
    aie.objectfifo @output(%core1, {%shim}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    %c0 = aie.core(%core0) {
      aie.end
    }
    %c1 = aie.core(%core1) {
      aie.end
    }
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

    // Small intratile fifos: 3 x 2KB = 6KB, well within 64KB
    aie.objectfifo @intra0(%coreA, {%coreA}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<512xi8>>
    aie.objectfifo @intra1(%coreA, {%coreA}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<512xi8>>

    aie.core(%coreA) { aie.end }
    aie.core(%coreB) { aie.end }
  }
}
