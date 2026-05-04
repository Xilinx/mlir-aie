//===- test_sa_objectfifo.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verify SA placer preserves ObjectFifo structure and handles linking patterns.

// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=42' %s | FileCheck %s

// ObjectFifo between core and shim preserved
// CHECK-LABEL: @simple_fifo
module @simple_fifo {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %core = aie.logical_tile<CoreTile>(?, ?)

    // CHECK: aie.objectfifo @in(
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    // CHECK: aie.objectfifo @out(
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.core(%core) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// ObjectFifo link (distribute pattern: L3 -> L2 -> 2xL1) preserved
// CHECK-LABEL: @fifo_link_distribute
module @fifo_link_distribute {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem = aie.logical_tile<MemTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @inL3(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>>
    aie.objectfifo @inL2_0(%mem, {%c0}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @inL2_1(%mem, {%c1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    // CHECK: aie.objectfifo.link [@inL3] -> [@inL2_0, @inL2_1]([] [0, 1024])
    aie.objectfifo.link [@inL3] -> [@inL2_0, @inL2_1]([] [0, 1024])

    aie.core(%c0) { aie.end }
    aie.core(%c1) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// ObjectFifo link (join pattern: 2xL1 -> L2 -> L3) preserved
// CHECK-LABEL: @fifo_link_join
module @fifo_link_join {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem = aie.logical_tile<MemTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @outL1_0(%c0, {%mem}, 2 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @outL1_1(%c1, {%mem}, 2 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @outL3(%mem, {%shim}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    // CHECK: aie.objectfifo.link [@outL1_0, @outL1_1] -> [@outL3]([0, 512] [])
    aie.objectfifo.link [@outL1_0, @outL1_1] -> [@outL3]([0, 512] [])

    aie.core(%c0) { aie.end }
    aie.core(%c1) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Broadcast: one producer to multiple consumers
// CHECK-LABEL: @fifo_broadcast
module @fifo_broadcast {
  aie.device(npu2) {
    %mem = aie.logical_tile<MemTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)

    // CHECK: aie.objectfifo @bcast(
    aie.objectfifo @bcast(%mem, {%c0, %c1, %c2, %c3}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.core(%c0) { aie.end }
    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
    aie.core(%c3) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Core-to-core ObjectFifo (pipeline pattern)
// CHECK-LABEL: @core_pipeline
module @core_pipeline {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @in(%shim, {%c0}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    // CHECK: aie.objectfifo @pipe01(
    aie.objectfifo @pipe01(%c0, {%c1}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    // CHECK: aie.objectfifo @pipe12(
    aie.objectfifo @pipe12(%c1, {%c2}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @out(%c2, {%shim}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.core(%c0) { aie.end }
    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Multiple MemTiles merge when DMA capacity allows
// CHECK-LABEL: @memtile_merge
module @memtile_merge {
  aie.device(npu2) {
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @of1(%mem1, {%c0}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @of2(%mem2, {%c0}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.core(%c0) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}
