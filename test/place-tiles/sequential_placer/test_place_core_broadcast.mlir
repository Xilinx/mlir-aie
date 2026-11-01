//===- test_place_core_broadcast.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles %s | FileCheck %s

// A core's fifo to several cores moves by DMA wherever they sit: one MM2S on
// the producer, routed to every consumer, and an S2MM on each. A broadcast to
// eight cores costs its producer one channel, not one per consumer, so it
// places; no neighbor saves a channel, so none is demanded.

// CHECK-LABEL: module @broadcast_to_eight
// CHECK: aie.objectfifo @row0(%{{.*}}, {%{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}}, 1 : i32)
// CHECK-NOT: aie.logical_tile
module @broadcast_to_eight {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %producer = aie.logical_tile<CoreTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    %c4 = aie.logical_tile<CoreTile>(?, ?)
    %c5 = aie.logical_tile<CoreTile>(?, ?)
    %c6 = aie.logical_tile<CoreTile>(?, ?)
    %c7 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @seed (%shim, {%producer}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @row0 (%producer, {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// Three broadcasts from one core are three MM2S, past a core's two.

module @three_broadcasts {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    // expected-error@+2 {{requires 1 input/3 output DMA channels, but only 2 input/2 output available}}
    // expected-note@+1 {{placer selected this tile}}
    %producer = aie.logical_tile<CoreTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @seed (%shim, {%producer}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @row0 (%producer, {%c0, %c1}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @row1 (%producer, {%c0, %c1}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @row2 (%producer, {%c0, %c1}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}
