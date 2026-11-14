//===- test_place_device_order.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie-place-tiles{ctrl-pkt-overlay=true})' %s | FileCheck %s

// Two devices share column 0's control overlay. @devA sends on shim MM2S 0
// and @devB on MM2S 1; the device placed first takes column 0's shim and the
// other moves to column 1. The devices go in their structural order, so
// @devA takes column 0 in whichever order the module lists them.
// CHECK-LABEL: module @a_first
// CHECK: @devA
// CHECK: aie.tile(0, 0)
// CHECK: @devB
// CHECK: aie.tile(1, 0)
// CHECK-LABEL: module @b_first
// CHECK: @devB
// CHECK: aie.tile(1, 0)
// CHECK: @devA
// CHECK: aie.tile(0, 0)

module @a_first {
  aie.device(npu2) @devA {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %x = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
  }
  aie.device(npu2) @devB {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @b(%s, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 1>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %x = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}

// -----

module @b_first {
  aie.device(npu2) @devB {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @b(%s, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 1>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %x = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
  aie.device(npu2) @devA {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %x = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
  }
}
