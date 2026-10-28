//===- test_place_overloaded_peer_room.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles %s | FileCheck %s --implicit-check-not='aie.logical_tile'

// Pinned producers fill column 0's four outgoing streams. The hinted owner
// needs another stream, so every candidate in column 0 is overloaded. It must
// still skip (0, 2), whose buffer-affinity neighbors are occupied, and use
// (0, 4), leaving (1, 4) for its buffer consumer.
// CHECK-DAG: %[[OWNER:.*]] = aie.tile(0, 4)
// CHECK-DAG: %[[CONSUMER:.*]] = aie.tile(1, 4)
// CHECK: %[[BUF:.*]] = aie.buffer(%[[OWNER]])
// CHECK: aie.core(%[[OWNER]])
// CHECK: aie.core(%[[CONSUMER]])
// CHECK: memref.load %[[BUF]]
module {
  aie.device(npu2) {
    %p0 = aie.logical_tile<CoreTile>(0, 3)
    %p1 = aie.logical_tile<CoreTile>(0, 5)
    %block = aie.logical_tile<CoreTile>(1, 2)
    %mem = aie.logical_tile<MemTile>(0, 1)
    aie.core(%p0) { aie.end }
    aie.core(%p1) { aie.end }
    aie.core(%block) { aie.end }
    aie.objectfifo @p0a(%p0, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @p0b(%p0, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @p1a(%p1, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @p1b(%p1, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %owner = aie.logical_tile<CoreTile>(0, ?)
    %buf = aie.buffer(%owner) : memref<16xi32>
    aie.objectfifo @out(%owner, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%owner) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%consumer) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
  }
}
