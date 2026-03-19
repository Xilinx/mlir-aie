//===- test_place_objectfifo.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// Multiple logical MemTiles merge to same physical tile
// CHECK-LABEL: @multi_fifo_same_tile
module @multi_fifo_same_tile {
  aie.device(npu1) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
    %core = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)

    // CHECK: aie.objectfifo @of1(%[[MEM]], {%[[CORE]]}, 2 : i32)
    aie.objectfifo @of1 (%mem1, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // CHECK: aie.objectfifo @of2(%[[MEM]], {%[[CORE]]}, 2 : i32)
    aie.objectfifo @of2 (%mem2, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Worker with multiple input ObjectFifos
// CHECK-LABEL: @worker_multiple_inputs
module @worker_multiple_inputs {
  aie.device(npu1) {
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
    %core = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @in1 (%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2 (%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// ObjectFifo with multiple consumers (broadcast)
// CHECK-LABEL: @multi_consumer
module @multi_consumer {
  aie.device(npu1) {
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    // CHECK-DAG: %[[CORE1:.*]] = aie.tile(0, 2)
    %core1 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[CORE2:.*]] = aie.tile(0, 3)
    %core2 = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @broadcast (%shim, {%core1, %core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core1) { aie.end }
    aie.core(%core2) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Linked fifos grouped to same column
// CHECK-LABEL: @objectfifo_link
module @objectfifo_link {
  aie.device(npu1) {
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    // CHECK-DAG: %[[CORE0:.*]] = aie.tile(0, 2)
    // CHECK-DAG: %[[CORE1:.*]] = aie.tile(0, 3)

    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem = aie.logical_tile<MemTile>(?, ?)
    %core0 = aie.logical_tile<CoreTile>(?, ?)
    %core1 = aie.logical_tile<CoreTile>(?, ?)

    // shim -> mem (splits to 2 outputs)
    aie.objectfifo @inA (%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @memA0 (%mem, {%core0}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @memA1 (%mem, {%core1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>

    // Link: 1 input, 2 outputs
    aie.objectfifo.link [@inA] -> [@memA0, @memA1] ([] [0, 1024])
    // CHECK-NOT: aie.logical_tile
  }
}
