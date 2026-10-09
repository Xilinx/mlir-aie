//===- shared_mem_transport.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectfifo-split --verify-diagnostics %s | FileCheck %s

// A shared_mem transport between neighbours gets one pool and no route.
// CHECK-LABEL: module @granted
// CHECK:       aie.objectfifo.pool @of_pool
// CHECK-NOT:   aie.route
module @granted {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    aie.objectfifo @of (%tile12, {%tile13}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

module @apart {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)
    // expected-error@+1 {{asks for a shared_mem transport, but its ends share no memory module}}
    aie.objectfifo @of (%tile12, {%tile33}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

module @repeats {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{asks for a shared_mem transport, but it repeats objects (`repeat_count`)}}
    aie.objectfifo @of (%tile12, {%tile13}, 2 : i32) {transport = #aie.transport<shared_mem>, repeat_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

module @two_consumers {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile22 = aie.tile(2, 2)
    // expected-error@+1 {{asks for a shared_mem transport, but it has more than one consumer}}
    aie.objectfifo @of (%tile12, {%tile13, %tile22}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

module @reshaped_out {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{asks for a shared_mem transport, but its producer reshapes the data (`dimensionsToStream`)}}
    aie.objectfifo @of (%tile12 dimensionsToStream [<size = 4, stride = 4>, <size = 4, stride = 1>], {%tile13}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

module @reshaped_in {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{asks for a shared_mem transport, but a consumer reshapes the data (`dimensionsFromStream`)}}
    aie.objectfifo @of (%tile12, {%tile13 dimensionsFromStream [<size = 4, stride = 4>, <size = 4, stride = 1>]}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

module @consumer_type {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{asks for a shared_mem transport, but its consumer takes a different element type}}
    aie.objectfifo @of (%tile12, {%tile13}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<16xi32>> -> !aie.objectfifo<memref<8xi32>>
 }
}

// -----

module @linked {
 aie.device(npu2) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile14 = aie.tile(1, 4)
    aie.objectfifo @in (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{asks for a shared_mem transport, but it is linked through a tile's DMAs}}
    aie.objectfifo @out (%tile13, {%tile14}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
 }
}
