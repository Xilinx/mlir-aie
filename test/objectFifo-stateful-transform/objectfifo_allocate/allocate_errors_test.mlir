//===- allocate_errors_test.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" -split-input-file --verify-diagnostics %s

aie.device(npu1) {
   %tile12 = aie.tile(1, 2)
   %tile33 = aie.tile(3, 3)
   aie.objectfifo @of1 (%tile12, {%tile33}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
   // expected-error@+1 {{'aie.objectfifo.allocate' op objectfifo has no shared memory access to delegate tile's memory module}}
   aie.objectfifo.allocate @of1 (%tile12)
}

// -----

aie.device(npu1) {
   %tile12 = aie.tile(1, 2)
   %tile33 = aie.tile(3, 3)
   aie.objectfifo @of1 (%tile12, {%tile33}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
   // expected-error@+1 {{'aie.objectfifo.allocate' op objectfifo has no shared memory access to delegate tile's memory module}}
   aie.objectfifo.allocate @of1 (%tile33)
}

// -----

aie.device(npu1) {
   %tile12 = aie.tile(1, 2)
   %tile32 = aie.tile(3, 2)
   %tile33 = aie.tile(3, 3)
   aie.objectfifo @of1 (%tile12, {%tile33}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
   // expected-error@+1 {{'aie.objectfifo.allocate' op objectfifo has no shared memory access to delegate tile's memory module}}
   aie.objectfifo.allocate @of1 (%tile32)
}

// -----

aie.device(npu1) {
   %tile12 = aie.tile(1, 2)
   %tile13 = aie.tile(1, 3)
   aie.objectfifo @of1 (%tile12, {%tile13}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
   aie.objectfifo.allocate @of1 (%tile12)
   // expected-error@+1 {{'aie.objectfifo.allocate' op objectfifo @of1 already has an allocate operation}}
   aie.objectfifo.allocate @of1 (%tile13)
}

// -----

// A linked fifo's objects are the link's pool on the link tile (0,4): the
// delegate (0,3) is reached by both ends, but the producer (0,2) cannot reach
// the pool. in0 owns the pool, so it cannot move by DMA instead.

aie.device(npu2_1col) {
   %tile00 = aie.tile(0, 0)
   %tile02 = aie.tile(0, 2)
   %tile03 = aie.tile(0, 3)
   %tile04 = aie.tile(0, 4)
   aie.objectfifo @in0 (%tile02, {%tile04}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
   aie.objectfifo @out (%tile04, {%tile00}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
   aie.objectfifo.link [@in0] -> [@out] ([] [])
   // expected-error@+1 {{'aie.objectfifo.allocate' op objectfifo has no shared memory access to the memory module of its link's tile, which holds the link's pool}}
   aie.objectfifo.allocate @in0 (%tile03)
}
