//===- acquire_rejected.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core can hold at most every object of a pool at once, and each of those
// objects has to be a buffer it can reach.

// RUN: aie-opt --aie-objectfifo-lower-cores --verify-diagnostics -split-input-file %s

module {
  aie.device(xcve2302) {
    %t = aie.tile(1, 2)
    aie.buffer @b0(%t) : memref<16xi32>
    aie.buffer @b1(%t) : memref<16xi32>
    aie.lock @free(%t) {init = 2 : i32}
    aie.lock @full(%t) {init = 0 : i32}
    aie.objectfifo.pool @pool(%t) {depth = 2 : i32, buffers = [@b0, @b1]} : memref<16xi32> {
      aie.objectfifo.segment @s0 {consumeLock = @full, offset = 0 : i32, produceLock = @free, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @writer(%t) fills @pool
    %core = aie.core(%t) {
      // expected-error@+1 {{acquires 3 objects from a pool of 2}}
      %e0, %e1, %e2 = aie.objectfifo.acquire @writer (3) : memref<16xi32>, memref<16xi32>, memref<16xi32>
      %c0 = arith.constant 0 : index
      %v = arith.constant 1 : i32
      memref.store %v, %e2[%c0] : memref<16xi32>
      aie.objectfifo.release @writer (3)
      aie.end
    }
  }
}

// -----

module {
  aie.device(xcve2302) {
    %t = aie.tile(1, 2)
    aie.buffer @b0(%t) : memref<16xi32>
    aie.lock @free(%t) {init = 2 : i32}
    aie.lock @full(%t) {init = 0 : i32}
    // expected-error@+1 {{expects every one of its 'depth' buffers to resolve}}
    aie.objectfifo.pool @pool(%t) {depth = 2 : i32, buffers = [@b0, @missing]} : memref<16xi32> {
      aie.objectfifo.segment @s0 {consumeLock = @full, offset = 0 : i32, produceLock = @free, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @writer(%t) fills @pool
    %core = aie.core(%t) {
      %e0 = aie.objectfifo.acquire @writer (1) : memref<16xi32>
      %c0 = arith.constant 0 : index
      %v = arith.constant 1 : i32
      memref.store %v, %e0[%c0] : memref<16xi32>
      aie.objectfifo.release @writer (1)
      aie.end
    }
  }
}

// -----

// Even unused acquisitions require resolved buffers, diagnosed once per pool.
module {
  aie.device(xcve2302) {
    %t = aie.tile(1, 2)
    aie.buffer @b0(%t) : memref<16xi32>
    aie.lock @free(%t) {init = 2 : i32}
    aie.lock @full(%t) {init = 0 : i32}
    // expected-error@+1 {{expects every one of its 'depth' buffers to resolve}}
    aie.objectfifo.pool @pool(%t) {depth = 2 : i32, buffers = [@b0, @missing]} : memref<16xi32> {
      aie.objectfifo.segment @s0 {consumeLock = @full, offset = 0 : i32, produceLock = @free, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @writer(%t) fills @pool
    %core = aie.core(%t) {
      %e0 = aie.objectfifo.acquire @writer (1) : memref<16xi32>
      aie.objectfifo.release @writer (1)
      %e1 = aie.objectfifo.acquire @writer (1) : memref<16xi32>
      aie.objectfifo.release @writer (1)
      aie.end
    }
  }
}
