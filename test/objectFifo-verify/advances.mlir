//===- advances.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core whose loop repeats forever and acquires from a pool each time, but
// never releases any of it, keeps the same objects for good: the end opposite
// stops once the pool is full.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectfifo-split --aie-objectfifo-verify %s

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    // expected-note@+1 {{the other end is here}}
    aie.objectfifo @in(%shim, {%tile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        // expected-error@+1 {{takes from @in_cons each time its loop repeats, but the loop releases none of them, so the core keeps the same objects forever and the end filling the pool stops once it is full}}
        %o = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// A producer that never releases sends nothing.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    // expected-note@+1 {{the other end is here}}
    aie.objectfifo @out(%tile, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        // expected-error@+1 {{takes from @out_prod each time its loop repeats, but the loop releases none of them, so nothing the core writes there is ever sent}}
        %o = aie.objectfifo.acquire @out (Produce, 1) : memref<16xi32>
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// A hand-written forever loop is an scf.for with a bound no design counts to.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    // expected-note@+1 {{the other end is here}}
    aie.objectfifo @in(%shim, {%tile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cmax = arith.constant 4294967295 : index
      scf.for %i = %c0 to %cmax step %c1 {
        // expected-error@+1 {{takes from @in_cons each time its loop repeats, but the loop releases none of them}}
        %o = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
      }
      aie.end
    }
  }
}

// -----

// Releasing what each repetition takes is the ordinary design, and a sliding
// window that releases one of the two it holds advances too.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%tile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @win(%shim, {%tile}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %o = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
        aie.objectfifo.release @in (Consume, 1)
        %w:2 = aie.objectfifo.acquire @win (Consume, 2) : memref<16xi32>, memref<16xi32>
        aie.objectfifo.release @win (Consume, 1)
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// Acquired once before the loop and held for the whole configuration: the
// loop never takes another, which is what the design means.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @w(%shim, {%tile}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in(%shim, {%tile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile) {
      %w = aie.objectfifo.acquire @w (Consume, 1) : memref<16xi32>
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %o = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
        aie.objectfifo.release @in (Consume, 1)
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// With an iteration count the filling end stops by itself, which the transfer
// budget check covers; this one leaves it alone.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%tile}, 2 : i32) {iter_count = 1 : i32} : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %o = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// Without synchronization the pool has no locks, so the DMA endpoint keeps
// moving objects whatever the core holds.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @out(%tile, {%shim}, 2 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %o = aie.objectfifo.acquire @out (Produce, 1) : memref<16xi32>
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// An acquire under a branch that is never taken never runs.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%tile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %false = arith.constant false
        scf.if %false {
          %o = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
        }
        scf.yield
      }
      aie.end
    }
  }
}
