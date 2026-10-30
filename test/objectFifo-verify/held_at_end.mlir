//===- held_at_end.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core that reaches its end still holding objects never gives their locks
// back, and locks keep their counts until a PDI load, so the next run of the
// configuration finds those objects taken. Only cores certain to end, with
// every access under static loops, are checked.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectfifo-split --aie-objectfifo-verify %s

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%tile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{'aie.core' op ends with 1 acquired from @in_cons never released, so their locks stay taken into the next run of this configuration}}
    %core = aie.core(%tile) {
      // expected-note@+1 {{acquired here}}
      %o = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
      aie.end
    }
  }
}

// -----

// Acquired before a finite loop and released after it: nothing is left held.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @w(%shim, {%tile}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in(%shim, {%tile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %w = aie.objectfifo.acquire @w (Consume, 1) : memref<16xi32>
      scf.for %i = %c0 to %c4 step %c1 {
        %o = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
        aie.objectfifo.release @in (Consume, 1)
      }
      aie.objectfifo.release @w (Consume, 1)
      aie.end
    }
  }
}

// -----

// A sliding window holds two and releases one per step, so one is still held
// after the loop.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%tile}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{'aie.core' op ends with 1 acquired from @in_cons never released}}
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        // expected-note@+1 {{acquired here}}
        %o:2 = aie.objectfifo.acquire @in (Consume, 2) : memref<16xi32>, memref<16xi32>
        aie.objectfifo.release @in (Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// Releasing the window's tail after the loop leaves nothing held.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%tile}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %o:2 = aie.objectfifo.acquire @in (Consume, 2) : memref<16xi32>, memref<16xi32>
        aie.objectfifo.release @in (Consume, 1)
      }
      aie.objectfifo.release @in (Consume, 1)
      aie.end
    }
  }
}

// -----

// A release under a condition may or may not run, so the count is unknown.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%tile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %flag = aie.buffer(%tile) : memref<1xi1>
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %o = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
      %cond = memref.load %flag[%c0] : memref<1xi1>
      scf.if %cond {
        aie.objectfifo.release @in (Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// A core whose loop runs forever never reaches its end, so what it holds
// across the loop stays held for the whole configuration.

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

// Without synchronization the pool has no locks, so nothing stays taken.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%tile}, 2 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile) {
      %o = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
      aie.end
    }
  }
}

// -----

// A pool that starts full of constants and is never refilled has no locks
// either.

module {
  aie.device(npu1_1col) {
    %tile = aie.tile(0, 2)
    aie.objectfifo @c(%tile, {%tile}, 1 : i32) : !aie.objectfifo<memref<4xi32>> = [dense<[1, 2, 3, 4]> : memref<4xi32>]
    %core = aie.core(%tile) {
      %o = aie.objectfifo.acquire @c (Consume, 1) : memref<4xi32>
      aie.end
    }
  }
}

// -----

// A fifo the runtime sequence re-arms gets its locks back each launch, so the
// next run does not find the held object taken.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%tile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile) {
      %o = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aiex.dma_channel_reset_for(@in)
    }
  }
}
