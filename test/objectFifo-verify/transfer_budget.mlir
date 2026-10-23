//===- transfer_budget.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A DMA endpoint with an iteration count moves `iterCount * depth` objects and
// stops. A core that must release more through the same pool waits forever
// for the rest, so the verifier rejects it. Only releases static loop bounds
// force are counted.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectfifo-split --aie-objectfifo-verify %s

// The consumer's loop runs twice as many times as the fifo delivers.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    // expected-note@+1 {{the DMA endpoint is here}}
    aie.objectfifo @in(%shim, {%tile}, 2 : i32) {iter_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %o = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
        // expected-error@+1 {{releases 8 objects through @in_cons over the run, but the pool's DMA endpoint stops after 4 (iterCount 2 x depth 2), so the core would wait forever for the rest}}
        aie.objectfifo.release @in (Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// Releasing exactly what the fifo delivers is the ordinary bounded design.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%tile}, 2 : i32) {iter_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %o = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
        aie.objectfifo.release @in (Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// Nested static loops multiply: 2 x 3 releases exceed the 4 delivered.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    // expected-note@+1 {{the DMA endpoint is here}}
    aie.objectfifo @in(%shim, {%tile}, 2 : i32) {iter_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      scf.for %i = %c0 to %c2 step %c1 {
        scf.for %j = %c0 to %c3 step %c1 {
          %o = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
          // expected-error@+1 {{releases 6 objects through @in_cons over the run, but the pool's DMA endpoint stops after 4}}
          aie.objectfifo.release @in (Consume, 1)
        }
      }
      aie.end
    }
  }
}

// -----

// A loop meant to run forever is not a bound: such a core stalls once the data
// stops, which is how the design ends.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%tile}, 2 : i32) {iter_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4294967295 = arith.constant 4294967295 : index
      scf.for %i = %c0 to %c4294967295 step %c1 {
        %o = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
        aie.objectfifo.release @in (Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// A release under a condition may never run, so it does not count.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%tile}, 2 : i32) {iter_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
    %flag = aie.buffer(%tile) : memref<1xi1>
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %cond = memref.load %flag[%c0] : memref<1xi1>
        scf.if %cond {
          %o = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
          aie.objectfifo.release @in (Consume, 1)
        }
      }
      aie.end
    }
  }
}

// -----

// The producer side is bounded the same way: its DMA drains
// `iter_count * depth` objects and stops, after which the pool never empties.

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    // expected-note@+1 {{the DMA endpoint is here}}
    aie.objectfifo @out(%tile, {%shim}, 2 : i32) {iter_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c5 = arith.constant 5 : index
      scf.for %i = %c0 to %c5 step %c1 {
        %o = aie.objectfifo.acquire @out (Produce, 1) : memref<16xi32>
        // expected-error@+1 {{releases 5 objects through @out_prod over the run, but the pool's DMA endpoint stops after 4}}
        aie.objectfifo.release @out (Produce, 1)
      }
      aie.end
    }
  }
}

// -----

// With repeat_count the sender replays each object, so the consumer sees
// `iter_count * depth * repeat_count` objects, 8 here, while the producer
// still hands over `iter_count * depth`, 4. Both cores release exactly that.

module {
  aie.device(npu1_1col) {
    %prod = aie.tile(0, 2)
    %cons = aie.tile(0, 4)
    aie.objectfifo @rep(%prod, {%cons}, 2 : i32) {iter_count = 2 : i32, repeat_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
    %core_prod = aie.core(%prod) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %o = aie.objectfifo.acquire @rep (Produce, 1) : memref<16xi32>
        aie.objectfifo.release @rep (Produce, 1)
      }
      aie.end
    }
    %core_cons = aie.core(%cons) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %o = aie.objectfifo.acquire @rep (Consume, 1) : memref<16xi32>
        aie.objectfifo.release @rep (Consume, 1)
      }
      aie.end
    }
  }
}
