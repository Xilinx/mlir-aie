//===- dispatch_volumes.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A Worker whose forever loop waits for each dispatch's parameters runs one
// pass of that loop per dispatch, so one pass must move through a fifo what
// one run of the runtime sequence moves through it. Checked when one runtime
// sequence syncs the parameters and its transfers on the fifo are static.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectfifo-split --aie-objectfifo-verify %s

// One object handed over per dispatch, two drained by the host.

module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @out(%tile, {%shim}, 2 : i32) : !aie.objectfifo<memref<2xi32>>
    %core = aie.core(%tile) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        aiex.wait_scratchpad_parameters
        // expected-error@+1 {{hands over 1 objects through @out_prod for each dispatch, one pass of a loop that waits for each dispatch's parameters, but each run of runtime sequence @sequence moves 2, so the core drifts out of step with the dispatches}}
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<2xi32>
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
    // expected-note@+1 {{the runtime sequence is here}}
    aie.runtime_sequence @sequence(%arg0: memref<4xi32>) {
      aiex.sync_scratchpad_parameters_from_host
      %t = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg0 : memref<4xi32> offset = 0 len = 4 sizes = [1, 1, 1, 4] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// Matching volumes are the ordinary design.

module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @out(%tile, {%shim}, 2 : i32) : !aie.objectfifo<memref<2xi32>>
    %core = aie.core(%tile) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        aiex.wait_scratchpad_parameters
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<2xi32>
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @sequence(%arg0: memref<2xi32>) {
      aiex.sync_scratchpad_parameters_from_host
      %t = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg0 : memref<2xi32> offset = 0 len = 2 sizes = [1, 1, 1, 2] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// Through a memtile distribute: 64 elements fill two 32-element objects, each
// split in halves, so each core receives two 16-element objects per dispatch
// and the core taking one per pass falls behind.

module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    aie.objectfifo @in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @a(%m, {%t2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%m, {%t3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@a, @b] ([] [0, 16])
    %core_b = aie.core(%t3) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        aiex.wait_scratchpad_parameters
        %o = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @b(Consume, 1)
        scf.yield
      }
      aie.end
    }
    %core = aie.core(%t2) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        aiex.wait_scratchpad_parameters
        // expected-error@+1 {{takes 1 objects through @a_cons for each dispatch, one pass of a loop that waits for each dispatch's parameters, but each run of runtime sequence @sequence moves 2}}
        %o = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @a(Consume, 1)
        scf.yield
      }
      aie.end
    }
    // expected-note@+1 {{the runtime sequence is here}}
    aie.runtime_sequence @sequence(%arg0: memref<64xi32>) {
      aiex.sync_scratchpad_parameters_from_host
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// The same distribute with 32 elements per dispatch hands each core one.

module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    aie.objectfifo @in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @a(%m, {%t2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%m, {%t3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@a, @b] ([] [0, 16])
    %core_b = aie.core(%t3) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        aiex.wait_scratchpad_parameters
        %o = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @b(Consume, 1)
        scf.yield
      }
      aie.end
    }
    %core = aie.core(%t2) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        aiex.wait_scratchpad_parameters
        %o = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @a(Consume, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @sequence(%arg0: memref<32xi32>) {
      aiex.sync_scratchpad_parameters_from_host
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// Syncing twice per run is two passes per run: two objects match.

module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @out(%tile, {%shim}, 2 : i32) : !aie.objectfifo<memref<2xi32>>
    %core = aie.core(%tile) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        aiex.wait_scratchpad_parameters
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<2xi32>
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @sequence(%arg0: memref<4xi32>) {
      aiex.sync_scratchpad_parameters_from_host
      aiex.sync_scratchpad_parameters_from_host
      %t = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg0 : memref<4xi32> offset = 0 len = 4 sizes = [1, 1, 1, 4] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// Without the wait, a pass is not tied to a dispatch, so nothing is checked.

module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @out(%tile, {%shim}, 2 : i32) : !aie.objectfifo<memref<2xi32>>
    %core = aie.core(%tile) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<2xi32>
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @sequence(%arg0: memref<4xi32>) {
      aiex.sync_scratchpad_parameters_from_host
      %t = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg0 : memref<4xi32> offset = 0 len = 4 sizes = [1, 1, 1, 4] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// Two sequences sync the parameters, and the host picks which runs, so no one
// run's volume is the dispatch's.

module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @out(%tile, {%shim}, 2 : i32) : !aie.objectfifo<memref<2xi32>>
    %core = aie.core(%tile) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        aiex.wait_scratchpad_parameters
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<2xi32>
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @first(%arg0: memref<4xi32>) {
      aiex.sync_scratchpad_parameters_from_host
      %t = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg0 : memref<4xi32> offset = 0 len = 4 sizes = [1, 1, 1, 4] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
    aie.runtime_sequence @second(%arg0: memref<4xi32>) {
      aiex.sync_scratchpad_parameters_from_host
    }
  }
}

// -----

// Only the chain from the task's entry runs: a block nothing reaches moves
// nothing, so one object per dispatch matches.

module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @out(%tile, {%shim}, 2 : i32) : !aie.objectfifo<memref<2xi32>>
    %core = aie.core(%tile) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        aiex.wait_scratchpad_parameters
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<2xi32>
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @sequence(%arg0: memref<4xi32>) {
      aiex.sync_scratchpad_parameters_from_host
      %t = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg0 : memref<4xi32> offset = 0 len = 2 sizes = [1, 1, 1, 2] strides = [0, 0, 0, 1])
        aie.end
      ^unreached:
        aie.dma_bd(%arg0 : memref<4xi32> offset = 2 len = 2 sizes = [1, 1, 1, 2] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// A task moves once per start: one object's task started twice moves the two
// objects each pass hands over.

module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @out(%tile, {%shim}, 2 : i32) : !aie.objectfifo<memref<2xi32>>
    %core = aie.core(%tile) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        aiex.wait_scratchpad_parameters
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<2xi32>
        aie.objectfifo.release @out(Produce, 1)
        %p = aie.objectfifo.acquire @out(Produce, 1) : memref<2xi32>
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @sequence(%arg0: memref<2xi32>) {
      aiex.sync_scratchpad_parameters_from_host
      %t = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg0 : memref<2xi32> offset = 0 len = 2 sizes = [1, 1, 1, 2] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// A sub-byte element takes a whole byte on the channel, as a buffer
// descriptor's length counts it: four i4 is one four-byte object.

module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @out(%tile, {%shim}, 2 : i32) : !aie.objectfifo<memref<4xi4>>
    %core = aie.core(%tile) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        aiex.wait_scratchpad_parameters
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<4xi4>
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @sequence(%arg0: memref<4xi4>) {
      aiex.sync_scratchpad_parameters_from_host
      %t = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg0 : memref<4xi4> offset = 0 len = 4 sizes = [1, 1, 1, 4] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
