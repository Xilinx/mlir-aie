//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform \
// RUN:         --aie-substitute-shim-dma-allocations \
// RUN:         --aie-assign-runtime-sequence-bd-ids='warn-unsafe-bd-reuse=true' \
// RUN:         --verify-diagnostics %s

// From an ObjectFIFO design to the BD-id pass: aie-objectfifo-verify proves
// the input consumed by the time the output arrives, allocation carries the
// proof to the input's shim allocation, substitution keeps the allocation on
// each task, and the BD-id pass credits the input once the output is awaited.

// The core takes one input object before releasing each output object, so
// freeing the input after awaiting the output is safe.
module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @rounds(%a: memref<32xi32>, %c: memref<32xi32>) {
      %in0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%a : memref<32xi32> offset = 0 len = 16)
        aie.end
      }
      aiex.dma_start_task(%in0)
      %out0 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%c : memref<32xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%a : memref<32xi32> offset = 16 len = 16)
        aie.end
      }
      aiex.dma_start_task(%in1)
      %out1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%c : memref<32xi32> offset = 16 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%in1)
    }
  }
}

// -----

// The core releases each output before it takes the input, so the output's
// arrival proves nothing and the reuse still warns.
module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @out(Produce, 1)
        %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @in(Consume, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @rounds(%a: memref<32xi32>, %c: memref<32xi32>) {
      %in0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%a : memref<32xi32> offset = 0 len = 16)
        aie.end
      }
      aiex.dma_start_task(%in0)
      %out0 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%c : memref<32xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task_for @in {
        // expected-warning@+1 {{reuses buffer descriptor ID 0 on tile (0,0) after it was released by an aiex.dma_free_task that had no completion guarantee}}
        aie.dma_bd(%a : memref<32xi32> offset = 16 len = 16)
        aie.end
      }
      aiex.dma_start_task(%in1)
      %out1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%c : memref<32xi32> offset = 16 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%in1)
    }
  }
}
