//===- test_place_shim_gates_proven.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie-place-tiles,aie.device(aie-objectFifo-stateful-transform))' %s | FileCheck %s
// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie-place-tiles{placer=sa_placer sa-seed=1},aie.device(aie-objectFifo-stateful-transform))' %s | FileCheck %s
// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie-place-tiles{placer=sa_placer sa-seed=1},aie.device(aie-objectFifo-stateful-transform))' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=QUIET --allow-empty

// Each core relays its own shim input to its output, and each core's
// transfers run in a sequence of their own, so aie-objectfifo-verify proves
// every output consumes its input, and allocation lets the three receiving
// ends (and the three sending ones) take turns on one shim's two channels.
// The placers find the same proofs on their copy of the design and place
// all six ends on the one shim.

// @proven_turns_unpinned_first: c0 is unpinned and declared before c1,
// which is pinned to (0, 2). The placers' copy places pinned tiles first, so
// c0 never takes c1's tile and the copy still proves every turn, whatever the
// order.
// CHECK-LABEL: @proven_turns_unpinned_first
// CHECK-COUNT-6: aie.shim_dma_allocation @{{(in|out)[0-2]}}__shim_alloc(%shim_noc_tile_0_0,
// QUIET-NOT: warning

// @proven_through_shared_memory: @in0's proof runs through @mid0, which the
// placement puts on shared memory (c0 and c1 are neighbors) while the
// placers' copy moved it by DMA. Allocation still finds the proof, so the
// turns the placers counted are the ones it grants.
// CHECK-LABEL: @proven_through_shared_memory
// CHECK-NOT:   mid0__cons__buf
// CHECK-COUNT-6: aie.shim_dma_allocation @{{(in|out)[0-2]}}__shim_alloc(%shim_noc_tile_0_0,

module @proven_turns_unpinned_first {
  aie.device(npu2_1col) {
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(0, 2)
    %c2 = aie.logical_tile<CoreTile>(0, 4)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in0(%shim, {%c0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out0(%c0, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in1(%shim, {%c1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out1(%c1, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2(%shim, {%c2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out2(%c2, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c0) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in0(Consume, 1) : memref<16xi32>
        %o = aie.objectfifo.acquire @out0(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.core(%c1) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in1(Consume, 1) : memref<16xi32>
        %o = aie.objectfifo.acquire @out1(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in1(Consume, 1)
        aie.objectfifo.release @out1(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.core(%c2) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in2(Consume, 1) : memref<16xi32>
        %o = aie.objectfifo.acquire @out2(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in2(Consume, 1)
        aie.objectfifo.release @out2(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @seq0(%a: memref<16xi32>, %b: memref<16xi32>) {
      %ti = aiex.dma_configure_task_for @in0 {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      %to = aiex.dma_configure_task_for @out0 {
        aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to)
      aiex.dma_start_task(%ti)
      aiex.dma_await_task(%to)
      aiex.dma_await_task(%ti)
    }
    aie.runtime_sequence @seq1(%a: memref<16xi32>, %b: memref<16xi32>) {
      %ti = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      %to = aiex.dma_configure_task_for @out1 {
        aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to)
      aiex.dma_start_task(%ti)
      aiex.dma_await_task(%to)
      aiex.dma_await_task(%ti)
    }
    aie.runtime_sequence @seq2(%a: memref<16xi32>, %b: memref<16xi32>) {
      %ti = aiex.dma_configure_task_for @in2 {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      %to = aiex.dma_configure_task_for @out2 {
        aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to)
      aiex.dma_start_task(%ti)
      aiex.dma_await_task(%to)
      aiex.dma_await_task(%ti)
    }
  }
}

// -----

module @proven_through_shared_memory {
  aie.device(npu2_1col) {
    %c0 = aie.logical_tile<CoreTile>(0, 2)
    %c1 = aie.logical_tile<CoreTile>(0, 3)
    %c2 = aie.logical_tile<CoreTile>(0, 4)
    %c3 = aie.logical_tile<CoreTile>(0, 5)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in0(%shim, {%c0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @mid0(%c0, {%c1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out0(%c1, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in1(%shim, {%c2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out1(%c2, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2(%shim, {%c3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out2(%c3, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c0) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in0(Consume, 1) : memref<16xi32>
        %o = aie.objectfifo.acquire @mid0(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @mid0(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.core(%c1) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @mid0(Consume, 1) : memref<16xi32>
        %o = aie.objectfifo.acquire @out0(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @mid0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.core(%c2) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in1(Consume, 1) : memref<16xi32>
        %o = aie.objectfifo.acquire @out1(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in1(Consume, 1)
        aie.objectfifo.release @out1(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.core(%c3) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in2(Consume, 1) : memref<16xi32>
        %o = aie.objectfifo.acquire @out2(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in2(Consume, 1)
        aie.objectfifo.release @out2(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @seq0(%a: memref<16xi32>, %b: memref<16xi32>) {
      %ti = aiex.dma_configure_task_for @in0 {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      %to = aiex.dma_configure_task_for @out0 {
        aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to)
      aiex.dma_start_task(%ti)
      aiex.dma_await_task(%to)
      aiex.dma_await_task(%ti)
    }
    aie.runtime_sequence @seq1(%a: memref<16xi32>, %b: memref<16xi32>) {
      %ti = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      %to = aiex.dma_configure_task_for @out1 {
        aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to)
      aiex.dma_start_task(%ti)
      aiex.dma_await_task(%to)
      aiex.dma_await_task(%ti)
    }
    aie.runtime_sequence @seq2(%a: memref<16xi32>, %b: memref<16xi32>) {
      %ti = aiex.dma_configure_task_for @in2 {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      %to = aiex.dma_configure_task_for @out2 {
        aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to)
      aiex.dma_start_task(%ti)
      aiex.dma_await_task(%to)
      aiex.dma_await_task(%ti)
    }
  }
}
