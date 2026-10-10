//===- arbiter_dispatch_merge_hub.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-objectFifo-stateful-transform --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s

// A mem tile dispatches to two cores over one channel and merges their results
// over another. Core a, stalled behind a full receiver, waits on the merge to
// take its result, and the merge takes it once a free object comes back: core
// b's packet into the merge is a DMA's, sent whole once it starts, so it never
// holds the merge waiting on core b. The flows route with no hazard.

// CHECK-LABEL: module @hub
// CHECK-NOT:   {{warning|error}}
// CHECK:       aie.packet_rules
module @hub {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo @in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @to_a(%m, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @to_b(%m, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@to_a, @to_b] ([] []) {mode = #aie.link_mode<time>}
    aie.objectfifo @from_a(%a, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @from_b(%b, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%m, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@from_a, @from_b] -> [@out] ([] []) {mode = #aie.link_mode<time>}
    aie.core(%a) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %x = aie.objectfifo.acquire @to_a(Consume, 1) : memref<16xi32>
        %y = aie.objectfifo.acquire @from_a(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @to_a(Consume, 1)
        aie.objectfifo.release @from_a(Produce, 1)
      }
      aie.end
    }
    aie.core(%b) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %x = aie.objectfifo.acquire @to_b(Consume, 1) : memref<16xi32>
        %y = aie.objectfifo.acquire @from_b(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @to_b(Consume, 1)
        aie.objectfifo.release @from_b(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%in : memref<128xi32>, %out : memref<128xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {metadata = @out, id = 1 : i64, issue_token = true} : memref<128xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {metadata = @in, id = 0 : i64} : memref<128xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}

// -----

// The same hub with core b writing its results to its stream port itself. A
// core can stop in the middle of a packet to wait for its next input, holding
// the merge, so the cycle through core a is real and routing refuses it.

// CHECK:       error: Flows can deadlock however they are routed: packet flow (0, 1) DMA:0 -> (0, 2) DMA:0 (id 0) can fill its receiver, and draining that waits on (0, 2) core, then (0, 2) MM2S 0, then (0, 1) S2MM 1, then (0, 3) core
module @hub_core_sender {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo @in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @to_a(%m, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @to_b(%m, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@to_a, @to_b] ([] []) {mode = #aie.link_mode<time>}
    aie.objectfifo @from_a(%a, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @from_b(%b, {%m}, 2 : i32) {prod_port = #aie.end_port<Core : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%m, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@from_a, @from_b] -> [@out] ([] []) {mode = #aie.link_mode<time>}
    aie.core(%a) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %x = aie.objectfifo.acquire @to_a(Consume, 1) : memref<16xi32>
        %y = aie.objectfifo.acquire @from_a(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @to_a(Consume, 1)
        aie.objectfifo.release @from_a(Produce, 1)
      }
      aie.end
    }
    aie.core(%b) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %x = aie.objectfifo.acquire @to_b(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @to_b(Consume, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%in : memref<128xi32>, %out : memref<128xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {metadata = @out, id = 1 : i64, issue_token = true} : memref<128xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {metadata = @in, id = 0 : i64} : memref<128xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}
