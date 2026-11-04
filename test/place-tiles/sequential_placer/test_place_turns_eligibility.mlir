//===- test_place_turns_eligibility.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s

// A core draining three mem tile fifos in turn has two input channels; the
// sequential placer counts the fifos as sharing one by turns only where
// aie-objectfifo-allocate can let them (ObjectFifoCreateOp::canTakeTurns).

// Plain fifos take turns on one channel.
// CHECK-LABEL: module @plain
// CHECK: turns = [
module @plain {
  aie.device(npu2) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %m = aie.logical_tile<MemTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%s0, {%m}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @a(%m, {%c}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in0] -> [@a] ([] [])
    aie.objectfifo @in1(%s1, {%m}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @b(%m, {%c}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in1] -> [@b] ([] [])
    aie.objectfifo @in2(%s2, {%m}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @d(%m, {%c}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in2] -> [@d] ([] [])
    aie.core(%c) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %x = aie.objectfifo.acquire @a(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @a(Consume, 1)
        %y = aie.objectfifo.acquire @b(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @b(Consume, 1)
        %z = aie.objectfifo.acquire @d(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @d(Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// A repeating fifo's draining end never takes turns, so the core needs three
// input channels, which placement reports.
module @repeating {
  aie.device(npu2) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %m = aie.logical_tile<MemTile>(?, ?)
    // expected-error@+2 {{requires 3 input/0 output DMA channels, but only 2 input/2 output available}}
    // expected-note@+1 {{placer selected this tile}}
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%s0, {%m}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @a(%m, {%c}, 2 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in0] -> [@a] ([] [])
    aie.objectfifo @in1(%s1, {%m}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @b(%m, {%c}, 2 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in1] -> [@b] ([] [])
    aie.objectfifo @in2(%s2, {%m}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @d(%m, {%c}, 2 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in2] -> [@d] ([] [])
    aie.core(%c) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %x = aie.objectfifo.acquire @a(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @a(Consume, 1)
        %y = aie.objectfifo.acquire @b(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @b(Consume, 1)
        %z = aie.objectfifo.acquire @d(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @d(Consume, 1)
      }
      aie.end
    }
  }
}
