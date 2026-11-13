//===- allocate_link_misfit.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectFifo-stateful-transform %s

// Where its ends reach the delegate and the link tile, an allocated linked
// fifo would keep its objects in the link's pool, which a merge's inputs and
// a dispatch's outputs reach only by taking turns on one DMA end, and which a
// fifo between two links cannot be for both. Before, the split aborted on
// "an endpoint the split made is not planned exactly once" or failed with an
// internal error.

module @merge_first_input {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %l = aie.tile(0, 3)
    %b = aie.tile(0, 4)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @in1(%b, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%l, {%s}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@in0, @in1] -> [@out] ([] []) {mode = #aie.link_mode<time>}
    // expected-error@+1 {{places @in0's objects in the link's pool, but @in0 is a merge's input, and a merge's inputs take turns filling the pool through one DMA end}}
    aie.objectfifo.allocate @in0 (%l)
  }
}

// -----

module @dispatch_first_output {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %l = aie.tile(0, 3)
    %b = aie.tile(0, 4)
    aie.objectfifo @in(%s, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out0(%l, {%a}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out1(%l, {%b}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@in] -> [@out0, @out1] ([] []) {mode = #aie.link_mode<time>}
    // expected-error@+1 {{places @out0's objects in the link's pool, but @out0 is a dispatch's output, and a dispatch's outputs take turns draining the pool through one DMA end}}
    aie.objectfifo.allocate @out0 (%l)
  }
}

// -----

module @between_links {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %l1 = aie.tile(0, 2)
    %l2 = aie.tile(0, 3)
    aie.objectfifo @in0(%s, {%l1}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @mid(%l1, {%l2}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%l2, {%s}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@in0] -> [@mid] ([] [])
    aie.objectfifo.link [@mid] -> [@out] ([] [])
    // expected-error@+1 {{places @mid's objects in the link's pool, but @mid is one link's output and another link's input, so its objects would be both links' pools}}
    aie.objectfifo.allocate @mid (%l2)
  }
}
