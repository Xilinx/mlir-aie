//===- objectfifo_link_slot_sizes.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A join input's objects, or a distribute output's, must match the slot its
// offsets give it in the shared object: otherwise the memtile's buffer
// descriptor waits for words that never come and the design hangs.

// RUN: aie-opt -split-input-file -verify-diagnostics %s

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %a = aie.tile(0, 2)
  %b = aie.tile(0, 3)
  aie.objectfifo @of_a(%a, {%mem}, 1 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @of_b(%b, {%mem}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @of_out(%mem, {%shim}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
  // expected-error@+1 {{input @of_a moves 32 B per object, but its slot in @of_out (32 elements) is 16 elements (64 B) from offsets [0, 16]; the transfer would never complete}}
  aie.objectfifo.link [@of_a, @of_b] -> [@of_out]([0, 16] [])
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %a = aie.tile(0, 2)
  %b = aie.tile(0, 3)
  aie.objectfifo @of_a(%a, {%mem}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @of_b(%b, {%mem}, 1 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @of_out(%mem, {%shim}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
  // expected-error@+1 {{input @of_b moves 32 B per object, but its slot in @of_out (32 elements) is 16 elements (64 B) from offsets [0, 16]; the transfer would never complete}}
  aie.objectfifo.link [@of_a, @of_b] -> [@of_out]([0, 16] [])
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %a = aie.tile(0, 2)
  %b = aie.tile(0, 3)
  aie.objectfifo @of_a(%a, {%mem}, 1 : i32) : !aie.objectfifo<memref<24xi32>>
  aie.objectfifo @of_b(%b, {%mem}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @of_out(%mem, {%shim}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
  // expected-error@+1 {{input @of_a moves 96 B per object, but its slot in @of_out (32 elements) is 16 elements (64 B) from offsets [0, 16]; the transfer would never complete}}
  aie.objectfifo.link [@of_a, @of_b] -> [@of_out]([0, 16] [])
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %a = aie.tile(0, 2)
  %b = aie.tile(0, 3)
  aie.objectfifo @of_in(%shim, {%mem}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
  aie.objectfifo @of_a(%mem, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @of_b(%mem, {%b}, 1 : i32) : !aie.objectfifo<memref<8xi32>>
  // expected-error@+1 {{output @of_b moves 32 B per object, but its slot in @of_in (32 elements) is 16 elements (64 B) from offsets [0, 16]; the transfer would never complete}}
  aie.objectfifo.link [@of_in] -> [@of_a, @of_b]([] [0, 16])
}

// -----

// Every participant matches its slot: a join, a distribute, a join input whose
// consumer side takes slot-sized pieces of a larger producer object, and a
// distribute output whose padding widens its slot to the output's object.
aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %mem2 = aie.tile(1, 1)
  %a = aie.tile(0, 2)
  %b = aie.tile(0, 3)
  %c = aie.tile(1, 2)
  %d = aie.tile(1, 3)
  aie.objectfifo @join_a(%a, {%mem}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @join_b(%b, {%mem}, 1 : i32) : !aie.objectfifo<memref<32xi32>> -> !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @join_out(%mem, {%shim}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
  aie.objectfifo.link [@join_a, @join_b] -> [@join_out]([0, 16] [])

  aie.objectfifo @dist_in(%shim, {%mem2}, 1 : i32) : !aie.objectfifo<memref<24xi32>>
  aie.objectfifo @dist_c(%mem2, {%c}, 1 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @dist_d(%mem2 dimensionsToStream [<size = 16, stride = 1>], {%d}, 1 : i32) {padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 2, const_pad_after = 2>]>} : !aie.objectfifo<memref<20xi32>>
  aie.objectfifo.link [@dist_in] -> [@dist_c, @dist_d]([] [0, 8])
}
