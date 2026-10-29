//===- bad_link_stream_walk.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %core = aie.tile(0, 2)
  aie.objectfifo @rows(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @windows(%mem dimensionsToStream [<size = 4, stride = 12>, <size = 3, stride = 8>, <size = 8, stride = 1>], {%core}, 2 : i32) : !aie.objectfifo<memref<24xi32>>
  // expected-error@+1 {{outer stride 12 is not a multiple of the 8-element object}}
  aie.objectfifo.link [@rows] -> [@windows] ([] [])
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %core = aie.tile(0, 2)
  aie.objectfifo @rows(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @windows(%mem dimensionsToStream [<size = 2, stride = 8>, <size = 2, stride = 8>, <size = 2, stride = 16>, <size = 4, stride = 1>], {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  // expected-error@+1 {{may step one inner dimension by exactly one 8-element object}}
  aie.objectfifo.link [@rows] -> [@windows] ([] [])
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %core = aie.tile(0, 2)
  aie.objectfifo @rows(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @windows(%mem dimensionsToStream [<size = 4, stride = 8>, <size = 3, stride = 8>, <size = 8, stride = 1>], {%core}, 2 : i32) {padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 1, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>]>} : !aie.objectfifo<memref<24xi32>>
  // expected-error@+1 {{pads only its outer dimension}}
  aie.objectfifo.link [@rows] -> [@windows] ([] [])
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %core = aie.tile(0, 2)
  aie.objectfifo @rows(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @windows(%mem dimensionsToStream [<size = 2, stride = 8>, <size = 3, stride = 8>, <size = 8, stride = 1>], {%core}, 2 : i32) {padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 3, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>]>} : !aie.objectfifo<memref<24xi32>>
  // expected-error@+1 {{leaves an output without data}}
  aie.objectfifo.link [@rows] -> [@windows] ([] [])
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %core = aie.tile(0, 2)
  aie.objectfifo @rows(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @evens(%mem dimensionsToStream [<size = 2, stride = 504>, <size = 8, stride = 1>], {%core}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  // expected-error@+1 {{a walk over a stream of 64 objects needs a lock counting that many, but locks count to 63}}
  aie.objectfifo.link [@rows] -> [@evens] ([] [])
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %core = aie.tile(0, 2)
  aie.objectfifo @rows(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @all(%mem dimensionsToStream [<size = 25, stride = 8>, <size = 8, stride = 1>], {%core}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  // expected-error@+1 {{lowers to one descriptor per output, 25, but a MemTile channel has 24}}
  aie.objectfifo.link [@rows] -> [@all] ([] [])
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %core = aie.tile(0, 2)
  aie.objectfifo @rows(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @windows(%mem dimensionsToStream [<size = 2, stride = 8>, <size = 3, stride = 8>, <size = 8, stride = 1>], {%core}, 2 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<24xi32>>
  // expected-error@+1 {{a walk over a stream does not repeat objects}}
  aie.objectfifo.link [@rows] -> [@windows] ([] [])
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %link = aie.tile(0, 2)
  %core = aie.tile(0, 3)
  aie.objectfifo @rows(%shim, {%link}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @windows(%link dimensionsToStream [<size = 2, stride = 8>, <size = 3, stride = 8>, <size = 8, stride = 1>], {%core}, 2 : i32) : !aie.objectfifo<memref<24xi32>>
  // expected-error@+1 {{a walk over a stream needs a MemTile link point}}
  aie.objectfifo.link [@rows] -> [@windows] ([] [])
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %core = aie.tile(0, 2)
  aie.objectfifo @rows(%shim, {%mem}, 2 : i32) {iter_count = 3 : i32} : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @windows(%mem dimensionsToStream [<size = 2, stride = 8>, <size = 3, stride = 8>, <size = 8, stride = 1>], {%core}, 2 : i32) : !aie.objectfifo<memref<24xi32>>
  // expected-error@+1 {{a walk over a stream runs without an iter_count}}
  aie.objectfifo.link [@rows] -> [@windows] ([] [])
}

// -----

// Part of each object, walked within one object: the object still leaves
// whole, so the walk would run on past itself.
aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %core = aie.tile(0, 2)
  aie.objectfifo @rows(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
  aie.objectfifo @heads(%mem dimensionsToStream [<size = 2, stride = 16>, <size = 4, stride = 1>], {%core}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  // expected-error@+1 {{@heads's walk covers 8 elements, but each transfer sends a whole 32-element object; to send part of each input object, walk the stream}}
  aie.objectfifo.link [@rows] -> [@heads] ([] [])
}
