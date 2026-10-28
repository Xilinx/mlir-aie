//===- objectfifo_link_slot_sizes.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A join input's slot, or a distribute output's, must hold a whole number of
// that participant's objects; otherwise an object straddles two shared objects
// and the DMA stalls on the device.

// RUN: aie-opt -split-input-file -verify-diagnostics %s

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %a = aie.tile(0, 2)
  %b = aie.tile(0, 3)
  aie.objectfifo @of_a(%a, {%mem}, 1 : i32) : !aie.objectfifo<memref<12xi32>>
  aie.objectfifo @of_b(%b, {%mem}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @of_out(%mem, {%shim}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
  // expected-error@+1 {{input @of_a has 48 B objects, but its 64 B slot in @of_out from offsets [0, 16] is not a whole number of them}}
  aie.objectfifo.link [@of_a, @of_b] -> [@of_out]([0, 16] [])
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %a = aie.tile(0, 2)
  %b = aie.tile(0, 3)
  aie.objectfifo @of_a(%a, {%mem}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @of_b(%b, {%mem}, 1 : i32) : !aie.objectfifo<memref<12xi32>>
  aie.objectfifo @of_out(%mem, {%shim}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
  // expected-error@+1 {{input @of_b has 48 B objects, but its 64 B slot in @of_out from offsets [0, 16] is not a whole number of them}}
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
  // expected-error@+1 {{input @of_a has 96 B objects, but its 64 B slot in @of_out from offsets [0, 16] is not a whole number of them}}
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
  aie.objectfifo @of_b(%mem, {%b}, 1 : i32) : !aie.objectfifo<memref<12xi32>>
  // expected-error@+1 {{output @of_b has 48 B objects, but its 64 B slot in @of_in from offsets [0, 16] is not a whole number of them}}
  aie.objectfifo.link [@of_in] -> [@of_a, @of_b]([] [0, 16])
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %a = aie.tile(0, 2)
  %b = aie.tile(0, 3)
  aie.objectfifo @of_in(%shim, {%mem}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
  aie.objectfifo @of_a(%mem, {%a}, 1 : i32) : !aie.objectfifo<memref<24xi32>>
  aie.objectfifo @of_b(%mem, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
  // expected-error@+1 {{output @of_a has 96 B objects, but its 64 B slot in @of_in from offsets [0, 16] is not a whole number of them}}
  aie.objectfifo.link [@of_in] -> [@of_a, @of_b]([] [0, 16])
}

// -----

// Every slot holds a whole number of its participant's objects:
// - 8-element inputs fill 16-element slots two objects each, first or last;
// - a GEMM-style join and distribute move 256-element bf16 objects through
//   2048-element slots, eight each;
// - a join input's consumer side takes slot-sized pieces of a larger producer
//   object;
// - a distribute output's padding widens its slot to the output's object.
aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %m0 = aie.tile(0, 1)
  %m1 = aie.tile(1, 1)
  %m2 = aie.tile(2, 1)
  %m3 = aie.tile(3, 1)
  %m4 = aie.tile(4, 1)
  %m5 = aie.tile(5, 1)
  %c00 = aie.tile(0, 2)
  %c01 = aie.tile(0, 3)
  %c10 = aie.tile(1, 2)
  %c11 = aie.tile(1, 3)
  %c20 = aie.tile(2, 2)
  %c21 = aie.tile(2, 3)
  %c30 = aie.tile(3, 2)
  %c31 = aie.tile(3, 3)
  %c40 = aie.tile(4, 2)
  %c41 = aie.tile(4, 3)
  %c50 = aie.tile(5, 2)
  %c51 = aie.tile(5, 3)

  aie.objectfifo @first_a(%c00, {%m0}, 1 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @first_b(%c01, {%m0}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @first_out(%m0, {%shim}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
  aie.objectfifo.link [@first_a, @first_b] -> [@first_out]([0, 16] [])

  aie.objectfifo @last_a(%c10, {%m1}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @last_b(%c11, {%m1}, 1 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @last_out(%m1, {%shim}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
  aie.objectfifo.link [@last_a, @last_b] -> [@last_out]([0, 16] [])

  aie.objectfifo @gemm_a(%c20, {%m2}, 2 : i32) : !aie.objectfifo<memref<256xbf16>>
  aie.objectfifo @gemm_b(%c21, {%m2}, 2 : i32) : !aie.objectfifo<memref<256xbf16>>
  aie.objectfifo @gemm_out(%m2, {%shim}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>
  aie.objectfifo.link [@gemm_a, @gemm_b] -> [@gemm_out]([0, 2048] [])

  aie.objectfifo @gemm_in(%shim, {%m3}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>
  aie.objectfifo @gemm_c(%m3, {%c30}, 2 : i32) : !aie.objectfifo<memref<256xbf16>>
  aie.objectfifo @gemm_d(%m3, {%c31}, 2 : i32) : !aie.objectfifo<memref<256xbf16>>
  aie.objectfifo.link [@gemm_in] -> [@gemm_c, @gemm_d]([] [0, 2048])

  aie.objectfifo @split_a(%c40, {%m4}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @split_b(%c41, {%m4}, 1 : i32) : !aie.objectfifo<memref<32xi32>> -> !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @split_out(%m4, {%shim}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
  aie.objectfifo.link [@split_a, @split_b] -> [@split_out]([0, 16] [])

  aie.objectfifo @pad_in(%shim, {%m5}, 1 : i32) : !aie.objectfifo<memref<36xi32>>
  aie.objectfifo @pad_c(%m5, {%c50}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  aie.objectfifo @pad_d(%m5 dimensionsToStream [<size = 16, stride = 1>], {%c51}, 1 : i32) {padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 2, const_pad_after = 2>]>} : !aie.objectfifo<memref<20xi32>>
  aie.objectfifo.link [@pad_in] -> [@pad_c, @pad_d]([] [0, 16])
}

// -----

// A block-float element is sized by the data layout: a v8bfp16ebs8 block is 9 B.
aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %a = aie.tile(0, 2)
  %b = aie.tile(0, 3)
  aie.objectfifo @of_a(%a, {%mem}, 1 : i32) : !aie.objectfifo<memref<6x!aiex.bfp<"v8bfp16ebs8">>>
  aie.objectfifo @of_b(%b, {%mem}, 1 : i32) : !aie.objectfifo<memref<8x!aiex.bfp<"v8bfp16ebs8">>>
  aie.objectfifo @of_out(%mem, {%shim}, 1 : i32) : !aie.objectfifo<memref<16x!aiex.bfp<"v8bfp16ebs8">>>
  // expected-error@+1 {{input @of_a has 54 B objects, but its 72 B slot in @of_out from offsets [0, 8] is not a whole number of them}}
  aie.objectfifo.link [@of_a, @of_b] -> [@of_out]([0, 8] [])
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  %a = aie.tile(0, 2)
  %b = aie.tile(0, 3)
  aie.objectfifo @of_a(%a, {%mem}, 1 : i32) : !aie.objectfifo<memref<8x!aiex.bfp<"v8bfp16ebs8">>>
  aie.objectfifo @of_b(%b, {%mem}, 1 : i32) : !aie.objectfifo<memref<8x!aiex.bfp<"v8bfp16ebs8">>>
  aie.objectfifo @of_out(%mem, {%shim}, 1 : i32) : !aie.objectfifo<memref<16x!aiex.bfp<"v8bfp16ebs8">>>
  aie.objectfifo.link [@of_a, @of_b] -> [@of_out]([0, 8] [])
}
