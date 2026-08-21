//===- peano_compat_float_decimal_merge.mlir - post-link float decimals --===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The merge is where a float constant crosses LLVM versions as text, so it is
// where the LLVM 24 spelling of one bites: the linker reprints the kernel with
// aiecc's LLVM, which writes a float/half constant as a short decimal whenever
// that decimal round-trips in the narrow type, and Peano's parser -- which
// still demands exact representability -- rejects it with "floating point
// constant invalid for type". This is the failure a fused-decode attention
// kernel hit on its exp2 lookup table.
//
// downgradeIRForPeano runs again after the link for exactly this reason, so
// check its output covers every position the constants appear in.

// REQUIRES: peano

// RUN: rm -rf %t && mkdir -p %t
// RUN: aiecc --tmpdir %t %s
// RUN: FileCheck %s --input-file %t/peano-linked_main_core_0_2.ll \
// RUN:   --implicit-check-not="float 3.141590e+00" \
// RUN:   --implicit-check-not="float 1.100000e-01" \
// RUN:   --implicit-check-not="float -3.236090e-03" \
// RUN:   --implicit-check-not="half 1.099850e-01" \
// RUN:   --implicit-check-not=", 1.100000e-01"

// Typed array: the three inexact literals in hex, the exact one untouched.
// CHECK-DAG: 0x400921FA00000000
// CHECK-DAG: 0xBF6A8292A0000000
// CHECK-DAG: float 2.500000e+00
// Bare operand, taking float from the instruction.
// CHECK-DAG: fmul float %{{.*}}, 0x3FBC28F5C0000000
// Half takes its own 16-bit form; the exact one is left alone.
// CHECK-DAG: half 0xH2F0A
// CHECK-DAG: half 2.500000e+00
// A double on a float-typed line keeps the spelling Peano accepts.
// CHECK-DAG: fptrunc double 1.100000e-01

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xf32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xf32>>

    func.func private @merge_kernel(memref<16xf32>, memref<16xf32>) attributes {link_with = "Inputs/peano_float_decimal_kernel.ll", link_with_mode = "merge"}

    %core_0_2 = aie.core(%tile_0_2) {
      %elem_in = aie.objectfifo.acquire @of_in (Consume, 1) : memref<16xf32>
      %elem_out = aie.objectfifo.acquire @of_out (Produce, 1) : memref<16xf32>
      func.call @merge_kernel(%elem_in, %elem_out) : (memref<16xf32>, memref<16xf32>) -> ()
      aie.objectfifo.release @of_in (Consume, 1)
      aie.objectfifo.release @of_out (Produce, 1)
      aie.end
    }
  }
}
