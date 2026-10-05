//===- peano_compat_float_decimal.mlir - downgradeIRForPeano float -------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression test: LLVM 24 prints a `float` constant as a short decimal
// whenever that decimal round-trips in float; older LLVM required it to round
// trip as a double and printed hex otherwise. Peano's opt still demands exact
// representability and rejects the decimal with "floating point constant
// invalid for type", so downgradeIRForPeano must rewrite it -- in the typed
// position a constant array gives (`[4 x float] [float 3.141590e+00, ...]`)
// and in the bare operand one an instruction gives (`fmul float %v,
// 1.100000e-01`), where the type comes from the instruction.
//
// The whole peano path runs here, so Peano's own opt decides: drop the rewrite
// and this fails at `opted_{0}.ll` with that error.

// REQUIRES: peano

// RUN: rm -rf %t && mkdir -p %t
// RUN: aiecc --tmpdir %t %s
// RUN: FileCheck %s --input-file %t/peano-compat_main_core_0_2.ll \
// RUN:   --implicit-check-not="float 3.141590e+00" \
// RUN:   --implicit-check-not="float 1.100000e-01" \
// RUN:   --implicit-check-not="float -3.236090e-03"

// The exactly representable literals Peano already accepts stay decimal, so
// the rewrite stays narrow.
// CHECK-DAG: 0x400921FA00000000
// CHECK-DAG: 0x3FBC28F5C0000000
// CHECK-DAG: 0xBF6A8292A0000000
// CHECK-DAG: 2.500000e+00

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %buf_in = aie.buffer(%tile) {sym_name = "in"} : memref<4xf32>
    %buf_out = aie.buffer(%tile) {sym_name = "out"} : memref<4xf32>

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      // Typed position: a dense constant prints each element behind its own
      // `float` keyword. 2.5 is exact as a double and must be left as printed;
      // the other three are not. A buffer's `initial_value` does not reach the
      // core's object, because a buffer lowers to a declaration and its
      // contents come from the device configuration.
      %lut = arith.constant dense<[3.141590e+00, 1.100000e-01, -3.236090e-03,
                                   2.500000e+00]> : vector<4xf32>
      // Bare operand position: the constant carries no type keyword of its own.
      %k = arith.constant 1.100000e-01 : f32
      scf.for %i = %c0 to %c4 step %c1 {
        %v = memref.load %buf_in[%i] : memref<4xf32>
        %l = vector.extract %lut[%i] : f32 from vector<4xf32>
        %s = arith.addf %v, %l : f32
        %r = arith.mulf %s, %k : f32
        memref.store %r, %buf_out[%i] : memref<4xf32>
      }
      aie.end
    }
  }
}
