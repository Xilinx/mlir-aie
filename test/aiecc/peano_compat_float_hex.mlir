//===- peano_compat_float_hex.mlir - downgradeIRForPeano float literals ---===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Regression test: downgradeIRForPeano must rewrite single-precision 'float'
// constants into the legacy 16-hex-digit form Peano's opt accepts. Recent main
// LLVM prints f32 constants two ways Peano's parser rejects:
//   - the compact 8-hex-digit 'float f0x########' form, and
//   - a decimal that is not exactly representable as f32 (e.g.
//     'float 8.208500e-02'), which Peano flags "floating point constant
//     invalid for type".
// The generated *.peano-compat.ll (the IR actually handed to Peano's opt) must
// therefore contain neither an 'f0x' literal nor a decimal-exponent float
// literal; every f32 constant must be a 0x-prefixed hex pattern.

// REQUIRES: peano

// RUN: aiecc --no-xchesscc --no-xbridge --tmpdir %t %s
// RUN: FileCheck %s --input-file %t/main_core_0_2.peano-compat.ll
// RUN: FileCheck %s --check-prefix=NOBAD --input-file %t/main_core_0_2.peano-compat.ll

// The exp_lut initializer must be all 0x hex floats after the downgrade.
// CHECK: @exp_lut = global [8 x float] [float 0x{{[0-9A-F]+}},

// Neither bad f32 form may survive: the compact 'f0x' hex literal nor a
// decimal-exponent literal (e.g. 'float 8.208500e-02').
// NOBAD-NOT: f0x
// NOBAD-NOT: float {{[-+.0-9]+e}}

module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    // Mix of values that LLVM prints as compact 'f0x' hex and as inexact
    // decimals; both forms must be normalized to 0x-hex by the downgrade.
    %lut = aie.buffer(%tile_0_2) {sym_name = "exp_lut"} : memref<8xf32> = dense<[0.018315639, 0.082085, 0.41686, 1.0, 2.117, 7.62361, 0.036425, 3.5]>
    %buf_c = aie.buffer(%tile_0_2) {sym_name = "buf_c"} : memref<8xf32>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index

      scf.for %i = %c0 to %c8 step %c1 {
        %v = memref.load %lut[%i] : memref<8xf32>
        memref.store %v, %buf_c[%i] : memref<8xf32>
      }
      aie.end
    }
  }
}
