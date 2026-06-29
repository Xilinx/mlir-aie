//===- peano_compat_f0x_float.mlir - downgradeIRForPeano f0x float --------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Regression test: downgradeIRForPeano must rewrite 'f0x<8hex>' typed float
// literals (introduced in LLVM 23) to the double-widened '0x<16hex>' form
// that Peano's LLVM 21 opt can parse.
//
// Without the fix, aiecc fails with:
//   opt: ...peano-compat.ll:N:M: error: expected value token
//     %r = fadd float %x, f0x3727C5AC
//
// The constant 9.99999974E-6 (float32 bits 0x3727C5AC, the layernorm epsilon)
// is a concrete trigger: LLVM 23's IR printer emits it as 'f0x3727C5AC'
// because it cannot be represented exactly as a short decimal literal.
// After downgrade the peano-compat.ll must use the equivalent double-widened
// form '0x3EE4F8B580000000' which Peano's LLVM 21 can parse.

// REQUIRES: peano

// RUN: aiecc --no-xchesscc --no-xbridge --tmpdir %t %s
// RUN: FileCheck %s --input-file %t/main_core_0_2.peano-compat.ll \
// RUN:   --implicit-check-not="f0x"
// RUN: FileCheck %s --input-file %t/main_core_0_2.peano-compat.ll \
// RUN:   --check-prefix=CHECK-HEX

// CHECK: define void @core_0_2()
// CHECK-HEX: 0x3EE4F8B580000000

module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %buf_in  = aie.buffer(%tile_0_2) {sym_name = "in"}  : memref<256xf32>
    %buf_out = aie.buffer(%tile_0_2) {sym_name = "out"} : memref<256xf32>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0   = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c1   = arith.constant 1 : index
      // This f32 constant (the closest float32 to 1e-5, used as layernorm
      // epsilon) has bit pattern 0x3727C5AC. LLVM 23 emits it as
      // 'f0x3727C5AC'; the downgrade must rewrite it to '0x3EE4F8B580000000'.
      %eps  = arith.constant 9.99999974E-6 : f32
      scf.for %i = %c0 to %c256 step %c1 {
        %val = memref.load %buf_in[%i] : memref<256xf32>
        %out = arith.addf %val, %eps : f32
        memref.store %out, %buf_out[%i] : memref<256xf32>
      }
      aie.end
    }
  }
}
