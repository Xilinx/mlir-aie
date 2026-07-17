//===- peano_compat_f0x_float.mlir - downgradeIRForPeano f0x float --------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression test: downgradeIRForPeano must rewrite 'f0x<8hex>' typed float
// literals (an LLVM 23 printing form) to the double-widened '0x<16hex>' form
// Peano's opt accepts. The trigger constant 9.99999974E-6 (float32 bits
// 0x3727C5AC, the layernorm epsilon) is printed as 'f0x3727C5AC'; the
// downgraded peano-compat.ll must use '0x3EE4F8B580000000' and contain no
// 'f0x' literal.

// REQUIRES: peano

// RUN: aiecc --no-xchesscc --no-xbridge --tmpdir %t %s
// RUN: FileCheck %s --input-file %t/peano-compat_main_core_0_2.ll \
// RUN:   --implicit-check-not="f0x"
// RUN: FileCheck %s --input-file %t/peano-compat_main_core_0_2.ll \
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
