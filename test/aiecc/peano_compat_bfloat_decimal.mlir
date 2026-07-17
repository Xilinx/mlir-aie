//===- peano_compat_bfloat_decimal.mlir - downgradeIRForPeano bfloat -----===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression test: downgradeIRForPeano must rewrite decimal bfloat16 literals
// (an LLVM 23 printing form) to the 0xR-prefixed bit-exact hex form Peano's opt
// accepts, e.g. `fmul bfloat %v, 1.445310e+00` -> 0xR3FB9 (the bfloat16
// approximation of log2(e)). The downgraded peano-compat.ll must use '0xR3FB9'
// and contain no bare decimal bfloat literals.

// REQUIRES: peano

// RUN: aiecc --no-xchesscc --no-xbridge --tmpdir %t %s
// RUN: FileCheck %s --input-file %t/peano-compat_main_core_0_2.ll \
// RUN:   --implicit-check-not="bfloat {{[0-9][0-9]*\.[0-9]}}"
// RUN: FileCheck %s --input-file %t/peano-compat_main_core_0_2.ll \
// RUN:   --check-prefix=CHECK-HEX

// CHECK: define void @core_0_2()
// CHECK-HEX: 0xR3FB9

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %buf_in  = aie.buffer(%tile) {sym_name = "in"}  : memref<32xbf16>
    %buf_out = aie.buffer(%tile) {sym_name = "out"} : memref<32xbf16>

    %core = aie.core(%tile) {
      %c0  = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1  = arith.constant 1 : index
      // bfloat16 approximation of log2(e) (bits 0x3FB9, decimal 1.445310e+00).
      // LLVM 23 emits this as 'bfloat 1.445310e+00'; the downgrade must
      // rewrite it to 'bfloat 0xR3FB9' for Peano's opt to parse.
      %log2e = arith.constant 1.445310e+00 : bf16
      scf.for %i = %c0 to %c32 step %c1 {
        %v = memref.load %buf_in[%i] : memref<32xbf16>
        %r = arith.mulf %v, %log2e : bf16
        memref.store %r, %buf_out[%i] : memref<32xbf16>
      }
      aie.end
    }
  }
}
