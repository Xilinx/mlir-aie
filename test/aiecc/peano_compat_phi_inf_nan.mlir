//===- peano_compat_phi_inf_nan.mlir - downgradeIRForPeano phi inf/nan ---===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression test: downgradeIRForPeano must rewrite *bare* inf/nan literals
// that LLVM 23 emits without a type prefix in phi operand lists, e.g.
//   %max = phi float [ %next, %body ], [ -inf, %entry ]
// to the double-widened hex form Peano's opt accepts. Such a phi arises from a
// max-reduction seeded with -inf (the scf.for iter_arg's initial value becomes
// the entry-edge operand). The downgraded peano-compat.ll must use
// '0xFFF0000000000000' and contain no bare '-inf' phi operand.

// REQUIRES: peano

// RUN: aiecc --no-xchesscc --no-xbridge --tmpdir %t %s
// RUN: FileCheck %s --input-file %t/peano-compat_main_core_0_2.ll \
// RUN:   --implicit-check-not="[ -inf" --implicit-check-not=", -inf"
// RUN: FileCheck %s --input-file %t/peano-compat_main_core_0_2.ll \
// RUN:   --check-prefix=CHECK-HEX

// CHECK: define void @core_0_2()
// CHECK-HEX: 0xFFF0000000000000

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %buf_in  = aie.buffer(%tile) {sym_name = "in"}  : memref<32xf32>
    %buf_out = aie.buffer(%tile) {sym_name = "out"} : memref<1xf32>

    %core = aie.core(%tile) {
      %c0  = arith.constant 0 : index
      %c1  = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      // -inf seed for a running max. As an scf.for iter_arg this becomes the
      // entry-edge operand of a phi: 'phi float [ ..., %body ], [ -inf, %entry ]'.
      // LLVM 23 prints the seed without a type prefix; the downgrade must
      // rewrite the bare '-inf' to 'float 0xFFF0000000000000'.
      %neg_inf = arith.constant 0xFF800000 : f32
      %max = scf.for %i = %c0 to %c32 step %c1
          iter_args(%acc = %neg_inf) -> (f32) {
        %v = memref.load %buf_in[%i] : memref<32xf32>
        %gt = arith.cmpf ogt, %v, %acc : f32
        %sel = arith.select %gt, %v, %acc : f32
        scf.yield %sel : f32
      }
      memref.store %max, %buf_out[%c0] : memref<1xf32>
      aie.end
    }
  }
}
