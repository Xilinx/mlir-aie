//===- peano_compat_minmax_expand.mlir - arith min/max must be expanded ---===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression test: the core pipeline must keep expanding arith's min/max ops
// into cmp+select. LLVM 24 moved those patterns behind the arith-expand options
// `include-min-max-f` / `include-min-max-i`, which default to false because a
// normal backend prefers the direct llvm.intr.{maxnum,minnum,smax,...}
// lowering. Peano's AIE2 GlobalISel has no rule for the resulting G_FMAXNUM /
// G_FMINNUM, so llc aborts with "unable to legalize instruction" and the whole
// core fails to build. The IR actually handed to Peano's opt must therefore
// carry no min/max intrinsic.

// REQUIRES: peano

// RUN: aiecc --tmpdir %t %s
// RUN: FileCheck %s --input-file %t/peano-compat_main_core_0_2.ll \
// RUN:   --implicit-check-not=llvm.maxnum --implicit-check-not=llvm.minnum

// CHECK: define void @core_0_2()

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %buf_in = aie.buffer(%tile_0_2) {sym_name = "buf_in"} : memref<256xf32>
    %buf_out = aie.buffer(%tile_0_2) {sym_name = "buf_out"} : memref<256xf32>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %lo = arith.constant -1.0 : f32
      %hi = arith.constant 1.0 : f32

      // Scalar f32 clamp: maxnumf then minnumf, the shape a softmax-style
      // reduction lowers to.
      scf.for %i = %c0 to %c256 step %c1 {
        %val = memref.load %buf_in[%i] : memref<256xf32>
        %mx = arith.maxnumf %val, %lo : f32
        %mn = arith.minnumf %mx, %hi : f32
        memref.store %mn, %buf_out[%i] : memref<256xf32>
      }
      aie.end
    }
  }
}
