//===- peano_compat_align_strip.mlir - downgradeIRForPeano align strip ----===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Regression test: downgradeIRForPeano must strip ', align <N>' attributes from
// the per-core LLVM IR. Retaining them makes Peano's capped-O1 opt skip
// vectorizing the reduction loop, scalarizing it into ~10x more program memory
// and overflowing AIE core program memory. The generated *.peano-compat.ll
// (the IR actually handed to Peano's opt) must contain no align attributes,
// while still being a real core function body.

// REQUIRES: peano

// RUN: aiecc --no-xchesscc --no-xbridge --tmpdir %t %s
// RUN: FileCheck %s --input-file %t/main_core_0_2.peano-compat.ll --implicit-check-not=", align "

// CHECK: define void @core_0_2()

module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %buf_a = aie.buffer(%tile_0_2) {sym_name = "buf_a"} : memref<256xbf16>
    %buf_b = aie.buffer(%tile_0_2) {sym_name = "buf_b"} : memref<256xbf16>
    %buf_c = aie.buffer(%tile_0_2) {sym_name = "buf_c"} : memref<256xbf16>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.0 : bf16

      scf.for %i = %c0 to %c256 step %c16 {
        %a = vector.transfer_read %buf_a[%i], %cst {in_bounds = [true]} : memref<256xbf16>, vector<16xbf16>
        %b = vector.transfer_read %buf_b[%i], %cst {in_bounds = [true]} : memref<256xbf16>, vector<16xbf16>
        %c = arith.addf %a, %b : vector<16xbf16>
        vector.transfer_write %c, %buf_c[%i] {in_bounds = [true]} : vector<16xbf16>, memref<256xbf16>
      }
      aie.end
    }
  }
}
