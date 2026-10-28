//===- pdi_deterministic.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bootgen leaves the image header's metaheader revoke_id uninitialized unless
// the BIF sets it. MALLOC_PERTURB_ fills fresh heap allocations with a chosen
// byte, so two fills produce different PDIs if any uninitialized heap reaches
// the output.

// RUN: env MALLOC_PERTURB_=1 %aiecc --get-pdi --pdi-name=a.pdi --output-dir=%t.a --tmpdir=%t.a.prj %s
// RUN: env MALLOC_PERTURB_=254 %aiecc --get-pdi --pdi-name=b.pdi --output-dir=%t.b --tmpdir=%t.b.prj %s
// RUN: cmp %t.a/a.pdi %t.b/b.pdi

module {
  aie.device(npu2) {
    %tile = aie.tile(1, 2)
    %buf = aie.buffer(%tile) : memref<256xi32>
  }
}
