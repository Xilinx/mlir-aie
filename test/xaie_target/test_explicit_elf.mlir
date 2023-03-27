//===- test_explicit_elf.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: mlir_aie_configure_cores
// CHECK: XAieTile_CoreControl(&(ctx->TileInst[3][3]), XAIE_DISABLE, XAIE_ENABLE);
// CHECK: XAieGbl_LoadElf(&(ctx->TileInst[3][3]), (u8*)"test.elf", XAIE_ENABLE);
// CHECK: mlir_aie_start_cores
// CHECK: XAieTile_CoreControl(&(ctx->TileInst[3][3]), XAIE_ENABLE, XAIE_DISABLE);
module @test_xaie0 {
 AIE.device(xcvc1902) {
  %t33 = AIE.tile(3, 3)
  AIE.core(%t33) {
    AIE.end
  } { elf_file = "test.elf" } 
 }
}
