//===- test_explicit_elf.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: mlir_aie_configure_cores
// CHECK: __mlir_aie_try(XAie_CoreReset(ctx->XAieDevInst, XAie_TileLoc(3,3)));
// CHECK: __mlir_aie_try(XAie_CoreDisable(ctx->XAieDevInst, XAie_TileLoc(3,3)));
// CHECK: XAie_LoadElf(ctx->XAieDevInst, XAie_TileLoc(3,3), (const char*)"test.elf",0);
// CHECK: mlir_aie_start_cores
// CHECK: __mlir_aie_try(XAie_CoreUnreset(ctx->XAieDevInst, XAie_TileLoc(3,3)));
// CHECK: __mlir_aie_try(XAie_CoreEnable(ctx->XAieDevInst, XAie_TileLoc(3,3)));

module @test_xaie0 {
 aie.device(xcvc1902) {
  %t33 = aie.tile(3, 3)
  aie.core(%t33) {
    aie.end
  } { elf_file = "test.elf" }
 }
}
