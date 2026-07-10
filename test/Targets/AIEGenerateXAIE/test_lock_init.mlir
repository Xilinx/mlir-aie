//===- test_lock_init.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s
// CHECK: __mlir_aie_try(XAie_LockSetValue(ctx->XAieDevInst, XAie_TileLoc(3,3), XAie_LockInit(0, 1)));

module @test_lock_init {
 aie.device(xcvc1902) {
  %t33 = aie.tile(3, 3)
  %l33_0 = aie.lock(%t33, 0) { init = 1 : i32 }
 }
}
