//===- test_lock_init.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
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
