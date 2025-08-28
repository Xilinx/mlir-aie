//===- tileDMA.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// aie.end is not the last block.

// CHECK: XAie_DmaDesc [[bd0:.*]];
// CHECK: __mlir_aie_try(XAie_DmaDescInit(ctx->XAieDevInst, &([[bd0]]), XAie_TileLoc(8,3)));
// CHECK: __mlir_aie_try(XAie_DmaSetNextBd(&([[bd0]]), {{.*}} 0, {{.*}} 1));
// CHECK: __mlir_aie_try(XAie_DmaWriteBd(ctx->XAieDevInst, &([[bd0]]), XAie_TileLoc(8,3), {{.*}} 0));

// CHECK: XAie_DmaDesc [[bd1:.*]];
// CHECK: __mlir_aie_try(XAie_DmaDescInit(ctx->XAieDevInst, &([[bd1]]), XAie_TileLoc(8,3)));
// CHECK: __mlir_aie_try(XAie_DmaSetNextBd(&([[bd1]]), {{.*}} 1, {{.*}} 1));
// CHECK: __mlir_aie_try(XAie_DmaWriteBd(ctx->XAieDevInst, &([[bd1]]), XAie_TileLoc(8,3), {{.*}} 1));


module @aie_module  {
 aie.device(xcvc1902) {
  %0 = aie.tile(8, 3)
  %24 = aie.buffer(%0) {address = 4096 : i32, sym_name = "buf6"} : memref<64xi32, 2>
  %25 = aie.lock(%0, 0)
  %26 = aie.buffer(%0) {address = 4352 : i32, sym_name = "buf7"} : memref<64xi32, 2>
  %27 = aie.lock(%0, 1)
  %28 = aie.mem(%0)  {
    %38 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    aie.use_lock(%25, Acquire, 0)
    aie.dma_bd(%24 : memref<64xi32, 2>, 0, 64)
    aie.use_lock(%25, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb3
    aie.end
  ^bb3:  // pred: ^bb0
    %39 = aie.dma_start(MM2S, 0, ^bb4, ^bb2)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    aie.use_lock(%27, Acquire, 1)
    aie.dma_bd(%26 : memref<64xi32, 2>, 0, 64)
    aie.use_lock(%27, Release, 0)
    aie.next_bd ^bb4
  }
 }
}
