//===- shim_axcache.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// A configured $axcache reaches the Cache argument of XAie_DmaSetAxi; an unset
// one falls back to AIETargetModel::getDefaultAxCache() == 2.

// CHECK-LABEL: int mlir_aie_configure_shimdma_70(aie_libxaie_ctx_t* ctx) {
// CHECK: XAie_DmaSetAxi(&(dma_tile70_bd0), /* smid */ 0, /* burstlen */ 4, /* QoS */ 0, /* Cache */ 15, /* Secure */ XAIE_ENABLE)
// CHECK: XAie_DmaSetAxi(&(dma_tile70_bd1), /* smid */ 0, /* burstlen */ 4, /* QoS */ 0, /* Cache */ 2, /* Secure */ XAIE_ENABLE)

module {
 aie.device(xcvc1902) {
  %buf = aie.external_buffer { sym_name = "buf" } : memref<32x32xi32>

  %tile70 = aie.tile(7, 0)
  %lock70 = aie.lock(%tile70, 0)

  %shimdma70 = aie.shim_dma(%tile70)  {
    aie.dma_start(MM2S, 0, ^bb1, ^bb3)
  ^bb1:
    %c1_ul1 = arith.constant 1 : i32
    aie.use_lock(%lock70, Acquire, %c1_ul1)
    aie.dma_bd(%buf : memref<32x32xi32> offset = 0 len = 1024) { axcache = 15 : i32 }
    %c0_ul2 = arith.constant 0 : i32
    aie.use_lock(%lock70, Release, %c0_ul2)
    aie.next_bd ^bb2
  ^bb2:
    %c1_ul3 = arith.constant 1 : i32
    aie.use_lock(%lock70, Acquire, %c1_ul3)
    aie.dma_bd(%buf : memref<32x32xi32> offset = 0 len = 1024)
    %c0_ul4 = arith.constant 0 : i32
    aie.use_lock(%lock70, Release, %c0_ul4)
    aie.next_bd ^bb1
  ^bb3:
    aie.end
  }
 }
}
