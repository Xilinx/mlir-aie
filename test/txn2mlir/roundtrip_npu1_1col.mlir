//===- roundtrip_npu1_1col.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate -aie-npu-instgen -aie-output-binary=true %s -o ./roundtrip_npu1_1col_cfg.bin
// RUN: %python txn2mlir.py -f ./roundtrip_npu1_1col_cfg.bin | FileCheck %s

// CHECK: aie.device(npu1_1col)
// CHECK: memref.global "private" constant @blockwrite_data : memref<2xi32> = dense<[4195328, 0]>
// CHECK: aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 2 : ui32, value = 2 : ui32}
// CHECK: aiex.npu.write32 {address = 2224128 : ui32, value = 2 : ui32}
// CHECK: aiex.npu.blockwrite(%0) {address = 2215936 : ui32} : memref<2xi32>
module {
  aie.device(npu1_1col) {
    memref.global "private" constant @blockwrite_data : memref<2xi32> = dense<[4195328, 0]>
    aiex.runtime_sequence() {
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 2224128 : ui32, value = 2 : ui32}
      %0 = memref.get_global @blockwrite_data : memref<2xi32>
      aiex.npu.blockwrite(%0) {address = 2215936 : ui32} : memref<2xi32>
    }
  }
}
