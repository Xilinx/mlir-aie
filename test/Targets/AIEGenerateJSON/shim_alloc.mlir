//===- shim_alloc.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: May 30th 2023
// 
//===----------------------------------------------------------------------===//

// RUN: aie-translate -aie-generate-json %s | FileCheck %s

// CHECK: {
// CHECK:   "of_in_0": {
// CHECK:     "channelDir": 1,
// CHECK:     "channelIndex": 0,
// CHECK:     "col": 2
// CHECK:   },
// CHECK:   "of_in_1": {
// CHECK:     "channelDir": 1,
// CHECK:     "channelIndex": 1,
// CHECK:     "col": 2
// CHECK:   },
// CHECK:   "of_out_0": {
// CHECK:     "channelDir": 0,
// CHECK:     "channelIndex": 0,
// CHECK:     "col": 2
// CHECK:   },
// CHECK:   "of_out_1": {
// CHECK:     "channelDir": 0,
// CHECK:     "channelIndex": 1,
// CHECK:     "col": 2
// CHECK:   }
// CHECK: }

module @alloc {
  AIE.device(xcve2302) {
    AIE.shimDMAAllocation(@of_out_1, S2MM, 1, 2)
    AIE.shimDMAAllocation(@of_in_1, MM2S, 1, 2)
    AIE.shimDMAAllocation(@of_out_0, S2MM, 0, 2)
    AIE.shimDMAAllocation(@of_in_0, MM2S, 0, 2)
  }
}