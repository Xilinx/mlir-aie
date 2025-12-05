//===- bad_release.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// CHECK:   error: 'aie.objectfifo' op cannot release from objectfifo stream port

module @bad_release {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2) 
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of_stream (%tile12, {%tile33}, 2 : i32) {aie_stream = 1 : i32, aie_stream_port = 0 : i32} : !aie.objectfifo<memref<16xi32>>

    %core33 = aie.core(%tile33) {
      aie.objectfifo.release @of_stream (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.end
    }
  }
}
