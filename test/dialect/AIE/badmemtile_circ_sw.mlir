//===- badmemtile_circ_sw.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s |& FileCheck %s
// CHECK: error{{.*}} 'AIE.connect' op illegal memtile stream switch connection

AIE.device(xcve2802) {
  %01 = AIE.tile(0, 1)
  AIE.switchbox(%01) {
    AIE.connect<North: 0, South: 1>
  }
}
