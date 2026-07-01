//===- badmemtile_circ_sw.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}} 'aie.connect' op illegal stream switch connection

aie.device(xcve2802) {
  %01 = aie.tile(0, 1)
  aie.switchbox(%01) {
    aie.connect<North: 0, South: 1>
  }
}
