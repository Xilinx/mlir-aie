//===- badmemtile_circ_sw.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
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
