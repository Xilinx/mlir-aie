//===- badmemtile_pkt_sw.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}} 'aie.amsel' op illegal stream switch connection

aie.device(xcve2802) {
  %01 = aie.tile(0, 1)
  aie.switchbox(%01) {
    %94 = aie.amsel<0> (0)
    %95 = aie.masterset(North : 1, %94)
    aie.packet_rules(South : 3) {
      aie.rule(31, 1, %94)
    }
  }
}
