//===- multi_source_packet_flow.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// Regression test: a packet_flow with multiple aie.packet_source ops is valid
// (fan-in: multiple source ports emit packets with the same flow ID).
// Verifier must not reject this. Regression for issue #2583 / PR #2919.

// CHECK-LABEL: module @multi_source_packet_flow
module @multi_source_packet_flow {
  aie.device(xcvc1902) {
    %t11 = aie.tile(1, 1)
    %t21 = aie.tile(2, 1)
    %t31 = aie.tile(3, 1)
    // CHECK: aie.packet_flow(0)
    aie.packet_flow(0x0) {
      aie.packet_source<%t11, West : 0>
      aie.packet_source<%t21, West : 0>
      aie.packet_dest<%t31, Core : 0>
    }
  }
}
