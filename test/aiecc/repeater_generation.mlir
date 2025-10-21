//===- fallback_routine_error.mlir -----------------------------*- MLIR -*-===//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %python aiecc.py --repeater-output-dir=. %s 2>&1 | FileCheck %s
// RUN: cat ./aiecc_repeater_*.sh | FileCheck --check-prefix=DIAG %s
// RUN: cat ./aiecc_failure_*.mlir | FileCheck --check-prefix=MLIR %s

// CHECK: AIECC COMPILATION FAILED
// CHECK: Intermediate MLIR saved to:
// CHECK: Repeater script generated:

// DIAG: 'aie.rule' op can lead to false packet id match for id 28, which is not supposed to pass through this port
// DIAG: PASS_PIPELINE='builtin.module(aie.device(aie-create-pathfinder-flows))'
// MLIR: aie.packet_flow(28)

// from test/create-packet-flows/badpacket_flow.mlir
aie.device(npu1_1col) {
  %03 = aie.tile(0, 3)
  %02 = aie.tile(0, 2)
  %00 = aie.tile(0, 0)
  aie.packet_flow(28) {
    aie.packet_source<%00, DMA : 0>
    aie.packet_dest<%02, TileControl : 0>
  }
  aie.packet_flow(29) {
    aie.packet_source<%00, DMA : 0>
    aie.packet_dest<%03, TileControl : 0>
  }
  aie.packet_flow(26) {
    aie.packet_source<%00, DMA : 0>
    aie.packet_dest<%03, DMA : 0>
  }
}
