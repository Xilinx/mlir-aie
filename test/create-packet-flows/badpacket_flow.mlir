//===- badpacket_flow.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s
// CHECK: error{{.*}} 'aie.rule' op can lead to false packet id match for id 28, which is not supposed to pass through this port
// CHECK: remark: Please consider changing all uses of packet id 28 to avoid deadlock.

aie.device(npu1_1col) {
  %03 = aie.tile(0, 3)
  %02 = aie.tile(0, 2)
  %00 = aie.tile(0, 0)
  aie.packet_flow(28) {
    aie.packet_source<%00, DMA : 0>
    aie.packet_dest<%02, Ctrl : 0>
  }
  aie.packet_flow(29) {
    aie.packet_source<%00, DMA : 0>
    aie.packet_dest<%03, Ctrl : 0>
  }
  aie.packet_flow(26) {
    aie.packet_source<%00, DMA : 0>
    aie.packet_dest<%03, DMA : 0>
  }
}
