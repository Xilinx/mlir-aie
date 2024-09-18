//===- convert_aie_to_ctrl_pkts.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -convert-aie-to-control-packets="elf-dir=%S/convert_aie_to_ctrl_pkts_elfs/" %s | FileCheck %s

// CHECK-label: aiex.runtime_sequence @configure
// CHECK-COUNT-5: aiex.control_packet {address = {{.*}} : ui32, data = array<i32: {{.*}}>
// CHECK-COUNT-2: aiex.control_packet {address = {{.*}} : ui32, data = array<i32: {{.*}}, {{.*}}, {{.*}}, {{.*}}>
// CHECK: aiex.control_packet {address = {{.*}} : ui32, data = array<i32: {{.*}}>
// CHECK-COUNT-36: aiex.control_packet {address = {{.*}} : ui32, data = array<i32: {{.*}}, {{.*}}, {{.*}}, {{.*}}>
// CHECK-COUNT-22: aiex.control_packet {address = {{.*}} : ui32, data = array<i32: {{.*}}>
aie.device(npu1_1col) {
  %12 = aie.tile(0, 2)
  %buf = aie.buffer(%12) : memref<256xi32>
  %4 = aie.core(%12)  {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 0 : index
    memref.store %0, %buf[%1] : memref<256xi32>
    aie.end
  }
}
