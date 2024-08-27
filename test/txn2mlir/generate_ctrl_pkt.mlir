//===- generate_ctrl_pkt.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate -aie-npu-instgen -aie-output-binary=true %s -o ./generate_ctrl_pkt_cfg.bin
// RUN: %python txn2mlir.py -f ./generate_ctrl_pkt_cfg.bin -generate-ctrl-pkt | FileCheck %s

// CHECK: aie.device(npu1_1col)
// CHECK: memref.global "private" constant @blockwrite_data : memref<6xi32> = dense<[4195328, 0, 0, 0, 0, 100941792]>
// CHECK: aiex.npu.control_packet {address = 2301952 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
// CHECK: aiex.npu.control_packet {address = 2224128 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
// CHECK: aiex.npu.control_packet {address = 2215936 : ui32, data = array<i32: 4195328, 0, 0, 0>, opcode = 0 : i32, stream_id = 0 : i32}
// CHECK: aiex.npu.control_packet {address = 2215952 : ui32, data = array<i32: 0, 100941792>, opcode = 0 : i32, stream_id = 0 : i32}
module {
  aie.device(npu1_1col) {
    memref.global "private" constant @blockwrite_data : memref<6xi32> = dense<[4195328, 0, 0, 0, 0, 100941792]>
    aiex.runtime_sequence() {
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 2224128 : ui32, value = 2 : ui32}
      %2 = memref.get_global @blockwrite_data : memref<6xi32>
      aiex.npu.blockwrite(%2) {address = 2215936 : ui32} : memref<6xi32>
    }
  }
}
