//===- roundtrip.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file %s | FileCheck %s

// CHECK: aie.device
// CHECK: aiex.npu.dma_wait {symbol = @out0}
aie.device(npu1) {
  memref.global "public" @out0 : memref<16xi32>
  aie.runtime_sequence() {
    aiex.npu.dma_wait {symbol = @out0}
  }
}

// -----

// CHECK: aie.device
// CHECK: aiex.npu.dma_wait {symbol = @out0}
aie.device(npu1) {
  memref.global "public" @out0 : memref<16xi32>
  aie.runtime_sequence() {
    aiex.npu.dma_wait {symbol = @out0}
  }
}

// -----

// CHECK: aie.device
// CHECK: %[[ARGPLUS:.*]] = arith.constant 0 : i32
// CHECK: aiex.npu.address_patch(%[[ARGPLUS]] : i32) {addr = 123 : ui32, arg_idx = 3 : i32}
aie.device(npu1) {
  aie.runtime_sequence() {
    %cst_npu_0 = arith.constant 0 : i32
    aiex.npu.address_patch(%cst_npu_0 : i32) {addr = 123 : ui32, arg_idx = 3 : i32}
  }
}

// -----

// CHECK: aie.device
// CHECK: runtime_sequence @seq(%arg0: memref<1xi32>)
// CHECK-DAG: %[[WADDR:.*]] = arith.constant 432 : i32
// CHECK-DAG: %[[WVAL:.*]] = arith.constant 1 : i32
// CHECK: aiex.npu.write32(%[[WADDR]], %[[WVAL]]) : i32, i32
aie.device(npu1) {
  aie.runtime_sequence @seq(%arg0 : memref<1xi32>) {
    %cst_npu_1 = arith.constant 432 : i32
    %cst_npu_2 = arith.constant 1 : i32
    aiex.npu.write32(%cst_npu_1, %cst_npu_2) : i32, i32
  }
}

// -----

// CHECK: aie.device
// CHECK: aiex.control_packet {address = 1234 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 1 : i32}
// CHECK: aiex.control_packet {address = 5678 : ui32, data = array<i32: 22, 42, 62, 72>, opcode = 2 : i32, stream_id = 7 : i32}
// CHECK: aiex.control_packet {address = 43981 : ui32, length = 3 : i32, opcode = 1 : i32, stream_id = 4 : i32}
aie.device(npu1) {
  aie.runtime_sequence() {
    aiex.control_packet {address = 1234 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 1 : i32}
    aiex.control_packet {address = 5678 : ui32, data = array<i32: 22, 42, 62, 72>, opcode = 2 : i32, stream_id = 7 : i32}
    aiex.control_packet {address = 0xABCD : ui32, length = 3 : i32, opcode = 1 : i32, stream_id = 4 : i32}
  }
}

// -----

// CHECK: aie.device
// CHECK: aiex.npu.load_pdi {id = 4 : i32}
// CHECK: aiex.npu.load_pdi {id = 7 : i32}
// CHECK: aiex.npu.load_pdi {address = 2 : ui64, id = 1 : i32, size = 3 : i32}
aie.device(npu1) {
  memref.global "private" constant @pdi_data : memref<8xi32> = dense<[-1, 1, 256, 1024, 41, 42, 43, 44]>
  aie.runtime_sequence @pdi_loader(%arg0 : memref<?xi32>) {
    %0 = memref.get_global @pdi_data : memref<8xi32>
    aiex.npu.load_pdi {id = 4 : i32}
    aiex.npu.load_pdi {id = 7 : i32}
    aiex.npu.load_pdi {id = 1 : i32, address = 2 : ui64, size = 3 : i32}
  }
}
