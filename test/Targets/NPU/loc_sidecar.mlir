//===- loc_sidecar.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Verifies aie-translate --aie-npu-to-binary --aie-npu-emit-locmap=<file> writes
// a JSON sidecar (to <file>) keying each transaction word's byte_offset to the
// source aiex.npu.* op's MLIR Location, including IRON-style NameLoc names from
// Phase 1 capture. The main output still holds the binary.

// RUN: aie-translate --aie-npu-to-binary --aie-npu-emit-locmap=%t.json %s
// RUN: FileCheck %s < %t.json

#user_w32 = loc("user.py":42:4)
#user_bw  = loc("user.py":50:4)
#user_zero = loc("user.py":60:4)
#name_w32 = loc("of_in"(#user_w32))
#name_bw  = loc("dma_task"(#user_bw))
#name_zero = loc("zero_reg"(#user_zero))

module {
  aie.device(npu1) {
    memref.global "private" constant @write_data : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.runtime_sequence(%arg0: memref<16xf32>) {
      aiex.npu.write32 { address = 0xabc00def : ui32, value = 0x42 : ui32 } loc(#name_w32)
      aiex.npu.write32 { address = 0x0 : ui32, value = 0x1 : ui32 } loc(#name_zero)
      %0 = memref.get_global @write_data : memref<4xi32>
      aiex.npu.blockwrite (%0) { address = 0x12345678 : ui32 } : memref<4xi32> loc(#name_bw)
    }
  }
}

// JSON header
// CHECK-DAG: "version": 1
// CHECK-DAG: "device":
// CHECK-DAG: "operations":

// Per-op entries: opcode, source op name, address, source location with
// the IRON NameLoc carrying the user file:line.
// CHECK-DAG: "opcode": "WRITE32"
// CHECK-DAG: "source_op": "aiex.npu.write32"
// CHECK-DAG: "address": "0xABC00DEF"
// CHECK-DAG: "name": "of_in"
// CHECK-DAG: "file": "user.py"
// CHECK-DAG: "line": 42

// Address zero is a real address, not the "no address" sentinel.
// CHECK-DAG: "address": "0x0"
// CHECK-DAG: "name": "zero_reg"
// CHECK-DAG: "line": 60

// CHECK-DAG: "opcode": "BLOCKWRITE"
// CHECK-DAG: "source_op": "aiex.npu.blockwrite"
// CHECK-DAG: "address": "0x12345678"
// CHECK-DAG: "name": "dma_task"
// CHECK-DAG: "line": 50
