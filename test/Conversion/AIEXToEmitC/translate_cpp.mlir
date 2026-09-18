//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate %s --aie-npu-to-cpp | FileCheck %s

// End-to-end: the npu ops become a standalone C++ function that assembles the
// TXN vector by calling TxnEncoding.h. The op_count (4) and device-info header
// are statically known and emitted as literals.

// CHECK: #include "aie/Runtime/TxnEncoding.h"
// CHECK: inline std::optional<std::vector<uint32_t>> generate_txn_main_seq() {
// CHECK:   std::vector<uint32_t> txn;
// CHECK:   aie_runtime::txn_init(txn);
// CHECK:   aie_runtime::txn_append_write32(txn,
// CHECK:   aie_runtime::txn_append_maskwrite32(txn,
// CHECK:   aie_runtime::txn_append_maskpoll32(txn,
// CHECK:   uint32_t [[ARR:v[0-9]+]][4] = {0x00000001u, 0x00000002u, 0x00000003u, 0x00000004u};
// CHECK:   aie_runtime::txn_append_blockwrite(txn, {{.*}}, [[ARR]],
// CHECK:   aie_runtime::txn_prepend_header(txn, 4u, {0, 1, 3, 6, 1, 1});
// CHECK:   return std::move(txn);
module {
  aie.device(npu1_1col) {
    memref.global "private" constant @blockdata : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.runtime_sequence @seq(%arg0: memref<8xi32>) {
      %addr = arith.constant 119300 : i32
      %val = arith.constant 42 : i32
      aiex.npu.write32(%addr, %val) : i32, i32
      %mask = arith.constant 255 : i32
      aiex.npu.maskwrite32(%addr, %val, %mask) : i32, i32, i32
      // Shim DMA_MM2S_Status_0, waiting for the 4-deep queue's depth bit to
      // clear -- the shape queue-depth enforcement emits.
      %status = arith.constant 119336 : i32
      %zero = arith.constant 0 : i32
      %depth_bit = arith.constant 4194304 : i32
      aiex.npu.maskpoll(%status, %zero, %depth_bit) : i32, i32, i32
      %d = memref.get_global @blockdata : memref<4xi32>
      aiex.npu.blockwrite(%d) {address = 119300 : ui32} : memref<4xi32>
    }
  }
}
