// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-dma-to-npu %s -o %t
// RUN: FileCheck %s < %t
// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false %t > /dev/null
// RUN: aie-translate --aie-npu-to-cpp %t | FileCheck %s --check-prefix=CPP

// CHECK-LABEL: aie.runtime_sequence
// CHECK-DAG: %[[VAL:.*]] = arith.constant 321 : i32
// CHECK-DAG: %[[MASK:.*]] = arith.constant 65535 : i32
// CHECK-DAG: %[[ADDR:.*]] = arith.constant 3147552 : i32
// CHECK: aiex.npu.maskpoll(%[[ADDR]], %[[VAL]], %[[MASK]]) : i32, i32, i32
// CPP: aie_runtime::txn_append_maskpoll32(
module {
  aie.device(npu1_1col) {
    %tile = aie.tile(0, 3)
    %buffer = aie.buffer(%tile) {address = 1024 : i32, sym_name = "status"} : memref<128xi32>
    aie.runtime_sequence() {
      %offset = arith.constant 200 : i32
      %value = arith.constant 321 : i32
      %mask = arith.constant 65535 : i32
      aiex.npu.maskpoll(%offset, %value, %mask) {buffer = @status} : i32, i32, i32
    }
  }
}
