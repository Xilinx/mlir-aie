// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: not aie-opt --aie-npu-to-cert --split-input-file %s 2>&1 | FileCheck %s

// CHECK: error: 'aiex.npu.maskpoll' op cannot lower to cert.maskpoll32 with non-constant address, mask, or value
aie.device(npu2) {
  aie.runtime_sequence @configure(%address: i32) {
    %zero = arith.constant 0 : i32
    %mask = arith.constant 0x1000000 : i32
    aiex.npu.maskpoll(%address, %zero, %mask) : i32, i32, i32
  }
}

// -----

// CHECK: error: 'aiex.npu.maskpoll' op cannot lower to cert.maskpoll32 with non-constant address, mask, or value
aie.device(npu2) {
  aie.runtime_sequence @configure(%value: i32) {
    %address = arith.constant 0x1d220 : i32
    %mask = arith.constant 0x1000000 : i32
    aiex.npu.maskpoll(%address, %value, %mask) : i32, i32, i32
  }
}

// -----

// CHECK: error: 'aiex.npu.maskpoll' op cannot lower to cert.maskpoll32 with non-constant address, mask, or value
aie.device(npu2) {
  aie.runtime_sequence @configure(%mask: i32) {
    %address = arith.constant 0x1d220 : i32
    %zero = arith.constant 0 : i32
    aiex.npu.maskpoll(%address, %zero, %mask) : i32, i32, i32
  }
}

// -----

// CHECK: error: referenced buffer must have address assigned
aie.device(npu2) {
  %tile = aie.tile(0, 3)
  %buffer = aie.buffer(%tile) {sym_name = "status"} : memref<1xi32>
  aie.runtime_sequence @configure() {
    %zero = arith.constant 0 : i32
    %mask = arith.constant 0x1000000 : i32
    aiex.npu.maskpoll(%zero, %zero, %mask) {buffer = @status} : i32, i32, i32
  }
}
