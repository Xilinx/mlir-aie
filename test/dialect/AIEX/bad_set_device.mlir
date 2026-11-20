//===- bad_set_device.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file --verify-diagnostics

module {
  aie.device(npu2) @cfg {
  }
  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.set_device @cfg
      %c0 = arith.constant 0 : i32
      aiex.control_packet {address = 0 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-error@+1 {{@missing is not a device}}
      aiex.set_device @missing
    }
  }
}

// -----

module {
  func.func private @not_device()
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-error@+1 {{@not_device is not a device}}
      aiex.set_device @not_device
    }
  }
}

// -----

module {
  aie.device(npu2) @cfg {
  }
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-error@+1 {{'aiex.set_device' op requires zero regions}}
      "aiex.set_device"() ({}) {device_ref = @cfg} : () -> ()
    }
  }
}