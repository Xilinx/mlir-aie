//===- bad_set_device.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file --verify-diagnostics
// RUN: aie-opt %s --split-input-file -aie-unroll-runtime-sequence-loops -canonicalize -aie-assign-runtime-sequence-bd-ids --verify-diagnostics

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

// -----

module {
  aie.device(npu2) @cfg {}
  aie.device(npu2) {
    aie.runtime_sequence() {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      scf.for %iteration = %zero to %one step %one {
        // expected-error@+1 {{'aiex.set_device' op expects parent op 'aie.runtime_sequence'}}
        aiex.set_device @cfg
      }
    }
  }
}

// -----

module {
  aie.device(npu2) @cfg {}
  aie.device(npu2) {
    aie.runtime_sequence() {
      %condition = arith.constant true
      scf.if %condition {
        // expected-error@+1 {{'aiex.set_device' op expects parent op 'aie.runtime_sequence'}}
        aiex.set_device @cfg
      }
    }
  }
}

// -----

module {
  aie.device(npu2) @cfg {}
  aie.device(npu2) {
    aie.runtime_sequence() {
      cf.br ^later
    ^later:
      // expected-error@+1 {{'aiex.set_device' op requires a single-block runtime sequence}}
      aiex.set_device @cfg
      cf.br ^later
    }
  }
}

// -----

module {
  aie.device(npu2_1col) @outer {
    %shim = aie.tile(0, 0)
    %data = aie.external_buffer : memref<16xi32>
    aie.shim_dma(%shim) {
      aie.dma_start(S2MM, 1, ^descriptor, ^end)
    ^descriptor:
      aie.dma_bd(%data : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^descriptor
    ^end:
      aie.end
    }
  }
  aie.device(npu2_1col) @inner {}
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.runtime_sequence(%data : memref<16xi32>) {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      aiex.set_device @outer
      scf.for %iteration = %zero to %one step %one {
        // expected-error@+1 {{'aiex.set_device' op expects parent op 'aie.runtime_sequence'}}
        aiex.set_device @inner
      }
      %task = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%data : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
      aiex.dma_await_task(%task)
    }
  }
}