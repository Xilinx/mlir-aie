//===- test_insert_trace_flows_option_ranges.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The trace DMA uses a shim S2MM channel (0 or 1) and a shim BD. Options
// outside those ranges are rejected up front instead of producing an invalid
// packet destination or aborting while encoding the task queue push.

// RUN: not aie-opt %s -aie-insert-trace-flows="shim-channel=2" 2>&1 | FileCheck %s --check-prefix=CHAN2
// RUN: not aie-opt %s -aie-insert-trace-flows="shim-channel=-1" 2>&1 | FileCheck %s --check-prefix=CHANNEG
// RUN: not aie-opt %s -aie-insert-trace-flows="default-bd-id=16" 2>&1 | FileCheck %s --check-prefix=BD16
// RUN: not aie-opt %s -aie-insert-trace-flows="default-bd-id=-1" 2>&1 | FileCheck %s --check-prefix=BDNEG

// CHAN2: error: shim-channel 2 is not a shim S2MM channel; expected 0 or 1
// CHANNEG: error: shim-channel -1 is not a shim S2MM channel; expected 0 or 1
// BD16: error: default-bd-id 16 is not a shim BD; expected [0, 16)
// BDNEG: error: default-bd-id -1 is not a shim BD; expected [0, 16)

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.trace @core_trace(%tile) {
      aie.trace.packet id = 10 type = core
      aie.trace.event <"INSTR_EVENT_0">
      aie.trace.start broadcast = 3
      aie.trace.stop broadcast = 4
    }
    aie.runtime_sequence(%arg0: memref<8xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @core_trace
    }
  }
}
