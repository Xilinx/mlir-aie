//===- test_insert_trace_flows_options.mlir -------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows="shim-channel=0" | FileCheck %s --check-prefix=CHAN0
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows="shim-channel=1" | FileCheck %s --check-prefix=CHAN1
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows="packet-id-start=10" | FileCheck %s --check-prefix=PKTID10
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows | FileCheck %s --check-prefix=DEFAULT

// Test: shim-channel and packet-id-start CLI options
module @options_single {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)
    aie.trace @core_trace(%tile02) {
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @core_trace
    }
    // CHAN0: aie.packet_dest<%{{.*}}, DMA : 0>
    // CHAN1: aie.packet_dest<%{{.*}}, DMA : 1>
    // PKTID10: aie.packet_flow(10)
    // DEFAULT: aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
  }
}

// -----

// Test: Default routing=single routes all traces to column 0 shim
// DEFAULT-LABEL: module @multi_column_single
module @multi_column_single {
  aie.device(npu1_2col) {
    %tile02 = aie.tile(0, 2)
    %tile12 = aie.tile(1, 2)
    %tile00 = aie.tile(0, 0)
    %tile10 = aie.tile(1, 0)
    aie.trace @trace_col0(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.trace @trace_col1(%tile12) {
      aie.trace.packet id=2 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @trace_col0
    }
    // DEFAULT: aiex.npu.writebd {{{.*}}column = 0{{.*}}}
    // DEFAULT: aiex.npu.address_patch
    // DEFAULT: aiex.npu.maskwrite32
    // DEFAULT-NOT: aiex.npu.writebd
    // DEFAULT-NOT: aiex.npu.address_patch
    // DEFAULT-NOT: aiex.npu.maskwrite32
    // Both traces go to column 0 shim
    // DEFAULT-DAG: aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    // DEFAULT-DAG: aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
  }
}

