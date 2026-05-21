//===- loc_preservation.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 AMD Inc.
//
//===----------------------------------------------------------------------===//

// Verifies --convert-aie-to-control-packets attaches the source DeviceOp's
// loc to the synthesized aie.runtime_sequence and aiex.control_packet ops.

// RUN: aie-opt --convert-aie-to-control-packets="elf-dir=%S/convert_aie_to_ctrl_pkts_elfs/" --mlir-print-debuginfo %s | FileCheck %s

#device_loc = loc("user_design.py":42:4)

aie.device(npu1_1col) {
  %12 = aie.tile(0, 2)
  %buf = aie.buffer(%12) : memref<256xi32>
  %4 = aie.core(%12) {
    aie.end
  } { elf_file = "core_0_2.elf" }
} loc(#device_loc)

// At least one synthesized aiex.control_packet inside the new
// runtime_sequence carries the device's loc, and the runtime_sequence
// itself does too. (The pass creates the runtime_sequence on demand.)
// CHECK-DAG: aie.runtime_sequence @configure() {{[{][[:space:]]*$}}
// CHECK-DAG: aiex.control_packet {{.*}} loc(#[[DEVLOC:loc[0-9]*]])
// CHECK-DAG: #[[DEVLOC]] = loc("user_design.py":42:4)
