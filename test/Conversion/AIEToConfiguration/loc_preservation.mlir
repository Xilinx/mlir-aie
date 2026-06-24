//===- loc_preservation.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 AMD Inc.
//
//===----------------------------------------------------------------------===//

// Verifies --convert-aie-to-control-packets propagates source op locations
// onto the synthesized aiex.control_packet ops via AIERTControl's
// instruction-range bracketing. The runtime_sequence itself inherits the
// device loc when synthesized on demand.

// RUN: aie-opt --convert-aie-to-control-packets="elf-dir=%S/convert_aie_to_ctrl_pkts_elfs/" --mlir-print-debuginfo %s | FileCheck %s

#device_loc = loc("user_design.py":42:4)
#core_user_loc = loc("user_design.py":80:4)
#core_named = loc("worker_2"(#core_user_loc))

aie.device(npu1_1col) {
  %12 = aie.tile(0, 2)
  %buf = aie.buffer(%12) : memref<256xi32>
  %4 = aie.core(%12) {
    aie.end
  } { elf_file = "core_0_2.elf" } loc(#core_named)
} loc(#device_loc)

// runtime_sequence is synthesized on demand and inherits the device loc.
// CHECK-DAG: aie.runtime_sequence @configure() {{[{][[:space:]]*$}}

// At least one synthesized control_packet carries the CoreOp's NameLoc —
// produced by AIERTControl::addCoreEnable's TxnLocBracket around the
// XAie_CoreEnable call.
// CHECK-DAG: #[[USERLOC:loc[0-9]*]] = loc("user_design.py":80:4)
// CHECK-DAG: #[[CORELOC:loc[0-9]*]] = loc("worker_2"(#[[USERLOC]]))
// CHECK-DAG: aiex.control_packet {{.*}} loc(#[[CORELOC]])
