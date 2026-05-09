//===- loc_roundtrip.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Verifies that --convert-aie-to-transaction preserves source op locations
// across the aie-rt round-trip. AIERTControl::TxnLocBracket scopes capture
// the lock op's loc around the XAie_LockSetValue call; AIEToConfiguration
// then reads ctl.getTxnInstrLocs() and applies those locations to the
// re-emitted aiex.npu.write32 ops instead of the device fallback location.

// RUN: aie-opt --convert-aie-to-transaction="elf-dir=%S/convert_aie_to_ctrl_pkts_elfs/" --mlir-print-debuginfo %s | FileCheck %s

#device_loc = loc("device.mlir":1:1)
#user_lock_loc = loc("user_program.py":17:8)
#user_lock_name = loc("of_in_lock0"(#user_lock_loc))

aie.device(npu1_1col) {
  %t02 = aie.tile(0, 2)
  %lock = aie.lock(%t02, 0) { init = 1 : i32, sym_name = "lk" } loc(#user_lock_name)
  %4 = aie.core(%t02) {
    aie.end
  } { elf_file = "core_0_2.elf" }
} loc(#device_loc)

// At least one synthesized aiex.npu.write32 (from XAie_LockSetValue) carries
// the IRON-style NameLoc pointing at user_program.py:17, NOT the device loc
// fallback. The bracketing in AIERT::initLocks attributes the cmds produced
// by the explicit-init lock walk to the LockOp's loc.
// CHECK-DAG: #[[USERLOC:loc[0-9]*]] = loc("user_program.py":17:8)
// CHECK-DAG: #[[LOCKLOC:loc[0-9]*]] = loc("of_in_lock0"(#[[USERLOC]]))
// CHECK-DAG: aiex.npu.write32{{.*}}loc(#[[LOCKLOC]])
