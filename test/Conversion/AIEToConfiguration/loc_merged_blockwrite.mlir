// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- loc_merged_blockwrite.mlir ---------------------*- MLIR -*-===//
//
//===----------------------------------------------------------------------===//

// aie-rt's transaction serializer collapses a run of address-contiguous
// XAIE_IO_BLOCKWRITE commands into one block-write operation, so recorded
// commands and serialized operations are not 1:1. AIERTControl models that in
// projectCmdLocsOntoSerializedOps; this checks the model holds: the two
// adjacent initialized buffers below become a single 8-word block write
// carrying the *first* buffer's loc, and the lock write32 that follows still
// carries its own loc rather than one shifted by the merge.

// RUN: aie-opt --convert-aie-to-transaction="elf-dir=%S/convert_aie_to_ctrl_pkts_elfs/" --mlir-print-debuginfo %s | FileCheck %s

#device_loc = loc("device.mlir":1:1)
#buf_a_loc = loc("user_program.py":10:1)
#buf_b_loc = loc("user_program.py":20:2)
#lock_loc = loc("user_program.py":30:3)

aie.device(npu1_1col) {
  %t02 = aie.tile(0, 2)
  %a = aie.buffer(%t02) { sym_name = "a", address = 1024 : i32,
        initial_value = dense<[1, 2, 3, 4]> : tensor<4xi32> } : memref<4xi32> loc(#buf_a_loc)
  %b = aie.buffer(%t02) { sym_name = "b", address = 1040 : i32,
        initial_value = dense<[5, 6, 7, 8]> : tensor<4xi32> } : memref<4xi32> loc(#buf_b_loc)
  %lock = aie.lock(%t02, 0) { init = 1 : i32, sym_name = "lk" } loc(#lock_loc)
  %core = aie.core(%t02) {
    aie.end
  } { elf_file = "core_0_2.elf" }
} loc(#device_loc)

// CHECK-DAG: #[[BUFALOC:loc[0-9]*]] = loc("user_program.py":10:1)
// CHECK-DAG: #[[LOCKLOC:loc[0-9]*]] = loc("user_program.py":30:3)

// Both buffers' initial values land in one block write, located at buffer a.
// CHECK-DAG: memref.global "private" constant @[[DATA:[a-zA-Z0-9_]+]] : memref<8xi32> = dense<[1, 2, 3, 4, 5, 6, 7, 8]>
// CHECK-DAG: aiex.npu.blockwrite({{.*}}) {address = 2098176 : ui32} : memref<8xi32> loc(#[[BUFALOC]])

// CHECK-DAG: aiex.npu.write32{{.*}}loc(#[[LOCKLOC]])
