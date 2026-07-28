//===- loc_preservation.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Verifies that --aie-objectFifo-stateful-transform forwards the source
// objectfifo.create op's Location onto the synthesized buffer / lock ops.
// See docs/SourceLocations.md for the broader plan.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=false" --aie-assign-lock-ids --mlir-print-debuginfo %s | FileCheck %s

#user_loc = loc("user_design.py":42:4)

module @loc_test {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2) loc(unknown)
    %tile13 = aie.tile(1, 3) loc(unknown)
    aie.objectfifo @of0 (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32>> loc(#user_loc)
 }
}

// MLIR pretty-prints with location aliases (loc(#locN)) at the end of the
// module. Capture the alias used on the source objectfifo, then assert that
// the synthesized buffers / locks reference the same alias and that the
// alias resolves to user_design.py:42:4 (and is *not* loc(unknown)).

// CHECK: aie.buffer({{.*}}) {sym_name = "of0_buff_0"} : memref<16xi32>{{ +}}loc(#[[USERLOC:loc[0-9]*]])
// CHECK: aie.buffer({{.*}}) {sym_name = "of0_buff_1"} : memref<16xi32>{{ +}}loc(#[[USERLOC]])
// CHECK: aie.lock({{.*}}) {init = 4 : i32, sym_name = "of0_prod_lock_0"} loc(#[[USERLOC]])
// CHECK: aie.lock({{.*}}) {init = 0 : i32, sym_name = "of0_cons_lock_0"} loc(#[[USERLOC]])
// CHECK: #[[USERLOC]] = loc("user_design.py":42:4)
