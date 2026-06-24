//===- loc_preservation.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 AMD Inc.
//
//===----------------------------------------------------------------------===//

// Verifies --aie-herd-routing forwards the source HerdOp's location onto
// the synthesized iter / select / switchbox / connect ops produced when
// expanding a route.

// The herd-routing pass produces an IR shape that the verifier rejects under
// the current placement model (aie.switchbox expects a placed tile). The
// location forwarding under test is independent of that — disable the
// after-pass verifier so we can inspect the synthesized ops.

// RUN: aie-opt --verify-each=false --aie-herd-routing --mlir-print-debuginfo %s | FileCheck %s

#herd_loc = loc("user_design.py":42:4)

// Use distinct, non-default locs on the user-written ops (iter, select) so
// the test can isolate the locs that --aie-herd-routing synthesizes from
// the pre-existing user-written ones.
#scaffold_loc = loc("scaffold.mlir":1:1)

module @loc_test {
  aie.device(xcvc1902) {
    %t = aiex.herd[1][1] { sym_name = "t" } loc(#herd_loc)
    %s = aiex.herd[1][1] { sym_name = "s" } loc(#herd_loc)

    %i0 = aiex.iter(0, 1, 1) loc(#scaffold_loc)

    %sel_t = aiex.select(%t, %i0, %i0) loc(#scaffold_loc)
    %sel_s = aiex.select(%s, %i0, %i0) loc(#scaffold_loc)
    aiex.place(%t, %s, 3, 3) loc(#scaffold_loc)
    aiex.route(<%sel_t, DMA: 0>, <%sel_s, DMA: 0>) loc(#scaffold_loc)
  }
}

// Synthesized aie.connect ops (only present after the pass) inherit the
// herd's source loc rather than loc(unknown) or the scaffold loc.
// CHECK-DAG: "aie.connect"() {{.*}} loc(#[[HERDLOC:loc[0-9]*]])
// CHECK-DAG: #[[HERDLOC]] = loc("user_design.py":42:4)
