//===- bad_trace_tile.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

// A block argument has no defining op, which the generic form can express.
"aie.device"() ({
^bb0(%tile: index):
  // expected-error @+1 {{'aie.trace' op tile operand must be a TileOp or LogicalTileOp}}
  "aie.trace"(%tile) <{sym_name = "tr"}> ({
    "aie.trace.event"() <{event = #aie.trace_event<"INSTR_EVENT_0">}> : () -> ()
    "aie.end"() : () -> ()
  }) : (index) -> ()
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()
