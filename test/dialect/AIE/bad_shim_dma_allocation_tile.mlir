//===- bad_shim_dma_allocation_tile.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// The tile operand of aie.shim_dma_allocation is an index, so any index value
// parses there. One that is not an aie.tile or aie.logical_tile is rejected.

aie.device(npu1) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{'aie.shim_dma_allocation' op tile operand must implement TileLike interface}}
  aie.shim_dma_allocation @alloc(%c0, MM2S, 0)
}

// -----

// A block argument has no defining op, which the generic form can express.
"aie.device"() ({
^bb0(%tile: index):
  // expected-error @+1 {{'aie.shim_dma_allocation' op tile operand must implement TileLike interface}}
  "aie.shim_dma_allocation"(%tile) <{sym_name = "alloc", channel_dir = 1 : i32, channel_index = 0 : i64, plio = false}> : (index) -> ()
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()
