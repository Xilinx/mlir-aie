//===- canonicalize_no_linearize_iter_bd.mlir -------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Regression test for LinearizeContiguousBDTransfer: a contiguous shim BD
// whose explicit `len` is smaller than the product of its dimension sizes
// uses the outermost dim as a hardware BD iteration dimension (len = per-
// iteration transfer size; outermost size/stride = repeat count and address
// advancement, programmed into the shim NoC tile's iteration_size /
// iteration_stride registers by AIEDMATasksToNPU).  Linearizing such a BD
// would collapse the iteration into len and drop the iteration stride,
// causing the NPU to repeatedly transfer only the first `len` elements from
// offset 0 instead of sweeping the whole buffer.
//
// Failing case exposed by IRON transpose[M_2048-N_64-m_64-n_64-s_8]: the
// input fill BD has dims [<32, 4096>, <1, 64>, <64, 64>, <64, 1>] and
// len = 4096, while the product of all sizes is 131072.  Before the fix,
// 89.5% of the output was wrong (only the first column-tile was correct).
//
// Positive linearization behavior (full coverage, len == product, true
// linear forms, attribute preservation, etc.) is already covered by
// canonicalize_linear_dma_bd.mlir; this file only tests the new bail-out.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --canonicalize --split-input-file %s | FileCheck %s --check-prefix=CANON
// RUN: aie-opt --pass-pipeline='any(aie.device(canonicalize,aie-dma-tasks-to-npu))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=LOWER

// -----

// 4D contiguous BD with per-iteration len (4096) != product of all dim
// sizes (32*1*64*64 = 131072).  The outermost dim is a BD iteration
// dimension.  Canonicalization must NOT linearize — the iteration stride
// (4096) carries the inter-iteration address advancement.

// CANON-LABEL: @iter_bd_no_linearize
// CANON:         aiex.dma_configure_task
// CANON:           aie.dma_bd(%arg0 : memref<131072xbf16>, 0, 4096,
// CANON-SAME:        [<size = 32, stride = 4096>, <size = 1, stride = 64>, <size = 64, stride = 64>, <size = 64, stride = 1>]

// End-to-end correctness: after canonicalize + aie-dma-tasks-to-npu, the
// iteration must show up in the NPU writebd registers — buffer_length is
// the per-iteration transfer (4096 bf16 = 2048 i32 words), iteration_size
// is the (32-1)-encoded repeat count, iteration_stride is the (2048-1)-
// encoded i32-word stride between repeats.  Before the fix, canonicalize
// dropped the iteration dim, yielding iteration_size = 0 (no repeat) and
// the wrong buffer_length, which is exactly the transpose bug.

// LOWER-LABEL: @iter_bd_no_linearize
// LOWER:         aiex.npu.writebd
// LOWER-SAME:      buffer_length = 2048
// LOWER-SAME:      iteration_size = 31
// LOWER-SAME:      iteration_stride = 2047
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_iter (%tile_0_0, MM2S, 0)
    aie.runtime_sequence @iter_bd_no_linearize(%arg0 : memref<131072xbf16>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<131072xbf16>, 0, 4096,
          [<size = 32, stride = 4096>, <size = 1, stride = 64>,
           <size = 64, stride = 64>, <size = 64, stride = 1>]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
