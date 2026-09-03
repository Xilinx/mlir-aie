//===- inline_symbols_in_loop.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences %s | FileCheck %s

// The symbol-inlining companion to inline_symbols.mlir: same shim DMA
// allocation, but referenced from an op *nested* in an scf.for rather than at
// the top level of the callee sequence.
//
// inlineReferencedSymbolDefinitions reads only the attributes of the op it is
// handed, and it was called once per top-level op of the callee body, so a
// reference inside a loop was never visited. The definition was then left
// behind in the callee while the reference travelled into the caller, and a
// later pass failed with "no shim DMA allocation found for symbol".
//
// Both allocations are checked deliberately: the top-level one already worked,
// and it working while the in-loop one did not is what the bug looked like.

module {
  aie.device(npu2) @main {
    %tile00 = aie.tile(0, 0)

    // Both definitions have to be inlined into the caller device, not just
    // the top-level one. They are placed ahead of the sequence that uses them.
    // CHECK: aie.shim_dma_allocation @buffer_top
    // CHECK: aie.shim_dma_allocation @buffer_in_loop

    // CHECK-LABEL: aie.runtime_sequence @main_seq
    aie.runtime_sequence @main_seq(%arg0: memref<64xi32>) {
      // CHECK: aiex.npu.load_pdi {device_ref = @config_with_symbols}
      // CHECK: aiex.npu.dma_memcpy_nd
      // CHECK-SAME: metadata = @buffer_top
      // CHECK: scf.for
      // CHECK:   aiex.npu.dma_memcpy_nd
      // CHECK-SAME: metadata = @buffer_in_loop
      aiex.configure @config_with_symbols {
        aiex.run @seq_with_dma(%arg0) : (memref<64xi32>)
      }
    }
  }

  // The originals stay put in the callee device.
  // CHECK: aie.device(npu2) @config_with_symbols
  // CHECK: aie.shim_dma_allocation @buffer_top
  // CHECK: aie.shim_dma_allocation @buffer_in_loop
  aie.device(npu2) @config_with_symbols {
    %tile20 = aie.tile(2, 0)
    %tile30 = aie.tile(3, 0)
    aie.shim_dma_allocation @buffer_top(%tile20, S2MM, 0)
    aie.shim_dma_allocation @buffer_in_loop(%tile30, S2MM, 1)

    aie.runtime_sequence @seq_with_dma(%arg0: memref<64xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @buffer_top, id = 0 : i64 } : memref<64xi32>
      scf.for %i = %c0 to %c2 step %c1 {
        aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @buffer_in_loop, id = 1 : i64 } : memref<64xi32>
      }
    }
  }
}
