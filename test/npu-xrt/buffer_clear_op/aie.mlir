//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Clears a core tile's resident accumulator via the aiex.buffer_clear op. See
// README.md.

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    %prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "prod_lock"}
    %cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "cons_lock"}

    // Pinned at local address 0 so the sequence can name it: buffer_clear
    // addresses data memory by offset, not by symbol.
    %acc = aie.buffer(%tile_0_2) {sym_name = "acc", address = 0 : i32} : memref<8xi32>

    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c1_i32 = arith.constant 1 : i32

      aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1_i32)
      // acc[i] = i + 1, distinct per word, so an all-zero readback can only
      // come from the clear.
      scf.for %i = %c0 to %c8 step %c1 {
        %i_i32 = arith.index_cast %i : index to i32
        %v = arith.addi %i_i32, %c1_i32 : i32
        memref.store %v, %acc[%i] : memref<8xi32>
      }
      aie.use_lock(%cons_lock, Release, %c1_i32)
      aie.end
    }

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      %c1_lk = arith.constant 1 : i32
      aie.use_lock(%cons_lock, AcquireGreaterEqual, %c1_lk)
      aie.dma_bd(%acc : memref<8xi32>) { len = 8 : i32 }
      aie.use_lock(%prod_lock, Release, %c1_lk)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    aie.shim_dma_allocation @out0 (%tile_0_0, S2MM, 0)

    aie.runtime_sequence @seq(%arg0: memref<16xi32>) {
      // Batch 1: the core has run once from load; collect acc = [1..8] into
      // arg0[0:8], proving the accumulator really holds nonzero data.
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out0}

      // Clears data memory in place, so no core re-run, unlike aiex.core_reset.
      aiex.buffer_clear(%tile_0_2, 0, 8)

      // ^bb1 is waiting on cons_lock >= 1; signalling it here fires the second
      // read with no core involvement, isolating the op under test.
      aiex.set_lock(%cons_lock, 1)

      // Batch 2: collect acc = [0..0] into arg0[8:16].
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 8][1, 1, 1, 8][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @out0} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out0}
    }
  }
}
