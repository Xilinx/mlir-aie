//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Core reset via the aiex.core_reset op (op-based analog of ../core). See README.md.

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    %prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "prod_lock"}
    %cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "cons_lock"}

    %counter = aie.buffer(%tile_0_2) {sym_name = "counter"} : memref<1xi32>
    %output_buffer = aie.buffer(%tile_0_2) {sym_name = "output_buffer"} : memref<8xi32>

    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c1_i32 = arith.constant 1 : i32

      aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1_i32)
      %n = memref.load %counter[%c0] : memref<1xi32>
      scf.for %i = %c0 to %c8 step %c1 {
        memref.store %n, %output_buffer[%i] : memref<8xi32>
      }
      %n1 = arith.addi %n, %c1_i32 : i32
      memref.store %n1, %counter[%c0] : memref<1xi32>
      aie.use_lock(%cons_lock, Release, %c1_i32)
      aie.end
    }

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      %c1_lk = arith.constant 1 : i32
      aie.use_lock(%cons_lock, AcquireGreaterEqual, %c1_lk)
      aie.dma_bd(%output_buffer : memref<8xi32>) { len = 8 : i32 }
      aie.use_lock(%prod_lock, Release, %c1_lk)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    aie.shim_dma_allocation @out0 (%tile_0_0, S2MM, 0)

    aie.runtime_sequence @seq(%arg0: memref<16xi32>) {
      // Batch 1: the core ran once from load; collect counter=N into arg0[0:8].
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out0}

      // Core reset on tile (0,2) via the merged runtime-sequence op. It lowers to
      // a mask-preserving reset pulse on CORE_CONTROL (assert bit 1, then clear
      // it) -- the reset->unreset half of ../core's raw sequence, clearing the PC
      // while preserving the other CORE_CONTROL fields.
      aiex.core_reset(%tile_0_2)

      // aiex.core_reset is reset-only (XAie_CoreReset + XAie_CoreUnreset): it does
      // not re-enable, by design assuming the core is still enabled. This core ran
      // to aie.end, so it is no longer enabled -- confirmed on-board: the op alone
      // leaves the core halted and batch 2 never arrives. Re-enable it with a
      // masked write of the ENABLE bit (CORE_CONTROL bit 0, mask 0x1), mirroring
      // aie-rt's XAie_CoreEnable (a MaskWrite32 of the enable field). op + this
      // write is the full XAie_CoreReset -> XAie_CoreUnreset -> XAie_CoreEnable
      // driver sequence, all masked so no other CORE_CONTROL field is clobbered.
      %cc = arith.constant 204800 : i32
      %enable = arith.constant 1 : i32
      %en_mask = arith.constant 1 : i32
      aiex.npu.maskwrite32(%cc, %enable, %en_mask) {column = 0 : i32, row = 2 : i32} : i32, i32, i32

      // Batch 2: the core re-ran from a clean PC and emitted counter=N+1.
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 8][1, 1, 1, 8][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @out0} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out0}
    }
  }
}
