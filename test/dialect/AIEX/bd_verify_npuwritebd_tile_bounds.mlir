//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NpuWriteBdOp::verify used to hardcode shim-tile bit widths for D0/D1
// size (10-bit, `> 0x3FF`) and D0/D1/D2/iteration stride (20-bit,
// `> 0xFFFFF`) regardless of the BD's actual tile type. Core tiles have an
// 8-bit wrap (real max 255) and memtile stepsize is 17-bit (real max
// 131071), both narrower than what the hardcoded checks enforced. Both
// bounds now come from the tile-aware target-model accessors
// (getDmaBdWrapBits / getDmaBdStepBits / getDmaBdIterBits).
//
// This defect (and these tests) exercise hand-authored `aiex.npu.writebd`
// directly: the normal front door (`aie.dma_bd` -> `dma_configure_task` ->
// `--aie-dma-tasks-to-npu`) already gets a tile-aware check first, via
// verifyStridesWraps, before NpuWriteBdOp is ever created.

// RUN: aie-opt --verify-diagnostics --split-input-file %s

// Control: d0_size = 255, the real core-tile 8-bit wrap maximum (column=0,
// row=2 is a core tile on npu2). d0_size is a wrap field (not stepsize),
// so it is not actual-1 biased. d1_size = 2 (not 1) forces
// isLinearTransfer = false so the D0 Size check actually runs. Must
// remain accepted.
module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 4 : i32, buffer_offset = 0 : i32, column = 0 : i32, row = 2 : i32, d0_size = 255 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 2 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    }
  }
}

// -----

// Negative companion: d0_size = 256, one past the real core-tile wrap
// maximum of 255. Must be rejected now that the tile-aware bound applies.
module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-error@+1 {{D0 Size exceeds the [0:255] range}}
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 4 : i32, buffer_offset = 0 : i32, column = 0 : i32, row = 2 : i32, d0_size = 256 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 2 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    }
  }
}

// -----

// Control: d0_stride (encoded) = 131071, the real memtile 17-bit stepsize
// maximum (actual stepsize 131072; stepsize fields are actual-1 biased,
// and the encoded attribute value here already IS that biased value).
// d1_size = 1, iteration_size = 0, so isLinearTransfer is true and the D0
// Size check is skipped, isolating the stride check. Must remain accepted.
module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 4 : i32, buffer_offset = 0 : i32, column = 0 : i32, row = 1 : i32, d0_size = 4 : i32, d0_stride = 131071 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 1 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    }
  }
}

// -----

// Negative companion: d0_stride (encoded) = 131072, one past the real
// memtile 17-bit stepsize maximum. Must be rejected now that the
// tile-aware bound applies.
module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-error@+1 {{D0 Stride exceeds the [0:131071] range}}
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 4 : i32, buffer_offset = 0 : i32, column = 0 : i32, row = 1 : i32, d0_size = 4 : i32, d0_stride = 131072 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 1 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    }
  }
}
