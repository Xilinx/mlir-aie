//===- buffer_clear_invalid.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -split-input-file -verify-diagnostics --aie-lower-buffer-clear %s

// Shim tiles have no data memory module; aie-rt's XAie_DataMemBlockWrite
// accepts only AIETILE and MEMTILE (driver/src/memory/xaie_mem.c).
module {
  aie.device(npu2) {
    %shim_tile = aie.tile(0, 0)
    aie.runtime_sequence() {
      // expected-error @+1 {{tile (0, 0) has no local data memory to clear (only core and mem tiles do)}}
      aiex.buffer_clear(%shim_tile, 0, 4)
    }
  }
}

// -----

// Rejected rather than accepted as a no-op, so a mis-specified size is caught.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    aie.runtime_sequence() {
      // expected-error @+1 {{length must be nonzero}}
      aiex.buffer_clear(%tile, 0, 0)
    }
  }
}

// -----

// npu.blockwrite writes whole 32-bit words and there is no read-modify-write
// path for a partial edge word, so a misaligned start cannot be rounded.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    aie.runtime_sequence() {
      // expected-error @+1 {{address 2 is not 4-byte (word) aligned}}
      aiex.buffer_clear(%tile, 2, 1)
    }
  }
}

// -----

// Core tile data memory is 0x10000 bytes (getLocalMemorySize, matching
// aie-rt's Aie2PTileMemMod.Size): the last word plus 2 words overruns by 4.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    aie.runtime_sequence() {
      // expected-error @+1 {{region [65532, 65540) exceeds tile (0, 2)'s local data memory size (65536 bytes)}}
      aiex.buffer_clear(%tile, 65532, 2)
    }
  }
}

// -----

// Same bound check against getMemTileSize, 0x80000 bytes.
module {
  aie.device(npu2) {
    %mem_tile = aie.tile(0, 1)
    aie.runtime_sequence() {
      // expected-error @+1 {{region [524284, 524292) exceeds tile (0, 1)'s local data memory size (524288 bytes)}}
      aiex.buffer_clear(%mem_tile, 524284, 2)
    }
  }
}

// -----

// The op lowers to an NPU instruction, as CoreResetOp and SetLockOp also do.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(0, 3)
    aie.runtime_sequence() {
      // expected-error @+1 {{not supported on AIE1}}
      aiex.buffer_clear(%tile, 0, 4)
    }
  }
}
