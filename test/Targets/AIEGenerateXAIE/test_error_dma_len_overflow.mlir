//===- test_error_dma_len_overflow.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie --verify-diagnostics %s

// `XAie_DmaSetAddrLen` / `XAie_DmaSetMultiDimAddr` take the byte length as a
// `u32`, so any length that does not fit in 32 bits would be silently
// truncated by the emitted C call. Verify the translator rejects it with an
// explicit op error instead.
//
// memref<536870912xi64>: 536870912 * 8 = 4294967296 bytes = 0x100000000 which
// does not fit in 32 bits.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    // expected-error @+1 {{does not fit in 32 bits}}
    aie.shim_dma(%tile_0_0) {
      %srcDma = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      %buf = aie.external_buffer { sym_name = "myBuffer_0_0_0" } : memref<536870912xi64>
      aie.dma_bd(%buf : memref<536870912xi64>, 0, 536870912)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
