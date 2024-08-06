//===- bad_rtp_write.mlir ---------------------------------------*- MLIR -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu -verify-diagnostics %s

aie.device(npu1_4col) {
  aiex.runtime_sequence() {
    // expected-error@+2 {{buffer 'RTP' not found in device}}
    // expected-error@+1 {{failed to legalize operation 'aiex.npu.rtp_write' that was explicitly marked illegal}}
    aiex.npu.rtp_write(@RTP, 4, 99)
  }
}
