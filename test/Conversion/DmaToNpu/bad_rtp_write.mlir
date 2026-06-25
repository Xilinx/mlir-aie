//===- bad_rtp_write.mlir ---------------------------------------*- MLIR -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu -verify-diagnostics %s

aie.device(npu1) {
  aie.runtime_sequence() {
    // expected-error@+2 {{buffer 'RTP' not found in device}}
    // expected-error@+1 {{failed to legalize operation 'aiex.npu.rtp_write' that was explicitly marked illegal}}
    %cst_npu_0 = arith.constant 99 : i32
    aiex.npu.rtp_write(@RTP, 4, %cst_npu_0) : i32
  }
}
