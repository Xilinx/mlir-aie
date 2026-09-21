//===- npu_blockwrite_values_error.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// npu.blockwrite_values carries a payload computed at TXN-build time, which has
// no static encoding -- it is produced only by the dynamic BD-pool lowering and
// consumed only by the C++ TXN target. Reaching the static binary path must be
// a clear diagnostic, not a silent drop from the emitted binary (the op would
// otherwise fall through the emitter's TypeSwitch and vanish, leaving a stream
// whose BDs are never configured).

// RUN: not aie-translate --aie-npu-to-binary -aie-output-binary=false %s 2>&1 | FileCheck %s

// CHECK: 'aiex.npu.blockwrite_values' op cannot translate a runtime-valued blockwrite payload to a static TXN binary
module {
  aie.device(npu1) {
    aie.runtime_sequence @seq(%arg0: memref<8xi32>) {
      %addr = arith.constant 119300 : i32
      %w0 = arith.constant 256 : i32
      %w1 = arith.constant 33554432 : i32
      aiex.npu.blockwrite_values(%addr : i32) values %w0, %w1 : i32, i32
    }
  }
}
