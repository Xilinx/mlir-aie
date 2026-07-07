//===- error_device_name.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --aie-npu-to-cert %s

// Test: --device-name defaults to "main", but no device has that symbol name

// expected-error@+1 {{no device found matching --device-name "main"}}
module {
  aie.device(npu2) @not_main {
    aie.runtime_sequence @configure() {
      aiex.npu.write32 {address = 12345 : ui32, value = 100 : ui32}
    }
  }
}
