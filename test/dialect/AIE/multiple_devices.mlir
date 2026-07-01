//===- multiple_devices.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s

module {
    aie.device(npu1) @dev_1 {
    }
    aie.device(npu1) @dev_2 {
    }
}
