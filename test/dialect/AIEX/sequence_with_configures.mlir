//===- sequence_with_configures.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//


// RUN: aie-opt %s

module {
    aie.device(npu1) @dev_a {
        aie.runtime_sequence @run_a (%x: memref<1xi32>) {
        }
        aie.runtime_sequence @run_b () {
        }
    }
    aie.device(npu1) { // main
        aie.runtime_sequence (%x : memref<1xi32>) {
            aiex.configure @dev_a {
                aiex.run @run_a (%x) : (memref<1xi32>)
            }
        }
    }
}