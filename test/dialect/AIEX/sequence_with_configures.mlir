//===- sequence_with_configures.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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