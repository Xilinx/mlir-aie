//===- sequence_with_configures_bad.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//


// RUN: aie-opt --verify-diagnostics --split-input-file %s

module {
    // expected-note@+1 {{This device does not have a 'run_x' runtime sequence}}
    aie.device(npu1) @dev_a {
        aiex.runtime_sequence @run_a (%x: memref<1xi32>) {
        }
        aiex.runtime_sequence @run_b () {
        }
    }
    aie.device(npu1) { // main
        aiex.runtime_sequence (%x : memref<1xi32>) {
            aiex.configure @dev_a {
                // expected-error@+1 {{No such runtime sequence for device 'dev_a': 'run_x'}}
                aiex.run @run_x (%x) : (memref<1xi32>)
            }
        }
    }
}

// -----

module {
    aie.device(npu1) @dev_a {
        aiex.runtime_sequence @run_a (%x: memref<1xi32>) {
        }
        aiex.runtime_sequence @run_b () {
        }
    }
    aie.device(npu2) { // main
        aiex.runtime_sequence (%x : memref<1xi32>) {
            // expected-error@+1 {{Device types do not match: 'npu2' vs. 'npu1'}}
            aiex.configure @dev_a {
                aiex.run @run_a (%x) : (memref<1xi32>)
            }
        }
    }
}

// -----

module {
    aie.device(npu1) @dev_a {
        aiex.runtime_sequence @run_a (%x: memref<1xi32>) {
        }
        aiex.runtime_sequence @run_b () {
        }
    }
    aie.device(npu2) { // main
        aiex.runtime_sequence (%x : memref<1xi32>) {
            // expected-error@+1 {{No such device: '@dev_x'}}
            aiex.configure @dev_x {
                aiex.run @run_a (%x) : (memref<1xi32>)
            }
        }
    }
}

// -----

module {
    memref.global "public" @in0 : memref<1024xi32>
    aie.device(npu2) { // main
        aiex.runtime_sequence (%x : memref<1xi32>) {
            // expected-error@+1 {{Not a device: '@in0'}}
            aiex.configure @in0 {
                aiex.run @run_x (%x) : (memref<1xi32>)
            }
        }
    }
}
