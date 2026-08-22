//===- register_reset_mutually_exclusive.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -verify-diagnostics "--aie-expand-load-pdi=register-reset=true ctrl-pkt=true" %s

// register-reset and ctrl-pkt pick different, incompatible preload sequences
// for the same load_pdi; requesting both is a pass-usage error, not a
// silent pick-one.

// expected-error @+1 {{ctrl-pkt and register-reset are mutually exclusive}}
module {
    aie.device(npu2_1col) @main {
        aie.runtime_sequence (%arg0: memref<1xi32>) {
        }
    }
}
