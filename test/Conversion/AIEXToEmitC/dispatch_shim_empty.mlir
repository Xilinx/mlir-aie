//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s --convert-aiex-to-emitc="emit-dispatch-shim=true" 2>&1 | FileCheck %s --check-prefix=SHIM
// RUN: aie-opt %s --convert-aiex-to-emitc | FileCheck %s --check-prefix=NOSHIM

// SHIM: error: 'builtin.module' op emit-dispatch-shim needs exactly one aie.runtime_sequence to wrap, found 0
// NOSHIM: module {
// NOSHIM-NEXT: }

module {
}
