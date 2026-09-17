//===- bad_sym_visibility.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Our symbol ops keep `sym_visibility` in a discardable attribute rather than a
// tablegen-declared argument, so upstream's symbol verifier -- which only reads
// the inherent attribute -- never sees it. Check that `AttrBasedSymbolVisibility`
// verifies it instead, rather than silently reading a malformed value as public.

// RUN: not aie-opt --split-input-file %s 2>&1 | FileCheck %s

// CHECK: error{{.*}}'aie.device' op visibility expected to be one of ["public", "private", "nested"], but got "privte"
aie.device(npu1) @bad_spelling {
} {sym_visibility = "privte"}

// -----

// CHECK: error{{.*}}'aie.device' op requires visibility attribute 'sym_visibility' to be a string attribute, but got 42 : i32
aie.device(npu1) @not_a_string {
} {sym_visibility = 42 : i32}
