//===- bad_scratchpad_offset_multiple.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

// The host refuses a value that is not a multiple of offset_multiple, which
// only a positive multiple can make sense of.
// expected-error@+1 {{attribute 'offset_multiple' failed to satisfy constraint}}
aiex.scratchpad_parameter @row : i32 {offset_multiple = 0 : i32}
