//===- stack_size_measured_kernel.cc ----------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_measured.mlir. entry_fn calls helper_fn directly
// inside one object, so the measured requirement is above the stack_size that
// the test declares. The test matches any byte count, which keeps it stable
// across compiler versions.

extern "C" {

static int helper_fn(int x) { return x + 1; }

void entry_fn(unsigned char *out) {
  for (int i = 0; i < 512; i++)
    out[i] = static_cast<unsigned char>(helper_fn(i));
}

} // extern "C"
