//===- stack_size_compiler_rt_kernel.cc ------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Float division and integer remainder lower to the compiler-rt calls
// __divsf3 and __modsi3.

extern "C" void div_mod(int *out) {
  float *f = reinterpret_cast<float *>(out);
  f[0] = f[1] / f[2];
  out[3] = out[4] % out[5];
}
