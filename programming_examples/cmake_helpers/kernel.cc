// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fixture kernel for helpers.lit's "mlir" CASE. It is never compiled -- that
// case is configure-only and asserts on the generated rule, not on its output.
// The file still has to exist, because add_aie_kernel_object() rejects a source
// it cannot find.

extern "C" {
void fixture_kernel(int *in, int *out, int n) {
  for (int i = 0; i < n; i++)
#ifdef GROUPA
    out[i] = in[i] + 1;
#else
    out[i] = in[i] * 2;
#endif
}
}
