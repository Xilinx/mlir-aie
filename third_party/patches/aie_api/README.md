<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Vendored aie_api patches

Applied to `third_party/aie_api` at CMake configure time (`cmake/apply_patches.cmake`).

- `0001-opaque-frac-address.patch`: aie_api's unaligned accessors compute a
  realigning shift as `address & mask` but access memory through the unfloored
  pointer, because `floor_ptr` is a no-op when `AIE_API_NATIVE == 0` (the
  hardware truncates the address). Peano nightlies from `2026111401+e587aea7`
  (upstream LLVM InferAlignment,
  [llvm/llvm-project#166935](https://github.com/llvm/llvm-project/pull/166935))
  read the alignment off those accesses and fold the shift to 0, so unaligned
  data is not realigned. The patch hides the address from the optimizer with
  `__builtin_annotation`, which the backend drops, and only on a compiler that
  has the `gcc_struct` attribute (LLVM from 2025-12-12); older Peano compiles
  to identical code. Remove this patch once aie_api or Peano fixes it.
