<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Vendored aie_api patches

Applied to `third_party/aie_api` at CMake configure time (`cmake/apply_patches.cmake`).

- `0001-floor-ptr-always-mask.patch`: aie_api's `load_unaligned` does `align 32`
  loads from an unfloored pointer, because `floor_ptr` is a no-op when
  `AIE_API_NATIVE == 0` (Peano). Peano nightlies from `2026111401+e587aea7`
  (upstream LLVM InferAlignment,
  [llvm/llvm-project#166935](https://github.com/llvm/llvm-project/pull/166935))
  fold `and(ptrtoint p, 31)` to 0 from that alignment, so the realigning
  `vshift` becomes a shift by 0 and unaligned loads return wrong data. Remove
  this patch once aie_api or Peano fixes it.
