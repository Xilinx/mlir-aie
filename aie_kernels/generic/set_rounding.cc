// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// Set the core's rounding mode register. Kernels that narrow an accumulator
// (an `srs` shift, a bf16 store) round in whatever mode the core is in, and a
// fresh core boots in floor; a design calls this once, before the first such
// kernel, with the mode that kernel's contract names (-DROUNDING_MODE=conv_even
// binds `set_rounding_conv_even`). See KernelContract.rounding_mode.

#include <aie_api/aie.hpp>

#ifndef ROUNDING_MODE
#error Please specify the mode at compile time, e.g. -DROUNDING_MODE=conv_even.
#endif

#define SET_ROUNDING_CAT(a, b) a##b
#define SET_ROUNDING_NAME(mode) SET_ROUNDING_CAT(set_rounding_, mode)

extern "C" {
void SET_ROUNDING_NAME(ROUNDING_MODE)() {
  ::aie::set_rounding(aie::rounding_mode::ROUNDING_MODE);
}
}
