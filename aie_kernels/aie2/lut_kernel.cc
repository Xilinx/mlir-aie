// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Factory-only translation unit: standalone kernels may link the LUT
// separately.
#include AIE_LUT_KERNEL_SOURCE
#include "lut_based_ops.cpp"
