//===- device_cache_kernel.cc -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for device_cache.mlir, built once per BUMP to give a linked
// object two different contents.

#include <stdint.h>

extern "C" void bump(int32_t *x) { x[0] += BUMP; }
