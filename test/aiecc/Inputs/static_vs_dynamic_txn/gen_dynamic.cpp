//===- gen_dynamic.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Wraps the dynamic `generate_txn_sequence(int32_t)` function emitted by
// aiecc into a uniquely-named symbol so it can be linked alongside one or
// more static wrappers.
//
//===----------------------------------------------------------------------===//

#include "dynamic_txn.h"

#include <cstdint>
#include <vector>

std::vector<uint32_t> dynamic_txn(int32_t n) {
  return generate_txn_sequence(n);
}
