//===- gen_static.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Wraps the static `generate_txn_sequence()` function emitted by aiecc into
// a uniquely-named symbol so it can be linked alongside the dynamic one.
//
// `STATIC_HEADER` is a -D quoted-include name set from the lit RUN line and
// `STATIC_NAME` is the wrapper symbol name (e.g. static_txn or
// static_txn_8192).
//
//===----------------------------------------------------------------------===//

#include STATIC_HEADER

#include <cstdint>
#include <vector>

std::vector<uint32_t> STATIC_NAME() { return generate_txn_sequence(); }
