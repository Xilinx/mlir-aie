//===- compare_main.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Harness for static-vs-dynamic TXN equivalence tests.  static_txn() and
// dynamic_txn(n) live in separate translation units (because both generated
// headers define a `generate_txn_sequence` symbol at namespace scope and
// cannot be #included into the same TU).  This file just compares the two
// `std::vector<uint32_t>` streams and reports the first divergence.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

std::vector<uint32_t> static_txn();
std::vector<uint32_t> dynamic_txn(int32_t n);

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <n>\n", argv[0]);
    return 2;
  }
  int32_t n = std::atoi(argv[1]);

  std::vector<uint32_t> a = static_txn();
  std::vector<uint32_t> b = dynamic_txn(n);

  if (a == b) {
    std::printf("equivalent: %zu words (n=%d)\n", a.size(), n);
    return 0;
  }

  std::fprintf(stderr,
               "TXN streams DIFFER: static=%zu words, dynamic=%zu words "
               "(n=%d)\n",
               a.size(), b.size(), n);

  size_t lim = std::min(a.size(), b.size());
  for (size_t i = 0; i < lim; ++i) {
    if (a[i] != b[i]) {
      std::fprintf(stderr,
                   "  first diff at word %zu: static=0x%08x dynamic=0x%08x\n",
                   i, a[i], b[i]);
      break;
    }
  }
  return 1;
}
