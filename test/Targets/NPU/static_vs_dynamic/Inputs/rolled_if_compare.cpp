//===- rolled_if_compare.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Proves a rolled dynamic scf.if programs the SAME hardware as the static
// branch it selects: replays generate_txn_main_rolled_if(cond) into a register
// map and asserts it equals the TAKEN static oracle for cond=true and the
// NOT-TAKEN (no-op) oracle for cond=false. TAKEN_HDR / NOTTAKEN_HDR /
// ROLLED_HDR select the generated headers.
//
//===----------------------------------------------------------------------===//

#include "aie/Runtime/TxnEncoding.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <vector>

#include ROLLED_HDR
#include TAKEN_HDR
#include NOTTAKEN_HDR

namespace {
using RegMap = std::map<uint64_t, uint32_t>;

RegMap replay(const std::vector<uint32_t> &t) {
  using namespace aie_runtime;
  RegMap r;
  size_t p = 4;
  uint64_t sk = 1ULL << 40;
  while (p < t.size()) {
    uint32_t o = t[p];
    if (o == TXN_OPC_WRITE) {
      r[t[p + 2]] = t[p + 4];
      p += 6;
    } else if (o == TXN_OPC_BLOCKWRITE) {
      uint32_t a = t[p + 2], bs = t[p + 3];
      size_t tot = bs / sizeof(uint32_t);
      for (size_t i = 4; i < tot; ++i)
        r[a + (i - 4) * 4] = t[p + i];
      p += tot;
    } else if (o == TXN_OPC_MASKWRITE) {
      r[t[p + 2]] = (r[t[p + 2]] & ~t[p + 5]) | (t[p + 4] & t[p + 5]);
      p += 7;
    } else if (o == TXN_OPC_TCT) {
      r[sk++] = t[p + 2];
      r[sk++] = t[p + 3];
      p += 4;
    } else if (o == TXN_OPC_DDR_PATCH) {
      r[sk++] = t[p + 6];
      r[sk++] = t[p + 8];
      r[sk++] = t[p + 10];
      p += 12;
    } else {
      std::fprintf(stderr, "unhandled opcode 0x%x\n", o);
      return {};
    }
  }
  return r;
}

int check(const char *name, std::optional<std::vector<uint32_t>> dyn,
          std::optional<std::vector<uint32_t>> oracle) {
  if (!dyn || !oracle) {
    std::fprintf(stderr, "%s: builder returned nullopt\n", name);
    return 1;
  }
  if (replay(*dyn) != replay(*oracle)) {
    std::fprintf(stderr, "%s: dynamic and static program different registers\n",
                 name);
    return 1;
  }
  return 0;
}
} // namespace

int main() {
  int rc = 0;
  rc |= check("cond=true", generate_txn_main_rolled_if(true),
              generate_txn_main_static_taken());
  rc |= check("cond=false", generate_txn_main_rolled_if(false),
              generate_txn_main_static_nottaken());
  if (rc == 0)
    std::printf(
        "equivalent: rolled_if(true)==taken, rolled_if(false)==nottaken\n");
  return rc;
}
