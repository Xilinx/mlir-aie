//===- rolled_loop_compare.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Proves the rolled dynamic loop programs the SAME hardware as N unrolled
// static configures: replays generate_txn_main_rolled(n) and the static
// generate_txn_main_static2 (a hand-unrolled 2-iteration ping-pong) into
// register maps and asserts they are equal for n = 2. A rolled loop over a
// runtime trip count is thus a drop-in for the unroll the static allocator
// would have produced. ROLLED_HDR / STATIC_HDR select the generated headers.
//
//===----------------------------------------------------------------------===//

#include "aie/Runtime/TxnEncoding.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <vector>

#include ROLLED_HDR
#include STATIC_HDR

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
} // namespace

int main() {
  auto rolled = generate_txn_main_rolled(2);
  auto stat = generate_txn_main_static2();
  if (!rolled || !stat) {
    std::fprintf(stderr, "builder returned nullopt\n");
    return 1;
  }
  RegMap a = replay(*rolled), b = replay(*stat);
  if (a != b) {
    std::fprintf(stderr, "rolled(2) and static2 program different registers\n");
    return 1;
  }
  std::printf("equivalent: rolled(2) == static2 (%zu registers)\n", a.size());
  return 0;
}
