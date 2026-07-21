//===- bd_pool_host.cpp - Standalone BdPool free-list test ---------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exercises the header-only runtime BD free-list pool in TxnEncoding.h the way
// generated host C++ consumes it: init a pool sized to a tile's BD count, then
// pop/push ids. Focus is the exhaustion backstop -- bd_pool_pop returns false
// when the working set exceeds the tile's BD count, which the generated builder
// turns into `return std::nullopt`. This is the only path the MLIR/e2e tests
// verify statically (the write32 emission) but never trigger at runtime.
//
//===----------------------------------------------------------------------===//

#include "aie/Runtime/TxnEncoding.h"

#include <cstdint>
#include <cstdio>

using namespace aie_runtime;

static int failures = 0;

static void check(bool cond, const char *label) {
  if (!cond) {
    std::printf("FAIL: %s\n", label);
    ++failures;
  }
}

int main() {
  // Pop hands out the lowest free id first (0, 1, ...), matching the static
  // allocator's order, so a fresh pool's first pop equals a pinned bd_id = 0.
  {
    BdPool p = bd_pool_init(4);
    uint32_t id = 99;
    for (uint32_t expect = 0; expect < 4; ++expect) {
      check(bd_pool_pop(p, id), "pop within capacity should succeed");
      check(id == expect, "pop should hand out lowest free id first");
    }
    // Exhaustion: the 5th pop on a 4-BD pool fails and leaves `out` untouched.
    id = 0xABCD;
    check(!bd_pool_pop(p, id), "pop on empty pool must return false");
    check(id == 0xABCD, "failed pop must not write out");
  }

  // Push returns an id for reuse; a subsequent pop hands it back out. A
  // rotating free inside a runtime loop relies on this recycling.
  {
    BdPool p = bd_pool_init(2);
    uint32_t a = 0, b = 0;
    check(bd_pool_pop(p, a) && a == 0, "first pop is id 0");
    check(bd_pool_pop(p, b) && b == 1, "second pop is id 1");
    check(!bd_pool_pop(p, a), "pool of 2 is now empty");
    bd_pool_push(p, 0); // return id 0
    uint32_t reused = 99;
    check(bd_pool_pop(p, reused), "pop after push succeeds");
    check(reused == 0, "pushed id is handed back out");
  }

  // n larger than the fixed array is clamped to kMaxBDsPerTile: the pool never
  // pops more ids than the backing storage holds.
  {
    BdPool p = bd_pool_init(kMaxBDsPerTile + 100);
    uint32_t id = 0, count = 0;
    while (bd_pool_pop(p, id))
      ++count;
    check(count == kMaxBDsPerTile, "pop count is clamped to kMaxBDsPerTile");
  }

  if (failures > 0) {
    std::printf("bd_pool: %d failure(s)\n", failures);
    return 1;
  }
  std::printf("bd_pool: all checks passed\n");
  return 0;
}
