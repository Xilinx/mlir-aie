//===- lock_test.cpp ----------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Tests for the lock module in lib/Lock.cpp. See that file for the register
// layout and the acquire/release semantics this exercises, and for why each
// is grounded the way it is.
//
// Device.cpp's makeAIE2PDevice is a separate, parallel task in this same
// shared worktree. At the time this test was written it had gone from a
// `return {}` stub to a real implementation (see test/device_test.cpp), but
// this test still builds its own DeviceModel by hand, following the existing
// convention in memory_test.cpp (`DeviceModel dev{}; dev.numCols = ...`)
// rather than calling makeAIE2PDevice: that keeps this test's obligations (a
// shim/memtile/core partition with plausible AIE2P lock counts) explicit at
// the call site and insulated from further concurrent edits to a file this
// task does not own.
//
// Similarly, the register offsets the "through the register file" section
// below computes are a small standalone table, not a reuse of Lock.cpp's
// constants: the lock_test executable links against the aie-sim library but
// does not get its PRIVATE third_party/aie-rt include path, so it could not
// include the vendored headers Lock.cpp uses even if it wanted to. The
// offsets are cited against the same aie-rt source Lock.cpp cites (see that
// file's header comment for exact file:line references); keeping them
// independent means a mismatch between the two shows up as a failing test
// rather than being hidden by sharing one set of numbers.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"
#include "aiesim/Components.h"

#include "TestSupport.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace aiesim;

namespace {

DeviceModel makeTestDevice() {
  // One column: shim (row 0), one memtile row (row 1), one core row (row
  // 2). Lock counts match the vendored AIE2P register tables
  // (Aie2PTileLockMod / Aie2PShimNocLockMod / Aie2PMemTileLockMod,
  // third_party/aie-rt/driver/src/global/xaie2pgbl_reginit.c:2384-2451):
  // 16 core, 16 shim, 64 memtile.
  DeviceModel dev{};
  dev.generation = Generation::AIE2P;
  dev.numCols = 1;
  dev.numRows = 3;
  dev.numMemTileRows = 1;
  dev.coreDataMemSize = 64 * 1024;
  dev.coreProgMemSize = 16 * 1024;
  dev.memTileMemSize = 512 * 1024;
  dev.numCoreLocks = 16;
  dev.numMemTileLocks = 64;
  dev.numShimLocks = 16;
  return dev;
}

// Core-tile lock register layout, AIE2P. See Lock.cpp's file header for the
// full citation list (xaie2pgbl_params.h / xaie2pgbl_reginit.c /
// xaie_locks_aieml.c); repeated here as plain numbers rather than an
// #include for the reason given in the file comment above.
constexpr uint32_t kCoreReqBase = 0x00040000;   // ...MODULE_LOCK_REQUEST
constexpr uint32_t kCoreValueBase = 0x0001F000; // ...MODULE_LOCK0_VALUE
constexpr uint32_t kLockIdOff = 0x400;
constexpr uint32_t kRelAcqOff = 0x200;
constexpr uint32_t kValueRegOff = 0x10;

uint32_t acquireAddr(uint32_t id, int32_t value) {
  uint32_t encoded = static_cast<uint32_t>(value) & 0x7Fu;
  return kCoreReqBase + id * kLockIdOff + kRelAcqOff + (encoded << 2);
}

uint32_t releaseAddr(uint32_t id, int32_t value) {
  uint32_t encoded = static_cast<uint32_t>(value) & 0x7Fu;
  return kCoreReqBase + id * kLockIdOff + (encoded << 2);
}

uint32_t valueRegAddr(uint32_t id) {
  return kCoreValueBase + id * kValueRegOff;
}

} // namespace

int main() {
  DeviceModel dev = makeTestDevice();
  Array array(dev, nullptr);

  // The default diagnostic handler (installed by Array's constructor)
  // prints and calls abort(), which a test cannot survive. Collect messages
  // instead, so the deliberate-error case near the end can be asserted on
  // and every earlier section can assert nothing unexpected fired.
  std::vector<std::string> diagnostics;
  array.setDiagnosticHandler(
      [&](const std::string &msg) { diagnostics.push_back(msg); });

  Tile *shim = array.tile(0, 0);
  Tile *memTile = array.tile(0, 1);
  Tile *core = array.tile(0, 2);
  AIESIM_CHECK(shim != nullptr);
  AIESIM_CHECK(memTile != nullptr);
  AIESIM_CHECK(core != nullptr);

  // --- Lock count matches the device, for all three tile types. ---
  AIESIM_CHECK(core->locks() != nullptr);
  AIESIM_CHECK(memTile->locks() != nullptr);
  AIESIM_CHECK(shim->locks() != nullptr);
  AIESIM_CHECK_EQ(core->locks()->count(), dev.numCoreLocks);
  AIESIM_CHECK_EQ(memTile->locks()->count(), dev.numMemTileLocks);
  AIESIM_CHECK_EQ(shim->locks()->count(), dev.numShimLocks);

  LockModule *lm = core->locks();

  // --- Round trip through the LockModule interface directly (lock 0). ---
  // Reset value is 0 (the LOCK_VALUE field's DEFVAL).
  AIESIM_CHECK_EQ(lm->value(0), int32_t{0});

  // Plain Acquire(0): matches the reset value immediately.
  AIESIM_CHECK(lm->tryAcquire(0, 0));
  AIESIM_CHECK_EQ(lm->value(0), int32_t{0});

  // Acquire fails when it must: nothing has released yet.
  AIESIM_CHECK(!lm->tryAcquire(0, 1));
  AIESIM_CHECK_EQ(lm->value(0), int32_t{0});

  // ...and succeeds once the matching release lands.
  lm->release(0, 1);
  AIESIM_CHECK_EQ(lm->value(0), int32_t{1});
  AIESIM_CHECK(lm->tryAcquire(0, 1));
  AIESIM_CHECK_EQ(lm->value(0), int32_t{0});

  // AcquireGreaterEqual (negative value): fails below the threshold,
  // succeeds at or above it, and decrements by the threshold on success.
  lm->release(0, 3);
  AIESIM_CHECK_EQ(lm->value(0), int32_t{3});
  AIESIM_CHECK(!lm->tryAcquire(0, -4)); // need >= 4, have 3: fails
  AIESIM_CHECK(lm->tryAcquire(0, -2));  // need >= 2, have 3: succeeds
  AIESIM_CHECK_EQ(lm->value(0), int32_t{1});
  AIESIM_CHECK(lm->tryAcquire(0, -1)); // need >= 1, have 1: succeeds
  AIESIM_CHECK_EQ(lm->value(0), int32_t{0});

  // --- The same acquire/release/value sequence, driven through register
  // writes on the tile's RegisterFile and observed through a register read
  // of the lock value -- the path a host program actually uses. Lock id 1,
  // kept separate from the interface-level lock 0 above. ---
  RegisterFile &regs = core->regs();

  AIESIM_CHECK_EQ(regs.read(valueRegAddr(1)), 0u);

  // Release(2): always succeeds (bit 0 set on read), then the value
  // register shows 2.
  AIESIM_CHECK_EQ(regs.read(releaseAddr(1, 2)), 1u);
  AIESIM_CHECK_EQ(regs.read(valueRegAddr(1)), 2u);

  // AcquireGreaterEqual 3 (encoded as -3): counter is 2, fails.
  AIESIM_CHECK_EQ(regs.read(acquireAddr(1, -3)), 0u);
  AIESIM_CHECK_EQ(regs.read(valueRegAddr(1)), 2u);

  // AcquireGreaterEqual 2 (encoded as -2): counter is 2, succeeds, drops to
  // 0.
  AIESIM_CHECK_EQ(regs.read(acquireAddr(1, -2)), 1u);
  AIESIM_CHECK_EQ(regs.read(valueRegAddr(1)), 0u);

  // Plain Acquire(0): matches immediately (counter is already 0).
  AIESIM_CHECK_EQ(regs.read(acquireAddr(1, 0)), 1u);

  // Writing the value register directly (lock initialisation) bypasses
  // acquire semantics entirely.
  regs.write(valueRegAddr(1), 5);
  AIESIM_CHECK_EQ(regs.read(valueRegAddr(1)), 5u);
  AIESIM_CHECK_EQ(lm->value(1), int32_t{5});

  AIESIM_CHECK(diagnostics.empty());

  // --- A write to the acquire/release request range is unmodelled and
  // must be a loud diagnostic, not a silent no-op. ---
  regs.write(acquireAddr(2, 0), 0);
  AIESIM_CHECK_EQ(diagnostics.size(), size_t{1});
  if (!diagnostics.empty())
    AIESIM_CHECK(diagnostics[0].find("request") != std::string::npos);

  return aiesim_test::summarize("lock");
}
