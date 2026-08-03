//===- core_lock_test.cpp -------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// A core executes acq and rel against the tile's real locks.
//
// These are the blocking lock PORTS, not memory-mapped registers, so they are
// the one path where the engine reaches the fabric outside CoreMemoryPort's
// read/write. lock_test drives the same module through the register bus; this
// drives it from instructions, which is the half that was untested.
//
// Both halves of the port contract are covered: an acquire whose counter is
// high enough retires and the core reaches DONE, and one whose counter is too
// low STALLS -- the core never reaches DONE and the counter is untouched. The
// second is what says the stall is a real wait rather than a skipped
// instruction, which would have let the program finish with the lock never
// taken.
//
// SKIPs (exit 77) with no engine configured.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"
#include "aiesim/Components.h"
#include "aiesim/Device.h"

#include "TestSupport.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

using namespace aiesim;

namespace {

/// Assembled with the pinned toolchain's llvm-mc, `-triple aie2p`:
///
///     mov r0, #N        ; N released
///     mov r1, #-2       ; AcquireGreaterEqual 2 (the sign picks the mode)
///     nop x7            ; def-to-use distance
///     rel #0, r0
///     nop x7
///     acq #0, r1
///     done
///
/// The two programs differ in ONE byte, the mov immediate, so nothing but the
/// released count separates the passing case from the blocking one.
const uint8_t ReleaseTwo[] = {
    0xb8, 0x04, 0x10, 0x18, 0xb8, 0xfc, 0x5f, 0x18, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x08,
    0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x02, 0x10, 0x18, 0x00, 0x08, 0x10,
};

const uint8_t ReleaseOne[] = {
    0xb8, 0x02, 0x10, 0x18, 0xb8, 0xfc, 0x5f, 0x18, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x08,
    0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x02, 0x10, 0x18, 0x00, 0x08, 0x10,
};

constexpr uint32_t kControl = 0x32000;
constexpr uint32_t kStatus = 0x32004;
constexpr uint32_t kEnable = 1u << 0;
constexpr uint32_t kReset = 1u << 1;
constexpr uint32_t kDone = 1u << 20;
constexpr uint32_t kErrorHalt = 1u << 19;

/// \returns the lock-0 counter after running \p Program to quiescence, with
/// \p ReachedDone and \p Faulted taken from CORE_STATUS. A fresh array per
/// program: lock counters carry over otherwise, and the starting count is the
/// whole variable under test.
int32_t run(const uint8_t *Program, size_t Size, bool &ReachedDone,
            bool &Faulted) {
  std::string error;
  std::unique_ptr<CoreEngineFactory> factory = loadCoreEngineFactory("", error);
  AIESIM_CHECK(factory != nullptr);
  DeviceModel dev;
  AIESIM_CHECK(makeDeviceFromName("npu2", dev, error));
  Array array(dev, std::move(factory));
  Tile *tile = array.tile(7, 3);
  AIESIM_CHECK(tile != nullptr);
  if (!tile) {
    ReachedDone = false;
    Faulted = true;
    return 0;
  }

  AIESIM_CHECK(tile->programMemory()->write(0, Program, Size));
  AIESIM_CHECK_EQ(tile->locks()->value(0), 0);

  tile->regs().write(kControl, kReset);
  tile->regs().write(kControl, kEnable);
  array.runUntilQuiescent(1000);
  const uint32_t status = tile->regs().read(kStatus);
  ReachedDone = (status & kDone) != 0;
  Faulted = (status & kErrorHalt) != 0;
  return tile->locks()->value(0);
}

} // namespace

int main() {
  const bool engineRequested = std::getenv("AIE_SIM_CORE_ENGINE") ||
                               std::getenv("PEANO_INSTALL_DIR");
  std::string error;
  std::unique_ptr<CoreEngineFactory> factory = loadCoreEngineFactory("", error);
  if (!factory) {
    if (engineRequested) {
      aiesim_test::reportFailure(__FILE__, __LINE__,
                                 "an engine was configured but did not load: " +
                                     error);
      return aiesim_test::summarize("core_lock_test");
    }
    std::printf("core_lock_test: SKIP (%s)\n", error.c_str());
    return 77;
  }

  // Released 2, acquires 2: the acquire retires and takes the count back down.
  bool done = false, faulted = false;
  AIESIM_CHECK_EQ(run(ReleaseTwo, sizeof(ReleaseTwo), done, faulted), 0);
  AIESIM_CHECK(done);
  AIESIM_CHECK(!faulted);

  // Released 1, acquires 2: the core is still waiting at the acquire, and the
  // one release it did perform is still standing. ERROR_HALT clear is what
  // separates waiting from having faulted -- both would leave the counter at
  // 1 and the core short of DONE.
  AIESIM_CHECK_EQ(run(ReleaseOne, sizeof(ReleaseOne), done, faulted), 1);
  AIESIM_CHECK(!done);
  AIESIM_CHECK(!faulted);

  return aiesim_test::summarize("core_lock_test");
}
