//===- core_datapath_test.cpp ---------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// A core that actually moves data: stores and loads through the core-local
// data band, into the tile's real Memory.
//
// core_exec_test only proves a core runs; every instruction in it is
// register-to-register. This is the first test in which the DATA path is
// executed -- program fetch and data access take different branches of
// TileCorePort's decode, and until now only the fetch branch had ever been
// reached by a running core.
//
// It is also the check that the core's 0x70000 really is this tile's own
// memory and not a private buffer: the test reads the results back through
// Tile::memory(), which the DMAs and the host see too.
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
#include <string>

using namespace aiesim;

namespace {

/// AIE2P, assembled with llvm-mc from:
///
///     movxm p0, #458752     ; 0x70000, the core-local own-memory band
///     mov   r0, #42
///     mov   r1, #7
///     st    r0, [p0], #4    ; post-increment: writes at the PRE-increment
///     nop                   ;   address, so 42 lands at 0x70000
///     st    r1, [p0], #4    ;   and 7 at 0x70004
///     nop
///     lda   r2, [p0, #-8]   ; read both back
///     lda   r3, [p0, #-4]
///     nop x6                ; load-use distance
///     add   r4, r2, r3      ; 49
///     st    r4, [p0, #0]    ; at 0x70008
///     done
///
/// Verified independently under llvm-aie-run with --scratch=0x70000:64,
/// which is a flat memory: r4 = 0x31 and mem = 2a 00 00 00 07 00 00 00
/// 31 00 00 00. Running it HERE must give the same answer through a real
/// tile, which is the point.
const uint8_t Program[] = {
    0x44, 0x00, 0xc0, 0x00, 0x07, 0x00, 0xb8, 0x54, 0x10, 0x18, 0xb8, 0x0e,
    0x50, 0x18, 0x98, 0x11, 0x1c, 0x08, 0x00, 0x00, 0x98, 0x31, 0x1c, 0x08,
    0x00, 0x00, 0x98, 0x56, 0xe4, 0x00, 0x98, 0x76, 0xf4, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x98, 0x30,
    0x88, 0x10, 0x98, 0x91, 0x04, 0x08, 0x18, 0x00, 0x08, 0x10, 0x04, 0x00,
    0x00, 0x00, 0x00, 0x00,
};

constexpr uint32_t kControl = 0x32000;
constexpr uint32_t kStatus = 0x32004;
constexpr uint32_t kEnable = 1u << 0;
constexpr uint32_t kReset = 1u << 1;
constexpr uint32_t kDone = 1u << 20;
constexpr uint32_t kR4 = 0x31040;

uint32_t memWord(Tile &tile, uint32_t off) {
  uint32_t value = 0;
  AIESIM_CHECK(tile.memory()->read(off, &value, sizeof(value)));
  return value;
}

} // namespace

int main() {
  const bool engineRequested = std::getenv("AIE_SIM_CORE_ENGINE") ||
                               std::getenv("PEANO_INSTALL_DIR");
  std::string error;
  std::unique_ptr<CoreEngineFactory> factory =
      loadCoreEngineFactory("", error);
  if (!factory) {
    if (engineRequested) {
      aiesim_test::reportFailure(__FILE__, __LINE__,
                                 "an engine was configured but did not load: " +
                                     error);
      return aiesim_test::summarize("core_datapath_test");
    }
    std::printf("core_datapath_test: SKIP (%s)\n", error.c_str());
    return 77;
  }

  DeviceModel dev;
  AIESIM_CHECK(makeDeviceFromName("npu2", dev, error));
  Array array(dev, std::move(factory));

  Tile *tile = array.tile(7, 3);
  AIESIM_CHECK(tile != nullptr);
  if (!tile)
    return aiesim_test::summarize("core_datapath_test");

  AIESIM_CHECK(tile->programMemory()->write(0, Program, sizeof(Program)));

  // Data memory starts clear, so a passing result cannot come from stale
  // contents.
  AIESIM_CHECK_EQ(memWord(*tile, 0), 0u);

  tile->regs().write(kControl, kReset);
  tile->regs().write(kControl, kEnable);
  AIESIM_CHECK(array.runUntilQuiescent(1000));
  AIESIM_CHECK_EQ(tile->regs().read(kStatus), kDone);

  // --- The stores landed in the TILE's memory, at the right offsets ---
  //
  // Core-local 0x70000 is tile-memory offset 0. Getting the band base wrong
  // would put these somewhere else, or fault.
  AIESIM_CHECK_EQ(memWord(*tile, 0), 42u);
  AIESIM_CHECK_EQ(memWord(*tile, 4), 7u);

  // --- The loads read back what the stores wrote ---
  //
  // 49 could only be computed by loading both values back through the port,
  // so this is the read direction as much as the arithmetic.
  AIESIM_CHECK_EQ(memWord(*tile, 8), 49u);
  AIESIM_CHECK_EQ(tile->regs().read(kR4), 49u);

  // --- Same answer as the flat-memory reference run ---
  //
  // llvm-aie-run gave 2a/07/31 for these three words with a plain scratch
  // buffer; a real tile must not differ.
  AIESIM_CHECK_EQ(memWord(*tile, 0), 0x2Au);
  AIESIM_CHECK_EQ(memWord(*tile, 4), 0x07u);
  AIESIM_CHECK_EQ(memWord(*tile, 8), 0x31u);

  return aiesim_test::summarize("core_datapath_test");
}
