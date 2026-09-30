//===- core_exec_test.cpp -------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Starting a core the way a host does: write CORE_CONTROL, let the array run,
// read CORE_STATUS and the register window.
//
// core_dispatch_test drives the engine directly, which proves the register
// bridge. This one never touches CoreEngine at all -- it only writes and reads
// registers, so it is the first test in which the simulator behaves like the
// device rather than like a library.
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

/// AIE2P `mov r0, #42; mov r1, #7; done`.
const uint8_t Program[] = {
    0xb8, 0x54, 0x10, 0x18, 0xb8, 0x0e, 0x50, 0x18, 0x18, 0x00, 0x08, 0x10,
};

// aie-rt: CORE_CONTROL ENABLE bit 0, RESET bit 1; CORE_STATUS ERROR_HALT bit
// 19, CORE_DONE bit 20. Written out here rather than shared with
// CoreModule.cpp so a changed constant fails a test.
constexpr uint32_t kControl = 0x32000;
constexpr uint32_t kStatus = 0x32004;
constexpr uint32_t kEnable = 1u << 0;
constexpr uint32_t kReset = 1u << 1;
constexpr uint32_t kDone = 1u << 20;
constexpr uint32_t kR0 = 0x31000;
constexpr uint32_t kR1 = 0x31010;

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
      return aiesim_test::summarize("core_exec_test");
    }
    std::printf("core_exec_test: SKIP (%s)\n", error.c_str());
    return 77;
  }

  DeviceModel dev;
  AIESIM_CHECK(makeDeviceFromName("npu2", dev, error));
  Array array(dev, std::move(factory));

  Tile *tile = array.tile(7, 3);
  AIESIM_CHECK(tile != nullptr);
  if (!tile)
    return aiesim_test::summarize("core_exec_test");

  // The host loads the program, then takes the core out of reset and enables
  // it -- the same two register writes aie-rt makes.
  AIESIM_CHECK(tile->programMemory()->write(0, Program, sizeof(Program)));

  // Out of reset, a core reports neither done nor halted.
  AIESIM_CHECK_EQ(tile->regs().read(kStatus), 0u);

  tile->regs().write(kControl, kReset);
  tile->regs().write(kControl, kEnable);

  // The array runs it. Quiescence is the core finishing, not a fixed count.
  AIESIM_CHECK(array.runUntilQuiescent(1000));

  // --- CORE_STATUS reports DONE, and nothing else ---
  AIESIM_CHECK_EQ(tile->regs().read(kStatus), kDone);

  // --- and the results are readable through the window ---
  AIESIM_CHECK_EQ(tile->regs().read(kR0), 42u);
  AIESIM_CHECK_EQ(tile->regs().read(kR1), 7u);

  // --- Enabling a finished core does not restart it ---
  //
  // The host has to reset first, which is what the hardware requires. Without
  // this, a status poll that re-enables would silently re-run the program.
  tile->regs().write(kControl, kEnable);
  AIESIM_CHECK(array.runUntilQuiescent(1000));
  AIESIM_CHECK_EQ(tile->regs().read(kStatus), kDone);

  // --- Reset clears DONE and rewinds, so the core can run again ---
  tile->regs().write(kControl, kReset);
  AIESIM_CHECK_EQ(tile->regs().read(kStatus), 0u);
  AIESIM_CHECK_EQ(tile->regs().read(kR0), 0u);

  tile->regs().write(kControl, kEnable);
  AIESIM_CHECK(array.runUntilQuiescent(1000));
  AIESIM_CHECK_EQ(tile->regs().read(kStatus), kDone);
  AIESIM_CHECK_EQ(tile->regs().read(kR0), 42u);

  // --- A core left disabled never runs ---
  if (Tile *idle = array.tile(6, 3)) {
    AIESIM_CHECK(idle->programMemory()->write(0, Program, sizeof(Program)));
    AIESIM_CHECK(array.runUntilQuiescent(1000));
    AIESIM_CHECK_EQ(idle->regs().read(kStatus), 0u);
    AIESIM_CHECK_EQ(idle->regs().read(kR0), 0u);
  }

  return aiesim_test::summarize("core_exec_test");
}
