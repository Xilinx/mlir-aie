//===- core_dispatch_test.cpp ---------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The whole chain, end to end: a host reads a tile's MMIO register window and
// gets back a value the instruction simulator computed.
//
//   register offset -> coreScalarRegister() -> CoreEngine::readRegister()
//                                           -> the C ABI -> llvm-aie
//
// Every link has its own test (core_register_map_test, tile_core_port_test,
// core_engine_test); this is the one that checks they are actually connected,
// which is the property none of them can see alone.
//
// SKIPs (exit 77) with no engine configured, for the same reason
// core_engine_test does: the engine is built by llvm-aie.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"
#include "aiesim/Components.h"
#include "aiesim/Device.h"

#include "TestSupport.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace aiesim;

namespace {

/// AIE2P `mov r0, #42; mov r1, #7; done`, assembled with llvm-mc. Same program
/// as core_engine_test; the encoding is guarded by llvm-aie's own lit suite.
const uint8_t Program[] = {
    0xb8, 0x54, 0x10, 0x18, // mov r0, #42
    0xb8, 0x0e, 0x50, 0x18, // mov r1, #7
    0x18, 0x00, 0x08, 0x10, // done
};

// XAIE2PGBL_CORE_MODULE_CORE_*, the offsets a host actually reads.
constexpr uint32_t kR0 = 0x31000;
constexpr uint32_t kR1 = 0x31010;
constexpr uint32_t kPC = 0x30E00;
constexpr uint32_t kWL0 = 0x30800; // in the window, NOT scalar-mapped

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
      return aiesim_test::summarize("core_dispatch_test");
    }
    std::printf("core_dispatch_test: SKIP (%s)\n", error.c_str());
    return 77;
  }

  DeviceModel dev;
  AIESIM_CHECK(makeDeviceFromName("npu2", dev, error));
  Array array(dev, std::move(factory));

  Tile *tile = array.tile(7, 3);
  AIESIM_CHECK(tile != nullptr);
  if (!tile)
    return aiesim_test::summarize("core_dispatch_test");

  // --- Before any engine exists, the window reads its reset zeros ---
  //
  // ensureCoreEngine() is lazy, so this also checks that merely READING the
  // window does not fabricate a value before the core has run.
  AIESIM_CHECK_EQ(tile->regs().read(kR0), 0u);

  // --- The host loads program memory, exactly as it does on hardware ---
  AIESIM_CHECK(tile->programMemory() != nullptr);
  AIESIM_CHECK(tile->programMemory()->write(0, Program, sizeof(Program)));

  // --- Run the core ---
  CoreEngine *engine = tile->ensureCoreEngine();
  AIESIM_CHECK(engine != nullptr);
  if (!engine)
    return aiesim_test::summarize("core_dispatch_test");

  engine->reset();
  engine->setProgramCounter(0);
  CoreStepResult result = CoreStepResult::Stalled;
  for (unsigned steps = 0; steps != 64; ++steps) {
    result = engine->step();
    if (result == CoreStepResult::Done || result == CoreStepResult::Fault)
      break;
  }
  if (result == CoreStepResult::Fault)
    aiesim_test::reportFailure(__FILE__, __LINE__,
                               "core faulted: " + engine->error());
  AIESIM_CHECK(result == CoreStepResult::Done);

  // --- THE POINT: read the results through the MMIO window ---
  //
  // Nothing here touches the engine directly. This is what
  // mlir_aie_read_buffer / a debugger / test_library.cpp would do.
  AIESIM_CHECK_EQ(tile->regs().read(kR0), 42u);
  AIESIM_CHECK_EQ(tile->regs().read(kR1), 7u);

  // PC is not in the engine's register file; it must still answer, via
  // getProgramCounter().
  AIESIM_CHECK_EQ(tile->regs().read(kPC), engine->getProgramCounter());

  // A window offset with no scalar mapping still reads 0 rather than faulting
  // or returning a neighbouring register: the window is wider than the map,
  // and a host walking the whole thing must not take the array down.
  AIESIM_CHECK_EQ(tile->regs().read(kWL0), 0u);

  // --- A tile with no core keeps its old behaviour ---
  //
  // installCore only runs on core tiles, so a shim tile must not have grown a
  // register window as a side effect of any of this.
  if (Tile *shim = array.tile(7, 0))
    AIESIM_CHECK(shim->ensureCoreEngine() == nullptr);

  return aiesim_test::summarize("core_dispatch_test");
}
