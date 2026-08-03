//===- core_stall_test.cpp ------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// A structural stall reaches the readings record.
//
// The engine counts cycles separately from retired bundles, because a bundle
// held at issue costs time without retiring anything. That number was only
// reachable from llvm-aie-run's stdout; this covers the whole chain the record
// depends on -- engine, the C ABI's cycle_counts, the loader, the producer.
//
// The hazard is the part-word store: a sub-word store holds PART_WORD_STORE as
// Required for 7 cycles and every other memory form holds it as Reserved, so a
// load one bundle later waits 6. Compiled code never does this; the scheduler
// models the hazard, which is why the program below is hand-written.
//
// SKIPs (exit 77) with no engine configured.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"
#include "aiesim/Components.h"
#include "aiesim/Device.h"
#include "aiesim/Readings.h"

#include "TestSupport.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

using namespace aiesim;
using namespace aiesim::readings;

namespace {

/// Assembled with the pinned toolchain's llvm-mc, `-triple aie2p`:
///
///     movxm p0, #0x70000     ; the core-local own-memory band
///     movxm p1, #0x70010
///     mov   r0, #0x11
///     mov   r1, #0x22
///     nop x7
///     st.s16 r0, [p0, #0]    ; holds PART_WORD_STORE, Required, 7 cycles
///     lda    r2, [p0, #0]    ; holds it Reserved -> waits 6
///     nop x8
///     st.s16 r1, [p1, #0]    ; the control: followed by nops, costs nothing
///     nop x9
///     done
///
/// Independently under llvm-aie-run with --scratch=0x70000:64: bundles 31,
/// cycles 37, stall-cycles 6.
const uint8_t Program[] = {
    0x44, 0x00, 0xc0, 0x00, 0x07, 0x00, 0x44, 0x20, 0xc0, 0x02, 0x07, 0x00,
    0xb8, 0x22, 0x10, 0x18, 0xb8, 0x44, 0x50, 0x18, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x17,
    0x04, 0x00, 0x98, 0x56, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x37,
    0x04, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x00, 0x08, 0x10,
};

constexpr uint32_t kControl = 0x32000;
constexpr uint32_t kStatus = 0x32004;
constexpr uint32_t kEnable = 1u << 0;
constexpr uint32_t kReset = 1u << 1;
constexpr uint32_t kDone = 1u << 20;

bool has(const std::string &haystack, const std::string &needle) {
  return haystack.find(needle) != std::string::npos;
}

CaptureConfig config() {
  CaptureConfig c;
  c.design = "core_stall_test";
  c.runId = "test-run";
  c.device = "npu2";
  c.provenance.simVersion = "test";
  return c;
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
      return aiesim_test::summarize("core_stall_test");
    }
    std::printf("core_stall_test: SKIP (%s)\n", error.c_str());
    return 77;
  }

  DeviceModel dev;
  AIESIM_CHECK(makeDeviceFromName("npu2", dev, error));
  Array array(dev, std::move(factory));
  Tile *tile = array.tile(7, 3);
  AIESIM_CHECK(tile != nullptr);
  if (!tile)
    return aiesim_test::summarize("core_stall_test");

  AIESIM_CHECK(tile->programMemory()->write(0, Program, sizeof(Program)));
  tile->regs().write(kControl, kReset);
  tile->regs().write(kControl, kEnable);
  AIESIM_CHECK(array.runUntilQuiescent(1000));
  AIESIM_CHECK_EQ(tile->regs().read(kStatus), kDone);

  // Straight off the engine first, so a failure says which link broke.
  const CoreEngine *eng = tile->attachedCoreEngine();
  AIESIM_CHECK(eng != nullptr);
  const CoreEngine::CycleCounts counts = eng->cycleCounts();
  AIESIM_CHECK(counts.tracked);
  AIESIM_CHECK_EQ(counts.stallCycles, 6u);
  // The identity the two counters exist to express.
  AIESIM_CHECK_EQ(counts.cycles, counts.retiredBundles + counts.stallCycles);

  // Then through the record, which is where a consumer reads it. The id and
  // the value are matched in ONE substring: separately, a future scalar that
  // happened to be 6 would let this pass with the stall reading absent.
  const std::string json = capture(array, config()).toJson();
  AIESIM_CHECK(has(json, "\"id\":\"scalar/core-stall-cycles\","
                        "\"label\":\"Core-cycles lost to structural hazards\","
                        "\"quantity\":{\"value\":6,\"unit\":\"cycles\""));

  return aiesim_test::summarize("core_stall_test");
}
