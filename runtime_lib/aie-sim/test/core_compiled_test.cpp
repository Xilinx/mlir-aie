//===- core_compiled_test.cpp ---------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// A tile runs code Peano actually compiled, rather than instructions written
// by hand to be easy to execute.
//
// This is the claim the whole component rests on. `aiecc --get-aiesim` forces
// `--xbridge` (tools/aiecc/CommandLineOptions.h), so the Vitis simulator can
// only ever run Chess output; there is no hardware-free way to run Peano's.
// Every other test here feeds the engine instructions chosen by a human, which
// cannot show that.
//
// What it adds beyond core_datapath_test, concretely: the compiler emits
// MULTI-SLOT bundles, and nothing had executed one. The first bundle of `body`
// is three instructions wide --
//
//     mova r0, #0x2a ; nopb ; movxm p0, #0x70000
//
// -- 12 bytes carrying a nested MCInst per issue slot. Hand-written tests all
// used one-instruction bundles, so the composite-decode path the design doc
// insists on ("an executor should consume the nested MCInst directly and must
// not re-derive slot boundaries") was untested against real scheduling.
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

/// `.text` of an AIE2P ELF built with the pinned Peano
/// (.cache/peano-local/integ-2026-07-31, clang 21.0.0git 0c8fe2df9d36):
///
///     void body(void) {
///       volatile int *p = (volatile int *)0x70000;
///       p[0] = 42;
///       p[1] = 7;
///       p[2] = p[0] + p[1];
///     }
///
/// compiled `--target=aie2p -O2 -c`, linked after a six-instruction entry stub
/// that sets sp, calls body at a pinned 0x20, and issues `done`. The stub is
/// hand-written because the aie2p assembler emits `jl` relocations with a null
/// symbol, so a symbolic call does not link; the BODY, which is what this test
/// is about, is untouched compiler output.
///
/// Verified independently under llvm-aie-run with --scratch=0x70000:64:
/// 21 bundles, mem = 2a 00 00 00 07 00 00 00 31 00 00 00.
const uint8_t Program[] = {
    0x44, 0x00, 0xe0, 0x09, 0x02, 0x00, 0x04, 0x01, 0x00, 0x10, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x00,
    0x08, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xb6, 0x10, 0x00, 0x30,
    0xc0, 0x01, 0x00, 0x20, 0x00, 0x00, 0x40, 0x05, 0x76, 0x10, 0x02, 0xb0,
    0xc0, 0x01, 0x80, 0x11, 0x04, 0x00, 0xe1, 0x00, 0x98, 0x31, 0x04, 0x09,
    0x98, 0x16, 0x04, 0x00, 0x98, 0x36, 0x04, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x18, 0x00, 0x28, 0x10, 0x00, 0x00, 0x44, 0x10, 0xc0, 0x00,
    0x07, 0x00, 0x98, 0x00, 0x40, 0x10, 0x98, 0x11, 0x04, 0x08, 0x00, 0x00,
};

constexpr uint32_t kControl = 0x32000;
constexpr uint32_t kStatus = 0x32004;
constexpr uint32_t kEnable = 1u << 0;
constexpr uint32_t kReset = 1u << 1;
constexpr uint32_t kDone = 1u << 20;

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
      return aiesim_test::summarize("core_compiled_test");
    }
    std::printf("core_compiled_test: SKIP (%s)\n", error.c_str());
    return 77;
  }

  DeviceModel dev;
  AIESIM_CHECK(makeDeviceFromName("npu2", dev, error));
  Array array(dev, std::move(factory));

  Tile *tile = array.tile(7, 3);
  AIESIM_CHECK(tile != nullptr);
  if (!tile)
    return aiesim_test::summarize("core_compiled_test");

  AIESIM_CHECK(tile->programMemory()->write(0, Program, sizeof(Program)));
  AIESIM_CHECK_EQ(memWord(*tile, 0), 0u);

  tile->regs().write(kControl, kReset);
  tile->regs().write(kControl, kEnable);
  AIESIM_CHECK(array.runUntilQuiescent(10000));

  // Reaching DONE means the call, its five delay slots, the multi-slot bundles
  // and the `ret lr` back into the stub all behaved -- a wrong return address
  // would run off into zeros and fault instead.
  AIESIM_CHECK_EQ(tile->regs().read(kStatus), kDone);

  // The C source's arithmetic, in the tile's own memory.
  AIESIM_CHECK_EQ(memWord(*tile, 0), 42u);
  AIESIM_CHECK_EQ(memWord(*tile, 4), 7u);
  AIESIM_CHECK_EQ(memWord(*tile, 8), 49u);

  return aiesim_test::summarize("core_compiled_test");
}
