//===- core_engine_test.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Drives a loadable core engine across the C ABI in aie_iss_c_abi.h.
//
// This is the only test that exercises the mlir-aie <-> llvm-aie boundary, so
// it is the one that catches an ABI drift between the two vendored copies of
// the header. Everything else in this directory models the array with the core
// held at reset.
//
// The engine is built by llvm-aie, not here, so the test SKIPS (exit 77) when
// none is configured rather than passing vacuously: a boundary test that
// silently succeeds when the far side is absent is worse than no test.
//
//===----------------------------------------------------------------------===//

#include "TestSupport.h"
#include "aiesim/CoreEngine.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace aiesim;

namespace {

/// AIE2P: `mov r0, #42; mov r1, #7; done`.
///
/// Assembled with `llvm-mc -triple aie2p -filetype=obj` and extracted with
/// `llvm-objcopy -O binary --only-section=.text`. Held as bytes rather than
/// assembled here so this test needs no Peano at build time; llvm-aie's own
/// `llvm/test/tools/llvm-aie-run` suite is what guards the encoding.
const uint8_t Program[] = {
    0xb8, 0x54, 0x10, 0x18, // mov r0, #42
    0xb8, 0x0e, 0x50, 0x18, // mov r1, #7
    0x18, 0x00, 0x08, 0x10, // done
};

/// Flat tile memory. The engine holds no memory of its own, so every fetch and
/// load the core makes lands here.
class FlatPort : public CoreMemoryPort {
public:
  FlatPort() : Bytes(4096, 0) {
    std::memcpy(Bytes.data(), Program, sizeof(Program));
  }

  bool read(uint32_t addr, void *data, uint32_t size) override {
    if (static_cast<uint64_t>(addr) + size > Bytes.size())
      return false;
    std::memcpy(data, Bytes.data() + addr, size);
    return true;
  }

  bool write(uint32_t addr, const void *data, uint32_t size) override {
    if (static_cast<uint64_t>(addr) + size > Bytes.size())
      return false;
    std::memcpy(Bytes.data() + addr, data, size);
    return true;
  }

  // This program touches no lock, stream or cascade. Refusing rather than
  // faking one keeps a future program that DOES touch them honest.
  bool tryAcquireLock(uint32_t, int32_t) override { return false; }
  void releaseLock(uint32_t, int32_t) override {}
  bool tryReadStream(uint32_t, uint32_t *, bool *) override { return false; }
  bool tryWriteStream(uint32_t, uint32_t, bool) override { return false; }
  bool tryReadCascade(void *) override { return false; }
  bool tryWriteCascade(const void *) override { return false; }
  void putChar(char c) override { Printed.push_back(c); }

  std::vector<uint8_t> Bytes;
  std::string Printed;
};

uint32_t readReg32(CoreEngine &Engine, const char *Name) {
  uint32_t Value = 0;
  AIESIM_CHECK(Engine.readRegister(Name, &Value, sizeof(Value)));
  return Value;
}

} // namespace

int main() {
  // "No engine configured" and "engine configured but unloadable" both come
  // back as a null factory, and they must not report the same way: the second
  // is exactly the ABI drift this test exists to catch, and skipping on it
  // would hide a version bump in one vendored copy of the header behind a
  // green run.
  const bool EngineRequested = std::getenv("AIE_SIM_CORE_ENGINE") ||
                               std::getenv("PEANO_INSTALL_DIR");

  std::string Error;
  std::unique_ptr<CoreEngineFactory> Factory =
      loadCoreEngineFactory("", Error);
  if (!Factory) {
    if (EngineRequested) {
      aiesim_test::reportFailure(__FILE__, __LINE__,
                                 "an engine was configured but did not load: " +
                                     Error);
      return aiesim_test::summarize("core_engine_test");
    }
    std::printf("core_engine_test: SKIP (%s)\n", Error.c_str());
    return 77;
  }
  std::printf("core_engine_test: engine is %s\n", Factory->name().c_str());

  FlatPort Port;
  std::unique_ptr<CoreEngine> Engine = Factory->create(CoreISA::AIE2P, Port);
  AIESIM_CHECK(Engine != nullptr);
  if (!Engine)
    return aiesim_test::summarize("core_engine_test");

  Engine->reset();
  Engine->setProgramCounter(0);
  AIESIM_CHECK_EQ(Engine->getProgramCounter(), 0u);

  // Bounded so a stalling or non-retiring engine fails rather than hangs.
  CoreStepResult Result = CoreStepResult::Stalled;
  unsigned Steps = 0;
  for (; Steps != 64; ++Steps) {
    Result = Engine->step();
    if (Result == CoreStepResult::Done || Result == CoreStepResult::Fault)
      break;
  }

  if (Result == CoreStepResult::Fault)
    aiesim_test::reportFailure(__FILE__, __LINE__,
                               "core faulted: " + Engine->error());
  AIESIM_CHECK(Result == CoreStepResult::Done);

  // The point of the whole boundary: values the ISS computed, read back by the
  // fabric through the C ABI.
  AIESIM_CHECK_EQ(readReg32(*Engine, "r0"), 42u);
  AIESIM_CHECK_EQ(readReg32(*Engine, "r1"), 7u);

  // An unknown name must fail rather than return a fabricated zero, which is
  // the same fault contract the register model holds on the fabric side.
  uint32_t Ignored = 0;
  AIESIM_CHECK(!Engine->readRegister("not_a_register", &Ignored, 4));

  // Write-then-read, since the core-debug register window needs both.
  uint32_t Poked = 0x1234;
  AIESIM_CHECK(Engine->writeRegister("r2", &Poked, sizeof(Poked)));
  AIESIM_CHECK_EQ(readReg32(*Engine, "r2"), 0x1234u);

  return aiesim_test::summarize("core_engine_test");
}
