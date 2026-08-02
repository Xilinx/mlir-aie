//===- CoreModule.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Drives a core from CORE_CONTROL and reports what it is doing in CORE_STATUS,
// which is how a host starts a core and how it learns the core finished. The
// engine itself only knows how to execute one bundle; deciding WHEN to do that
// is the array's job, and this is where that decision lives.
//
// Field positions are from aie-rt's parameter headers, not inferred:
// CORE_CONTROL ENABLE at bit 0 and RESET at bit 1 (RESET_DEFVAL 0x1, so a core
// comes out of reset held), CORE_STATUS CORE_DONE at bit 20 and ERROR_HALT at
// bit 19. AIE2 and AIE2P agree on all four.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Components.h"

#include <algorithm>
#include <string>

using namespace aiesim;

namespace {

constexpr uint32_t kControlEnableBit = 1u << 0;
constexpr uint32_t kControlResetBit = 1u << 1;
constexpr uint32_t kStatusErrorHaltBit = 1u << 19;
constexpr uint32_t kStatusDoneBit = 1u << 20;

/// One core's execution state.
///
/// Deliberately does NOT hold the engine pointer: the engine is created on
/// first use, and a design may write CORE_CONTROL before anything else has
/// asked for one. Fetching it per step keeps the laziness in one place.
class CoreExec final : public Steppable {
public:
  explicit CoreExec(Tile &tile) : tile(tile) {}

  /// A host wrote CORE_CONTROL.
  void control(uint32_t value) {
    const bool wantReset = value & kControlResetBit;
    const bool wantEnable = value & kControlEnableBit;

    if (wantReset) {
      // Reset wins over enable: a core held in reset does not run, whatever
      // the enable bit says.
      if (CoreEngine *engine = tile.ensureCoreEngine()) {
        engine->reset();
        engine->setProgramCounter(0);
      }
      running = false;
      done = false;
      faulted = false;
      return;
    }

    if (!wantEnable) {
      running = false;
      return;
    }

    // Enabling a core that already finished does not restart it; the host has
    // to reset it first, which is what the hardware requires too.
    if (done || faulted)
      return;
    if (!tile.ensureCoreEngine()) {
      // A host enables every core tile in its start-up sequence, including
      // tiles a design put no code on -- harmless on hardware, since an empty
      // core has nothing to run. Faulting on those would refuse designs the
      // model handles perfectly well, so the reaction depends on whether there
      // is actually a program to skip.
      if (!hasProgram()) {
        running = false;
        return;
      }
      // There IS code here and nothing can execute it, so every later result
      // would be a guess.
      if (!reportedNoEngine) {
        reportedNoEngine = true;
        tile.getArray().error(
            "tile (" + std::to_string(tile.getCol()) + ", " +
            std::to_string(tile.getRow()) +
            "): CORE_CONTROL enabled a core with a program loaded, but no core "
            "engine is available (set AIE_SIM_CORE_ENGINE or "
            "PEANO_INSTALL_DIR)");
      }
      return;
    }
    running = true;
    tile.getArray().wake(this);
  }

  uint32_t status() const {
    uint32_t value = 0;
    if (done)
      value |= kStatusDoneBit;
    if (faulted)
      value |= kStatusErrorHaltBit;
    return value;
  }

  bool step() override {
    if (!running)
      return false;
    CoreEngine *engine = tile.ensureCoreEngine();
    if (!engine) {
      running = false;
      return false;
    }

    switch (engine->step()) {
    case CoreStepResult::Retired:
      return true;
    case CoreStepResult::Stalled:
      // Busy but idle: waiting on a lock or stream is not observable work, and
      // reporting it as work would stop the array ever looking quiescent.
      return false;
    case CoreStepResult::Done:
      running = false;
      done = true;
      return true;
    case CoreStepResult::Fault: {
      running = false;
      faulted = true;
      const std::string why = engine->error();
      // Record the gap as data before reporting it as text, so a sweep can
      // accumulate "which instructions have no semantics" instead of grepping
      // diagnostics. The engine spells a missing instruction "<NAME>: no
      // semantics"; anything else is kept whole rather than mis-parsed, so an
      // engine that changes the wording degrades to a coarser key, never to a
      // wrong one.
      static constexpr char kNoSemantics[] = ": no semantics";
      const std::string::size_type at = why.find(kNoSemantics);
      tile.getArray().recordUnmodelledOpcode(
          tile.getCol(), tile.getRow(),
          at == std::string::npos ? why : why.substr(0, at));
      // An unmodelled instruction means every later result would be a guess,
      // so this is fatal rather than a status bit alone.
      tile.getArray().error("tile (" + std::to_string(tile.getCol()) + ", " +
                            std::to_string(tile.getRow()) +
                            "): core faulted: " + why);
      return true;
    }
    }
    return false;
  }

  /// Still scheduled while enabled, even on a cycle that stalled -- otherwise
  /// a core waiting on a lock another component is about to release would be
  /// dropped from the active set and never resume.
  bool busy() const override { return running; }

private:
  /// Whether anything was loaded into this tile's program memory. A design with
  /// no `aie.core` on a tile leaves it all zero, and the host still enables it.
  bool hasProgram() const {
    Memory *prog = tile.programMemory();
    if (!prog)
      return false;
    const uint8_t *bytes = prog->data();
    return std::any_of(bytes, bytes + prog->size(),
                       [](uint8_t b) { return b != 0; });
  }

  Tile &tile;
  bool running = false;
  bool done = false;
  bool faulted = false;
  bool reportedNoEngine = false;
};

} // namespace

void aiesim::installCoreExecution(Tile &tile, uint32_t controlOff,
                                  uint32_t statusOff) {
  auto exec = std::make_unique<CoreExec>(tile);
  CoreExec *raw = exec.get();
  tile.setCoreModule(std::move(exec));
  // Before the write handler below, which calls wake(): wake() silently
  // ignores a component the array has never been told about.
  tile.getArray().addSteppable(raw);

  tile.regs().onWrite(controlOff, controlOff + 4,
                      [raw](uint32_t, uint32_t value) { raw->control(value); });
  tile.regs().onRead(statusOff, statusOff + 4,
                     [raw](uint32_t) { return raw->status(); });
}
