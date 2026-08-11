//===- tile_core_port_test.cpp --------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Tests the core-local address decode a real tile presents to its engine
// (lib/TileCorePort.cpp).
//
// The constants below are written out from their sources -- the ld script for
// program memory, AIETargetModel.h for the band bases -- rather than taken from
// TileCorePort.cpp, so a wrong table fails a test instead of agreeing with
// itself. This is the decode that decides whether a core reads its own data or
// a neighbour's, so a plausible wrong answer here is the worst failure mode the
// model has.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"
#include "aiesim/Components.h"
#include "aiesim/Device.h"

#include "TestSupport.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace aiesim;

namespace {

// lib/Targets/AIETargetLdScript.cpp: `program (RX) : ORIGIN = 0`.
constexpr uint32_t kProgramBase = 0x00000;
// AIETargetModel.h AIE2TargetModel::getMem*BaseAddress, and
// getMemInternalBaseAddress returns East, so East is a tile's OWN memory.
constexpr uint32_t kSouthBase = 0x40000;
constexpr uint32_t kWestBase = 0x50000;
constexpr uint32_t kNorthBase = 0x60000;
constexpr uint32_t kOwnBase = 0x70000;

/// Runs `body` with diagnostics captured instead of aborting, and reports
/// whether one fired. The port signals an unmodelled access by diagnosing, so
/// "did this fault" is the property under test.
bool faulted(Array &array, const std::function<void()> &body) {
  bool fired = false;
  array.setDiagnosticHandler([&](const std::string &) { fired = true; });
  body();
  return fired;
}

} // namespace

int main() {
  DeviceModel dev;
  std::string error;
  AIESIM_CHECK(makeDeviceFromName("npu2", dev, error));
  Array array(dev, nullptr);

  // A core tile: row 0 is the shim, so anything above it in a populated
  // column. 08_tile_locks uses (7, 3), which exists on npu2.
  Tile *tile = array.tile(7, 3);
  AIESIM_CHECK(tile != nullptr);
  if (!tile)
    return aiesim_test::summarize("tile_core_port_test");
  AIESIM_CHECK(tile->getType() == TileType::Core);

  std::unique_ptr<CoreMemoryPort> port = makeTileCorePort(*tile);
  AIESIM_CHECK(port != nullptr);

  // --- Program memory reads back what was loaded into it ---
  //
  // The host loads program memory with ordinary MMIO writes; the core then
  // FETCHES from core-local 0. Those are different address spaces, and this is
  // the check that the port bridges them.
  const uint8_t bundle[] = {0xb8, 0x54, 0x10, 0x18};
  AIESIM_CHECK(tile->programMemory() != nullptr);
  AIESIM_CHECK(tile->programMemory()->write(0, bundle, sizeof(bundle)));

  uint8_t got[sizeof(bundle)] = {};
  AIESIM_CHECK(port->read(kProgramBase, got, sizeof(got)));
  AIESIM_CHECK(std::memcmp(got, bundle, sizeof(bundle)) == 0);

  // --- The tile's own data memory is the EAST band, not band zero ---
  //
  // This is the assertion that catches the tempting mistake of treating
  // 0x40000 as "the local one" because it is the first band and because
  // aie-rt's DataMemAddr is 0x40000 from the HOST side.
  const uint32_t marker = 0xA5A5'1234;
  AIESIM_CHECK(port->write(kOwnBase, &marker, sizeof(marker)));
  uint32_t readBack = 0;
  AIESIM_CHECK(port->read(kOwnBase, &readBack, sizeof(readBack)));
  AIESIM_CHECK_EQ(readBack, marker);

  // And it really is the tile's own memory, not a private buffer.
  uint32_t viaTile = 0;
  AIESIM_CHECK(tile->memory()->read(0, &viaTile, sizeof(viaTile)));
  AIESIM_CHECK_EQ(viaTile, marker);

  // --- The three neighbour bands fault rather than aliasing to own memory ---
  //
  // Aliasing is the dangerous outcome: the core would read a plausible value
  // that belongs to the wrong tile.
  for (uint32_t base : {kSouthBase, kWestBase, kNorthBase}) {
    uint32_t sink = 0;
    AIESIM_CHECK(faulted(array, [&] { port->read(base, &sink, 4); }));
    AIESIM_CHECK(!port->read(base, &sink, 4));
  }

  // --- Past the end of a region faults ---
  {
    uint32_t sink = 0;
    // Program memory is 16 KB, and program space runs to the first data band,
    // so both of these are unbacked addresses inside it and must fault rather
    // than wrap onto something.
    AIESIM_CHECK(faulted(array, [&] { port->read(0x10000, &sink, 4); }));
    AIESIM_CHECK(faulted(array, [&] { port->read(0x30000, &sink, 4); }));
    // Past the own band's 64 KB.
    AIESIM_CHECK(faulted(array, [&] { port->read(kOwnBase + 0x10000, &sink, 4); }));
  }

  // --- Writing program memory faults ---
  //
  // Self-modifying code would land in memory but not in the engine's decode
  // cache, so it must refuse rather than silently diverge.
  {
    uint32_t word = 0;
    AIESIM_CHECK(faulted(array, [&] { port->write(0, &word, 4); }));
  }

  // --- Locks route to the tile's own lock module ---
  array.setDiagnosticHandler(nullptr);
  AIESIM_CHECK(tile->locks() != nullptr);
  if (tile->locks()) {
    // A core-issued lock id is BANDED by direction like an address, so the
    // tile's own lock 0 is id 3*count (48 with 16 locks per module), not 0.
    // Acquire-with-0 on a fresh lock succeeds; the port must not invent its
    // own lock state.
    const uint32_t own0 = 3 * tile->locks()->count();
    AIESIM_CHECK(port->tryAcquireLock(own0, 0));
    port->releaseLock(own0, 1);
    AIESIM_CHECK_EQ(tile->locks()->value(0), 1);
  }

  // A neighbour's band and an id past the last band are both faults, not a
  // false return: false means "not yet" and the core would retry forever.
  {
    const uint32_t count = tile->locks()->count();
    AIESIM_CHECK(faulted(array, [&] { port->tryAcquireLock(0, 0); }));
    AIESIM_CHECK(faulted(array, [&] { port->tryAcquireLock(4 * count, 0); }));
    AIESIM_CHECK(faulted(array, [&] { port->releaseLock(count, 1); }));
  }

  return aiesim_test::summarize("tile_core_port_test");
}
