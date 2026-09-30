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
///
/// The flag is STATIC because the handler installed here outlives this call:
/// Array has no way to remove one, and every check below deliberately faults
/// again outside a faulted() call. Capturing a local by reference left the
/// handler writing a byte into a dead frame -- harmless under -O2, and under
/// -O0 it landed on a saved rbp and crashed with SIGBUS.
bool faulted(Array &array, const std::function<void()> &body) {
  static bool fired;
  fired = false;
  array.setDiagnosticHandler([](const std::string &) { fired = true; });
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

  // --- Each neighbour band lands in the tile it names, and no other ---
  //
  // Aliasing is the dangerous outcome: the core would read a plausible value
  // that belongs to the wrong tile. So each band is checked against the tile
  // its DIRECTION names, taken from AIE2TargetModel::getMem* rather than from
  // the code under test -- south is (col, row-1), west is (col-1, row), north
  // is (col, row+1).
  //
  // A row-3 core's own linker script is the same claim from the other side:
  // op2_ElementwiseAdd places out_0_buff_0 at 0x40400 and out_2_buff_0 at
  // 0x60400, so the objectFIFO it runs writes into two neighbours' memories.
  {
    struct { uint32_t base; Tile *owner; uint32_t marker; } bands[] = {
        {kSouthBase, array.tile(7, 2), 0x5007'0001},
        {kWestBase, array.tile(6, 3), 0x5007'0002},
        {kNorthBase, array.tile(7, 4), 0x5007'0003},
    };
    for (auto &b : bands) {
      AIESIM_CHECK(b.owner != nullptr);
      // Written at a non-zero offset so a band that resolved to the right tile
      // but the wrong base still fails.
      const uint32_t off = 0x400;
      AIESIM_CHECK(port->write(b.base + off, &b.marker, sizeof(b.marker)));
      uint32_t viaTile = 0;
      AIESIM_CHECK(b.owner->memory()->read(off, &viaTile, sizeof(viaTile)));
      AIESIM_CHECK_EQ(viaTile, b.marker);

      uint32_t viaPort = 0;
      AIESIM_CHECK(port->read(b.base + off, &viaPort, sizeof(viaPort)));
      AIESIM_CHECK_EQ(viaPort, b.marker);
    }
    // None of it reached own memory, which is the aliasing failure.
    uint32_t ownAt400 = 0;
    AIESIM_CHECK(tile->memory()->read(0x400, &ownAt400, sizeof(ownAt400)));
    AIESIM_CHECK_EQ(ownAt400, 0u);
  }

  // --- A band whose neighbour does not exist still faults ---
  //
  // The band is in every core tile's address map; whether it is backed depends
  // on where the tile sits. These three are the exclusions getMemSouth and
  // isValidTile carry, and the first is the one designs actually hit: a row-2
  // core's south is the memtile, which has no memory adjacency to it.
  {
    struct { Tile *core; uint32_t base; } absent[] = {
        {array.tile(7, 2), kSouthBase}, // south is the memtile on row 1
        {array.tile(0, 3), kWestBase},  // column 0 has nothing west
        {array.tile(7, 5), kNorthBase}, // row 5 is the top core row
    };
    for (auto &a : absent) {
      AIESIM_CHECK(a.core != nullptr);
      AIESIM_CHECK(a.core->getType() == TileType::Core);
      std::unique_ptr<CoreMemoryPort> edge = makeTileCorePort(*a.core);
      uint32_t sink = 0;
      AIESIM_CHECK(faulted(array, [&] { edge->read(a.base, &sink, 4); }));
      AIESIM_CHECK(!edge->read(a.base, &sink, 4));
    }
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

  // --- Locks band by direction exactly as addresses do ---
  array.setDiagnosticHandler(nullptr);
  AIESIM_CHECK(tile->locks() != nullptr);
  if (tile->locks()) {
    // A core-issued lock id is BANDED by direction like an address, so the
    // tile's own lock 0 is id 3*count (48 with 16 locks per module), not 0.
    // Acquire-with-0 on a fresh lock succeeds; the port must not invent its
    // own lock state.
    const uint32_t count = tile->locks()->count();
    const uint32_t own0 = 3 * count;
    AIESIM_CHECK(port->tryAcquireLock(own0, 0));
    port->releaseLock(own0, 1);
    AIESIM_CHECK_EQ(tile->locks()->value(0), 1);

    // The other three bands reach the NEIGHBOUR's module, which is half of what
    // an objectFIFO across two tiles needs: its buffers are in the neighbour's
    // memory and the locks guarding them are in the neighbour's lock module.
    // Base indices are getLockLocalBaseIndex's: south 0, west count, north
    // 2*count, east 3*count.
    struct { uint32_t base; Tile *owner; int32_t release; } bands[] = {
        {0 * count, array.tile(7, 2), 3},
        {1 * count, array.tile(6, 3), 4},
        {2 * count, array.tile(7, 4), 5},
    };
    for (auto &b : bands) {
      AIESIM_CHECK(b.owner != nullptr && b.owner->locks() != nullptr);
      // Lock 1 rather than 0, so an id that lost its within-band index is
      // caught as well as one that lost its band.
      port->releaseLock(b.base + 1, b.release);
      AIESIM_CHECK_EQ(b.owner->locks()->value(1), b.release);
      // AcquireGreaterEqual the whole count, which only succeeds against the
      // module that just took the release.
      AIESIM_CHECK(port->tryAcquireLock(b.base + 1, -b.release));
      AIESIM_CHECK_EQ(b.owner->locks()->value(1), 0);
    }
    // And none of it touched this tile's own lock 1.
    AIESIM_CHECK_EQ(tile->locks()->value(1), 0);
  }

  // An id past the last band, and a band with no neighbour, are both faults
  // rather than a false return: false means "not yet" and the core would retry
  // a lock that can never come.
  {
    const uint32_t count = tile->locks()->count();
    AIESIM_CHECK(faulted(array, [&] { port->tryAcquireLock(4 * count, 0); }));

    Tile *rowTwo = array.tile(7, 2); // no south neighbour: row 1 is the memtile
    AIESIM_CHECK(rowTwo != nullptr);
    std::unique_ptr<CoreMemoryPort> edge = makeTileCorePort(*rowTwo);
    AIESIM_CHECK(faulted(array, [&] { edge->tryAcquireLock(0, 0); }));
    AIESIM_CHECK(faulted(array, [&] { edge->releaseLock(0, 1); }));
  }

  return aiesim_test::summarize("tile_core_port_test");
}
