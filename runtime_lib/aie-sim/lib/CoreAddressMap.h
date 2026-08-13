//===- CoreAddressMap.h -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The core's view of memory: where program memory sits and which window maps
// onto which tile's data memory.
//
// Shared so the two things that need it agree by construction. TileCorePort
// routes a running core's loads and stores through it; ElfLoader routes an
// ELF's PT_LOAD segments through the same map, because a loader must place a
// segment exactly where the core will later address it.
//
//===----------------------------------------------------------------------===//

#ifndef AIESIM_CORE_ADDRESS_MAP_H
#define AIESIM_CORE_ADDRESS_MAP_H

#include "aiesim/Components.h"

#include <cstdint>
#include <iterator>
#include <optional>

namespace aiesim {

/// Core-local program memory, which starts the core's address map.
constexpr uint32_t kProgramBase = 0x00000;

/// One 64 KB window onto some tile's data memory, as the core addresses it.
struct MemBand {
  uint32_t base;
  MemDirection dir;
};

// AIE2/AIE2P. `getMemInternalBaseAddress` returns East unconditionally on this
// generation, so East is the tile's OWN memory and the other three are
// neighbours -- not a symmetric four-way split.
constexpr MemBand kAIE2Bands[] = {
    {0x40000, MemDirection::South},
    {0x50000, MemDirection::West},
    {0x60000, MemDirection::North},
    {0x70000, MemDirection::Own},
};

constexpr uint32_t kBandSize = 0x10000;

/// A lock id as the core issues it: which module it names, and the index within
/// that module.
struct LockBand {
  MemDirection dir;
  uint32_t local;
};

/// Split a core-issued lock id into its band and local index.
///
/// The lock id space bands by direction exactly as the memory windows do, and
/// in the same order. The authority is `AIETargetModel::getLockLocalBaseIndex`,
/// which for a core tile returns `getNumLocks() * {0,1,2,3}` for South, West,
/// North and East -- so with 16 locks per core tile, 48 is the tile's OWN lock
/// 0, not an out-of-range id. Deriving the band width from the module's own
/// `count()` rather than a literal 16 keeps this right if a device reports a
/// different number.
///
/// Returns nullopt for an id past the last band, which is a genuine fault: the
/// core named a module that does not exist.
inline std::optional<LockBand> splitLockId(uint32_t id, uint32_t numLocks) {
  if (numLocks == 0)
    return std::nullopt;
  const uint32_t band = id / numLocks;
  if (band >= std::size(kAIE2Bands))
    return std::nullopt;
  return LockBand{kAIE2Bands[band].dir, id % numLocks};
}

/// The tile whose data memory and locks a band names, or null where the
/// hardware puts nothing.
///
/// `AIE2TargetModel::getMem{South,West,North,East}`
/// (lib/Dialect/AIE/IR/AIETargetModel.cpp:752-784), keeping both exclusions: a
/// band that would leave the array has no backing, and a core tile's SOUTH band
/// is empty when the tile below is a memtile, which `isLegalMemAffinity` states
/// separately (`IsMemSouth && !isMemTile(memCol, memRow)`).
///
/// The generated linker scripts say the same from the other side, which is what
/// makes this checkable rather than asserted: `op2_ElementwiseAdd`'s row-2 core
/// carries `/* No tile with memory exists to the south. */` and leaves 0x4xxxx
/// empty, while the identical kernel one row up places objectFIFO buffers at
/// 0x40400 because row 3 does have a core tile below it.
inline Tile *bandTile(Tile &core, MemDirection dir) {
  Array &array = core.getArray();
  const uint32_t col = core.getCol();
  const uint32_t row = core.getRow();
  // Array::tile bounds-checks against the partition, so the wrap at col 0 or
  // row 0 lands outside it and answers null.
  switch (dir) {
  case MemDirection::Own:
    return &core;
  case MemDirection::West:
    return array.tile(col - 1, row);
  case MemDirection::North:
    return array.tile(col, row + 1);
  case MemDirection::South:
    if (Tile *south = array.tile(col, row - 1))
      return south->getType() == TileType::Core ? south : nullptr;
    return nullptr;
  }
  return nullptr;
}

} // namespace aiesim

#endif // AIESIM_CORE_ADDRESS_MAP_H
