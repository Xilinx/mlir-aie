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

} // namespace aiesim

#endif // AIESIM_CORE_ADDRESS_MAP_H
