//===- ElfLoader.h ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Place a compiled AIE core ELF into a tile's memories, the way the host's
// loader places it on hardware.
//
// This is what lets the model run a real kernel rather than a hand-assembled
// body: the core's first lock acquire needs the array around it, so a kernel
// enters through a Tile, not through the standalone core runner.
//
//===----------------------------------------------------------------------===//

#ifndef AIESIM_ELF_LOADER_H
#define AIESIM_ELF_LOADER_H

#include "aiesim/Array.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace aiesim {

/// Load the PT_LOAD segments of an AIE core ELF image into `tile`, routing each
/// by address through the core's own memory map: program memory, or this tile's
/// data memory. A segment addressed at a NEIGHBOUR's data band is rejected --
/// the host loader writes one tile at a time, and silently accepting it would
/// place data the neighbour's own load would then overwrite.
///
/// On success `entry` receives the ELF entry address. Returns false and sets
/// `error` on a malformed image or a segment the map cannot place.
bool loadCoreElf(Tile &tile, const void *image, size_t size, uint32_t &entry,
                 std::string &error);

/// As above, reading the image from `path`.
bool loadCoreElfFile(Tile &tile, const std::string &path, uint32_t &entry,
                     std::string &error);

} // namespace aiesim

#endif // AIESIM_ELF_LOADER_H
