//===- Memory.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"

#include <cstring>

using namespace aiesim;

bool Memory::read(uint32_t off, void *dst, uint32_t len) const {
  if (!inRange(off, len))
    return false;
  std::memcpy(dst, bytes.data() + off, len);
  return true;
}

bool Memory::write(uint32_t off, const void *src, uint32_t len) {
  if (!inRange(off, len))
    return false;
  std::memcpy(bytes.data() + off, src, len);
  if (!written.empty() && len) {
    uint32_t first = off / kTrackGranule;
    uint32_t last = (off + len - 1) / kTrackGranule;
    for (uint32_t g = first; g <= last; ++g)
      written[g / 64] |= uint64_t(1) << (g % 64);
  }
  return true;
}

void Memory::trackWrites() {
  uint32_t granules = (size() + kTrackGranule - 1) / kTrackGranule;
  written.assign((granules + 63) / 64, 0);
}

uint32_t Memory::touchedBytes() const {
  uint32_t granules = 0;
  for (uint64_t word : written)
    for (; word; word &= word - 1)
      ++granules;
  // Cannot exceed the memory even when the last granule is partial.
  uint64_t bytesTouched = uint64_t(granules) * kTrackGranule;
  return static_cast<uint32_t>(bytesTouched < size() ? bytesTouched : size());
}
