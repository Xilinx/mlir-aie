//===- RegisterFile.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"

using namespace aiesim;

bool RegisterFile::isClaimed(uint32_t off) const {
  for (const Range &r : ranges)
    if (off >= r.begin && off < r.end)
      return true;
  return false;
}

uint32_t RegisterFile::read(uint32_t off) const {
  // A computed register wins over stored state, wherever it was registered:
  // a component may claim a block for storage and separately expose a status
  // word inside it.
  for (const Range &r : ranges)
    if (r.read && off >= r.begin && off < r.end)
      return r.read(off);

  for (const Range &r : ranges) {
    if (off < r.begin || off >= r.end)
      continue;
    if (r.reservedWhy)
      return 0;
    auto it = stored.find(off);
    return it == stored.end() ? 0u : it->second;
  }

  // Unclaimed. Reading back a value the host itself wrote is a pass-through,
  // not an invention, so it is allowed. Reading an offset that was never
  // written AND is modelled by nothing is the dangerous case: a design polling
  // a status register we do not implement would spin on a fabricated zero.
  auto it = stored.find(off);
  if (it != stored.end())
    return it->second;
  if (onUnclaimed)
    onUnclaimed(off, /*isRead=*/true);
  return 0;
}

void RegisterFile::write(uint32_t off, uint32_t value) {
  stored[off] = value;
  bool claimed = false;
  // Handlers run after the store, so a handler that reads neighbouring
  // registers (a DMA channel control register reading its own BD, say) sees a
  // consistent window.
  for (const Range &r : ranges) {
    if (off < r.begin || off >= r.end)
      continue;
    claimed = true;
    if (r.write)
      r.write(off, value);
  }
  if (!claimed && onUnclaimed)
    onUnclaimed(off, /*isRead=*/false);
}

void RegisterFile::onWrite(uint32_t begin, uint32_t end, WriteHandler h) {
  ranges.push_back(Range{begin, end, std::move(h), nullptr, nullptr});
}

void RegisterFile::onRead(uint32_t begin, uint32_t end, ReadHandler h) {
  ranges.push_back(Range{begin, end, nullptr, std::move(h), nullptr});
}

void RegisterFile::claim(uint32_t begin, uint32_t end) {
  ranges.push_back(Range{begin, end, nullptr, nullptr, nullptr});
}

void RegisterFile::reserve(uint32_t begin, uint32_t end, const char *why) {
  ranges.push_back(Range{begin, end, nullptr, nullptr, why});
}
