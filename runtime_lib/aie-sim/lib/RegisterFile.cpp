//===- RegisterFile.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"

using namespace aiesim;

uint32_t RegisterFile::read(uint32_t off) const {
  for (const Range &r : ranges)
    if (r.read && off >= r.begin && off < r.end)
      return r.read(off);
  auto it = stored.find(off);
  return it == stored.end() ? 0u : it->second;
}

void RegisterFile::write(uint32_t off, uint32_t value) {
  stored[off] = value;
  // Handlers run after the store, so a handler that reads neighbouring
  // registers (a DMA channel control register reading its own BD, say) sees a
  // consistent window.
  for (const Range &r : ranges)
    if (r.write && off >= r.begin && off < r.end)
      r.write(off, value);
}

void RegisterFile::onWrite(uint32_t begin, uint32_t end, WriteHandler h) {
  ranges.push_back(Range{begin, end, std::move(h), nullptr});
}

void RegisterFile::onRead(uint32_t begin, uint32_t end, ReadHandler h) {
  ranges.push_back(Range{begin, end, nullptr, std::move(h)});
}
