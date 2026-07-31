//===- Array.cpp ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"
#include "aiesim/Components.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace aiesim;

//===----------------------------------------------------------------------===//
// Tile
//===----------------------------------------------------------------------===//

Tile::Tile(Array &array, uint32_t col, uint32_t row, TileType type)
    : array(array), col(col), row(row), type(type) {
  const DeviceModel &dev = array.device();
  switch (type) {
  case TileType::Core:
    mem = std::make_unique<Memory>(dev.coreDataMemSize);
    progMem = std::make_unique<Memory>(dev.coreProgMemSize);
    break;
  case TileType::MemTile:
    mem = std::make_unique<Memory>(dev.memTileMemSize);
    break;
  case TileType::Shim:
  case TileType::Invalid:
    break;
  }
}

Tile::~Tile() = default;

void Tile::setLocks(std::unique_ptr<LockModule> m) { lockModule = std::move(m); }
void Tile::setStreamSwitch(std::unique_ptr<StreamSwitchModule> m) {
  switchModule = std::move(m);
}
void Tile::setDma(std::unique_ptr<DmaModule> m) { dmaModule = std::move(m); }

//===----------------------------------------------------------------------===//
// Array
//===----------------------------------------------------------------------===//

Array::Array(const DeviceModel &dev, std::unique_ptr<CoreEngineFactory> engines)
    : dev(dev), engines(std::move(engines)) {
  diag = [](const std::string &message) {
    std::fprintf(stderr, "aie-sim: %s\n", message.c_str());
    std::abort();
  };

  tiles.resize(static_cast<size_t>(dev.numRows) * dev.numCols);
  for (uint32_t row = 0; row < dev.numRows; ++row)
    for (uint32_t col = 0; col < dev.numCols; ++col)
      tiles[static_cast<size_t>(row) * dev.numCols + col] =
          std::make_unique<Tile>(*this, col, row, dev.tileTypeAt(row));

  // Installation order matters: DMA looks up locks and stream ports, and the
  // core looks up all three. Steppable registration order follows, which is
  // what makes the cycle-by-cycle interleaving reproducible.
  for (auto &t : tiles)
    installLocks(*t);
  for (auto &t : tiles)
    installStreamSwitch(*t);
  for (auto &t : tiles)
    installDma(*t);
  for (auto &t : tiles)
    installCore(*t);
}

Array::~Array() = default;

Tile *Array::tile(uint32_t col, uint32_t row) {
  if (!dev.contains(col, row))
    return nullptr;
  return tiles[static_cast<size_t>(row) * dev.numCols + col].get();
}

void Array::error(const std::string &message) {
  if (diag)
    diag(message);
}

//===----------------------------------------------------------------------===//
// Register access
//===----------------------------------------------------------------------===//

// Every host register access costs one simulated cycle, which is both roughly
// right (an MMIO access is not free) and the mechanism that makes host polling
// loops make progress without threads.
static constexpr uint64_t kCyclesPerHostAccess = 1;

void Array::write32(uint64_t addr, uint32_t value) {
  DecodedAddress d = decodeAddress(dev, addr);
  if (!d.valid) {
    char buf[128];
    std::snprintf(buf, sizeof(buf),
                  "write32 to unmapped address 0x%llx (outside the %u x %u "
                  "partition)",
                  static_cast<unsigned long long>(addr), dev.numCols,
                  dev.numRows);
    error(buf);
    return;
  }
  tile(d.col, d.row)->regs().write(d.regOff, value);
  advance(kCyclesPerHostAccess);
}

uint32_t Array::read32(uint64_t addr) {
  DecodedAddress d = decodeAddress(dev, addr);
  if (!d.valid) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "read32 from unmapped address 0x%llx",
                  static_cast<unsigned long long>(addr));
    error(buf);
    return 0;
  }
  // Advance before the read, so a poll loop observes state that has moved on
  // since the previous poll.
  advance(kCyclesPerHostAccess);
  return tile(d.col, d.row)->regs().read(d.regOff);
}

void Array::writeCmd(uint8_t command, uint8_t col, uint8_t row, uint32_t word0,
                     uint32_t word1, const char *str) {
  // aie-rt emits exactly two commands, both simulator conveniences that carry
  // no architectural state: SETSTACK (0) and LOADSYM (1). See
  // third_party/aie-rt/driver/src/core/xaie_elfloader.c:44-45. Nothing to do,
  // but an unknown command means aie-rt grew a case we have not read.
  (void)col;
  (void)row;
  (void)word0;
  (void)word1;
  (void)str;
  if (command > 1) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "unknown CmdWrite command %u", command);
    error(buf);
  }
}

//===----------------------------------------------------------------------===//
// DDR
//===----------------------------------------------------------------------===//

// DDR is sparse: designs place buffers at large physical addresses and we must
// not allocate the space between them.
static constexpr uint64_t kPageShift = 16; // 64 KiB
static constexpr uint64_t kPageSize = 1ull << kPageShift;

bool Array::ddrWrite(uint64_t addr, const void *data, uint64_t size) {
  const uint8_t *src = static_cast<const uint8_t *>(data);
  while (size) {
    uint64_t page = addr >> kPageShift;
    uint64_t off = addr & (kPageSize - 1);
    uint64_t n = std::min(size, kPageSize - off);
    auto it = ddrPages.find(page);
    if (it == ddrPages.end())
      it = ddrPages.emplace(page, std::vector<uint8_t>(kPageSize, 0)).first;
    std::memcpy(it->second.data() + off, src, n);
    addr += n;
    src += n;
    size -= n;
  }
  return true;
}

bool Array::ddrRead(uint64_t addr, void *data, uint64_t size) {
  uint8_t *dst = static_cast<uint8_t *>(data);
  while (size) {
    uint64_t page = addr >> kPageShift;
    uint64_t off = addr & (kPageSize - 1);
    uint64_t n = std::min(size, kPageSize - off);
    auto it = ddrPages.find(page);
    if (it == ddrPages.end())
      std::memset(dst, 0, n);
    else
      std::memcpy(dst, it->second.data() + off, n);
    addr += n;
    dst += n;
    size -= n;
  }
  return true;
}

void Array::writeGlobal(uint64_t addr, const void *data, uint64_t size) {
  ddrWrite(addr, data, size);
  advance(kCyclesPerHostAccess);
}

void Array::readGlobal(uint64_t addr, void *data, uint64_t size) {
  advance(kCyclesPerHostAccess);
  ddrRead(addr, data, size);
}

//===----------------------------------------------------------------------===//
// Clock
//===----------------------------------------------------------------------===//

void Array::advance(uint64_t n) {
  for (uint64_t i = 0; i < n; ++i) {
    for (Steppable *s : steppables)
      s->step();
    ++cycles;
  }
}

bool Array::runUntilQuiescent(uint64_t maxCycles) {
  for (uint64_t i = 0; i < maxCycles; ++i) {
    bool worked = false;
    for (Steppable *s : steppables)
      worked |= s->step();
    ++cycles;
    if (!worked)
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// The process-wide array
//===----------------------------------------------------------------------===//

namespace {
std::unique_ptr<Array> theArray;
} // namespace

void aiesim::setCurrentArray(std::unique_ptr<Array> array) {
  theArray = std::move(array);
}

Array &aiesim::currentArray() {
  if (!theArray) {
    const char *name = std::getenv("AIE_SIM_DEVICE");
    DeviceModel dev;
    std::string err;
    if (!makeDeviceFromName(name ? name : "npu2", dev, err)) {
      std::fprintf(stderr, "aie-sim: %s\n", err.c_str());
      std::abort();
    }
    std::string loadError;
    auto engines = loadCoreEngineFactory("", loadError);
    // A missing engine is not fatal here. Designs with no core code simulate
    // fine, and a design that does enable a core gets a clear diagnostic at
    // that point instead of an unexplained failure at startup.
    theArray = std::make_unique<Array>(dev, std::move(engines));
  }
  return *theArray;
}
