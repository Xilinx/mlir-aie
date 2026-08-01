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
  registers.setUnclaimedHandler([this](uint32_t off, bool isRead) {
    if (isRead) {
      char buf[192];
      std::snprintf(buf, sizeof(buf),
                    "read of unmodelled register 0x%x on tile (%u, %u): "
                    "nothing claims this offset, so any value returned would "
                    "be invented",
                    off, this->col, this->row);
      this->array.error(buf);
      return;
    }
    this->array.recordUnclaimedWrite(this->col, this->row, off);
  });
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
  if (const char *s = std::getenv("AIE_SIM_STRICT"))
    strictMode = s[0] == '1';
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

void Array::recordUnclaimedWrite(uint32_t col, uint32_t row, uint32_t regOff) {
  uint64_t key = (static_cast<uint64_t>(col) << 48) |
                 (static_cast<uint64_t>(row) << 32) | regOff;
  if (!unclaimedSeen.insert(key).second)
    return;
  unclaimed.push_back({col, row, regOff});
  if (strictMode) {
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  "write to unmodelled register 0x%x on tile (%u, %u) "
                  "(AIE_SIM_STRICT=1)",
                  regOff, col, row);
    error(buf);
  }
}

std::string Array::unclaimedReport() const {
  if (unclaimed.empty())
    return "aie-sim: every register written was modelled\n";
  std::string out = "aie-sim: " + std::to_string(unclaimed.size()) +
                    " distinct register(s) written but modelled by nothing:\n";
  char buf[96];
  for (const UnclaimedWrite &u : unclaimed) {
    std::snprintf(buf, sizeof(buf), "  tile (%u, %u) offset 0x%x\n", u.col,
                  u.row, u.regOff);
    out += buf;
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Register access
//===----------------------------------------------------------------------===//

// Host READS advance the clock; host writes do not.
//
// Advancing on reads is what makes a polling loop progress without threads,
// and polling is how aie-rt implements every wait. Leaving writes free is not
// laziness, it is what keeps a read-modify-write atomic: aie-rt's masked
// register update (XAie_SimIO_MaskWrite32,
// third_party/aie-rt/driver/src/io_backend/ext/xaie_sim.c) is a Read32
// followed by a Write32, and if the write also advanced, the array could
// evolve between the sampled value and the applied one and clobber a
// hardware-driven bit. Real hardware treats that pair as back-to-back at bus
// speed. We cannot see that the two accesses are a pair, so the fix is to make
// the second one cost nothing.
//
// Consequence worth knowing: a burst of configuration writes takes zero
// simulated time. That is closer to the truth than charging each one a core
// cycle, since configuration runs far faster than the array it configures.
static constexpr uint64_t kCyclesPerHostRead = 1;

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
  advance(kCyclesPerHostRead);
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
}

void Array::readGlobal(uint64_t addr, void *data, uint64_t size) {
  advance(kCyclesPerHostRead);
  ddrRead(addr, data, size);
}

//===----------------------------------------------------------------------===//
// Clock
//===----------------------------------------------------------------------===//

void Array::wake(Steppable *s) {
  auto it = steppableIndex.find(s);
  if (it == steppableIndex.end())
    return; // Not a registered component: addSteppable() always precedes any
            // write handler that could call wake(), so this is unreachable
            // outside a bug in a component itself.
  active.insert(it->second);
}

// Two phases per cycle, over the active set only. Every active component
// reads the same published state during step(), regardless of registration
// order, and only then does commit() make the new state visible -- see the
// comment on Steppable. `active` is a std::set<size_t> of registration
// indices, so both loops below run in registration order regardless of
// wake() call order: that is the determinism guarantee.
//
// The set consumed by both phases is snapshotted into `current` up front. A
// component can still call wake() on ANOTHER component from inside step()/
// commit() this same cycle -- a stream switch handing a word to a neighbour
// tile's switch, or a DMA channel depositing into its tile's switch, both do
// (StreamSwitch.cpp, Dma.cpp): that neighbour may otherwise have nothing of
// its own to report busy() about yet. Such a wake() only ever targets a
// component outside `current` (everything inside it is re-evaluated below
// regardless), so the fix-up below must REMOVE non-requalifiers from
// `active` rather than rebuild it from `current` -- a rebuild would discard
// exactly those mid-cycle wake() calls.
void Array::stepOneCycle() {
  std::vector<size_t> current(active.begin(), active.end());
  std::vector<bool> stepped(current.size());
  for (size_t i = 0; i < current.size(); ++i)
    stepped[i] = steppables[current[i]]->step();
  for (size_t idx : current)
    steppables[idx]->commit();

  // A component that was active this cycle stays active iff it did work or
  // still has work outstanding; otherwise it leaves the set until something
  // wakes it again.
  for (size_t i = 0; i < current.size(); ++i)
    if (!stepped[i] && !steppables[current[i]]->busy())
      active.erase(current[i]);
  ++cycles;
}

void Array::advance(uint64_t n) {
  while (n > 0) {
    if (active.empty()) {
      // Nothing outstanding and nothing to wake it: every remaining cycle is
      // identical to this one, so skip straight to the target instead of
      // looping n times over an empty active set.
      cycles += n;
      return;
    }
    stepOneCycle();
    --n;
  }
}

bool Array::runUntilQuiescent(uint64_t maxCycles) {
  for (uint64_t i = 0; i < maxCycles; ++i) {
    stepOneCycle();
    if (active.empty())
      return true;
  }
  return active.empty();
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
