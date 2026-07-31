//===- Array.h --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The simulated array: tiles, memories, registers, DDR, and the clock.
//
// Everything the host does arrives as a register or memory access at the
// ess_* entry points, so `Array` is deliberately the only object that knows
// how those map onto state. Components (locks, DMA, stream switch, core) read
// their configuration out of the tile register file rather than being
// configured through a side API, which is what keeps the model honest: it
// simulates what the configuration code actually programmed.
//
// Time. There is one global cycle counter and no threads. `advance(n)` steps
// every component n times in a fixed order, and the ess_* entry points call it
// so that a host polling loop (which is how aie-rt implements every wait)
// makes progress. This is what makes runs reproducible.
//
//===----------------------------------------------------------------------===//

#ifndef AIESIM_ARRAY_H
#define AIESIM_ARRAY_H

#include "aiesim/CoreEngine.h"
#include "aiesim/Device.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace aiesim {

class Tile;
class Array;

/// A flat, zero-initialised byte-addressable memory.
class Memory {
public:
  explicit Memory(uint32_t sizeBytes) : bytes(sizeBytes, 0) {}

  uint32_t size() const { return static_cast<uint32_t>(bytes.size()); }
  bool inRange(uint32_t off, uint32_t len) const {
    return static_cast<uint64_t>(off) + len <= bytes.size();
  }

  bool read(uint32_t off, void *dst, uint32_t len) const;
  bool write(uint32_t off, const void *src, uint32_t len);

  uint8_t *data() { return bytes.data(); }
  const uint8_t *data() const { return bytes.data(); }

private:
  std::vector<uint8_t> bytes;
};

/// The per-tile register window.
///
/// Registers are stored sparsely: a register that was never written reads back
/// as zero, which matches how the tests use them and keeps a full array cheap.
/// Registers with side effects (lock requests, core control, DMA channel
/// control, stream-switch config) additionally run a handler, registered by
/// the component that owns them.
class RegisterFile {
public:
  using WriteHandler = std::function<void(uint32_t off, uint32_t value)>;
  using ReadHandler = std::function<uint32_t(uint32_t off)>;

  uint32_t read(uint32_t off) const;
  void write(uint32_t off, uint32_t value);

  /// Registers in [begin, end) get `h` called after the stored value updates.
  void onWrite(uint32_t begin, uint32_t end, WriteHandler h);
  /// Registers in [begin, end) are computed rather than stored.
  void onRead(uint32_t begin, uint32_t end, ReadHandler h);

private:
  struct Range {
    uint32_t begin;
    uint32_t end;
    WriteHandler write;
    ReadHandler read;
  };
  std::map<uint32_t, uint32_t> stored;
  std::vector<Range> ranges;
};

/// One component of the array that is stepped once per simulated cycle.
class Steppable {
public:
  virtual ~Steppable() = default;
  /// Advance one cycle. Returns true if the component did observable work,
  /// which the array uses to decide whether it has gone quiescent.
  virtual bool step() = 0;
};

/// One tile. Which sub-objects are present depends on the tile type.
class Tile {
public:
  Tile(Array &array, uint32_t col, uint32_t row, TileType type);
  ~Tile();

  Array &getArray() const { return array; }
  uint32_t getCol() const { return col; }
  uint32_t getRow() const { return row; }
  TileType getType() const { return type; }

  RegisterFile &regs() { return registers; }
  /// Data memory, or null for a shim tile.
  Memory *memory() { return mem.get(); }
  /// Program memory, or null for anything but a core tile.
  Memory *programMemory() { return progMem.get(); }

  /// Components install themselves with these; the tile owns their lifetime.
  /// Each kind is set once, by its installer in Components.h.
  void setLocks(std::unique_ptr<class LockModule> m);
  void setStreamSwitch(std::unique_ptr<class StreamSwitchModule> m);
  void setDma(std::unique_ptr<class DmaModule> m);

  class LockModule *locks() { return lockModule.get(); }
  class StreamSwitchModule *streamSwitch() { return switchModule.get(); }
  class DmaModule *dma() { return dmaModule.get(); }

private:
  Array &array;
  uint32_t col;
  uint32_t row;
  TileType type;
  RegisterFile registers;
  std::unique_ptr<Memory> mem;
  std::unique_ptr<Memory> progMem;
  std::unique_ptr<class LockModule> lockModule;
  std::unique_ptr<class StreamSwitchModule> switchModule;
  std::unique_ptr<class DmaModule> dmaModule;
};

/// How the simulator reports a problem. Nothing here is recoverable: an
/// unmapped access or an unmodelled instruction means the simulation result
/// would be a lie, so the default handler prints and aborts. Tests override it
/// to assert on the message instead.
using DiagnosticHandler = std::function<void(const std::string &message)>;

class Array {
public:
  Array(const DeviceModel &dev, std::unique_ptr<CoreEngineFactory> engines);
  ~Array();

  const DeviceModel &device() const { return dev; }

  /// Tile at (col, row), or null if outside the partition.
  Tile *tile(uint32_t col, uint32_t row);

  /// The ess_* entry points land here.
  void write32(uint64_t addr, uint32_t value);
  uint32_t read32(uint64_t addr);
  void writeGlobal(uint64_t addr, const void *data, uint64_t size);
  void readGlobal(uint64_t addr, void *data, uint64_t size);
  void writeCmd(uint8_t command, uint8_t col, uint8_t row, uint32_t word0,
                uint32_t word1, const char *str);

  /// Step every component `cycles` times.
  void advance(uint64_t cycles);
  /// Step until nothing does observable work, or until `maxCycles` elapse.
  /// Returns false if the budget ran out, which is how a deadlocked design is
  /// reported rather than hanging.
  bool runUntilQuiescent(uint64_t maxCycles);

  uint64_t cycle() const { return cycles; }

  void setDiagnosticHandler(DiagnosticHandler h) { diag = std::move(h); }
  /// Report a fatal modelling problem.
  void error(const std::string &message);

  /// Registers a component to be stepped once per cycle, in registration
  /// order. Order is fixed so runs are reproducible.
  void addSteppable(Steppable *s) { steppables.push_back(s); }

  CoreEngineFactory *coreEngines() { return engines.get(); }

  /// DDR behind ess_WriteGM / ess_ReadGM. Sparse, page-backed, so a design may
  /// use large physical addresses without allocating them.
  bool ddrRead(uint64_t addr, void *data, uint64_t size);
  bool ddrWrite(uint64_t addr, const void *data, uint64_t size);

private:
  DeviceModel dev;
  std::unique_ptr<CoreEngineFactory> engines;
  std::vector<std::unique_ptr<Tile>> tiles; // row-major, numRows * numCols
  std::vector<Steppable *> steppables;
  std::map<uint64_t, std::vector<uint8_t>> ddrPages;
  DiagnosticHandler diag;
  uint64_t cycles = 0;
};

/// The array the ess_* symbols operate on. A process simulates one array; the
/// host program's aie-rt calls have no way to say which one they mean, so this
/// is a deliberate singleton rather than a design flaw to fix later.
///
/// Created on first use from $AIE_SIM_DEVICE (default "npu2"), overridable by
/// a test that calls setCurrentArray first.
Array &currentArray();
void setCurrentArray(std::unique_ptr<Array> array);

} // namespace aiesim

#endif // AIESIM_ARRAY_H
