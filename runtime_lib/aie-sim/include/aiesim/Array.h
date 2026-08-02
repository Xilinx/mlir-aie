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
// Time. There is one global cycle counter and no threads. Only components
// with outstanding work are stepped each cycle (the active set, see
// Steppable::busy() and Array::wake()); order within that set is always
// registration order, which is what keeps runs reproducible. The ess_* entry
// points call advance() so that a host polling loop (which is how aie-rt
// implements every wait) makes progress.
//
//===----------------------------------------------------------------------===//

#ifndef AIESIM_ARRAY_H
#define AIESIM_ARRAY_H

#include "aiesim/CoreEngine.h"
#include "aiesim/Device.h"

#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <unordered_map>
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

  /// Record which granules get written, for the readings memory map. Off by
  /// default so write() costs nothing in a normal run, and enabling it changes
  /// no scheduling -- an observation that alters the thing observed would cost
  /// the determinism guarantee this model is built on.
  void trackWrites();
  bool tracksWrites() const { return !written.empty(); }
  /// Bytes in granules that have been written at least once, rounded UP to the
  /// granule. An upper bound on live data, not an exact figure; the honest use
  /// is "how much of this memory did the design touch", not byte accounting.
  uint32_t touchedBytes() const;
  /// Same, restricted to [off, off+len). Granule-rounded, so a region shorter
  /// than a granule reports the whole granule.
  uint32_t touchedBytesIn(uint32_t off, uint32_t len) const;
  static constexpr uint32_t kTrackGranule = 32;

private:
  std::vector<uint8_t> bytes;
  std::vector<uint64_t> written; // granule bitmap; empty when not tracking
};

/// The per-tile register window.
///
/// Every offset falls in exactly one of three buckets, and which bucket decides
/// whether an access is trustworthy:
///
///   * CLAIMED, because a component said it owns the range. Reads return the
///     component's handler value or the stored word.
///   * RESERVED, an explicit allow-list of ranges known to read as zero. These
///     must come from the vendor's own field documentation, never from "the
///     tests seem happy".
///   * UNCLAIMED, meaning nothing models it.
///
/// An unclaimed read of an offset that was NEVER WRITTEN is a fault. That is
/// the direction which produces silent wrongness: a design polling a status
/// register nobody modelled would spin on a fabricated zero forever, hanging
/// with no diagnostic. Reading back a value the host itself wrote is a
/// pass-through rather than an invention, so it is allowed.
///
/// An earlier version returned zero for any unwritten register and justified it
/// with "matches how the tests use them", which is circular: the tests were
/// written against this model.
///
/// An unclaimed WRITE is recorded rather than fatal by default, because the
/// model will never claim every register and refusing to run until it does is
/// useless. It is still a real risk (the design configured something we will
/// not honour), so every one is reported through Array's coverage set, and
/// AIE_SIM_STRICT=1 promotes it to a fault.
class RegisterFile {
public:
  using WriteHandler = std::function<void(uint32_t off, uint32_t value)>;
  using ReadHandler = std::function<uint32_t(uint32_t off)>;
  /// Called for an access nothing claims. `isRead` distinguishes the fatal
  /// direction from the recorded one. Unset on a bare RegisterFile, which is
  /// then permissive; a Tile always sets it.
  using UnclaimedHandler = std::function<void(uint32_t off, bool isRead)>;

  uint32_t read(uint32_t off) const;
  void write(uint32_t off, uint32_t value);

  /// Registers in [begin, end) get `h` called after the stored value updates.
  /// Implies claim().
  void onWrite(uint32_t begin, uint32_t end, WriteHandler h);
  /// Registers in [begin, end) are computed rather than stored. Implies claim().
  void onRead(uint32_t begin, uint32_t end, ReadHandler h);

  /// Declare ownership of [begin, end) with no handler: plain storage a
  /// component reads back itself, such as a buffer-descriptor block.
  void claim(uint32_t begin, uint32_t end);
  /// Declare [begin, end) as architecturally reading zero. `why` must cite the
  /// evidence, and is quoted in diagnostics.
  void reserve(uint32_t begin, uint32_t end, const char *why);

  bool isClaimed(uint32_t off) const;

  void setUnclaimedHandler(UnclaimedHandler h) { onUnclaimed = std::move(h); }

private:
  struct Range {
    uint32_t begin;
    uint32_t end;
    WriteHandler write;
    ReadHandler read;
    const char *reservedWhy = nullptr;
  };
  std::map<uint32_t, uint32_t> stored;
  std::vector<Range> ranges;
  UnclaimedHandler onUnclaimed;
};

/// One component of the array that is stepped once per simulated cycle.
///
/// A cycle has TWO phases, and the split is load bearing rather than
/// decoration. `step()` may read any component's published state and may stage
/// its own outputs, but must not make an output visible to another component.
/// `commit()` then publishes what was staged. Every component sees the same
/// state during `step()`, whatever order they happen to run in.
///
/// Without this the model is a combinational mesh, not a synchronous one: a
/// component that pushed straight into a neighbour would be seen by any
/// neighbour that had not run yet this cycle, so data would cross an unbounded
/// number of tiles in one cycle travelling one way and exactly one tile per
/// cycle travelling the other. That is not slightly-wrong timing, it is a
/// model in which identical hardware behaves differently depending on which
/// way the data flows.
class Steppable {
public:
  virtual ~Steppable() = default;
  /// Advance one cycle: read published state, stage outputs. Returns true if
  /// the component did observable work, which the array uses to decide whether
  /// it has gone quiescent.
  virtual bool step() = 0;
  /// Publish whatever `step()` staged. Components with no cross-component
  /// outputs need not override this.
  virtual void commit() {}

  /// True while this component has work outstanding, INCLUDING work it
  /// cannot currently make progress on (a DMA channel stalled on a lock).
  /// Distinct from step()'s return value, which says whether this cycle did
  /// observable work: a stalled component is busy but did nothing.
  virtual bool busy() const { return false; }
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
  /// What drives this tile's core from CORE_CONTROL. A plain Steppable slot
  /// rather than its own module type: everything a caller needs to observe
  /// (running, done, faulted) is already visible through CORE_STATUS, which is
  /// also where the hardware puts it.
  void setCoreModule(std::unique_ptr<Steppable> m);

  class LockModule *locks() { return lockModule.get(); }
  class StreamSwitchModule *streamSwitch() { return switchModule.get(); }
  class DmaModule *dma() { return dmaModule.get(); }

  /// This tile's core engine, created on FIRST USE from the array's factory.
  ///
  /// Lazy rather than eager for two reasons: an engine carries a whole MC
  /// layer per core, which is wasted on the many tiles a design leaves empty;
  /// and a missing engine should be reported when a core is actually needed,
  /// not at array construction, so core-free designs still simulate. Returns
  /// null when the array has no factory or the factory declined, and it is
  /// only attempted once.
  CoreEngine *ensureCoreEngine();

  /// What this core's linker script allocated where. Optional: a design whose
  /// script was never handed over simulates exactly as before, it just cannot
  /// tell a stack overrun from a legitimate buffer write.
  void setRegionMap(class RegionMap map);
  const class RegionMap *regionMap() const { return regions.get(); }

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
  // Members destruct in reverse declaration order, and each of these three
  // depends on the one above it: the engine holds a reference to the port, and
  // the exec module drives the engine. So they are declared port, engine,
  // module and torn down module, engine, port.
  std::unique_ptr<CoreMemoryPort> corePort;
  std::unique_ptr<CoreEngine> coreEngine;
  std::unique_ptr<Steppable> coreExecModule;
  std::unique_ptr<class RegionMap> regions;
  bool coreEngineAttempted = false;
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

  /// Step the active set forward `cycles` times. An idle array (empty active
  /// set) jumps the cycle counter straight to the target instead of looping,
  /// so a design that configures nothing costs O(1) regardless of `cycles`.
  void advance(uint64_t cycles);
  /// Step until the active set is empty, or until `maxCycles` elapse. Returns
  /// false if the budget ran out, which is how a deadlocked design is
  /// reported rather than hanging.
  bool runUntilQuiescent(uint64_t maxCycles);

  /// Marks `s` active: it is stepped and committed starting next cycle.
  /// Idempotent. A component must call this on itself from its own register
  /// write handler when that write gives it new work, since nothing else
  /// will notice it needs to run -- see Steppable::busy() for the other half
  /// of how a component stays scheduled once it is already active.
  void wake(Steppable *s);

  uint64_t cycle() const { return cycles; }

  void setDiagnosticHandler(DiagnosticHandler h) { diag = std::move(h); }
  /// Report a fatal modelling problem.
  void error(const std::string &message);

  /// A write to a register nothing models. Recorded, not fatal, unless strict
  /// mode is on. See the RegisterFile comment for why the two directions are
  /// treated differently.
  void recordUnclaimedWrite(uint32_t col, uint32_t row, uint32_t regOff);

  /// Strict mode promotes an unclaimed write to a fault. Set from
  /// AIE_SIM_STRICT at construction; a test can override it.
  bool strict() const { return strictMode; }
  void setStrict(bool s) { strictMode = s; }

  /// Every distinct (col, row, regOff) written but modelled by nothing, in
  /// first-seen order. This is what makes "unmodelled" a number rather than a
  /// surprise; see docs/AIESimulator.md.
  struct UnclaimedWrite {
    uint32_t col;
    uint32_t row;
    uint32_t regOff;
  };
  const std::vector<UnclaimedWrite> &unclaimedWrites() const {
    return unclaimed;
  }
  /// Human-readable coverage summary, one line per distinct site.
  std::string unclaimedReport() const;

  /// An instruction the core engine has no semantics for. Recorded so the
  /// semantics gap is a number in the readings record rather than a string in
  /// a diagnostic, which is what lets a sweep accumulate it across designs.
  struct UnmodelledOpcode {
    uint32_t col;
    uint32_t row;
    /// Opcode name when the engine reported one, otherwise its whole message.
    std::string name;
  };
  void recordUnmodelledOpcode(uint32_t col, uint32_t row, std::string name);
  const std::vector<UnmodelledOpcode> &unmodelledOpcodes() const {
    return unmodelled;
  }

  /// Registers a component. Its index in `steppables` is both its wake()
  /// identity and its sort key in the active set, so registration order is
  /// what makes the order components step in reproducible -- see wake().
  void addSteppable(Steppable *s) {
    steppableIndex[s] = steppables.size();
    steppables.push_back(s);
  }

  CoreEngineFactory *coreEngines() { return engines.get(); }

  /// DDR behind ess_WriteGM / ess_ReadGM. Sparse, page-backed, so a design may
  /// use large physical addresses without allocating them.
  bool ddrRead(uint64_t addr, void *data, uint64_t size);
  bool ddrWrite(uint64_t addr, const void *data, uint64_t size);

private:
  // One cycle over exactly the components active at its start, in
  // registration order for both phases. See advance()/runUntilQuiescent().
  void stepOneCycle();

  DeviceModel dev;
  std::unique_ptr<CoreEngineFactory> engines;
  std::vector<std::unique_ptr<Tile>> tiles; // row-major, numRows * numCols
  std::vector<Steppable *> steppables;
  std::unordered_map<Steppable *, size_t> steppableIndex; // -> steppables idx
  // Registration indices of components due to be stepped, ordered ascending
  // so iteration order is always registration order -- the determinism
  // guarantee is this sort key, not whatever order wake() happened to be
  // called in.
  std::set<size_t> active;
  std::map<uint64_t, std::vector<uint8_t>> ddrPages;
  DiagnosticHandler diag;
  uint64_t cycles = 0;
  bool strictMode = false;
  std::vector<UnclaimedWrite> unclaimed;
  std::set<uint64_t> unclaimedSeen;
  std::vector<UnmodelledOpcode> unmodelled;
  std::set<std::string> unmodelledSeen;
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
