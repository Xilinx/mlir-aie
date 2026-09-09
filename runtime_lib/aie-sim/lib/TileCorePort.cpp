//===- TileCorePort.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The CoreMemoryPort a real tile presents to its core engine.
//
// Addresses here are CORE-LOCAL: what the instruction stream computes, which is
// not the tile's MMIO offset. The two regions the core can reach are disjoint,
// which is why one address space serves both instruction fetch and data access:
//
//   program  [0, 0x20000)   ld script's `program (RX) : ORIGIN = 0`
//                           (lib/Targets/AIETargetLdScript.cpp), of which only
//                           the first ProgMemSize bytes physically exist
//   data     0x40000 +      four 64 KB neighbour bands, south/west/north/east
//
// Band bases are AIETargetModel.h's getMem{South,West,North,East}BaseAddress
// for AIE2/AIE2P, and aie-rt agrees on where data starts from the host side
// (DataMemAddr = 0x40000, xaie2pgbl_reginit.c). AIE1 packs the same four bands
// at 0x20000 with 32 KB each; it is out of scope here.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Components.h"
#include "CoreAddressMap.h"

#include <cstdio>
#include <cstring>
#include <optional>

using namespace aiesim;

namespace {


class TileCorePort final : public CoreMemoryPort {
public:
  explicit TileCorePort(Tile &tile) : tile(tile) {}

  bool read(uint32_t addr, void *data, uint32_t size) override {
    if (Memory *m = resolve(addr, size)) {
      return m->read(addr - regionBase, data, size);
    }
    return false;
  }

  bool write(uint32_t addr, const void *data, uint32_t size) override {
    if (Memory *prog = tile.programMemory();
        prog && addr - kProgramBase < prog->size()) {
      // A core writing its own program memory is self-modifying code. The
      // model does not support it: the engine caches decoded bundles, so the
      // write would land but execution would not see it.
      fault("core wrote program memory at 0x%08X; self-modifying code is not "
            "modelled (the engine caches decoded bundles)",
            addr);
      return false;
    }
    if (Memory *m = resolve(addr, size))
      return m->write(addr - regionBase, data, size);
    return false;
  }

  bool tryAcquireLock(uint32_t lockId, int32_t value) override {
    if (std::optional<ResolvedLock> lock = resolveLock(lockId, "acquire"))
      return lock->module->tryAcquire(lock->id, value);
    return false;
  }

  void releaseLock(uint32_t lockId, int32_t value) override {
    if (std::optional<ResolvedLock> lock = resolveLock(lockId, "release"))
      lock->module->release(lock->id, value);
  }

  /// Which lock module a core-issued id lands in, and its index there.
  struct ResolvedLock {
    LockModule *module;
    uint32_t id;
  };

  /// A core-issued lock id names a module by band, exactly as an address does.
  /// Resolve it, or fault -- returning false would be wrong here, because false
  /// means "not available yet" and the core would retry a lock that can never
  /// come.
  std::optional<ResolvedLock> resolveLock(uint32_t lockId, const char *what) {
    LockModule *locks = tile.locks();
    if (!locks) {
      fault("core tried to %s lock %u on a tile with no lock module", what,
            lockId);
      return std::nullopt;
    }
    // The band width is the ISSUING tile's lock count, not the target's:
    // getLockLocalBaseIndex bands by getNumLocks(localCol, localRow).
    std::optional<LockBand> band = splitLockId(lockId, locks->count());
    if (!band) {
      fault("core tried to %s lock %u, past the last band (%u locks per "
            "module, %zu bands)",
            what, lockId, locks->count(), std::size(kAIE2Bands));
      return std::nullopt;
    }
    Tile *owner = bandTile(tile, band->dir);
    LockModule *module = owner ? owner->locks() : nullptr;
    if (!module) {
      fault("core tried to %s lock %u, the %s band's lock %u, where this tile "
            "has no neighbour",
            what, lockId, directionName(band->dir), band->local);
      return std::nullopt;
    }
    return ResolvedLock{module, band->local};
  }

  // Core stream and cascade ports are not wired to the stream switch yet.
  // Refusing is a stall, which would hang a design that uses them -- so these
  // fault loudly instead of quietly stalling forever.
  bool tryReadStream(uint32_t port, uint32_t *, bool *) override {
    fault("core read stream port %u; core stream ports are not wired to the "
          "stream switch yet",
          port);
    return false;
  }
  bool tryWriteStream(uint32_t port, uint32_t, bool) override {
    fault("core wrote stream port %u; core stream ports are not wired to the "
          "stream switch yet",
          port);
    return false;
  }
  bool tryReadCascade(void *) override {
    fault("core read the cascade; the cascade is not modelled");
    return false;
  }
  bool tryWriteCascade(const void *) override {
    fault("core wrote the cascade; the cascade is not modelled");
    return false;
  }

  /// The debug channel test_library.cpp's designs print through, and what the
  /// array tests FileCheck.
  void putChar(char c) override { std::fputc(c, stdout); }

private:
  /// \returns the memory backing \p addr, setting regionBase to its core-local
  /// base, or null when nothing does. Null is a genuine fault: the core
  /// computed an address the hardware would not have answered either.
  Memory *resolve(uint32_t addr, uint32_t size) {
    // Program space is everything below the first data band. Nothing in the
    // core's address map names a tighter boundary: the ld script's
    // `program (RX) : LENGTH = 0x0020000` is a LINKER region, emitted
    // identically for every target even though program memory is not, and it
    // does not line up with the first band at 0x40000 either. So the model
    // bounds program space by its own band table and reports the backed size
    // from the device, which are the two numbers it can actually source.
    if (addr < kAIE2Bands[0].base) {
      regionBase = kProgramBase;
      Memory *prog = tile.programMemory();
      if (prog && prog->inRange(addr - kProgramBase, size))
        return prog;
      fault("core fetched 0x%08X, past the %u bytes of program memory this "
            "tile has",
            addr, prog ? prog->size() : 0u);
      return nullptr;
    }

    for (const MemBand &band : kAIE2Bands) {
      if (addr < band.base || addr >= band.base + kBandSize)
        continue;
      regionBase = band.base;
      Tile *owner = bandTile(tile, band.dir);
      Memory *mem = owner ? owner->memory() : nullptr;
      if (!mem) {
        // Not a gap: the band exists in the address map on every core tile, but
        // the tile it names does not exist on this one, so hardware would not
        // have answered either. A row-2 core reaching into 0x4xxxx is the case
        // that shows up in practice -- its south neighbour is the memtile.
        fault("core accessed the %s band at 0x%08X, which names no memory from "
              "this tile",
              directionName(band.dir), addr);
        return nullptr;
      }
      if (mem->inRange(addr - band.base, size))
        return mem;
      fault("core accessed 0x%08X, past the %u bytes of data memory in tile "
            "(%u, %u)",
            addr, mem->size(), owner->getCol(), owner->getRow());
      return nullptr;
    }

    fault("core accessed 0x%08X, which is past the last data band "
          "[0x%X, 0x%X)",
          addr, kAIE2Bands[0].base, kAIE2Bands[3].base + kBandSize);
    return nullptr;
  }

  static const char *directionName(MemDirection d) {
    switch (d) {
    case MemDirection::South:
      return "south";
    case MemDirection::West:
      return "west";
    case MemDirection::North:
      return "north";
    case MemDirection::Own:
      return "own";
    }
    return "?";
  }

  template <typename... Args>
  void fault(const char *fmt, Args... args) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), fmt, args...);
    tile.getArray().error(std::string("tile (") +
                             std::to_string(tile.getCol()) + ", " +
                             std::to_string(tile.getRow()) + "): " + buf);
  }

  Tile &tile;
  uint32_t regionBase = 0;
};

} // namespace

std::unique_ptr<CoreMemoryPort> aiesim::makeTileCorePort(Tile &tile) {
  return std::make_unique<TileCorePort>(tile);
}

uint32_t aiesim::ownMemoryBandBase(Generation gen) {
  (void)gen; // AIE2 and AIE2P share this layout; AIE1 does not and is not modelled.
  for (const MemBand &b : kAIE2Bands)
    if (b.dir == MemDirection::Own)
      return b.base;
  return 0;
}

uint32_t aiesim::memoryBandSize() { return kBandSize; }
