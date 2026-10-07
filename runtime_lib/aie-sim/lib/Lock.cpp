//===- Lock.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// AIE2/AIE2P lock module.
//
// aie-rt never calls a lock API here: it reads and writes plain registers, and
// the semantics live in how those registers are wired. An acquire is a READ --
// the signed request value is folded into address bits and success comes back
// in bit 0. Offsets, field encodings and the sign convention are derived from
// the vendored aie-rt tables in docs/AIESimulator.md 8a.3, which also records
// the one corner (plain Acquire's decrement) that no single source settles.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Components.h"

#include "xaie2pgbl_params.h"
#include "xaiemlgbl_params.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace aiesim;

namespace {

// Field layout shared by AIE2 and AIE2P (see the file header for the
// per-generation macro citations that feed layoutFor()).
constexpr uint32_t kLockIdOff = 0x400;      // XAie_LockMod::LockIdOff
constexpr uint32_t kRelAcqOff = 0x200;      // XAie_LockMod::RelAcqOff
constexpr uint32_t kValueRegOff = 0x10;     // XAie_LockMod::LockSetValOff
constexpr uint32_t kReqFieldMask = 0x7Fu;   // XAIEML_LOCK_VALUE_MASK
constexpr uint32_t kReqFieldShift = 2u;     // XAIEML_LOCK_VALUE_SHIFT
constexpr uint32_t kReqSuccessBit = 0x1u;   // XAIEML_LOCK_RESULT_MASK/SUCCESS
constexpr uint32_t kValueFieldMask = 0x3Fu; // ..._LOCK0_VALUE_LOCK_VALUE_MASK

// Base addresses of the two register ranges for one tile type. Only two
// numbers are generation-dependent inputs (the rest of the layout above is
// shared), and even these turn out to be numerically identical between AIE2
// and AIE2P; layoutFor() still selects the generation-specific macro rather
// than assuming that, so a future submodule bump that lets them diverge does
// not silently go unnoticed.
struct LockLayout {
  uint32_t requestBase; // XAie_LockMod::BaseAddr
  uint32_t valueBase;   // XAie_LockMod::LockSetValBase
};

LockLayout layoutFor(Generation gen, TileType type) {
  switch (gen) {
  case Generation::AIE2:
    switch (type) {
    case TileType::Core:
      return {XAIEMLGBL_MEMORY_MODULE_LOCK_REQUEST,
              XAIEMLGBL_MEMORY_MODULE_LOCK0_VALUE};
    case TileType::Shim:
      return {XAIEMLGBL_NOC_MODULE_LOCK_REQUEST,
              XAIEMLGBL_NOC_MODULE_LOCK0_VALUE};
    case TileType::MemTile:
      return {XAIEMLGBL_MEM_TILE_MODULE_LOCK_REQUEST,
              XAIEMLGBL_MEM_TILE_MODULE_LOCK0_VALUE};
    case TileType::Invalid:
      break;
    }
    break;
  case Generation::AIE2P:
    switch (type) {
    case TileType::Core:
      return {XAIE2PGBL_MEMORY_MODULE_LOCK_REQUEST,
              XAIE2PGBL_MEMORY_MODULE_LOCK0_VALUE};
    case TileType::Shim:
      return {XAIE2PGBL_NOC_MODULE_LOCK_REQUEST,
              XAIE2PGBL_NOC_MODULE_LOCK0_VALUE};
    case TileType::MemTile:
      return {XAIE2PGBL_MEM_TILE_MODULE_LOCK_REQUEST,
              XAIE2PGBL_MEM_TILE_MODULE_LOCK0_VALUE};
    case TileType::Invalid:
      break;
    }
    break;
  }
  return {0, 0}; // Unreachable: installLocks() handles Invalid earlier.
}

// The counter is a 6-bit hardware field (kValueFieldMask); fold every
// mutation into that range the same way the real register would.
int32_t wrap6(int32_t v) {
  return static_cast<int32_t>(static_cast<uint32_t>(v) & kValueFieldMask);
}

class LockModuleImpl final : public LockModule {
public:
  LockModuleImpl(Tile &tile, uint32_t numLocks)
      : tile(tile), counters(numLocks, 0) {}

  bool tryAcquire(uint32_t id, int32_t value) override {
    if (!checkId(id))
      return false;
    int32_t &counter = counters[id];
    int32_t subtract;
    if (value >= 0) {
      // Plain Acquire: exact match. See the file header for why this also
      // subtracts the matched value rather than leaving the counter alone;
      // that half of the rule is inferred from design coherence, not from a
      // single witnessed source line.
      if (counter != value)
        return false;
      subtract = value;
    } else {
      // AcquireGreaterEqual: negative request, magnitude is the threshold.
      int32_t threshold = -value;
      if (counter < threshold)
        return false;
      subtract = threshold;
    }
    counter = wrap6(counter - subtract);
    return true;
  }

  void release(uint32_t id, int32_t value) override {
    if (!checkId(id))
      return;
    // AIEOps.td:1476: "In AIE2, increment the lock by `value`." Never fails.
    counters[id] = wrap6(counters[id] + value);
  }

  int32_t value(uint32_t id) const override {
    if (!checkId(id))
      return 0;
    return counters[id];
  }

  uint32_t count() const override {
    return static_cast<uint32_t>(counters.size());
  }

  // The direct LockN_Value register write: bypasses acquire/release
  // entirely (xaie_locks_aieml.c:150-164, _XAieMl_LockSetValue), used to set
  // a lock's initial value at configuration time rather than at runtime.
  void setRawValue(uint32_t id, int32_t rawValue) {
    if (!checkId(id))
      return;
    counters[id] = wrap6(rawValue);
  }

  // Locks have no autonomous per-cycle behaviour: every path that touches
  // one is a register read or write with an immediate side effect. Nothing
  // to do on a bare clock edge, and never anything outstanding either, so
  // Steppable::busy()'s default (false) is left as-is.
  bool step() override { return false; }

private:
  bool checkId(uint32_t id) const {
    if (id < counters.size())
      return true;
    char buf[128];
    std::snprintf(buf, sizeof(buf),
                  "lock id %u out of range (this module has %zu locks)", id,
                  counters.size());
    tile.getArray().error(buf);
    return false;
  }

  Tile &tile;
  std::vector<int32_t> counters;
};

} // namespace

void aiesim::installLocks(Tile &tile) {
  TileType type = tile.getType();
  if (type == TileType::Invalid)
    return;

  const DeviceModel &dev = tile.getArray().device();
  uint32_t numLocks = 0;
  switch (type) {
  case TileType::Core:
    numLocks = dev.numCoreLocks;
    break;
  case TileType::MemTile:
    numLocks = dev.numMemTileLocks;
    break;
  case TileType::Shim:
    numLocks = dev.numShimLocks;
    break;
  case TileType::Invalid:
    return; // Unreachable, handled above.
  }

  LockLayout layout = layoutFor(dev.generation, type);
  auto module = std::make_unique<LockModuleImpl>(tile, numLocks);
  LockModuleImpl *lm = module.get();

  // Acquire/release request range. One onRead handler covers both halves
  // (release below RelAcqOff, acquire at or above it); see the file header
  // for the address formula.
  uint32_t reqBegin = layout.requestBase;
  uint32_t reqEnd = layout.requestBase + numLocks * kLockIdOff;
  tile.regs().onRead(
      reqBegin, reqEnd, [lm, reqBegin](uint32_t off) -> uint32_t {
        uint32_t rel = off - reqBegin;
        uint32_t id = rel / kLockIdOff;
        uint32_t within = rel % kLockIdOff;
        bool isAcquire = within >= kRelAcqOff;
        uint32_t fieldOff = isAcquire ? within - kRelAcqOff : within;
        uint32_t encoded = (fieldOff >> kReqFieldShift) & kReqFieldMask;
        // encoded is the low 7 bits of a signed byte; bit 6 set means the
        // original value was negative (xaie_locks_aieml.c:34, MASK = 0x7F over
        // a cast-to-u8 s8).
        int32_t value = (encoded & 0x40u) ? static_cast<int32_t>(encoded) - 128
                                          : static_cast<int32_t>(encoded);
        bool ok = isAcquire ? lm->tryAcquire(id, value)
                            : (lm->release(id, value), true);
        return ok ? kReqSuccessBit : 0u;
      });
  // No aie-rt path ever writes into the request range: XAie_SimIO_MaskPoll
  // (xaie_sim.c:205-225) only reads it, for both acquire and release. A
  // write here means either a new caller this model has not seen, or a bug
  // upstream of us; either way it should stop loudly rather than silently
  // accept a write that has no modelled meaning.
  tile.regs().onWrite(reqBegin, reqEnd, [&tile](uint32_t off, uint32_t) {
    char buf[160];
    std::snprintf(
        buf, sizeof(buf),
        "write to lock acquire/release request register at tile-local "
        "offset 0x%x; aie-rt only reads this range (XAie_SimIO_MaskPoll) "
        "to perform an acquire or release, a write here is unmodelled",
        off);
    tile.getArray().error(buf);
  });

  // Plain LockN_Value register: one exact 4-byte register per lock, leaving
  // the reserved bytes between them (the stride is 0x10, the register is
  // 32 bits) unclaimed rather than aliased onto it.
  for (uint32_t id = 0; id < numLocks; ++id) {
    uint32_t off = layout.valueBase + id * kValueRegOff;
    tile.regs().onWrite(off, off + 4, [lm, id, &tile](uint32_t,
                                                       uint32_t regValue) {
      lm->setRawValue(id, static_cast<int32_t>(regValue & kValueFieldMask));
      tile.getArray().wake(lm); // Idle lock module receiving a write: rejoin
                                // the active set (Array.h's wake() contract).
    });
    tile.regs().onRead(off, off + 4, [lm, id](uint32_t) -> uint32_t {
      return static_cast<uint32_t>(lm->value(id)) & kValueFieldMask;
    });
  }

  tile.setLocks(std::move(module));
  tile.getArray().addSteppable(lm);
}
