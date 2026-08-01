//===- Lock.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// AIE2/AIE2P lock module.
//
// aie-rt never "calls a lock API" on the simulator: it reads and writes plain
// registers, and the lock semantics live in how those particular registers are
// wired. Two disjoint per-tile register ranges matter, both taken from the
// vendored, MIT-licensed AIE2/AIE2P tables in third_party/aie-rt rather than
// re-derived (docs/AIESimulator.md section 2):
//
// 1. The acquire/release REQUEST range. A read at an address that folds the
//    signed request value into address bits performs the operation as a side
//    effect and returns success in bit 0. This is not a simulator shortcut:
//    it is the documented hardware mechanism (third_party/aie-rt/driver/src/
//    global/xaiegbl_regdef.h:681-690, the comment on `struct XAie_LockMod`:
//    "the lock is acquired by reading a register"), and it is exactly what
//    the sim IO backend's blocking poll does
//    (third_party/aie-rt/driver/src/io_backend/ext/xaie_sim.c:205-225,
//    XAie_SimIO_MaskPoll: a plain Read32 loop, no writes). The address
//    arithmetic and the field encoding come from
//    third_party/aie-rt/driver/src/locks/xaie_locks_aieml.c:66-133
//    (_XAieMl_LockAcquire / _XAieMl_LockRelease):
//
//      acquire off = BaseAddr + LockId*LockIdOff + RelAcqOff + (v7 << 2)
//      release off = BaseAddr + LockId*LockIdOff              + (v7 << 2)
//      v7 = (uint8_t)LockVal & 0x7F        (xaie_locks_aieml.c:34, MASK)
//      shift = 2                            (xaie_locks_aieml.c:35, SHIFT)
//      success = bit 0 of the read value    (xaie_locks_aieml.c:37-39)
//
//    BaseAddr, LockIdOff (0x400) and RelAcqOff (0x200) are per tile-type
//    fields of the vendored `XAie_LockMod` instances; they are identical
//    between AIE2 and AIE2P (compared field by field against both reginit
//    files below), so this file does not branch on generation for them:
//
//      third_party/aie-rt/driver/src/global/xaie2pgbl_reginit.c:2384-2401
//      (core), :2409-2426 (shim/noc), :2434-2451 (memtile); the AIE2
//      equivalents at third_party/aie-rt/driver/src/global/
//      xaiemlgbl_reginit.c:2448-2513 carry the same numbers.
//
//    The BaseAddr values themselves are read straight out of the vendored
//    param headers (included below) rather than copied, so this file tracks
//    the submodule if it is ever bumped:
//
//      (core)    XAIE2PGBL_MEMORY_MODULE_LOCK_REQUEST   params.h:11005
//      (shim)    XAIE2PGBL_NOC_MODULE_LOCK_REQUEST      params.h:19195
//      (memtile) XAIE2PGBL_MEM_TILE_MODULE_LOCK_REQUEST params.h:33546
//      (core)    XAIEMLGBL_MEMORY_MODULE_LOCK_REQUEST   params.h:11076
//      (shim)    XAIEMLGBL_NOC_MODULE_LOCK_REQUEST      params.h:19226
//      (memtile) XAIEMLGBL_MEM_TILE_MODULE_LOCK_REQUEST params.h:33533
//    (xaie2pgbl_params.h / xaiemlgbl_params.h, both under
//    third_party/aie-rt/driver/src/global/)
//
// 2. The plain VALUE range (`LockN_Value`), one 32-bit register per lock at
//    LockSetValBase + id*0x10, a 6-bit field (mask 0x3F, DEFVAL 0), a direct
//    read/write with no acquire semantics
//    (third_party/aie-rt/driver/src/locks/xaie_locks_aieml.c:150-198,
//    _XAieMl_LockSetValue / _XAieMl_LockGetValue). DEFVAL 0 is also the
//    lock's reset value, matching the all-zero row
//    `chess_compiler_tests_aie2/08_tile_locks` expects before any lock is
//    touched. Bases, again read from the vendored headers:
//
//      (core)    XAIE2PGBL_MEMORY_MODULE_LOCK0_VALUE   params.h:10645
//      (shim)    XAIE2PGBL_NOC_MODULE_LOCK0_VALUE      params.h:15919
//      (memtile) XAIE2PGBL_MEM_TILE_MODULE_LOCK0_VALUE params.h:32410
//      (core)    XAIEMLGBL_MEMORY_MODULE_LOCK0_VALUE   params.h:10716
//      (shim)    XAIEMLGBL_NOC_MODULE_LOCK0_VALUE      params.h:15950
//      (memtile) XAIEMLGBL_MEM_TILE_MODULE_LOCK0_VALUE params.h:32397
//    (same two header files as above)
//
// Acquire/AcquireGreaterEqual/Release semantics (what the sign of the request
// value means, and what happens to the counter) are not spelled out in
// aie-rt's C comments, so they are grounded one level up, in how mlir-aie
// itself produces that signed value:
//
//   include/aie/Dialect/AIE/IR/AIEOps.td:1472-1476 (UseLockOp doc):
//     "Acquire: blocks execution until the lock is set to `value`."
//     "AcquireGreaterEqual (only AIE2): blocks execution until the lock is
//      set to `value` or greater. Then, the value of the lock is decremented
//      by `value`."
//     "Release: ... In AIE2, increment the lock by `value`."
//   lib/Targets/AIERT.cpp:322-333 (BD lock lowering):
//     "if (op.acquireGE()) acqValue.value() = -acqValue.value();"
//   test/unit_tests/chess_compiler_tests_aie2/08_tile_locks/test.cpp:51-52:
//     mlir_aie_acquire_done_lock_1(_xaie, -2, 10000) -- "wait for 2 releases"
//     passed as a literal negative value, matching the negation above
//     (lib/Targets/AIETargetXAIEV2.cpp:855-861 shows the generated wrapper
//     forwards `value` to XAie_LockAcquire completely unchanged).
//
// Together these three call sites agree: AcquireGreaterEqual is encoded as a
// NEGATIVE request value (magnitude = the threshold), and plain Acquire is a
// non-negative value (an exact match). Components.h's `LockModule` doc
// comment states this same polarity and cites the same three sources; the
// two files agree.
//
// One corner is NOT fully nailed down by a single source line: whether a
// successful plain (non-GE) Acquire also decrements the counter by the
// matched value, the way AcquireGreaterEqual explicitly does. AIEOps.td is
// silent on it for plain Acquire. This file decrements in both cases (i.e.
// runs one compare-then-subtract datapath, gated only by the sign of the
// request), because (a) the register encoding gives both modes exactly one
// signed 7-bit field with no separate opcode, (b) Components.h defines
// exactly one `tryAcquire` entry point for both, and (c) on an exact match
// the subtraction always lands on zero, so it can never be observed to do
// anything unsound. This is a coherence argument, not a witnessed test, and
// it is called out again at the point in the code where it matters.
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
