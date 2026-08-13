//===- txn_replay_test.cpp ------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The TXN container and its operations, against a blob built here.
//
// Built rather than read from a design file, same reasoning as
// cdo_replay_test: a checked-in fixture proves the parser agrees with one
// sample, an emitter written from the struct declarations proves it agrees
// with the format. The builder below packs each operation at the offsets
// XAie_TxnHeader and the XAie_*Hdr family put them, INCLUDING their padding,
// which is where this format is easy to get wrong -- two of the four scalar
// ops carry four bytes of it before their address, and patch_op_t carries
// four more before its u64s.
//
// The cases a hand-written fixture would never reach, and the ones that
// silently mis-parse everything after them, are the point: an unknown opcode
// (no length, so no next operation), and a block write whose payload length is
// implied by the operation size rather than stated.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"
#include "aiesim/Components.h"
#include "aiesim/Device.h"
#include "aiesim/TxnReplay.h"

#include "TestSupport.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace aiesim;

namespace {

/// Accumulates operations, then closes with XAie_TxnHeader.
class TxnBuilder {
public:
  void write32(uint64_t addr, uint32_t value) {
    size_t at = begin(0);
    put32(at + 8, static_cast<uint32_t>(addr));
    put32(at + 12, static_cast<uint32_t>(addr >> 32));
    put32(at + 16, value);
    end(at, 24);
  }
  void maskWrite32(uint64_t addr, uint32_t value, uint32_t mask) {
    size_t at = begin(3);
    put32(at + 8, static_cast<uint32_t>(addr));
    put32(at + 12, static_cast<uint32_t>(addr >> 32));
    put32(at + 16, value);
    put32(at + 20, mask);
    end(at, 28);
  }
  void maskPoll(uint64_t addr, uint32_t value, uint32_t mask) {
    size_t at = begin(4);
    put32(at + 8, static_cast<uint32_t>(addr));
    put32(at + 12, static_cast<uint32_t>(addr >> 32));
    put32(at + 16, value);
    put32(at + 20, mask);
    end(at, 28);
  }
  /// RegOff is a 32-bit field here, not the 64-bit one the scalar ops carry.
  void blockWrite32(uint32_t addr, const std::vector<uint32_t> &data) {
    size_t at = begin(1);
    put32(at + 8, addr);
    size_t size = 16 + data.size() * 4;
    for (size_t i = 0; i < data.size(); ++i)
      put32(at + 16 + 4 * i, data[i]);
    end(at, size);
  }
  void addressPatch(uint64_t regAddr, uint64_t argIdx, uint64_t argPlus) {
    size_t at = begin(129);
    put64(at + 24, regAddr);
    put64(at + 32, argIdx);
    put64(at + 40, argPlus);
    end(at, 48);
  }
  void sync(uint32_t col, uint32_t row, uint32_t dir, uint32_t channel,
            uint32_t colCount, uint32_t rowCount) {
    size_t at = begin(128);
    put32(at + 8, (col << 16) | (row << 8) | dir);
    put32(at + 12, (channel << 24) | (colCount << 16) | (rowCount << 8));
    end(at, 16);
  }
  void raw(uint8_t opcode, size_t size) {
    size_t at = begin(opcode);
    end(at, size);
  }

  std::vector<uint8_t> finish(uint8_t major = 0, uint8_t minor = 1) const {
    std::vector<uint8_t> out(16 + body.size(), 0);
    out[0] = major;
    out[1] = minor;
    out[2] = 4; // DevGen, AIE2P.
    out[3] = 6; // NumRows.
    out[4] = 8; // NumCols.
    out[5] = 1; // NumMemTileRows.
    uint32_t n = ops, size = static_cast<uint32_t>(out.size());
    std::memcpy(out.data() + 8, &n, 4);
    std::memcpy(out.data() + 12, &size, 4);
    std::memcpy(out.data() + 16, body.data(), body.size());
    return out;
  }

  /// Corrupts the declared operation count, to check the replayer notices.
  std::vector<uint8_t> finishWithOpCount(uint32_t n) const {
    std::vector<uint8_t> out = finish();
    std::memcpy(out.data() + 8, &n, 4);
    return out;
  }

private:
  size_t begin(uint8_t opcode) {
    size_t at = body.size();
    body.push_back(opcode);
    return at;
  }
  void end(size_t at, size_t size) {
    body.resize(at + size, 0);
    // Both size fields, since which one an opcode uses depends on its header:
    // the custom ops carry theirs in XAie_CustomOpHdr's word at +4, the scalar
    // ops after their own fields. Writing only the used one would make this
    // builder encode what the parser already assumes.
    uint32_t s = static_cast<uint32_t>(size);
    switch (body[at]) {
    case 0:
      put32(at + 20, s);
      break;
    case 1:
      put32(at + 12, s);
      break;
    case 3:
    case 4:
      put32(at + 24, s);
      break;
    default:
      put32(at + 4, s);
      break;
    }
    ++ops;
  }
  void put32(size_t at, uint32_t v) {
    if (body.size() < at + 4)
      body.resize(at + 4, 0);
    std::memcpy(body.data() + at, &v, 4);
  }
  void put64(size_t at, uint64_t v) {
    if (body.size() < at + 8)
      body.resize(at + 8, 0);
    std::memcpy(body.data() + at, &v, 8);
  }
  std::vector<uint8_t> body;
  uint32_t ops = 0;
};

DeviceModel device() {
  DeviceModel dev;
  std::string error;
  AIESIM_CHECK(makeDeviceFromName("npu2", dev, error));
  return dev;
}

/// A tile-relative offset, WITHOUT the host base: a TXN carries addresses
/// built against the compiler's device instance, whose base is zero, and the
/// replayer is what adds the array's.
uint64_t tileOff(const DeviceModel &dev, uint32_t col, uint32_t row,
                 uint32_t off) {
  return (static_cast<uint64_t>(col) << dev.colShift) |
         (static_cast<uint64_t>(row) << dev.rowShift) | off;
}

/// True when the whole image was applied. The tests that expect a refusal
/// check the outcome directly, since "declined" and "malformed" are different
/// answers and a bool would fold them together.
bool replay(Array &array, const std::vector<uint8_t> &image, uint64_t base,
            TxnReplayStats &stats, std::string &error) {
  return replayTxn(array, image.data(), image.size(), base, stats, error,
                   /*waitCycles=*/64) == TxnOutcome::Applied;
}

//===----------------------------------------------------------------------===//
// Test 1: each operation lands where the design said, with the host base
// added. A TXN whose base was dropped decodes every address as unmapped, so
// the base is checked by using a non-zero one throughout.
//===----------------------------------------------------------------------===//

void testOperationsLand() {
  DeviceModel dev = device();
  Array array(dev, nullptr);
  Tile *core = array.tile(0, 2);
  AIESIM_CHECK(core != nullptr);

  constexpr uint32_t kLock0 = 0x0001F000;
  constexpr uint32_t kLock1 = 0x0001F010;

  TxnBuilder b;
  b.write32(tileOff(dev, 0, 2, kLock0), 5);
  b.maskWrite32(tileOff(dev, 0, 2, kLock0), 0x2, 0x3);
  b.write32(tileOff(dev, 0, 2, kLock1), 9);
  b.blockWrite32(static_cast<uint32_t>(tileOff(dev, 0, 2, 0x40)),
                 {0xAA, 0xBB, 0xCC});

  TxnReplayStats stats;
  std::string error;
  AIESIM_CHECK(replay(array, b.finish(), dev.baseAddr, stats, error));
  AIESIM_CHECK(error.empty());

  AIESIM_CHECK_EQ(stats.write32, 2u);
  AIESIM_CHECK_EQ(stats.maskWrite32, 1u);
  AIESIM_CHECK_EQ(stats.blockWrite32, 1u);
  AIESIM_CHECK_EQ(stats.blockWriteWords, static_cast<uint64_t>(3));

  AIESIM_CHECK_EQ(core->locks()->value(0), 6); // 5 & ~3 | 2.
  AIESIM_CHECK_EQ(core->locks()->value(1), 9);
  uint32_t word = 0;
  AIESIM_CHECK(core->memory()->read(0x40, &word, sizeof(word)));
  AIESIM_CHECK_EQ(word, 0xAAu);
  AIESIM_CHECK(core->memory()->read(0x48, &word, sizeof(word)));
  AIESIM_CHECK_EQ(word, 0xCCu);
}

//===----------------------------------------------------------------------===//
// Test 2: an address patch resolves a BD's buffer address from a kernel
// argument, across BOTH words -- and the high one is a masked write, because a
// shim BD shares it with the packet fields.
//===----------------------------------------------------------------------===//

void testAddressPatch() {
  DeviceModel dev = device();
  Array array(dev, nullptr);
  Tile *shim = array.tile(0, 0);
  AIESIM_CHECK(shim != nullptr);

  // Shim BD 0: word 1 is AddrLow, word 2 is AddrHigh in its low 16 bits with
  // packet type/id above (Dma.cpp's makeShimLayout).
  constexpr uint32_t kBd0AddrLow = 0x0001D004;
  constexpr uint32_t kBd0AddrHigh = 0x0001D008;
  constexpr uint32_t kPacketBits = 0x40070000u; // EnPkt + a packet type.
  const uint64_t argPlus = 0x2000;

  TxnBuilder b;
  // The block write pre-loads the offset, exactly as every design's does, and
  // sets packet bits in the high word so the masked write has something to
  // preserve.
  b.blockWrite32(static_cast<uint32_t>(tileOff(dev, 0, 0, 0x0001D000)),
                 {0x100, static_cast<uint32_t>(argPlus), kPacketBits});
  b.addressPatch(tileOff(dev, 0, 0, kBd0AddrLow), /*argIdx=*/3, argPlus);

  TxnReplayStats stats;
  std::string error;
  AIESIM_CHECK(replay(array, b.finish(), dev.baseAddr, stats, error));
  AIESIM_CHECK(error.empty());
  AIESIM_CHECK_EQ(stats.addressPatch, 1u);
  // The pre-load matched, which is what makes replace and add agree here.
  AIESIM_CHECK_EQ(stats.addressPatchPreloadMismatch, 0u);

  const uint64_t expected = argumentBase(3) + argPlus;
  AIESIM_CHECK_EQ(shim->regs().read(kBd0AddrLow),
                  static_cast<uint32_t>(expected));
  // Argument 3 is 3 << 32, so the high half is 3 -- and the packet bits above
  // it survived.
  AIESIM_CHECK_EQ(shim->regs().read(kBd0AddrHigh), kPacketBits | 3u);
}

//===----------------------------------------------------------------------===//
// Test 3: the mismatch counter fires when the pre-loaded word is NOT the
// offset the patch carries -- the only case where "replace the word" and "add
// the argument base to it" compute different addresses.
//===----------------------------------------------------------------------===//

void testAddressPatchPreloadMismatchIsCounted() {
  DeviceModel dev = device();
  Array array(dev, nullptr);

  TxnBuilder b;
  b.blockWrite32(static_cast<uint32_t>(tileOff(dev, 0, 0, 0x0001D000)),
                 {0x100, 0x1234, 0});
  b.addressPatch(tileOff(dev, 0, 0, 0x0001D004), /*argIdx=*/1,
                 /*argPlus=*/0x5678);

  TxnReplayStats stats;
  std::string error;
  AIESIM_CHECK(replay(array, b.finish(), dev.baseAddr, stats, error));
  AIESIM_CHECK(error.empty());
  AIESIM_CHECK_EQ(stats.addressPatchPreloadMismatch, 1u);
  // Replace wins: the register holds base + argplus, not preload + base.
  AIESIM_CHECK_EQ(array.tile(0, 0)->regs().read(0x0001D004),
                  static_cast<uint32_t>(argumentBase(1) + 0x5678));
}

//===----------------------------------------------------------------------===//
// Test 4: one argument per 4 GiB, so no offset can carry one argument into
// another. This is the property the whole buffer model rests on, and it is the
// reason the unresolved signedness of argplus does not change what a replay
// measures.
//===----------------------------------------------------------------------===//

void testArgumentsCannotAlias() {
  // The largest offset in the corpus, as an unsigned value.
  const uint64_t largest = 0xCCFFB3C0ull;
  for (uint32_t arg = 0; arg < 8; ++arg) {
    AIESIM_CHECK(argumentBase(arg) + largest < argumentBase(arg + 1));
    // And read as a negative int32 instead, it stays at or above its own base
    // for every argument that has room below it.
    if (arg > 0) {
      uint64_t low = argumentBase(arg) + static_cast<uint64_t>(
                                             static_cast<int64_t>(-2147483648));
      AIESIM_CHECK(low > argumentBase(arg - 1));
    }
  }
}

//===----------------------------------------------------------------------===//
// Test 5: an unknown opcode STOPS the replay. It has no length field this
// parser knows, so there is no way to find the next operation -- continuing
// would apply whatever the payload bytes happen to decode as.
//===----------------------------------------------------------------------===//

void testUnknownOpcodeStops() {
  DeviceModel dev = device();
  Array array(dev, nullptr);

  TxnBuilder b;
  b.write32(tileOff(dev, 0, 2, 0x0001F000), 5);
  b.raw(/*opcode=*/200, 16);
  b.write32(tileOff(dev, 0, 2, 0x0001F010), 9);

  TxnReplayStats stats;
  std::string error;
  AIESIM_CHECK(!replay(array, b.finish(), dev.baseAddr, stats, error));
  AIESIM_CHECK(error.find("unknown TXN opcode 200") != std::string::npos);
  // The operation before it was applied; the one after it was not.
  AIESIM_CHECK_EQ(stats.write32, 1u);
  AIESIM_CHECK_EQ(array.tile(0, 2)->locks()->value(1), 0);
}

//===----------------------------------------------------------------------===//
// Test 6: the header is checked before the operations are trusted. A version
// this parser does not model, a declared size that disagrees with the file,
// and a declared operation count that disagrees with what parsing found --
// the last being the one that catches a length field read from the wrong
// offset, since the walk would still terminate.
//===----------------------------------------------------------------------===//

void testHeaderMismatchesAreRejected() {
  DeviceModel dev = device();
  TxnBuilder b;
  b.write32(tileOff(dev, 0, 2, 0x0001F000), 5);

  {
    Array array(dev, nullptr);
    TxnReplayStats stats;
    std::string error;
    AIESIM_CHECK(!replay(array, b.finish(/*major=*/1, /*minor=*/0), 0, stats,
                         error));
    AIESIM_CHECK(error.find("version is 1.0") != std::string::npos);
  }
  {
    Array array(dev, nullptr);
    TxnReplayStats stats;
    std::string error;
    std::vector<uint8_t> image = b.finish();
    image.push_back(0); // Now longer than the header says.
    AIESIM_CHECK(!replay(array, image, 0, stats, error));
    AIESIM_CHECK(error.find("declares") != std::string::npos);
  }
  {
    Array array(dev, nullptr);
    TxnReplayStats stats;
    std::string error;
    AIESIM_CHECK(!replay(array, b.finishWithOpCount(7), dev.baseAddr, stats,
                         error));
    AIESIM_CHECK(error.find("7 operations but 1") != std::string::npos);
  }
}

//===----------------------------------------------------------------------===//
// Test 7: the two waits give up rather than hanging, and say they did.
//
// A TXN carries no budget of its own -- on hardware both block until the
// device answers -- so a model that inherited that would turn any unmodelled
// corner into a hang with no diagnostic. Both are also the operations that
// let time pass during a replay, which is what makes a sequence's task pushes
// actually run.
//===----------------------------------------------------------------------===//

void testWaitsTimeOutRatherThanHanging() {
  DeviceModel dev = device();
  Array array(dev, nullptr);

  TxnBuilder b;
  // Nothing will ever complete a BD on this channel, and nothing will ever
  // make the polled register match.
  b.sync(/*col=*/0, /*row=*/0, /*dir=*/0, /*channel=*/0, /*colCount=*/1,
         /*rowCount=*/1);
  b.maskPoll(tileOff(dev, 0, 2, 0x0001F000), /*value=*/0xF, /*mask=*/0xF);

  TxnReplayStats stats;
  std::string error;
  AIESIM_CHECK(replay(array, b.finish(), dev.baseAddr, stats, error));
  AIESIM_CHECK(error.empty());
  AIESIM_CHECK_EQ(stats.sync, 1u);
  AIESIM_CHECK_EQ(stats.syncTimedOut, 1u);
  AIESIM_CHECK_EQ(stats.maskPoll, 1u);
  AIESIM_CHECK_EQ(stats.maskPollTimedOut, 1u);
}

//===----------------------------------------------------------------------===//
// Test 8: a wait that CAN be satisfied returns as soon as it is, and does not
// count as timed out. Driven by a shim MM2S BD the sequence itself pushes --
// the whole point of the file, in one operation list: program the BD, resolve
// its address, push the task, wait for it.
//===----------------------------------------------------------------------===//

void testSyncCompletesWhenTheDmaDoes() {
  DeviceModel dev = device();
  Array array(dev, nullptr);
  Tile *shim = array.tile(0, 0);

  // Point the mux at the DMA and route MM2S channel 0's south slave port into
  // the Ctrl master, so the words have somewhere to go.
  const uint32_t port =
      static_cast<uint32_t>(shimDmaSouthPort(DmaDirection::MM2S, 0));
  shim->regs().write(0x0001F000, 1u << (8 + 2 * (port > 3 ? port - 4
                                                          : port - 2)));
  uint32_t slvReg = 0x0003F108 + 4 * port;
  shim->regs().write(slvReg, 1u << 31);
  shim->regs().write(0x0003F000, (1u << 31) | ((slvReg - 0x0003F100) / 4));

  TxnBuilder b;
  // BD 0: four words, valid, no lock. Word 7 bit 25 is ValidBd.
  b.blockWrite32(static_cast<uint32_t>(tileOff(dev, 0, 0, 0x0001D000)),
                 {4, 0, 0, 0, 0, 0, 0, 1u << 25});
  b.addressPatch(tileOff(dev, 0, 0, 0x0001D004), /*argIdx=*/0, /*argPlus=*/0);
  // MM2S channel 0's start queue: shim ctrl base 0x1D200, MM2S at +0x10.
  b.write32(tileOff(dev, 0, 0, 0x0001D214), 0x80000000u); // BD 0, token on.
  b.sync(/*col=*/0, /*row=*/0, /*dir=*/1, /*channel=*/0, /*colCount=*/1,
         /*rowCount=*/1);

  TxnReplayStats stats;
  std::string error;
  // A larger budget than the other tests: this one is meant to SUCCEED, so it
  // needs room for the transfer rather than just for the timeout.
  std::vector<uint8_t> image = b.finish();
  AIESIM_CHECK(replayTxn(array, image.data(), image.size(), dev.baseAddr, stats,
                         error, /*waitCycles=*/2000) == TxnOutcome::Applied);
  AIESIM_CHECK(error.empty());
  AIESIM_CHECK_EQ(stats.sync, 1u);
  AIESIM_CHECK_EQ(stats.syncTimedOut, 0u);
  AIESIM_CHECK_EQ(shim->dma()->completedBds(DmaDirection::MM2S, 0), 1u);
  AIESIM_CHECK_EQ(array.streamTraffic().ddrRead, static_cast<uint64_t>(16));
}

//===----------------------------------------------------------------------===//
// Test 9: a sequence that loads its own partition images is declined WHOLE,
// before anything is applied.
//
// Two things are checked, and the second is the one that matters: that the
// outcome is Declined rather than Malformed (the operations parse fine -- it
// is the run they describe that is not being performed), and that the array is
// untouched, since a half-applied orchestrator would leave buffer descriptors
// belonging to a design that was never loaded.
//===----------------------------------------------------------------------===//

void testSequenceThatLoadsPartitionsIsDeclinedWhole() {
  DeviceModel dev = device();
  Array array(dev, nullptr);

  TxnBuilder b;
  b.write32(tileOff(dev, 0, 2, 0x0001F000), 7);
  b.raw(/*opcode=*/8, 16); // LOAD_PDI, after an operation that would apply.

  TxnReplayStats stats;
  std::string error;
  std::vector<uint8_t> image = b.finish();
  AIESIM_CHECK(replayTxn(array, image.data(), image.size(), dev.baseAddr, stats,
                         error, /*waitCycles=*/64) == TxnOutcome::Declined);
  AIESIM_CHECK(error.find("loads 1 partition image") != std::string::npos);
  AIESIM_CHECK_EQ(stats.loadPdi, 1u);
  // Nothing applied, including the write that came BEFORE the load-pdi.
  AIESIM_CHECK_EQ(stats.write32, 0u);
  AIESIM_CHECK_EQ(array.tile(0, 2)->locks()->value(0), 0);
}

} // namespace

int main() {
  testOperationsLand();
  testSequenceThatLoadsPartitionsIsDeclinedWhole();
  testAddressPatch();
  testAddressPatchPreloadMismatchIsCounted();
  testArgumentsCannotAlias();
  testUnknownOpcodeStops();
  testHeaderMismatchesAreRejected();
  testWaitsTimeOutRatherThanHanging();
  testSyncCompletesWhenTheDmaDoes();
  return aiesim_test::summarize("txn_replay");
}
