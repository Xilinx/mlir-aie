//===- cdo_replay_test.cpp ------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The CDO container and its five commands, against a blob built here.
//
// Built rather than read from a design file for the usual reason: a fixture
// checked in as bytes proves the parser agrees with one sample, while an
// emitter written from bootgen's own field order proves it agrees with the
// format. The emitter below packs each command exactly as
// third_party/bootgen/cdo-driver/cdo_driver.c writes it, including the two
// places the layout is easy to get wrong -- the address is high word FIRST,
// and the header's Length counts payload words only.
//
// The escape for a long block write (Length 255, real length in an extra word)
// is covered because it is the one path a small hand-written fixture would
// never reach, and the one that silently mis-parses every command after it.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"
#include "aiesim/CdoReplay.h"
#include "aiesim/Components.h"
#include "aiesim/Device.h"

#include "TestSupport.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace aiesim;

namespace {

constexpr uint32_t kIdentWord = 0x004F4443;

/// Accumulates commands, then closes with the header FileHeader() writes.
class CdoBuilder {
public:
  void write32(uint64_t addr, uint32_t data) {
    body.push_back((0x03u << 16) | 0x108u);
    pushAddr(addr);
    body.push_back(data);
  }
  void maskWrite32(uint64_t addr, uint32_t mask, uint32_t data) {
    body.push_back((0x04u << 16) | 0x107u);
    pushAddr(addr);
    body.push_back(mask);
    body.push_back(data);
  }
  void blockWrite32(uint64_t addr, const std::vector<uint32_t> &data) {
    uint32_t length = static_cast<uint32_t>(data.size()) + 2;
    if (length < 255) {
      body.push_back((length << 16) | 0x105u);
    } else {
      body.push_back((255u << 16) | 0x105u);
      body.push_back(length);
    }
    pushAddr(addr);
    body.insert(body.end(), data.begin(), data.end());
  }
  void maskPoll(uint64_t addr, uint32_t mask, uint32_t expected) {
    body.push_back((0x05u << 16) | 0x106u);
    pushAddr(addr);
    body.push_back(mask);
    body.push_back(expected);
    body.push_back(0); // Timeout in ms, ignored by the replayer.
  }
  void noOp(uint32_t payloadWords) {
    body.push_back((payloadWords << 16) | 0x111u);
    body.insert(body.end(), payloadWords, 0u);
  }

  std::vector<uint32_t> finish() const {
    std::vector<uint32_t> out;
    uint32_t cdoLength = static_cast<uint32_t>(body.size());
    out.push_back(4); // NumWords: the four header words that follow.
    out.push_back(kIdentWord);
    out.push_back(0x200); // Version.
    out.push_back(cdoLength);
    out.push_back(~(4u + kIdentWord + 0x200u + cdoLength));
    out.insert(out.end(), body.begin(), body.end());
    return out;
  }

private:
  void pushAddr(uint64_t addr) {
    body.push_back(static_cast<uint32_t>(addr >> 32));
    body.push_back(static_cast<uint32_t>(addr));
  }
  std::vector<uint32_t> body;
};

bool replay(Array &array, const std::vector<uint32_t> &image, uint64_t base,
            CdoReplayStats &stats, std::string &error) {
  return replayCdo(array, image.data(), image.size() * 4, base, stats, error,
                   /*maskPollCycles=*/64);
}

DeviceModel device() {
  DeviceModel dev;
  std::string error;
  AIESIM_CHECK(makeDeviceFromName("npu2", dev, error));
  return dev;
}

uint64_t tileAddr(const DeviceModel &dev, uint32_t col, uint32_t row,
                  uint32_t off) {
  return dev.baseAddr | (static_cast<uint64_t>(col) << dev.colShift) |
         (static_cast<uint64_t>(row) << dev.rowShift) | off;
}

//===----------------------------------------------------------------------===//
// Each command lands where the design said, on a real tile.
//===----------------------------------------------------------------------===//

void testCommandsLand() {
  DeviceModel dev = device();
  Array array(dev, nullptr);
  Tile *core = array.tile(0, 2);
  AIESIM_CHECK(core != nullptr);

  // A core tile's LOCK0_VALUE / LOCK1_VALUE, which is what a design's init CDO
  // uses to set a lock's starting count -- a register with real behaviour
  // behind it, so this checks the write reached the MODEL and not just the
  // register file.
  constexpr uint32_t kLock0 = 0x0001F000;
  constexpr uint32_t kLock1 = 0x0001F010;

  CdoBuilder b;
  b.noOp(3);
  b.write32(tileAddr(dev, 0, 2, kLock0), 5);
  // Mask-write is read-modify-write: 5 with the low two bits replaced by 2.
  b.maskWrite32(tileAddr(dev, 0, 2, kLock0), 0x3, 0x2);
  b.write32(tileAddr(dev, 0, 2, kLock1), 9);
  // Straight into data memory, the way an ELF CDO loads a section.
  b.blockWrite32(tileAddr(dev, 0, 2, 0x40), {0xAA, 0xBB, 0xCC});

  CdoReplayStats stats;
  std::string error;
  AIESIM_CHECK(replay(array, b.finish(), 0, stats, error));
  AIESIM_CHECK(error.empty());

  AIESIM_CHECK_EQ(stats.write32, 2u);
  AIESIM_CHECK_EQ(stats.maskWrite32, 1u);
  AIESIM_CHECK_EQ(stats.blockWrite32, 1u);
  AIESIM_CHECK_EQ(stats.blockWriteWords, static_cast<uint64_t>(3));
  AIESIM_CHECK_EQ(stats.noOp, 1u);

  AIESIM_CHECK_EQ(core->locks()->value(0), 6); // 5 & ~3 | 2.
  AIESIM_CHECK_EQ(core->locks()->value(1), 9);
  uint32_t word = 0;
  AIESIM_CHECK(core->memory()->read(0x40, &word, sizeof(word)));
  AIESIM_CHECK_EQ(word, 0xAAu);
  AIESIM_CHECK(core->memory()->read(0x48, &word, sizeof(word)));
  AIESIM_CHECK_EQ(word, 0xCCu);
}

//===----------------------------------------------------------------------===//
// A block write longer than the 8-bit Length field, and the commands after it.
//===----------------------------------------------------------------------===//

void testLongBlockWrite() {
  DeviceModel dev = device();
  Array array(dev, nullptr);
  Tile *core = array.tile(1, 3);
  AIESIM_CHECK(core != nullptr);

  std::vector<uint32_t> payload(400);
  for (uint32_t i = 0; i < payload.size(); ++i)
    payload[i] = 0x1000 + i;

  CdoBuilder b;
  b.blockWrite32(tileAddr(dev, 1, 3, 0), payload);
  // Only reached if the long length was consumed correctly; a short read would
  // resume mid-payload and fail on an unknown command instead.
  b.write32(tileAddr(dev, 1, 3, 0x0001F000), 7);

  CdoReplayStats stats;
  std::string error;
  AIESIM_CHECK(replay(array, b.finish(), 0, stats, error));
  AIESIM_CHECK(error.empty());
  AIESIM_CHECK_EQ(stats.blockWriteWords, static_cast<uint64_t>(400));
  AIESIM_CHECK_EQ(core->locks()->value(0), 7);

  uint32_t word = 0;
  AIESIM_CHECK(core->memory()->read(399 * 4, &word, sizeof(word)));
  AIESIM_CHECK_EQ(word, 0x1000u + 399u);
}

//===----------------------------------------------------------------------===//
// The core ELF image path: a block write into program memory reaches program
// memory, which nothing outside the tile could do before that window was
// claimed.
//===----------------------------------------------------------------------===//

void testProgramMemoryIsReachable() {
  DeviceModel dev = device();
  Array array(dev, nullptr);
  Tile *core = array.tile(0, 2);

  CdoBuilder b;
  b.blockWrite32(tileAddr(dev, 0, 2, dev.progMemHostOffset), {1, 2, 3, 4});

  CdoReplayStats stats;
  std::string error;
  AIESIM_CHECK(replay(array, b.finish(), 0, stats, error));
  AIESIM_CHECK(error.empty());
  AIESIM_CHECK_EQ(array.unclaimedWrites().size(), static_cast<size_t>(0));

  uint32_t word = 0;
  AIESIM_CHECK(core->programMemory()->read(12, &word, sizeof(word)));
  AIESIM_CHECK_EQ(word, 4u);
}

//===----------------------------------------------------------------------===//
// A poll whose condition holds passes through; one that never holds is counted
// rather than hanging.
//===----------------------------------------------------------------------===//

void testMaskPoll() {
  DeviceModel dev = device();
  Array array(dev, nullptr);

  constexpr uint32_t kLock0 = 0x0001F000;
  CdoBuilder b;
  b.write32(tileAddr(dev, 0, 2, kLock0), 3);
  b.maskPoll(tileAddr(dev, 0, 2, kLock0), 0x3F, 3);
  b.maskPoll(tileAddr(dev, 0, 2, kLock0), 0x3F, 4);

  CdoReplayStats stats;
  std::string error;
  AIESIM_CHECK(replay(array, b.finish(), 0, stats, error));
  AIESIM_CHECK(error.empty());
  AIESIM_CHECK_EQ(stats.maskPoll, 2u);
  AIESIM_CHECK_EQ(stats.maskPollTimedOut, 1u);
}

//===----------------------------------------------------------------------===//
// Malformed images are refused by name, not half-applied.
//===----------------------------------------------------------------------===//

void testRejectsMalformed() {
  DeviceModel dev = device();

  CdoBuilder b;
  b.write32(tileAddr(dev, 0, 2, 0x0001F000), 1);
  std::vector<uint32_t> good = b.finish();

  {
    Array array(dev, nullptr);
    std::vector<uint32_t> bad = good;
    bad[1] = 0xDEADBEEF; // Ident.
    CdoReplayStats stats;
    std::string error;
    AIESIM_CHECK(!replay(array, bad, 0, stats, error));
    AIESIM_CHECK(error.find("not a CDO image") != std::string::npos);
  }
  {
    Array array(dev, nullptr);
    std::vector<uint32_t> bad = good;
    bad[4] ^= 1; // Checksum.
    CdoReplayStats stats;
    std::string error;
    AIESIM_CHECK(!replay(array, bad, 0, stats, error));
    AIESIM_CHECK(error.find("checksum") != std::string::npos);
  }
  {
    // An unknown command stops the replay: the commands after it would be read
    // at the wrong offset, so skipping is not a lesser evil than refusing.
    Array array(dev, nullptr);
    std::vector<uint32_t> bad = good;
    bad[5] = (0x03u << 16) | 0x1FFu; // Same length, so the checksum still holds.
    CdoReplayStats stats;
    std::string error;
    AIESIM_CHECK(!replay(array, bad, 0, stats, error));
    AIESIM_CHECK(error.find("unknown CDO command") != std::string::npos);
    AIESIM_CHECK(array.tile(0, 2)->locks()->value(0) == 0);
  }
}

//===----------------------------------------------------------------------===//
// The `base` argument: a blob emitted against address 0 lands on an array that
// decodes against a host base.
//===----------------------------------------------------------------------===//

void testBaseIsAdded() {
  DeviceModel dev = device();
  Array array(dev, nullptr);

  CdoBuilder b;
  b.write32(tileAddr(dev, 2, 4, 0x0001F000) - dev.baseAddr, 11);

  CdoReplayStats stats;
  std::string error;
  AIESIM_CHECK(replay(array, b.finish(), dev.baseAddr, stats, error));
  AIESIM_CHECK(error.empty());
  AIESIM_CHECK_EQ(array.tile(2, 4)->locks()->value(0), 11);
}

} // namespace

int main() {
  testCommandsLand();
  testLongBlockWrite();
  testProgramMemoryIsReachable();
  testMaskPoll();
  testRejectsMalformed();
  testBaseIsAdded();
  return aiesim_test::summarize("cdo_replay");
}
