//===- core_window_test.cpp -----------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Tests the core architectural register window installed by installCore()
// (lib/CoreEngineLoader.cpp). The ranges below are written out independently
// of that file's tables, following the convention in stream_switch_test.cpp:
// if the two disagree, a test fails rather than both sharing one mistake.
//
// Both generations are checked because their windows collide -- the same
// offset is a valid but different register on each -- so a table applied to
// the wrong generation returns a plausible wrong value instead of faulting.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"
#include "aiesim/Device.h"

#include "TestSupport.h"

#include <cstdint>
#include <string>

using namespace aiesim;

namespace {

struct Range {
  uint32_t begin, end;
};

// XAIE2PGBL_CORE_MODULE_CORE_*: 230 registers, stride 0x10.
const Range kAIE2P[] = {
    {0x30000, 0x30500}, {0x30800, 0x30B00}, {0x30C00, 0x30C40},
    {0x30D00, 0x30DC0}, {0x30E00, 0x30EA0}, {0x31000, 0x314C0},
};
// XAIEMLGBL_CORE_MODULE_CORE_*: 210 registers, stride 0x10.
const Range kAIE2[] = {
    {0x30000, 0x30480},
    {0x30800, 0x30B00},
    {0x30C00, 0x310C0},
    {0x31100, 0x311A0},
    {0x31200, 0x31240},
};

constexpr uint32_t kCoreControl = 0x32000;
constexpr uint32_t kCoreControlReset = 0x2;

size_t checkWindow(const char *deviceName, const Range *ranges, size_t n,
                   uint32_t pc, uint32_t gapOffset) {
  DeviceModel dev;
  std::string error;
  AIESIM_CHECK(makeDeviceFromName(deviceName, dev, error));
  if (!error.empty())
    return 0;
  Array array(dev, nullptr);

  uint32_t coreRow = dev.numMemTileRows + 1;
  Tile *core = array.tile(0, coreRow);
  AIESIM_CHECK(core != nullptr);
  if (!core)
    return 0;
  AIESIM_CHECK(core->getType() == TileType::Core);
  RegisterFile &regs = core->regs();

  // Every register in the window is claimed, and with no engine installed
  // reads its DEFVAL of 0.
  size_t count = 0;
  for (size_t i = 0; i < n; ++i)
    for (uint32_t off = ranges[i].begin; off < ranges[i].end; off += 0x10) {
      AIESIM_CHECK(regs.isClaimed(off));
      AIESIM_CHECK_EQ(regs.read(off), 0u);
      ++count;
    }

  // The gaps between ranges are genuinely unmapped and must stay that way:
  // claiming them would trade a fault for an invented value.
  AIESIM_CHECK(!regs.isClaimed(gapOffset));

  // The window must not have swallowed CORE_CONTROL, which carries a reset
  // value rather than reading zero.
  AIESIM_CHECK(regs.isClaimed(kCoreControl));
  AIESIM_CHECK_EQ(regs.read(kCoreControl), kCoreControlReset);

  // PC is in the window at this generation's own offset.
  AIESIM_CHECK(regs.isClaimed(pc));

  // Shim and memtile have no core, so no window.
  Tile *shim = array.tile(0, 0);
  AIESIM_CHECK(shim != nullptr);
  if (shim)
    AIESIM_CHECK(!shim->regs().isClaimed(pc));

  return count;
}

} // namespace

int main() {
  // PC is 0x30E00 on AIE2P and 0x31100 on AIE2; each is a different, valid
  // register on the other generation (R16 and M0 respectively).
  size_t aie2p = checkWindow("npu2", kAIE2P, std::size(kAIE2P),
                             /*pc=*/0x30E00, /*gap=*/0x30600);
  AIESIM_CHECK_EQ(aie2p, size_t{230});

  size_t aie2 = checkWindow("xcve2802", kAIE2, std::size(kAIE2),
                            /*pc=*/0x31100, /*gap=*/0x30500);
  AIESIM_CHECK_EQ(aie2, size_t{210});

  return aiesim_test::summarize("core_window");
}
