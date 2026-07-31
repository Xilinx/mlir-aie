//===- memory_test.cpp ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Memory, RegisterFile and the sparse DDR window: the parts of the array model
// that everything else is built on.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"
#include "TestSupport.h"

#include <cstdint>
#include <vector>

using namespace aiesim;

static void testMemoryBounds() {
  Memory m(64);
  uint32_t v = 0xdeadbeef;
  AIESIM_CHECK(m.write(0, &v, 4));
  AIESIM_CHECK(m.write(60, &v, 4));
  // One byte past the end must fail rather than corrupt the heap.
  AIESIM_CHECK(!m.write(61, &v, 4));
  AIESIM_CHECK(!m.read(64, &v, 1));

  uint32_t back = 0;
  AIESIM_CHECK(m.read(60, &back, 4));
  AIESIM_CHECK_EQ(back, 0xdeadbeefu);

  // Unwritten memory reads as zero.
  AIESIM_CHECK(m.read(8, &back, 4));
  AIESIM_CHECK_EQ(back, 0u);
}

static void testRegisterFile() {
  RegisterFile rf;
  // An untouched register reads back zero rather than faulting: that is what
  // the hardware does for the reserved bits the driver reads through.
  AIESIM_CHECK_EQ(rf.read(0x1000), 0u);

  rf.write(0x1000, 0x1234);
  AIESIM_CHECK_EQ(rf.read(0x1000), 0x1234u);

  // A write handler sees the value after it is stored, so a handler that reads
  // neighbouring registers observes a consistent window.
  std::vector<std::pair<uint32_t, uint32_t>> seen;
  uint32_t observedDuringHandler = 0;
  rf.onWrite(0x2000, 0x2010, [&](uint32_t off, uint32_t value) {
    seen.emplace_back(off, value);
    observedDuringHandler = rf.read(off);
  });
  rf.write(0x2004, 0x55);
  rf.write(0x1004, 0x66); // outside the range, no handler
  AIESIM_CHECK_EQ(seen.size(), size_t{1});
  AIESIM_CHECK_EQ(seen[0].first, 0x2004u);
  AIESIM_CHECK_EQ(seen[0].second, 0x55u);
  AIESIM_CHECK_EQ(observedDuringHandler, 0x55u);

  // A read handler wins over stored state, which is how status registers are
  // modelled.
  rf.onRead(0x3000, 0x3004, [](uint32_t) { return 0xa5a5u; });
  rf.write(0x3000, 0x1);
  AIESIM_CHECK_EQ(rf.read(0x3000), 0xa5a5u);
}

static void testSparseDdr() {
  DeviceModel dev{};
  dev.numCols = 1;
  dev.numRows = 1;
  Array array(dev, nullptr);

  // Two buffers megabytes apart must not allocate the space between them, and
  // must not alias.
  const uint64_t a = 0x1000;
  const uint64_t b = 0x8000'0000ull;
  std::vector<uint32_t> src(1024);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<uint32_t>(i * 7 + 1);

  array.ddrWrite(a, src.data(), src.size() * 4);
  std::vector<uint32_t> other(1024, 0xffffffffu);
  array.ddrWrite(b, other.data(), other.size() * 4);

  std::vector<uint32_t> back(1024, 0);
  array.ddrRead(a, back.data(), back.size() * 4);
  bool same = true;
  for (size_t i = 0; i < src.size(); ++i)
    same &= back[i] == src[i];
  AIESIM_CHECK(same);

  // Never-written DDR reads as zero rather than as uninitialised memory, so a
  // test that forgets to fill a buffer fails deterministically.
  std::vector<uint32_t> zeros(16, 0x1234u);
  array.ddrRead(0x4000'0000ull, zeros.data(), zeros.size() * 4);
  bool allZero = true;
  for (uint32_t z : zeros)
    allZero &= z == 0;
  AIESIM_CHECK(allZero);

  // A transfer that crosses the internal page boundary must still be correct.
  const uint64_t crossing = (1ull << 16) - 8;
  array.ddrWrite(crossing, src.data(), 64);
  std::vector<uint32_t> crossBack(16, 0);
  array.ddrRead(crossing, crossBack.data(), 64);
  bool crossSame = true;
  for (size_t i = 0; i < 16; ++i)
    crossSame &= crossBack[i] == src[i];
  AIESIM_CHECK(crossSame);
}

static void testClockIsDeterministic() {
  DeviceModel dev{};
  dev.numCols = 1;
  dev.numRows = 1;
  Array array(dev, nullptr);
  AIESIM_CHECK_EQ(array.cycle(), uint64_t{0});
  array.advance(10);
  AIESIM_CHECK_EQ(array.cycle(), uint64_t{10});

  // With nothing installed the array is quiescent immediately, and says so
  // rather than burning the whole budget.
  AIESIM_CHECK(array.runUntilQuiescent(1000));
  AIESIM_CHECK_EQ(array.cycle(), uint64_t{11});
}

int main() {
  testMemoryBounds();
  testRegisterFile();
  testSparseDdr();
  testClockIsDeterministic();
  return aiesim_test::summarize("memory");
}
