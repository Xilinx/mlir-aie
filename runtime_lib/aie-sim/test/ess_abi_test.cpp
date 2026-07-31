//===- ess_abi_test.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The load-bearing integration claim of this whole component is that defining
// a handful of C symbols is enough to replace the Vitis simulator, with no
// change to aie-rt. This test is the guard on that claim.
//
// The declarations below are copied VERBATIM from the two places that declare
// them, with their citations. They are not paraphrased and must not be
// "cleaned up": the point is that if aie-rt ever changes a signature, or adds
// a symbol, this file stops compiling or stops linking, and the mismatch
// surfaces here instead of as a confusing link failure in somebody's host
// program.
//
//   third_party/aie-rt/driver/src/io_backend/ext/xaie_sim.c:45-50
//   runtime_lib/test_lib/memory_allocator.cpp:15-16
//
// This is deliberately a LINK test as much as a behaviour test. Compiling is
// the signature check; linking is the completeness check.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"
#include "TestSupport.h"

#include <cstdint>
#include <string>
#include <vector>

// --- verbatim from xaie_sim.c:41 (the typedef the declarations below use) ---
typedef unsigned int uint;

extern "C" {

// --- verbatim from xaie_sim.c:45-50 ---
void ess_Write32(uint64_t Addr, uint Data);
uint ess_Read32(uint64_t Addr);
void ess_WriteCmd(unsigned char Command, unsigned char ColId,
                  unsigned char RowId, unsigned int CmdWd0, unsigned int CmdWd1,
                  unsigned char *CmdStr);

void ess_NpiWrite32(uint64_t Addr, uint Data);
uint ess_NpiRead32(uint64_t Addr);

// --- verbatim from memory_allocator.cpp:15-16 ---
void ess_WriteGM(uint64_t addr, const void *data, uint64_t size);
void ess_ReadGM(uint64_t addr, void *data, uint64_t size);

} // extern "C"

using namespace aiesim;

// Builds a tile address the way aie-rt does (_XAie_GetTileAddr,
// third_party/aie-rt/driver/src/common/xaie_helper.h:145) so this test drives
// the same addresses a real host program would.
static uint64_t tileAddr(const DeviceModel &dev, uint32_t col, uint32_t row,
                         uint32_t regOff) {
  return dev.baseAddr + (static_cast<uint64_t>(col) << dev.colShift) +
         (static_cast<uint64_t>(row) << dev.rowShift) + regOff;
}

int main() {
  DeviceModel dev;
  std::string err;
  if (!makeDeviceFromName("npu2_1col", dev, err)) {
    std::fprintf(stderr, "ess_abi: %s\n", err.c_str());
    return 1;
  }
  setCurrentArray(std::make_unique<Array>(dev, nullptr));

  // A register write through the ess_* entry points must be visible through
  // the matching read, at the tile the address decodes to.
  const uint64_t reg = tileAddr(dev, 0, 2, 0x00001000);
  ess_Write32(reg, 0xcafef00du);
  AIESIM_CHECK_EQ(ess_Read32(reg), 0xcafef00du);

  // Different tiles must not alias. This is the check that catches a decode
  // that drops the row or column field, which is exactly the bug aie-rt's own
  // inverse helpers have (see Device.h).
  const uint64_t other = tileAddr(dev, 0, 3, 0x00001000);
  ess_Write32(other, 0x12345678u);
  AIESIM_CHECK_EQ(ess_Read32(reg), 0xcafef00du);
  AIESIM_CHECK_EQ(ess_Read32(other), 0x12345678u);

  // The DDR window round-trips, including across the model's internal page
  // boundary.
  std::vector<uint32_t> out(64);
  for (size_t i = 0; i < out.size(); ++i)
    out[i] = static_cast<uint32_t>(i * 3 + 11);
  const uint64_t ddr = (1ull << 16) - 32;
  ess_WriteGM(ddr, out.data(), out.size() * 4);
  std::vector<uint32_t> back(64, 0);
  ess_ReadGM(ddr, back.data(), back.size() * 4);
  bool same = true;
  for (size_t i = 0; i < out.size(); ++i)
    same &= back[i] == out[i];
  AIESIM_CHECK(same);

  // The NPI block has its own address space and must not fall through to the
  // tile decoder, which would either fault or corrupt a tile register.
  ess_NpiWrite32(0x40, 0xa5a5a5a5u);
  AIESIM_CHECK_EQ(ess_NpiRead32(0x40), 0xa5a5a5a5u);

  // The only two commands aie-rt emits are SETSTACK (0) and LOADSYM (1)
  // (third_party/aie-rt/driver/src/core/xaie_elfloader.c:44-45). Both are
  // simulator conveniences carrying no architectural state, so these must be
  // accepted silently rather than faulting.
  unsigned char sym[] = "some_symbol";
  ess_WriteCmd(0, 0, 2, 0x1000, 0x2000, nullptr);
  ess_WriteCmd(1, 0, 2, 0, 0, sym);

  // Time must have advanced: the whole reason host polling loops make progress
  // without threads is that the ess_* entry points step the array.
  AIESIM_CHECK(currentArray().cycle() > 0);

  // The fault contract, which is the point of the register file. Reading an
  // offset that nothing models AND nobody wrote must be a named failure, not a
  // fabricated zero: that is the case where a design polling an unimplemented
  // status register would otherwise spin forever with no diagnostic.
  std::string caught;
  currentArray().setDiagnosticHandler(
      [&](const std::string &m) { caught = m; });
  (void)ess_Read32(tileAddr(dev, 0, 2, 0x00007f00));
  AIESIM_CHECK(caught.find("unmodelled register") != std::string::npos);

  // Writes to something unmodelled are recorded rather than fatal, so a model
  // that does not claim every register is still usable, and what it missed is
  // a number rather than a surprise.
  caught.clear();
  ess_Write32(tileAddr(dev, 0, 2, 0x00007f80), 1);
  AIESIM_CHECK(caught.empty());
  AIESIM_CHECK(!currentArray().unclaimedWrites().empty());
  AIESIM_CHECK(currentArray().unclaimedReport().find("0x7f80") !=
               std::string::npos);

  // Strict mode promotes that write to a fault.
  currentArray().setStrict(true);
  ess_Write32(tileAddr(dev, 0, 2, 0x00007fc0), 1);
  AIESIM_CHECK(caught.find("AIE_SIM_STRICT") != std::string::npos);

  return aiesim_test::summarize("ess_abi");
}
