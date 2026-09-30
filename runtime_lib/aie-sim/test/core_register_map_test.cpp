//===- core_register_map_test.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Checks the core-debug window's offset -> engine-register mapping.
//
// Expectations here are written out from aie-rt's parameter headers directly
// rather than derived from CoreEngineLoader.cpp's runs, following the
// convention stream_switch_test and core_window_test set: if the two disagree
// a test fails, instead of both sharing one mistake.
//
//===----------------------------------------------------------------------===//

#include "TestSupport.h"
#include "aiesim/Components.h"

#include <cstring>
#include <string>

using namespace aiesim;

namespace {

std::string nameAt(Generation gen, uint32_t off) {
  CoreRegisterMapping m = coreScalarRegister(gen, off);
  if (m.isProgramCounter)
    return "<pc>";
  return m.name;
}

void checkFamily(Generation gen, uint32_t base, unsigned count,
                 const char *prefix) {
  for (unsigned i = 0; i != count; ++i) {
    std::string want = std::string(prefix) + std::to_string(i);
    AIESIM_CHECK(nameAt(gen, base + i * 0x10) == want);
  }
  // One slot past the end must NOT continue the family.
  AIESIM_CHECK(nameAt(gen, base + count * 0x10) != std::string(prefix) +
                                                       std::to_string(count));
}

} // namespace

int main() {
  // --- AIE2P, from xaie2pgbl_params.h ---
  checkFamily(Generation::AIE2P, 0x31000, 32, "r");
  checkFamily(Generation::AIE2P, 0x31200, 8, "m");
  checkFamily(Generation::AIE2P, 0x31400, 8, "p");
  checkFamily(Generation::AIE2P, 0x31480, 4, "s");
  AIESIM_CHECK(nameAt(Generation::AIE2P, 0x30E00) == "<pc>");
  AIESIM_CHECK(nameAt(Generation::AIE2P, 0x30E20) == "sp");
  AIESIM_CHECK(nameAt(Generation::AIE2P, 0x30E30) == "lr");
  AIESIM_CHECK(nameAt(Generation::AIE2P, 0x30E40) == "ls");
  AIESIM_CHECK(nameAt(Generation::AIE2P, 0x30E50) == "le");
  AIESIM_CHECK(nameAt(Generation::AIE2P, 0x30E60) == "lc");

  // --- AIE2, from xaiemlgbl_params.h ---
  checkFamily(Generation::AIE2, 0x30C00, 32, "r");
  checkFamily(Generation::AIE2, 0x30E00, 8, "m");
  checkFamily(Generation::AIE2, 0x31000, 8, "p");
  checkFamily(Generation::AIE2, 0x31080, 4, "s");
  AIESIM_CHECK(nameAt(Generation::AIE2, 0x31100) == "<pc>");
  AIESIM_CHECK(nameAt(Generation::AIE2, 0x31120) == "sp");
  AIESIM_CHECK(nameAt(Generation::AIE2, 0x31160) == "lc");

  // --- The collisions, stated as such ---
  //
  // These three offsets are a valid register on BOTH generations and a
  // DIFFERENT one on each, so a shared table would return a plausible wrong
  // value rather than fault. This is the case core_window_test cannot catch,
  // because both generations claim the offset.
  AIESIM_CHECK(nameAt(Generation::AIE2P, 0x30E00) == "<pc>");
  AIESIM_CHECK(nameAt(Generation::AIE2, 0x30E00) == "m0");

  AIESIM_CHECK(nameAt(Generation::AIE2P, 0x31000) == "r0");
  AIESIM_CHECK(nameAt(Generation::AIE2, 0x31000) == "p0");

  AIESIM_CHECK(nameAt(Generation::AIE2P, 0x31100) == "r16");
  AIESIM_CHECK(nameAt(Generation::AIE2, 0x31100) == "<pc>");

  // --- What must stay unmapped ---
  //
  // Not gaps in the table: llvm-aie splits each of these across several named
  // control registers (crSat, crRnd, srsSign0, ...), so one offset is an
  // assembly rather than a rename. Probed against a real engine -- fc, cr1,
  // cr2, sr and AIE2's dp all fail readRegister. Mapping them by name would
  // fabricate a value.
  for (uint32_t off : {0x30E10u /*fc*/, 0x30E70u /*cr1*/, 0x30E80u /*cr2*/,
                       0x30E90u /*sr*/})
    AIESIM_CHECK(!coreScalarRegister(Generation::AIE2P, off).mapped());
  for (uint32_t off : {0x31110u /*fc*/, 0x31170u /*cr*/, 0x31180u /*sr*/,
                       0x31190u /*dp*/})
    AIESIM_CHECK(!coreScalarRegister(Generation::AIE2, off).mapped());

  // Vector and accumulator families belong to the vector phase: they need the
  // part-assembly rule, so a name here would be wrong rather than missing.
  AIESIM_CHECK(!coreScalarRegister(Generation::AIE2P, 0x30000).mapped());
  AIESIM_CHECK(!coreScalarRegister(Generation::AIE2P, 0x30800).mapped());

  // --- Sub-slot offsets are not registers ---
  //
  // A slot is one 32-bit value per 16 bytes; the bytes after the first word
  // are that value's tail. Naming them would let a misaligned read look like a
  // different register.
  for (uint32_t byte = 1; byte != 0x10; ++byte)
    AIESIM_CHECK(!coreScalarRegister(Generation::AIE2P, 0x31000 + byte)
                      .mapped());

  // --- Outside the window entirely ---
  AIESIM_CHECK(!coreScalarRegister(Generation::AIE2P, 0x00000).mapped());
  AIESIM_CHECK(!coreScalarRegister(Generation::AIE2P, 0x32000).mapped());

  return aiesim_test::summarize("core_register_map_test");
}
