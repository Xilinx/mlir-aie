//===- EssAbi.cpp -----------------------------------------------*- C -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The seven symbols aie-rt's simulation IO backend calls, and the whole of the
// interface the array model presents to a host program.
//
// aie-rt declares these in
// third_party/aie-rt/driver/src/io_backend/ext/xaie_sim.c:45-50 and leaves
// them undefined; mlir-aie already builds libxaienginecdo with __AIESIM__
// (runtime_lib/xaiengine/lib/CMakeLists.txt:57), so linking a host program
// against this library is the entire integration. Nothing in aie-rt changes.
//
// ess_Write128 / ess_Read128 are defined too because the Vitis wrapper exports
// them; no aie-rt path calls them.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"

#include <cstdint>

using namespace aiesim;

extern "C" {

void ess_Write32(uint64_t Addr, uint32_t Data) {
  currentArray().write32(Addr, Data);
}

uint32_t ess_Read32(uint64_t Addr) { return currentArray().read32(Addr); }

void ess_Write128(uint64_t Addr, uint32_t *Data) {
  for (int i = 0; i < 4; ++i)
    currentArray().write32(Addr + 4u * i, Data[i]);
}

void ess_Read128(uint64_t Addr, uint32_t *Data) {
  for (int i = 0; i < 4; ++i)
    Data[i] = currentArray().read32(Addr + 4u * i);
}

void ess_WriteCmd(uint8_t Command, uint8_t ColId, uint8_t RowId, uint32_t CmdWd0,
                  uint32_t CmdWd1, uint8_t *CmdStr) {
  currentArray().writeCmd(Command, ColId, RowId, CmdWd0, CmdWd1,
                          reinterpret_cast<const char *>(CmdStr));
}

// The NPI register block configures partition-level things (clock gating,
// column reset) that sit outside the tile address map. Modelling it is not
// needed for any behaviour the tests observe, but the accesses must not fall
// through to the tile decoder, so they get their own trivial store.
static uint32_t npiRegs[256];

void ess_NpiWrite32(uint64_t Addr, uint32_t Data) {
  npiRegs[(Addr >> 2) & 0xffu] = Data;
}

uint32_t ess_NpiRead32(uint64_t Addr) { return npiRegs[(Addr >> 2) & 0xffu]; }

void ess_WriteGM(uint64_t addr, const void *data, uint64_t size) {
  currentArray().writeGlobal(addr, data, size);
}

void ess_ReadGM(uint64_t addr, void *data, uint64_t size) {
  currentArray().readGlobal(addr, data, size);
}

} // extern "C"
