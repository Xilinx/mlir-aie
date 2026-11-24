// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// aie-codegen's xaie_cdo.c includes the Vitis-only "cdo_rts.h" under
// __AIECDO__. These are the cdo_* entry points it uses; they are provided by
// bootgen's cdo-driver (cdo_driver_mlir_aie).
#ifndef MLIR_AIE_CDO_RTS_H
#define MLIR_AIE_CDO_RTS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void cdo_Write32(uint64_t, uint32_t);
void cdo_MaskWrite32(uint64_t, uint32_t, uint32_t);
void cdo_MaskPoll(uint64_t, uint32_t, uint32_t, uint32_t);
void cdo_BlockWrite32(uint64_t, const uint32_t *, uint32_t);
void cdo_BlockSet32(uint64_t, uint32_t, uint32_t);

#ifdef __cplusplus
}
#endif

#endif
