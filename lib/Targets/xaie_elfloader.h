//===- xaie_elfloader.h -------------------------------------------*- C -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_XAIE_ELFLOADER_H
#define AIE_XAIE_ELFLOADER_H

#include "xaiengine/xaie_helper.h"
#include "xaiengine/xaiegbl.h"
#include "xaiengine/xaiegbl_defs.h"

AieRC XAie_LoadElf(XAie_DevInst *DevInst, XAie_LocType Loc, const char *ElfPtr,
                   u8 LoadSym);

#endif // AIE_XAIE_ELFLOADER_H
