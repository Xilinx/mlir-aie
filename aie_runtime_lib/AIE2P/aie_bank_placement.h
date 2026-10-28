//===- aie_bank_placement.h -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Puts a static in a named memory bank, which an `aie::lut<4>` pair needs.
// Peano discards `chess_storage`, so it gets a section instead -- one the
// linker script gives a region per bank, matching bankSectionName().
//===----------------------------------------------------------------------===//

#ifndef AIE_BANK_PLACEMENT_H
#define AIE_BANK_PLACEMENT_H

#if defined(__chess__)

#define AIE_BANK_A chess_storage(DM_bankA)
#define AIE_BANK_B chess_storage(DM_bankB)
#define AIE_BANK_C chess_storage(DM_bankC)
#define AIE_BANK_D chess_storage(DM_bankD)

#else

#define AIE_BANK_A __attribute__((section(".aie.bank0")))
#define AIE_BANK_B __attribute__((section(".aie.bank1")))
#define AIE_BANK_C __attribute__((section(".aie.bank2")))
#define AIE_BANK_D __attribute__((section(".aie.bank3")))

#endif

#endif // AIE_BANK_PLACEMENT_H
