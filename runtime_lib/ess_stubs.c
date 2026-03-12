/*
 * Weak stubs for aiesimulator ESS functions.
 *
 * xaie_sim.c forward-declares these symbols (ess_Write32, ess_Read32, etc.)
 * which are provided at runtime by the aiesimulator SystemC process via
 * dlopen. When linking aiecc statically, these symbols are unresolved.
 *
 * These weak definitions satisfy the linker. If aiecc is loaded by
 * aiesimulator, the real (strong) symbols override them. If someone
 * calls aiecc --aiesim outside of aiesimulator, the sim backend init
 * will fail gracefully before these are reached, but the stubs abort()
 * as a safety net.
 *
 * This file is licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * (c) Copyright 2026 Advanced Micro Devices, Inc.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef unsigned int uint;

__attribute__((weak)) void ess_Write32(uint64_t Addr, uint Data) {
  (void)Addr;
  (void)Data;
  fprintf(stderr, "FATAL: ess_Write32 called outside aiesimulator\n");
  abort();
}

__attribute__((weak)) uint ess_Read32(uint64_t Addr) {
  (void)Addr;
  fprintf(stderr, "FATAL: ess_Read32 called outside aiesimulator\n");
  abort();
  __builtin_unreachable();
}

__attribute__((weak)) void
ess_WriteCmd(unsigned char Command, unsigned char ColId, unsigned char RowId,
             unsigned int CmdWd0, unsigned int CmdWd1, unsigned char *CmdStr) {
  (void)Command;
  (void)ColId;
  (void)RowId;
  (void)CmdWd0;
  (void)CmdWd1;
  (void)CmdStr;
  fprintf(stderr, "FATAL: ess_WriteCmd called outside aiesimulator\n");
  abort();
}

__attribute__((weak)) void ess_NpiWrite32(uint64_t Addr, uint Data) {
  (void)Addr;
  (void)Data;
  fprintf(stderr, "FATAL: ess_NpiWrite32 called outside aiesimulator\n");
  abort();
}

__attribute__((weak)) uint ess_NpiRead32(uint64_t Addr) {
  (void)Addr;
  fprintf(stderr, "FATAL: ess_NpiRead32 called outside aiesimulator\n");
  abort();
  __builtin_unreachable();
}
