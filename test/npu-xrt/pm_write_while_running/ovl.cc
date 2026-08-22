//===- ovl.cc -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Resident code for the program-memory-write-while-running experiment.
//
// sel_a and sel_b are the "overlay" pair. Peano compiles each to a single
// 32-byte, 16-byte-aligned, branch-free block that differs in exactly one word
// (the `mova r0, #N` feeding the store), so sel_b's bytes can be copied over
// sel_a's address with no relinking -- AIE2P encodes data as inline immediates
// and only control transfers as absolute addresses. 32 bytes is two whole
// program-memory lines, so the write cannot straddle an ECC granule.
//
// They write through an out-pointer rather than returning a value because
// iron.ExternalFunction declares argument types only; it has no return type.
//
// The attributes are all load-bearing:
//   noinline, weak  keep the caller from constant-propagating the stored value
//                   (`noinline` alone does not stop IPSCCP; without `weak` the
//                   call site folds to `mova r0, #0x7` and patching does
//                   nothing observable)
//   used, retain    keep sel_b, which nothing calls, alive against
//                   -Wl,--gc-sections (emits SHF_GNU_RETAIN)

#include <cstdint>

extern "C" __attribute__((noinline, used, retain, weak)) void
sel_a(int32_t *out) {
  *out = 7;
}

extern "C" __attribute__((noinline, used, retain, weak)) void
sel_b(int32_t *out) {
  *out = 9;
}

// Spin until the host sets the flag, then clear it for the next round. The
// pointer must be volatile: this has to stay a real fetch loop so that the core
// is provably enabled and fetching when the program-memory write lands.
// Expressing the spin as an IRON/MLIR loop is not safe against LICM.
extern "C" void ovl_wait(volatile int32_t *f) {
  while (*f == 0)
    ;
  *f = 0;
}
