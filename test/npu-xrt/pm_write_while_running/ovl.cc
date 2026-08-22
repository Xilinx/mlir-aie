//===- ovl.cc -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Resident code for the program-memory-write-while-running experiment.
//
// sel_*_a and sel_*_b are the "overlay" pairs. Peano compiles each to a single
// 32-byte, 16-byte-aligned, branch-free block that differs in exactly one word
// (the `mova r0, #N` feeding the store), so sel_b's bytes can be copied over
// sel_a's address with no relinking -- AIE2P encodes data as inline immediates
// and only control transfers as absolute addresses. 32 bytes is two whole
// program-memory lines, so the write cannot straddle an ECC granule.
//
// There are two pairs because "can program memory be written under a running
// core" turned out to depend on *where*. The near pair sits next to the spin
// loop; the far pair sits thousands of bytes away, which is the geometry a real
// overlay load has. See README.md.
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

#define SEL(name, val)                                                         \
  extern "C" __attribute__((noinline, used, retain, weak)) void name(          \
      int32_t *out) {                                                          \
    *out = val;                                                                \
  }

// Two interchangeable pairs at opposite ends of the program, so a variant can
// choose how far the write lands from the program counter. Emitted before the
// filler, so these come first in .text and end up thousands of bytes below the
// spin loop.
SEL(sel_far_a, 7)
SEL(sel_far_b, 9)

// Filler, purely to separate the two pairs. Each is a distinct retained
// function, so the linker cannot fold them together. The trailing `;` on each
// invocation is an empty-declaration; without it clang-format runs the
// following declarations on as if they were one expression.
#define FILL8(n)                                                               \
  SEL(fill_##n##0, 1)                                                          \
  SEL(fill_##n##1, 1)                                                          \
  SEL(fill_##n##2, 1)                                                          \
  SEL(fill_##n##3, 1)                                                          \
  SEL(fill_##n##4, 1)                                                          \
  SEL(fill_##n##5, 1)                                                          \
  SEL(fill_##n##6, 1)                                                          \
  SEL(fill_##n##7, 1)

#define FILL64(n)                                                              \
  FILL8(n##0)                                                                  \
  FILL8(n##1)                                                                  \
  FILL8(n##2)                                                                  \
  FILL8(n##3)                                                                  \
  FILL8(n##4)                                                                  \
  FILL8(n##5)                                                                  \
  FILL8(n##6)                                                                  \
  FILL8(n##7)

FILL64(a);
FILL64(b);
FILL64(c);
FILL64(d);

// Emitted last, so these land immediately below ovl_wait -- the adjacent case.
SEL(sel_near_a, 7)
SEL(sel_near_b, 9)

// Spin until the host sets the flag, then clear it for the next round. The
// pointer must be volatile: this has to stay a real fetch loop so that the core
// is provably enabled and fetching when the program-memory write lands.
// Expressing the spin as an IRON/MLIR loop is not safe against LICM.
extern "C" void ovl_wait(volatile int32_t *f) {
  while (*f == 0)
    ;
  *f = 0;
}
