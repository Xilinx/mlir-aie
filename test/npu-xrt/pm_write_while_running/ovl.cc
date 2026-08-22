//===- ovl.cc -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Resident code for the program-memory-write experiment.
//
// Each sel_dN_a / sel_dN_b is an "overlay" pair. Peano compiles each to a
// single 32-byte, 16-byte-aligned, branch-free block that differs in exactly
// one word (the `mova r0, #N` feeding the store), so sel_dN_b's bytes can be
// copied over sel_dN_a's address with no relinking -- AIE2P encodes data as
// inline immediates and only control transfers as absolute addresses. 32 bytes
// is two whole program-memory lines, so the write cannot straddle an ECC
// granule.
//
// There are several pairs because whether a write lands turned out to depend on
// how far it is from the program counter, not on the core's state. The N in
// each name is the intended distance in bytes from ovl_wait, where the core
// spins; overlay_elf.py checks the linked addresses actually match. The fillers
// between them exist only to create that spacing.
//
// The attributes are all load-bearing:
//   noinline, weak  keep the caller from constant-propagating the stored value
//                   (`noinline` alone does not stop IPSCCP; without `weak` the
//                   call site folds to `mova r0, #0x7` and patching does
//                   nothing observable)
//   used, retain    keep the sel_dN_b halves, which nothing calls, alive
//   against
//                   -Wl,--gc-sections (emits SHF_GNU_RETAIN)

#include <cstdint>

#define SEL(name, val)                                                         \
  extern "C" __attribute__((noinline, used, retain, weak)) void name(          \
      int32_t *out) {                                                          \
    *out = val;                                                                \
  }

#define PAIR(d)                                                                \
  SEL(sel_d##d##_a, 7)                                                         \
  SEL(sel_d##d##_b, 9)

// Filler. Each is a distinct retained function, so the linker cannot fold them
// together. FILL8 is 8 * 32 = 256 bytes. The trailing `;` on each invocation is
// an empty-declaration; without it clang-format runs the following declarations
// on as if they were one expression.
#define FILL8(n)                                                               \
  SEL(fill_##n##0, 1)                                                          \
  SEL(fill_##n##1, 1)                                                          \
  SEL(fill_##n##2, 1)                                                          \
  SEL(fill_##n##3, 1)                                                          \
  SEL(fill_##n##4, 1)                                                          \
  SEL(fill_##n##5, 1)                                                          \
  SEL(fill_##n##6, 1)                                                          \
  SEL(fill_##n##7, 1)

#define FILL2(n)                                                               \
  SEL(fill_##n##0, 1)                                                          \
  SEL(fill_##n##1, 1)

#define FILL32(n)                                                              \
  FILL8(n##0)                                                                  \
  FILL8(n##1)                                                                  \
  FILL8(n##2)                                                                  \
  FILL8(n##3)

#define FILL64(n)                                                              \
  FILL32(n##0)                                                                 \
  FILL32(n##1)

#define FILL128(n)                                                             \
  FILL64(n##0)                                                                 \
  FILL64(n##1)

// Shifts every absolute address up without changing any pair's distance from
// ovl_wait. Set to confirm that what matters is which 4 KB region of program
// memory a write lands in, not how far it is from the program counter: with a
// shift, the same distance falls on the other side of a region boundary.
#ifndef PM_SHIFT_FILL
#define PM_SHIFT_FILL 0
#endif
#if PM_SHIFT_FILL
FILL64(z);
#endif

// A second spin loop, emitted near the bottom of .text so it sits in a
// different program-memory region than ovl_wait at the top. Selecting which one
// the core spins in moves the program counter without moving any of the pairs,
// which is what distinguishes "the conflict follows the PC" from "the conflict
// is a property of the addresses".
extern "C" void ovl_wait_lo(volatile int32_t *f) {
  while (*f == 0)
    ;
  *f = 0;
}

// Emitted farthest-first: .text follows source order and ovl_wait is last, so
// each pair's distance is the total size of everything after it.
PAIR(8320)
FILL128(a);
PAIR(4160)
FILL64(b);
PAIR(2048)
FILL8(c0);
FILL8(c1);
FILL2(c2);
PAIR(1408)
FILL2(c3);
PAIR(1280)
FILL2(c4);
PAIR(1152)
FILL2(c5);
PAIR(1024)
PAIR(960)
PAIR(896)
FILL2(d);
PAIR(768)
FILL2(e);
PAIR(640)
FILL2(f);
PAIR(512)
FILL2(g);
PAIR(384)
FILL8(h);
PAIR(64)

// Spin until the host sets the flag, then clear it for the next round. The
// pointer must be volatile: this has to stay a real fetch loop so that the core
// is provably enabled and fetching when the program-memory write lands.
// Expressing the spin as an IRON/MLIR loop is not safe against LICM.
extern "C" void ovl_wait(volatile int32_t *f) {
  while (*f == 0)
    ;
  *f = 0;
}
