<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rewriting AIE core program memory from a runtime sequence

## The question

AIE core program memory (PM) is 16 KB on every generation, including AIE2P. That
is the hardest capacity wall in the architecture: no I-cache, no paging, no spill.
A natural way around it is the classic double-buffering trick applied to
instruction memory -- reserve a resident region, split the rest into two overlay
slots, and stream the next kernel into the idle slot while the core executes from
the active one.

Whether that works at all rests on one hardware fact that neither aie-rt, the AIE2P
register spec, nor mlir-aie documents anywhere:

> **Can program memory be written while the core is enabled?**

This experiment answers it.

## The measurement

Each `sel_dN_a()` / `sel_dN_b()` in `ovl.cc` is an interchangeable overlay pair.
Peano compiles each to a single 32-byte, 16-byte-aligned, branch-free block that
differs in exactly one word:

```
00002050 <sel_d960_a>:             00002070 <sel_d960_b>:
    2050: ...  ret lr; ...             2070: ...  ret lr; ...
    2066: mova r0, #0x7                2086: mova r0, #0x9
    206a: st   r0, [p0, #0]            208a: st   r0, [p0, #0]
```

That makes the patch as small and as unambiguous as it can be, and 32 bytes is two
whole PM lines (`PROGRAM_MEMORY_WIDTH` is 128 bits), so the write can never
straddle an ECC granule. AIE2P encodes data as inline immediates and only control
transfers as absolute addresses, so a branch-free leaf is byte-for-byte position
independent and the `_b` bytes can be dropped onto the `_a` address with no
relinking at all. The experiment therefore tests the hardware and nothing else --
no linker script, no trampoline, no new dialect.

The `N` in each name is that pair's distance in bytes from `ovl_wait`, where the
core spins; `overlay_elf.py` asserts the linked addresses really match, and the
filler functions between the pairs exist only to create the spacing.

The core runs two rounds. Each round it waits on `gate`, spins in `ovl_wait()`
until the host sets `flag`, then calls *every* pair and reports one word each into
an ObjectFifo buffer. Round 0 is unpatched and must read 7 everywhere. Between the
rounds the runtime sequence overwrites one chosen pair via a single
`aiex.npu.blockwrite` to `0x20000 + addr`, so that word reads 9 if the write took
effect -- and every other pair is a control in the same run.

Two wait points put the core in either state that matters, enabled in both cases.
Releasing `gate` before the write leaves the core spinning in `ovl_wait` --
enabled and *actively fetching*. Releasing it after leaves the core parked on the
lock acquire -- enabled but *not fetching*. `flag` is always released last, so no
`sel_*` has been entered when the write lands.

## Results

AMD Strix (`npu2_1col`), XRT 2.20.0, amdxdna 2.20.0, 20 runs per variant.

### Core state (write 64 B from the PC)

| Core state | Landed |
|---|---|
| no write (negative control) | 0/20 |
| **enabled, fetching** | **10/20** |
| debug-halted (`DEBUG_CONTROL0` bit 0) | 20/20 |
| disabled (`CORE_CONTROL` bit 0) | 20/20 |
| enabled, stalled on a lock acquire | 20/20 |
| enabled, fetching, via ECC-bypass `0x24000` | **10/20** |

### Write distance from the PC, core running

Core spinning in `ovl_wait` at 9488; the `0x2000` boundary is 1296 bytes below.

| Distance | Address | Half vs PC | Landed |
|---|---|---|---|
| 64 | 9168 | same | 12/20 |
| 384 | 8848 | same | 13/20 |
| 512 | 8720 | same | 10/20 |
| 960 | 8528 | same | 15/20 |
| 1152 | 8336 | same | 6/20 |
| 1280 | 8208 | same | 13/20 |
| **1408** | **8080** | **different** | **20/20** |
| 2048 | 7440 | different | 20/20 |
| 4160 | 5072 | different | 20/20 |
| 8320 | 912 | different | 20/20 |

1280 and 1408 are 128 bytes apart and land on opposite sides of the `0x2000`
boundary, and that is where the behaviour changes — not at any particular
distance.

### Same distances, every address shifted up 2048 bytes

Built with `-DPM_SHIFT_FILL=1`, which pads the top of `.text` so absolute
addresses move while every pair's distance from the PC stays identical. The
boundary now sits 3344 bytes from the PC instead of 1296:

| Distance | Half vs PC (before → after) | Landed (before → after) |
|---|---|---|
| 1408 | different → **same** | 20/20 → **7/20** |
| 2048 | different → **same** | 20/20 → **9/20** |
| 4160 | different → different | 20/20 → 20/20 |

Identical distances, opposite outcomes. Distance is not the variable.

### Moving the program counter instead of the targets

`--spin lo` parks the core in a second spin loop near the bottom of `.text`, so
the PC moves while every pair stays where it was. This is what fixes the region
size: with the PC at 1168, an 8 KB split and a 4 KB split predict opposite results
for the middle pairs.

| Address | Distance from PC | 8 KB half | 4 KB region | Predicted (8 KB / 4 KB) | Landed |
|---|---|---|---|---|---|
| 1232 | 64 | same | same | race / race | 8/20 |
| **5392** | **4224** | **same** | different | **race** / land | **14/20** |
| **7504** | **6336** | **same** | different | **race** / land | **8/20** |
| 8272 | 7104 | different | different | land / land | 20/20 |
| 9488 | 8320 | different | different | land / land | 20/20 |

The decisive comparison is the middle two rows against the fourth: a target
**6336 bytes away races** while one **7104 bytes away is perfect**, because the
first is in the same 8 KB half as the PC and the second is not.

### Across the whole 16 KB

Everything above fits in the bottom 9.5 KB, so only the split at `0x2000` had been
exercised — while a real ping-pong slot B lives in the top half. `-DPM_SHIFT_FILL=3`
pushes the design up to 15808 bytes of `.text`, filling program memory, and puts
the spin loop at 15632.

| Address | Half | 4 KB region | Predicted (8 KB / 4 KB) | Landed |
|---|---|---|---|---|
| 15568 | same | same | race / race | 7/20 |
| 13584 | same | same | race / race | 9/20 |
| **11472** | **same** | **different** | **race** / land | **9/20** |
| 7312 | different | different | land / land | 20/20 |

The third row rules out a boundary at `0x3000`: the top half is one region, not
two. The fourth is the ping-pong geometry itself — the core executing at the very
top of program memory while the other half is rewritten underneath it.

### A realistically sized load

Every case above writes the pair's 32 bytes. A real overlay load moves kilobytes,
takes proportionally longer, and has correspondingly more opportunity to collide
with instruction fetch. `--block N` patches N bytes of program memory around the
pair instead, clamped to the pair's half, with the same observable.

| Write size | Landed |
|---|---|
| 32 B | 20/20 |
| 1 KB | 20/20 |
| 4 KB | 20/20 |

A 4 KB cross-half write lands every time with the core running.

Every run patches exactly one pair, so the rest are controls; they read 7 in every
run of every table above.

## What this means

**Program memory behaves as two 8 KB halves, split at `0x2000`. A write to the
half the core is currently fetching from races; a write to the other half always
lands.**

- **It is a half conflict, not a distance.** Two independent experiments show it:
  shifting every address while holding distances fixed flips outcomes, and moving
  the PC while holding addresses fixed flips them back.
- **A write to the other half lands every time with the core running** — no halt,
  no disable, no stall. That is the geometry a real overlay load has.
- **A same-half write is a coin flip** at any distance, from 64 bytes to 6336. It
  never hangs and never returns a torn value; the new bytes simply do not take
  effect, which reads as the core having already fetched them rather than the
  write being dropped.
- **It is not ECC.** The ECC-check-disabled mirror at `0x24000` races identically.
- **Halting, disabling or stalling fixes the same-half case**, so those remain the
  fallback when a layout cannot avoid the conflict.

So **double-buffered ("ping-pong") program memory works on AIE2P**, and the design
rule is precise:

> An overlay slot must not share an 8 KB half of program memory with the code that
> is executing while it is written.

With 16 KB there are exactly two halves, so a ping-pong design puts one slot in
each. The code that waits for the swap and dispatches into the slot has to live in
the *executing* half too, which means it is duplicated into both halves rather
than sitting in a shared resident region -- a shared region would, by definition,
be in the same half as one of the slots.

So a balanced ping-pong slot is **just under 8 KB**: half the program memory,
minus the per-half dispatch stub. A kernel larger than that cannot be
ping-ponged at all and has to fall back to the halt/stall path.

### Not established

- Whether the same split holds on other AIE generations, or on npu1.
- Whether a *partially* overlapping write (a slot straddling the boundary, with
  only part of it in the executing half) fails proportionally or wholly.
  `overlay_block` refuses to build one, since the design rule forbids it anyway.
- Whether `CORE_STATUS` (`0x32004`) bit 21 `CORE_PROCESSOR_BUS_STALL` is set during
  a same-region write, which would give a direct mechanical read on arbitration.
- Whether what matters is the *dynamic* PC at the instant the write arrives, or
  the region containing the code the core is running in general. Every measurement
  here has the core in a tight spin loop, so the two are indistinguishable.

## Layout

| | |
|---|---|
| `aie2.py` | the design, in IRON, and the lit RUN lines. Emits pass 1 (no `--elf`) or a patched case (`--variant X --pair N --elf <tmpdir>`). `--spin lo` moves the program counter to the bottom of `.text` without moving any pair. |
| `overlay_elf.py` | reads a pair out of a core ELF, checks the two halves are interchangeable, and owns `PAIR_DISTANCES`. Also `--check A B`, the drift guard. |
| `ovl.cc` | the `sel_dN_*` pairs, the filler that spaces them, and two spin loops. `-DPM_SHIFT_FILL=1` shifts every address without changing any distance. |
| `test.cpp` | host side; the pair to expect patched comes from `PM_PATCHED_DIST` (a distance, not an index). |

The design is ordinary IRON -- `ObjectFifo`, `Worker`, `Lock`, `Buffer`, `Kernel`,
`TaskGroup` -- with three raw `aiex.npu.*` ops in the runtime sequence for the
things IRON has no verb for: `blockwrite` to program memory, `maskwrite32` to the
core control registers, and `set_lock`. Those work inside an IRON sequence body
because `Runtime.resolve` runs it under the sequence's `InsertionPoint`. The one
other gap is a module-scope `memref.global` for the patch payload, which
`blockwrite()` places by walking out to the enclosing `aie.device`.

The core is *not* pinned to the first-pass ELF with `elf_file`: an empty core body
and an ObjectFifo endpoint contradict each other, and the objectfifo lowering drops
the pinned core's region outright. The second build recompiles the core instead, so
each variant is followed by `overlay_elf.py --check p1 pX`, which fails loudly if
the core moved and the patch would land on the wrong address. In practice it does
not move, but that is asserted, not assumed -- and `--check` additionally verifies
that every pair sits exactly its nominal distance from `ovl_wait`, so a compiler
change that perturbs the layout fails the build rather than silently invalidating
the sweep.

## Reproducing

The lit test covers only the deterministic cases: no write, the same-region write
with the core halted / disabled / stalled, and the different-region write with the
core running. Same-region writes under a running core are the racy ones and would
make it flaky. To reproduce the sweep by hand, build a case and run it repeatedly:

```
python aie2.py --dev npu2 --out design.mlir
aiecc --tmpdir=p1 --get-xclbin --xclbin-name=p1.xclbin \
      --get-npu-insts --npu-insts-name=p1.bin ./design.mlir
python aie2.py --dev npu2 --variant B --pair 1280 --elf p1 --out final_B.mlir
```

Peano ships no `llvm-objcopy`, so `overlay_elf.py` parses the ELF with `struct`.

## Still unmeasured

- **Fetch-contention cost.** Whether a concurrent PM write steals fetch cycles from
  a core executing out of the *other* half of PM, and how many. This sets the real
  break-even for overlays and is not answered by the pass/fail above.
- **Sustained bandwidth into the tile control port.** Whether it accepts one word
  per cycle, which bounds a full 16 KB reload at roughly 6 us.
- **The cross-slot call ABI tax.** An overlay entered through an opaque
  absolute-address call cannot be inlined into its caller, and AIE2P has 1024-bit
  accumulators and zero-overhead-loop registers to spill. That cost is paid every
  iteration, not just on swaps, and could exceed the swap cost outright.
- **Preemption.** If NPU firmware restores context by reloading PM from the PDI,
  an overlay-mutated PM would be silently reverted while data-memory state survives
  -- wrong answers, not a hang. See `aiex.npu.preempt`.
