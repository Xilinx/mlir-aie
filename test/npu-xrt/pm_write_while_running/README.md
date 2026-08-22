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

`ovl_wait` (where the core spins) at `0x2510`; the 4 KB boundary below it is
`0x2000`, i.e. 1296 bytes away.

| Distance | Address | 4 KB region vs PC | Landed |
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

### The same distances, with every address shifted up 2048 bytes

Built with `-DPM_SHIFT_FILL=1`, which pads the top of `.text` so absolute
addresses move while every pair's distance from the PC stays identical. The
boundary now sits 3344 bytes from the PC instead of 1296:

| Distance | Region vs PC (before → after) | Landed (before → after) |
|---|---|---|
| 1408 | different → **same** | 20/20 → **7/20** |
| 2048 | different → **same** | 20/20 → **9/20** |
| 4160 | different → different | 20/20 → 20/20 |

Identical distances, opposite outcomes. Distance is not the variable.

Every run patches exactly one pair, so the rest are controls; they read 7 in every
run of every table above.

## What this means

**Program memory behaves as four 4 KB regions. A write to the region the core is
currently fetching from races; a write to any other region always lands.**

- **It is a region conflict, not a distance.** The shift experiment is the proof:
  moving the code without changing any distance flips 1408 and 2048 from perfect
  to racing, exactly as their region membership flips.
- **A write to a different region lands every time with the core running** — no
  halt, no disable, no stall. That is the geometry a real overlay load has.
- **A same-region write is a coin flip.** It never hangs and never returns a torn
  value; the new bytes simply do not take effect, which reads as the core having
  already fetched them rather than the write being dropped.
- **It is not ECC.** The ECC-check-disabled mirror at `0x24000` races identically.
- **Halting, disabling or stalling fixes the same-region case**, so those remain
  the fallback when a layout cannot avoid the conflict.

So **double-buffered ("ping-pong") program memory works on AIE2P**, and the design
rule is precise:

> An overlay slot must not share a 4 KB region of program memory with the code
> that is executing while it is written.

With 16 KB of program memory that is four regions to allocate — for example
resident in region 0, slot A in region 1, slot B in region 2. Slots larger than
4 KB have to occupy whole regions disjoint from the executing slot's, which caps
how big a ping-pong slot can be: two 4 KB slots plus a 4 KB resident fits with one
region to spare, but a single kernel needing three regions cannot be ping-ponged
at all and must fall back to the halt/stall path.

### Not established

- Whether the region size is exactly 4 KB on other AIE generations, or whether
  AIE2P's program memory is literally banked that way — only that 4 KB predicts
  every measurement here, across two different code layouts.
- Whether a *partially* overlapping write (a slot straddling a boundary, with only
  part of it in the executing region) fails proportionally or wholly.
- Whether `CORE_STATUS` (`0x32004`) bit 21 `CORE_PROCESSOR_BUS_STALL` is set during
  a same-region write, which would give a direct mechanical read on arbitration.
- Whether what matters is the *dynamic* PC at the instant the write arrives, or
  the region containing the code the core is running in general. Every measurement
  here has the core in a tight spin loop, so the two are indistinguishable.

## Layout

| | |
|---|---|
| `aie2.py` | the design, in IRON, and the lit RUN lines. Emits pass 1 (no `--elf`) or a patched case (`--variant X --pair N --elf <tmpdir>`). |
| `overlay_elf.py` | reads a pair out of a core ELF, checks the two halves are interchangeable, and owns `PAIR_DISTANCES`. Also `--check A B`, the drift guard. |
| `ovl.cc` | the `sel_dN_*` pairs, the filler that spaces them, and `ovl_wait`. `-DPM_SHIFT_FILL=1` shifts every address without changing any distance. |
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
