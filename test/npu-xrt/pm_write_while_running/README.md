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

`sel_a()` and `sel_b()` in `ovl.cc` are the overlay pair. Peano compiles each to a
single 32-byte, 16-byte-aligned, branch-free block that differs in exactly one
word:

```
00000180 <sel_a>:                  000001a0 <sel_b>:
     180: ...  ret lr; ...              1a0: ...  ret lr; ...
     196: mova r0, #0x7                 1b6: mova r0, #0x9
     19a: st   r0, [p0, #0]             1ba: st   r0, [p0, #0]
```

That makes the patch as small and as unambiguous as it can be, and 32 bytes is two
whole PM lines (`PROGRAM_MEMORY_WIDTH` is 128 bits), so the write can never
straddle an ECC granule. AIE2P encodes data as inline immediates and only control
transfers as absolute addresses, so a branch-free leaf is byte-for-byte position
independent and `sel_b`'s bytes can be dropped onto `sel_a`'s address with no
relinking at all. The experiment therefore tests the hardware and nothing else --
no linker script, no trampoline, no new dialect.

The core runs two rounds. Each round it waits on `gate`, spins in `ovl_wait()`
until the host sets `flag`, then calls `sel_a()` and fans the result across an
ObjectFifo buffer. Round 0 is unpatched and must read 7. Between the rounds the
runtime sequence overwrites `sel_a`'s PM with `sel_b`'s bytes via a single
`aiex.npu.blockwrite` to `0x20000 + addr(sel_a)`, so round 1 reads 9 if the write
took effect.

Two wait points put the core in either state that matters, enabled in both cases.
Releasing `gate` before the write leaves the core spinning in `ovl_wait` --
enabled and *actively fetching*. Releasing it after leaves the core parked on the
lock acquire -- enabled but *not fetching*. `flag` is always released last, so
`sel_a` cannot have been entered when the write lands.

## Results

AMD Strix (`npu2_1col`), XRT 2.20.0, amdxdna 2.20.0, 20 runs per variant.

| | Core state | Write target | Landed | Verdict |
|---|---|---|---|---|
| **A** | fetching | no write (control) | 0/20 | control is clean |
| **B** | **fetching** | **near** (64 B from the PC) | **10/20** | **races, silently** |
| **C** | debug-halted | near | 20/20 | reliable |
| **D** | disabled | near | 20/20 | reliable |
| **E** | fetching | near, via ECC-bypass `0x24000` | **10/20** | same race |
| **F** | lock-stalled | near | 20/20 | reliable |
| **G** | **fetching** | **far** (8320 B from the PC) | **20/20** | **reliable** |
| **H** | debug-halted | far | 20/20 | reliable |

Every variant patches exactly one of two interchangeable pairs, so the other half
of each round is a per-run control. It read 7 in all 140 runs above: the write is
surgical, and the harness is not simply corrupting program memory.

B and E are genuinely nondeterministic — the rate wanders between builds and runs
(8/20, 10/20, 11/20, 12/20, 13/20 across revisions). A, C, D, F, G, H were 0/20 or
20/20 every time.

## What this means

**What decides the outcome is the write's distance from the program counter, not
whether the core is running.**

- **A write far from the PC lands every time, with the core running.** Variant G
  writes 8320 bytes from the spin loop while the core is fetching, and lands 20/20
  — no halt, no disable, no stall. This is the geometry a real overlay load has:
  you write the *inactive* slot, never the code you are executing.
- **A write next to the PC is a coin flip.** Variant B is identical to G except the
  target sits 64 bytes from the spin loop, and it lands about half the time. It
  never hangs and never returns a torn value; the new bytes simply do not take
  effect. The natural reading is that the core had already fetched them — a
  staleness effect, not a dropped write.
- **It is not ECC.** Variant E drives the identical near write through
  `PROGRAM_MEMORY_ERROR_INJECTION` at `0x24000`, the ECC-check-disabled mirror of
  PM, and races at the same rate as B.
- **Stalling or halting fixes the near case** (C, D, F all 20/20), which is worth
  knowing, but G shows it is not required when the write is far away.

So **double-buffered ("ping-pong") program memory works on AIE2P**: the load can
overlap compute, because the slot being written is by construction far from the
slot being executed. The rule is not "interlock the load with a stall" — it is
**"never write near the program counter"**.

### The one number this experiment does not give you

**Where the boundary between "near" and "far" is.** 64 bytes races, 8320 bytes does
not; everything between is unmeasured. That matters for a real design: with two
adjacent slots, a PC at the end of slot A is only a few bytes from the start of
slot B. Until the threshold is measured, keep a guard band between slots — or put
the resident region between them — rather than assuming any non-zero distance is
enough.

Two other things worth pinning down before relying on this: whether
`CORE_STATUS` (`0x32004`) bit 21 `CORE_PROCESSOR_BUS_STALL` is set during a write
(direct evidence for or against fetch arbitration), and whether the effect depends
on the *dynamic* PC at the instant of the write rather than static distance.

## Layout

| | |
|---|---|
| `aie2.py` | the design, in IRON, and the lit RUN lines. Emits pass 1 (no `--elf`) or a patched variant (`--variant X --elf <tmpdir>`). |
| `overlay_elf.py` | reads the overlay pair out of a core ELF and checks the two are interchangeable. Also `--check A B`, the drift guard. |
| `ovl.cc` | `sel_a`/`sel_b`/`ovl_wait`. |
| `test.cpp` | host side; expected round-1 value comes from `PM_EXPECT_ROUND1`. |

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
not move -- `sel_a` is at `0x180` in every build -- but that is asserted, not
assumed.

## Reproducing

The lit test covers only the deterministic variants (A, C, D, F); asserting B or E
either way would be flaky. To reproduce the full matrix by hand, build a variant
and run it repeatedly:

```
python aie2.py --dev npu2_1col --out design.mlir
aiecc --tmpdir=p1 --get-xclbin --xclbin-name=p1.xclbin \
      --get-npu-insts --npu-insts-name=p1.bin ./design.mlir
python aie2.py --dev npu2_1col --variant B --elf p1 --out final_B.mlir
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
