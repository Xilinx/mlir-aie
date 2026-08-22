<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Program-memory overlays

An AIE core holds 16 KB of code, with no I-cache, no paging and no spill. A
design whose kernels do not collectively fit has to split across tiles, shrink,
or reconfigure the whole device. These tests write kernels into a core's program
memory at run time, so which code a core runs is decided while it runs — and the
program it executes over its lifetime is not bounded by what it can hold at once.

```
0x0000  resident   the wait loop, the call site, and anything overlays call
                   back into -- always present
0x2000  slot       whichever overlay was most recently written here
```

## Why the split is where it is

Not arbitrary, and not a guess. A configuration write to the 8 KB granule of
program memory the core is currently fetching from is **silently discarded**
about half the time, while a write to the other granule always lands — measured
in [`../pm_write_while_running`](../pm_write_while_running/README.md). Keeping
the resident in the low granule and the slot in the high one means the core is
never fetching from the granule being written, so the load needs no halt.

`pmlib/geometry.py` is the one place that rule lives. It reads the granule from
the target model, and refuses a device where it has never been characterised
rather than assuming npu2's behaviour holds — `get_program_memory_write_granule()`
returns `None` on npu1, and a layout there is rejected, not guessed at.

The rule is stated per slot: each `Slot` records `core_in`, the address the core
executes from while that slot is written. For one slot that is the resident at
0. For ping-pong it is the *other* slot, which is what makes ping-pong
expressible at all — phrasing it as "must not share a granule with the resident"
rules it out by definition rather than by evidence.

`hw/guard_band.lit` measures the case that rule leaves open: the core spinning
192 bytes below the slot while 123 payloads are written to it. All 123 land, so
the ASSERT stays at `SIZEOF(.text) <= slot` with no guard band.

## How a kernel becomes an overlay

No compiler changes.

1. **The slot address reaches the linker as a symbol.** `slot.ld` says
   `overlay_entry = 0x2000;`. It rides in as a `Kernel`'s `link_with`, the
   generated ld script turns that into an `INPUT()`, and `ld.lld` parses an input
   it does not recognise as a linker script. The core's call compiles to a bare
   jump; no definition is ever linked.
2. **Each overlay links at that address** against the resident's *defined*
   symbols, so it can call resident functions and reach resident buffers by name
   instead of carrying copies.
3. **The bytes are embedded and written** as a `memref.global` and an
   `aiex.npu.blockwrite`, ordered before the release so the slot is populated by
   the time the core jumps into it.

Two passes: pass 1 builds the resident, the overlays link against it, pass 2
embeds them. `pm.py check` asserts the resident did not move — and that every
symbol the overlays *import* still exists, which is a different question.

## What only travels in `.text`

Only `.text` is swapped. `AIETargetLdScript.cpp` routes `.rodata`, `.data` and
`.bss` into **data** memory, and nothing swaps that. So an overlay carrying a
constant table would be two pieces landing in two memories, and only one of them
travels — the table would never arrive and the kernel would read whatever the
previous overlay left.

`pmlib/link.py` refuses anything but a single allocatable `.text`, and refuses
static constructors separately so the message names the real problem: `crt0`
walks the constructor list once at core start, long before any overlay reaches
the slot.

Swapping `.rodata` too would need a data-memory arena partitioned across every
overlay, or a second swap path with its own coherence story. Neither exists.

## Layout

```
pmlib/      geometry.py  the placement rules, and named recipes
            design.py    the one IRON design, parameterized by a Config
            link.py      linking into a slot, and what is refused
            elf.py       just enough ELF32
            workload.py  generated kernels of an exact size
pm.py       the CLI the RUN lines drive
build/      REQUIRES: peano          -- no NPU, runs everywhere
hw/         REQUIRES: ryzen_ai_npu2  -- the mechanism on a device
```

`build/` holds most of the matrix, because most of what can go wrong here is
geometric or structural: placement rules, both size directions, the ASSERT
boundary, the jump into the slot, write-before-release ordering, payload
round-trip and link determinism, program-memory overflow, and the stack budget.

## Sizes are exact, in both directions

Boundary tests that sit *near* a boundary do not test it.

```
overlay  .text = 32*N + kernel     SEL blocks, 32 branch-free bytes each,
                                   retained through --gc-sections
resident .text = 304 + 64*N        N straight-line phase bodies
```

The resident figure has a trap in it. IRON's `range_()` emits an `scf.for` whose
size depends on whether LLVM's unroller expands it: linear to about N=16, then it
collapses to a real loop and stays near 900 bytes however large N gets. Driving
the resident with `range_()` would cap these tests around 1.5 KB and never reach
the boundary they exist to test. A plain Python loop emits the bodies
straight-line, and the size is linear and unbounded.

## Workloads

Generated ones by default: each adds its own tag to every element, so what ran
in a phase is readable straight off the output and distinctness holds by
construction rather than being checked. A workload whose output can go constant
makes a distinctness check vacuous, which is not hypothetical — an earlier cut
had two layers emitting all-zero rows that compared equal to everything.

`hw/real_kernels.lit` runs three real `aie_kernels` in bfloat16 against float
references, because generated workloads are branch-free padding plus a loop and
exercise very little of what a kernel does. A tag proves *which* overlay ran; a
numeric check proves it ran *correctly*.

**No scalar byte-store loops anywhere.** The pinned Peano miscompiles them,
dropping the last store of each unrolled iteration so three bytes in four are
written, silently. See [`../peano_scalar_store_canary`](../peano_scalar_store_canary/run.lit).

## Negative controls

Injected when the design is generated — `--corrupt`, `--skip-write`,
`--wrong-address` — not patched into a finished artifact. Locating a payload in a
built instruction stream needs a multi-word signature, breaks whenever codegen
shifts, and can land on padding that never reaches the output.

Every slot is poisoned during setup, and that earns its place: skipping phase 0
*with* the poison reports "ran the poison fill"; without it the identical design
gives "Kernel did not complete", because the core jumps into whatever the
previous xclbin left in program memory. npu-xrt runs serialized, so that outcome
depends on which test ran last.

## Ping-pong

The geometry and both call sites are checked in
`build/pingpong_geometry.lit`. The hardware half is not here, and the reason is
structural: the core returns from a slot into the resident at 0, so on the phase
where the low granule is written the core would be executing the granule being
written. It needs a dispatch path in the upper granule, and there is no mechanism
for one — `AIETargetLdScript.cpp` emits a single `.text ... > program` and aiecc
links with `--orphan-handling=error`, so a `link_with` fragment arriving as an
`INPUT()` cannot introduce a second output section. Either a write-once bootstrap
overlay or named-section placement for core code would do it.
