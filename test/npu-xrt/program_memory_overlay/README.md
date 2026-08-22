<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Program-memory overlays

An AIE core holds 16 KB of code and there is no I-cache, no paging and no spill,
so a design whose kernels do not collectively fit has to split across tiles,
shrink, or reconfigure the whole device. This runs three real `aie_kernels` --
`silu`, `gelu`, `softmax` -- on one core by rewriting its program memory between
phases: which kernel the core runs is decided at run time, not at link time.

```
0x0000  resident   the wait loop, anything overlays call back into, and the
                   call site -- always present
0x2000  slot       whichever overlay was most recently written here
```

Phase *k* writes overlay *k* into the slot, releases the core, and collects a
row of output. All three rows are checked against references, so they can only
all be right if all three kernels were really loaded and run.

## Why the split is where it is

Not arbitrary, and not a guess. A configuration write to the 8 KB half of program
memory the core is currently fetching from is **silently discarded** about half
the time, while a write to the other half always lands — measured in
[`../pm_write_while_running`](../pm_write_while_running/README.md). Keeping the
resident in the low half and the slot in the high one means the core is never
fetching from the half being written, so the load is reliable with the core
running and needs no halt or stall.

`slot.ld`'s `ASSERT` fails the build if the resident ever grows into the slot.

## How a kernel becomes an overlay

The mechanism needs no compiler changes.

1. **The slot address reaches the linker as a symbol.** `slot.ld` says
   `overlay_entry = 0x2000;`. It is attached as a `Kernel`'s `link_with`, so
   `aie-assign-core-link-files` puts it in the core's `link_files`, the generated
   ld script emits `INPUT(slot.ld)`, and `ld.lld` parses an input it does not
   recognise as a linker script. The core's call compiles to a bare jump:

   ```
   62: 04 01 00 00 10 00      jl  #0x2000
   ```

   No definition is ever linked. The body arrives at run time.

2. **Each overlay is linked at that address** — `overlay.py link` — from its
   wrapper plus the kernel it calls, against the resident's *defined* symbols.
   That last part is why an overlay can call back into always-present code
   instead of carrying its own copy; absolute symbols are excluded, or the slot
   symbol itself would collide with the overlay's definition of it.

3. **The bytes are embedded and written.** `aie2.py` reads each overlay's `.text`
   into a `memref.global` and emits an `aiex.npu.blockwrite` to
   `0x20000 + 0x2000`, ordered before the release so the slot is populated by the
   time the core jumps into it.

Two passes are needed: pass 1 builds the resident, the overlays are linked
against it, pass 2 embeds them. Pass 2 recompiles the core rather than reusing
pass 1's ELF, so `overlay.py check` asserts the resident did not move between
passes — every overlay holds pass 1's addresses for the resident symbols it
calls.

## What is checked

`overlay.py link` fails the build unless each overlay is exactly one allocatable
section named `.text`, at the slot address, no larger than the slot, a whole
number of 128-bit program-memory lines, and free of static constructors. Those
are the ways an overlay silently misbehaves rather than failing to build:

- **anything but `.text`** — `.rodata` and `.data` live in *data* memory, so a
  kernel with a constant table needs a home that persists across every overlay.
  v1 refuses rather than pretending.
- **wrong address** — the resident jumps to a fixed address; if the entry is not
  there, the core jumps into the middle of something.
- **static constructors** — would never run.

The negative control is direct: corrupting a single word inside overlay 1's
payload in the instruction stream makes the run fail. Note that the payload has
to be located by a multi-word signature -- a single word matches elsewhere in the
instruction stream and corrupts a header instead, which the test does not notice.

## What this does *not* yet show

`overlay.py sizes` reports what the design would need if it were all resident at
once, and the number is the honest state of things:

```
  resident               544 bytes
  ovl0.elf               208 bytes    silu
  ovl1.elf               304 bytes    gelu
  ovl2.elf               752 bytes    softmax
  total                 1808 bytes of 16384 (11% of program memory)
```

**This set does not exceed program memory**, so it demonstrates the mechanism
rather than the capacity win. That is not a shortcut: AIE kernels are genuinely
small. The whole of `aie_kernels/aie2p` compiles to roughly 14 KB, so no handful
of activations reaches 16 KB — the entire library barely would.

Getting past the limit needs either many shapes of one kernel (`mm.cc` is about
900 bytes per tile shape after `--gc-sections`, so on the order of eighteen of
them) or a real application's layer library. yolo26n is the motivating case:
38 kernel objects totalling 94.8 KB of `.text`, nearly six times program memory.
Its largest single kernel is 9376 bytes, which is bigger than one 8 KB slot, so
kernels that large need the halt/stall path rather than this one. `overlay.py
sizes --require-exceeds` fails the build if a design that claims to exceed
program memory stops doing so.

Also not addressed here: only one slot, so the load cannot overlap compute. A
balanced two-slot ping-pong is possible — a write to the half not being executed
is reliable — but it requires the wait/dispatch code to be duplicated into both
halves, since a shared resident region would by definition sit in the same half
as one of the slots.
