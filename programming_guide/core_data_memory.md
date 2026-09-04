<!---//===- core_data_memory.md ------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Core Data Memory

Every AIE compute tile has one small block of local data memory, 64 kB on npu2
for example. Three things share that block:

- the **stack**, at the bottom;
- the **`aie.buffer`s** that the buffer allocator places on the tile: the L1
  storage behind ObjectFifos, and any hand-declared buffer;
- the core's **own compiled sections** (`.data`, `.rodata`, `.bss`): the
  globals, the constants and the zero-initialized statics of the code that
  runs on the core, including its kernels.

The buffer allocator places the `aie.buffer`s above the stack reservation, and
places them so that the largest single run of free memory stays as large as it
can. The core compiler, Peano or Chess, then places the core's own sections
into that run. Buffer placement therefore decides whether the core links.

This page covers the **stack** and the **core's own sections**: how each is
sized, which attributes and flags control them, and what to do when a
diagnostic fires.

One rule governs the stack check: *the compiler measures and reports, you
declare and rebuild.* `aiecc` never writes `stack_size`. A value you set
explicitly, an explicit `0` included, stays as you wrote it, and a core that
leaves `stack_size` absent keeps the target default. When the measured
requirement exceeds the declared value, the build reports the number to set
and stops.

The core's own sections follow the opposite rule. Nothing declares their size
up front: the allocator leaves the largest run it can, and the core link
reports a shortfall against it.

## The stack: `stack_size`

`stack_size` is a per-core attribute on `aie.core`. A core that leaves it
absent uses the target default from
`AIETargetModel::getDefaultCoreStackSize()`, currently 1024 bytes. IRON spells
it on the `Worker`:

```python
Worker(core_fn, [args], stack_size=4096)
```

The stack sits directly below the buffers with no clearance, so a core whose
frames exceed the reservation overwrites the buffers above it. `aiecc`
therefore measures each core's stack requirement and checks `stack_size`
against that number. It measures the **linked core ELF**: the linker decides
which objects a core contains, so its output covers the kernels and the
toolchain's own startup code. The analysis reads the `.stack_sizes` section for
each frame, follows the relocations for the call edges, and takes the
**maximum over all root-to-leaf call paths** from the ELF entry point. One call
chain is live at a time, so the maximum bounds the requirement. Static data
differs: all of it coexists, whatever the control flow.

The walk starts at `__start` (crt0), which calls `_main_init` (crt1), which
calls the core body, which calls its kernels. `_main_init` holds a frame that
stays live across the whole chain, so it counts. `__start` establishes the
stack pointer, so its own frame counts as 0.

The check runs once, after each core is linked, and the Peano link keeps the
relocations (`-Wl,--emit-relocs`) that the walk needs. `aiecc` writes the
result to the `measured_stack_size` attribute on the `aie.core`, so you can
inspect it:

```mlir
aie.core(%tile_0_2) { ... } {stack_size = 8192 : i32, measured_stack_size = 4128 : i32}
```

Pass `--get=measured_stack_sizes.mlir` to dump that module. A `stack_size`
below `measured_stack_size` fails the build. `measured_stack_size` stays absent
when `aiecc` cannot measure the core; the next section lists those cases.

## Overriding a kernel's stack contribution: `stack_size_override`

The analysis cannot size some symbols:

- **recursion**: a cycle in the call graph is unbounded;
- **an indirect call through a function pointer**: the analysis fans out
  conservatively and still misses some targets;
- **a kernel compiled without `-fstack-size-section`**: the linked core carries
  no frame for it. A Chess-compiled object and a pre-compiled object both hit
  this case.

For these, declare the answer with `stack_size_override`. It lives on the
**kernel's `func.func`**, at function granularity, and not on the core. Two
reasons drive that placement. Several cores often link one kernel. And the
problematic symbol usually sits inside a kernel object that MLIR never reads,
so the override has to address the one granularity MLIR does read: the
external-function declaration. `aiecc` takes the declared value as the
requirement of that kernel's whole call subtree and stops the walk there. The
value is a declaration, not a clamp: an explicit value replaces the computed
one, even when it is smaller. An explicit `0` is legal.

IRON spells it as a keyword on the kernel declaration:

```python
Kernel("recursive_kernel", "recursive.o", [...], stack_size_override=4096)
ExternalFunction("my_kernel", ..., stack_size_override=4096)
external_func("my_kernel", ..., stack_size_override=4096)
```

## The core's own sections: the data region

`.data`, `.rodata` and `.bss` do not go wherever there is room. The generated
linker script grants the core compiler exactly **one** contiguous region, so
the number that matters is the largest single free run on the tile, not the
total free memory.

The allocator therefore scores every candidate buffer address by the largest
free run that address leaves behind. It writes the resulting region to the
`aie.core`, tile-relative, and `aie-translate` emits it as the `data` region:

```mlir
aie.core(%tile_0_2) { ... } {data_origin = 13312 : i32, data_length = 52224 : i32}
```

The start is aligned, because the linker begins `.data` at a multiple of its
strongest section alignment and an unaligned origin loses that much of the
region to padding.

Both attributes are outputs. `aie-assign-buffer-addresses` replaces whatever it
finds in them, so writing them by hand achieves nothing. To declare memory the
allocator must keep clear of, use a buffer at a fixed address; see [Cores built
ahead of time](#cores-built-ahead-of-time) below.

### `reserved_data_size`

`reserved_data_size` is a per-core attribute that sets a **floor** on that run.
Placement already maximizes the run it leaves, so a floor changes nothing until
the layout drops below it. Set one when you know a core needs more than the
layout happens to leave:

```mlir
aie.core(%tile_0_2) { ... } {reserved_data_size = 8192 : i32}
```

A floor the layout already clears costs nothing. A floor the allocator cannot
satisfy fails during buffer allocation and names the tile.

A floor also constrains placement, so a large one costs freedom
and can fail an allocation the linker would have accepted. Reach for it after a
link reports the `data` region overflowing, and set it to the number that
report gives.

### Cores built ahead of time

A core that carries an `elf_file` attribute comes linked, and its `.data` and
`.bss` sit at the addresses that link chose. `aiecc` reads nothing back out of
that ELF, so nothing tells the buffer allocator which bytes of the tile the ELF
holds, and a buffer can land on top of them.

`reserved_data_size` does not express this. It is a size, and what the ELF needs
is a specific range. Declare that range as a buffer at a fixed address on the
same tile:

```mlir
%prebaked = aie.buffer(%tile_0_3) {sym_name = "prebaked_data", address = 8192 : i32} : memref<4096xi8>
```

The allocator treats a fixed-address buffer as occupied space, keeps every
buffer it places clear of it, reports a collision against it by name, and lists
it in the memory map when a tile runs out of room.

With `--xchesscc`/`--xbridge`, `aiecc` compiles and links an `elf_file` core, so
that core gets a `data` region like any other.

## Escape hatches and allocation control

One flag disables a measurement, for a build that has to skip it and for
debugging the analysis:

- **`--no-measure-stack-size`** drops the measurement and the check, so no
  `measured_stack_size` reaches the IR.

A design-wide stand-in for the built-in default covers any core that leaves
`stack_size` absent:

- **`--default-stack-size=<bytes>`** assumes this many bytes in place of
  `AIETargetModel::getDefaultCoreStackSize()` for any core without an explicit
  `stack_size`. The rest of the build then treats that core as if it declared
  `stack_size` explicitly, and the diagnostics call the value assumed. A core
  with an explicit `stack_size` keeps it.

Separate flags control the allocation strategy:

- **`--alloc-scheme=<basic-sequential|bank-aware>`** picks the scheme for the
  whole design. Without it, the allocator runs bank-aware first and falls back
  to basic-sequential when bank-aware runs out of memory. Bank-aware spreads
  buffers across banks to limit DMA contention, but only where the spread does
  not cost the core its contiguous region: a large free run ranks above a
  spread.
- The per-tile **`allocation_scheme`** attribute picks the scheme for one tile
  and overrides `--alloc-scheme` there. IRON spells it
  `Worker(allocation_scheme="basic-sequential")`.
- **`Buffer(mem_bank=...)`** pins a buffer to a bank. Under bank-aware the pin
  is a hard constraint: the allocator reports an error when the bank cannot
  hold the buffer. Basic-sequential has no notion of banks, ignores the pin and
  warns that it dropped it, so a design that depends on `mem_bank` must not
  select that scheme.

## What to do when you hit a diagnostic

**`cannot determine this core's stack requirement: ...` (error).** The call
graph has a cycle, and recursion is unbounded. Set `stack_size_override` on
the affected kernel's `external_func()` or `func.func` declaration, to a value
large enough for the deepest recursion. Pass `--no-measure-stack-size` to skip
the check instead.

**`cannot determine this core's stack requirement: ...; stack_size is not being
validated for this core` (warning).** The linked core is unreadable, its
`.stack_sizes` data is malformed, or the link kept no relocations, so the call
graph is unavailable. The chess/BCF link produces such an ELF, so a
`--xbridge` build leaves `stack_size` unchecked and writes no
`measured_stack_size`.

**`no stack size information for N function(s) this core reaches, so its
requirement is at least M bytes and may be higher` (warning).** The linked core
carries no `.stack_sizes` entry for the functions the diagnostic names, and
their frames count as 0. `M` is therefore a lower bound: `aiecc` still fails a
core that declares less than `M`, and writes no `measured_stack_size`. Compile
the named source with `-fstack-size-section`, or set `stack_size_override` on
the affected kernel.

**`stack_size is absent, so this core uses the device default of M bytes, but
it needs N bytes` (error).** Set `stack_size = N`, or `Worker(stack_size=N)`,
and rebuild.

**`stack_size = M is insufficient: this core needs N bytes` (error).** The same
case, with `stack_size` already set explicitly to a value below the
requirement. Increase it to `N` and rebuild.

At this point the requirement is known, and `--no-measure-stack-size` silences a
proven overflow. Reach for it only when you believe the measurement itself is
wrong, and please file an issue in that case.

**`section '.bss' will not fit in region 'data'` from the linker, followed by
`this core's own .data/.rodata/.bss exceed the region reserved for them`.**
The linker reports the numbers, including how many bytes over. The region is
the largest gap between the stack and this tile's buffers. Free space by
shrinking or moving buffers, lower `stack_size`, or raise `reserved_data_size`
so that the allocator reserves more.

**`will not fit in region 'program'`, followed by `this core's code exceeds
the tile's program memory`.** Program memory is fixed and the region covers all
of it, so only the code can shrink. Split the work across more cores, remove
unused kernels from `link_files`, or lower the optimisation level.

**`section '.bss' will not fit in region 'data'` (linker error).** This core's
own sections do not fit the run the allocator left. `aiecc` follows the linker's
message with the core's name and the script that declares the region. Shrink or
move the tile's buffers, lower `stack_size`, or set `reserved_data_size` to the
number the linker reports and let placement work around it.

**`cannot reserve N contiguous bytes for this core's data sections` (error).**
The allocator cannot satisfy `reserved_data_size`. The message names the
largest run it found. At that point only the stack and this tile's pinned
buffers are placed, so one of those, or the reservation itself, has to give.

**`basic-sequential allocation ignores mem_bank; dropping the pin on: "b"`
(warning).** That scheme has no notion of banks. Either remove the `mem_bank`
request or let the tile use bank-aware allocation.
