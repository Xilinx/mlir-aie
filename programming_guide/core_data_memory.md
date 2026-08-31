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

The buffer allocator places the `aie.buffer`s above the stack reservation. The
core compiler, Peano or Chess, then places the core's own sections into the
memory that the buffers leave free. Buffer placement therefore decides whether
the core links.

This page covers the **stack** region: how the compiler measures a core's
stack requirement, which attributes and flags control that measurement, and
what to do when a diagnostic fires. A separate mechanism reserves and checks
the core's own compiled sections.

One rule governs the whole check: *the compiler measures and reports, you
declare and rebuild.* `aiecc` never writes `stack_size`. A value you set
explicitly, an explicit `0` included, stays as you wrote it, and a core that
leaves `stack_size` absent keeps the target default. When the measured
requirement exceeds the declared value, the build reports the number to set
and stops.

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
therefore computes each core's stack requirement from its call graph and
checks `stack_size` against that number. The analysis parses the
`.stack_sizes` section of every kernel object, builds a call graph across the
core's `link_files`, and takes the **maximum over all root-to-leaf call
paths**. One call chain is live at a time, so the maximum bounds the
requirement. Static data differs: all of it coexists, whatever the control
flow.

The check runs at two points, because the two halves of the requirement become
known at different times:

- Early in the build, `aiecc` knows the subtrees of the callees. The core
  body's own top-level frame appears only after codegen of the core, so this
  early number is a *lower bound*. A lower bound above `stack_size` proves a
  problem, and `aiecc` warns. A lower bound below `stack_size` proves nothing,
  and `aiecc` stays silent.
- After the build, `aiecc` reads the compiled frame of the core body from its
  object and adds the lower bound. That sum is the full requirement, so a
  smaller `stack_size` fails the build, an explicit one included.

## Overriding a kernel's stack contribution: `stack_size_override`

The analysis cannot size some symbols:

- **recursion**: a cycle in the call graph is unbounded;
- **an indirect call through a function pointer**: the analysis fans out
  conservatively and still misses some targets;
- **a Chess-compiled object**: it carries no `.stack_sizes` section;
- **an archive or a bitcode file** in `link_files`: the analysis skips both;
- **a `link_with_mode="merge"` kernel**: it merges into the core's own LLVM
  module before codegen, so the analysis reads it as part of the core.

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

## Escape hatches and allocation control

One flag disables the check, for a build that has to skip it and for debugging
the analysis:

- **`--no-auto-stack-size`** skips every `stack_size` check, both the early
  warning and the later error.

A design-wide stand-in for the built-in default covers any core that leaves
`stack_size` absent:

- **`--default-stack-size=<bytes>`** assumes this many bytes in place of
  `AIETargetModel::getDefaultCoreStackSize()` for any core without an explicit
  `stack_size`. The rest of the build then treats that core as if it declared
  `stack_size` explicitly, and the diagnostics call the value assumed. A core
  with an explicit `stack_size` keeps it.

Separate flags control the allocation strategy:

- **`--alloc-scheme=<basic-sequential|bank-aware>`** picks the scheme for the
  whole design. Without it, the allocator runs bank-aware first, which spreads
  buffers across banks to limit DMA contention, and falls back to
  basic-sequential when bank-aware runs out of memory.
- The per-tile **`allocation_scheme`** attribute picks the scheme for one tile
  and overrides `--alloc-scheme` there. IRON spells it
  `Worker(allocation_scheme="basic-sequential")`. `Buffer(mem_bank=...)`
  requests a bank per buffer.

## What to do when you hit a diagnostic

**`cannot determine this core's stack requirement: ...` (error).** The call
graph has a cycle, and recursion is unbounded. Set `stack_size_override` on
the affected kernel's `external_func()` or `func.func` declaration, to a value
large enough for the deepest recursion. Pass `--no-auto-stack-size` to skip
the check instead.

**`cannot determine this core's stack requirement: ...; stack_size is not being
validated for this core` (warning).** The analysis cannot measure a symbol,
because of a missing `.stack_sizes` section, a Chess-compiled object, an
archive or bitcode entry, or a `link_merge_files` dependency. A pre-compiled
kernel object hits this case often, and the build continues: the compiler
leaves `stack_size` unchecked for this core. Set `stack_size_override` on the
affected kernel to give the analysis a number to work from.

**`stack_size is absent ... but this core's real requirement is N bytes; set
stack_size = N explicitly on this aie.core ... and rebuild` (error).** The
buffer placement assumed the device default, and the finished core needs more.
Do what the message says: set `stack_size = N`, or `Worker(stack_size=N)`, and
rebuild.

**`stack_size = K is insufficient ... but this core's real requirement is N
bytes; increase stack_size to N ... and rebuild` (error).** The same case, with
`stack_size` already set explicitly to a value that turned out too small.
`aiecc` leaves an explicit value alone, so this stays a hard failure. Increase
it to `N` and rebuild.

At this point the requirement is known, and `--no-auto-stack-size` silences a
proven overflow. Reach for it only when you believe the measurement itself is
wrong, and please file an issue in that case.
