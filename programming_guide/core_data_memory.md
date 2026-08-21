<!---//===- core_data_memory.md ------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Core Data Memory

Every AIE compute tile has one small block of local data memory (for example,
64 kB on npu2), and three different things share it:

- the **stack**, at the bottom;
- the **`aie.buffer`s** placed on the tile by the buffer allocator — the L1
  storage behind ObjectFifos and any hand-declared buffers;
- the core's **own compiled sections** (`.data`/`.rodata`/`.bss`) — the
  globals, constants and zero-initialized statics of the code that runs on the
  core, including its kernels.

These three used to be decided by parties that never spoke to each other. The
buffer allocator placed `aie.buffer`s knowing only about the stack reservation;
the core compiler (Peano or Chess) later placed the core's own sections into
whatever memory the buffers left behind. Buffer placement therefore silently
decided whether the core would link — and it had no idea it was doing so. The
failure modes were correspondingly opaque: a stack that quietly overwrote the
buffers directly above it, or a linker "could not find free space" error that
named neither the tile nor the buffers responsible.

The compiler now measures and checks all three regions. This page describes the
attributes and flags that control that, and what to do when a check fires.

The guiding rule throughout is *the compiler measures and reports; you declare
and rebuild.* Any value you set explicitly — including an explicit `0` — is
never overwritten. The auto-measurement only fills in values you left absent.

## The stack: `stack_size`

`stack_size` is a per-core attribute on `aie.core`. When absent, the core uses
the target default from `AIETargetModel::getDefaultCoreStackSize()`, currently
1024 bytes. In IRON it is spelled on the `Worker`:

```python
Worker(core_fn, [args], stack_size=4096)
```

The stack sits directly below the buffers with no clearance, so a core whose
frames exceed the reservation overwrites the buffers above it. To catch that,
`aiecc` computes each core's stack requirement from its call graph and validates
`stack_size` against it. The analysis parses the `.stack_sizes` sections emitted
by every kernel object, builds a call graph across the core's `link_files`, and
takes the **maximum over all root-to-leaf call paths** — not the sum, because
only one call chain is live at a time. (Static data, by contrast, all coexists,
which is why `reserved_data_size` below *is* a sum.)

The check happens in two places, because the two halves of the requirement
become knowable at different times:

- Early in the build, `aiecc` knows the callees' subtrees but not the core
  body's own top-level frame (that only exists after the core's own codegen).
  This early number is a *lower bound*: a value that already exceeds
  `stack_size` is a proven problem, so it warns; a value that fits proves
  nothing, so it stays silent.
- After the build, `aiecc` reads the core's now-compiled frame back from its
  object and combines it with the lower bound to get the true requirement. If
  `stack_size` was left absent and the device default turns out to be
  insufficient, this is a hard build failure that names the exact value to set.

This asymmetry is only in the *early* check, which only ever warns because its
number is a lower bound that proves nothing when it happens to fit. The later,
post-build check has the true total, so it fails the build on an explicit
`stack_size` that is provably too small exactly as it does on an absent one —
a warning there would ship a proven overflow.

## Overriding a kernel's stack contribution: `stack_size_override`

Some symbols cannot be sized automatically:

- **recursion** (a cycle in the call graph is unbounded);
- **indirect / function-pointer calls** (the analysis fans out conservatively
  but cannot always resolve the target);
- **Chess-compiled objects**, which carry no `.stack_sizes`;
- **archives and bitcode** in `link_files`, which are not scanned;
- **`link_with_mode="merge"` kernels**, merged into the core's own LLVM module
  before codegen, so the analysis never sees them as a separate object.

For these, you declare the answer with `stack_size_override`. Crucially it lives
on the **kernel's `func.func`**, at function granularity, *not* on the core.
That placement is deliberate: the same kernel is often linked into several
cores, and the actually-problematic symbol is usually internal to a kernel
object that MLIR never saw — so the override has to be addressable at the one
granularity MLIR does see, the external-function declaration. `aiecc` treats the
declared value as the answer for that kernel's entire call subtree and does not
descend into it. It is a declaration, not a clamp: an explicit value wins even
if it is smaller than what the analysis would compute. An explicit `0` is legal.

In IRON it is a keyword on the kernel declaration:

```python
Kernel("recursive_kernel", "recursive.o", [...], stack_size_override=4096)
ExternalFunction("my_kernel", ..., stack_size_override=4096)
external_func("my_kernel", ..., stack_size_override=4096)
```

## The core's own sections: `reserved_data_size`

`reserved_data_size` is a per-core attribute on `aie.core` stating how many
bytes of *contiguous* data memory the core's own `.data`/`.rodata`/`.bss` need
above the stack. Contiguity is the point: the generated linker script hands the
core compiler exactly one region — the single largest gap left between the stack
and the placed buffers — so a design can have plenty of total free memory and
still fail to link because no single run is large enough for one big `.bss`
symbol. This is what made several designs pin themselves to
`--alloc-scheme=basic-sequential`, whose bottom-up packing happens to leave one
big run.

`aiecc` auto-measures this from the `.data`/`.rodata`/`.bss` sizes of the core's
`link_files` objects (plus a small margin), summed. Those objects are fully
compiled long before addresses are assigned, so their sections can simply be
read off. Only relocatable objects are measured; archives and bitcode are
skipped with a warning rather than approximated, since summing members the link
would not actually pull in would over-count without bound.

The user always wins. An explicit `reserved_data_size` — including an explicit
`0`, a legal way to say "reserve nothing" — is never overwritten. A core with
nothing measurable is left untouched, so absent still means "reserve nothing",
exactly today's behavior. Note that auto-measurement only sees `link_files`, not
`link_merge_files` (bitcode merged into the core's module at compile time) nor
the core's own compiled globals.

The buffer allocator now treats "every buffer was placed" and "a large enough
contiguous run survives" as a single acceptance test. Its strategy portfolio
gained a tight-packing entry that gives up bank spreading entirely, so a large
reservation only costs bank parallelism when it actually has to.

In IRON it is a keyword on the `Worker`:

```python
Worker(core_fn, [args], reserved_data_size=8192)
```

## Escape hatches and allocation control

Two flags disable the automatic behavior, for when you want the pre-existing
behavior back or are debugging the analysis itself:

- **`--no-auto-reserved-data`** — skip auto-measuring `reserved_data_size`;
  cores without an explicit value keep reserving nothing.
- **`--no-auto-stack-size`** — skip *all* `stack_size` validation (both the
  early warning and the later hard error); cores are not checked at all.

A design-wide stand-in for the target's built-in `stack_size` default is
available too, for any core that leaves `stack_size` absent:

- **`--default-stack-size=<bytes>`** — assume this many bytes instead of
  `AIETargetModel::getDefaultCoreStackSize()` for any core with no explicit
  `stack_size`. Once applied, that core is treated as if it had written
  `stack_size` explicitly for the rest of the build (diagnostics describe it as
  an assumed value, not as "absent"). A core with its own explicit `stack_size`
  is never affected.

Allocation strategy is controlled independently:

- **`--alloc-scheme=<basic-sequential|bank-aware>`** picks the scheme for the
  whole design. When unset, the allocator tries bank-aware first (which spreads
  buffers across banks to limit DMA contention) and falls back to
  basic-sequential if bank-aware runs out of memory.
- The per-tile **`allocation_scheme`** attribute overrides the scheme for one
  tile. In IRON this is `Worker(allocation_scheme="basic-sequential")`, and it
  overrides any scheme set on the tile. Bank placement can also be requested per
  buffer with `Buffer(mem_bank=...)`.

## What to do when you hit a diagnostic

**`cannot determine this core's stack requirement: ...` (error).** The call
graph has a cycle (recursion), which is unbounded. Set `stack_size_override` on
the affected kernel's `external_func()`/`func.func` declaration to a value large
enough for the deepest recursion, or pass `--no-auto-stack-size` to skip the
check entirely.

**`cannot determine this core's stack requirement: ...; stack_size is not being
validated for this core` (warning).** A symbol could not be measured — a missing
`.stack_sizes` section, a Chess-compiled object, an archive/bitcode entry, or a
`link_merge_files` dependency. This is the common case for pre-existing kernel
objects and is *not* fatal; the compiler simply cannot validate `stack_size` for
this core. If you want validation, set `stack_size_override` on the affected
kernel so the analysis has a number to trust. (Note the deliberate severity
split: an *unmeasurable* symbol only warns, but a genuine *cycle* is a hard
error — undercounting the stack corrupts memory, whereas an unknown value can
safely be left unchecked.)

**`stack_size is absent ... but this core's real requirement is N bytes; set
stack_size = N explicitly on this aie.core ... and rebuild` (error).** The
device default was assumed when the buffers were placed, but the finished core
needs more. Do exactly what it says — set `stack_size = N` (or
`Worker(stack_size=N)`) and rebuild — or pass `--no-auto-stack-size` to skip
the check.

**`stack_size = K is insufficient ... but this core's real requirement is N
bytes; increase stack_size to N ... and rebuild` (error).** Same as above, but
`stack_size` was already set explicitly to a value that turned out too small.
An explicit value is never silently changed, so this is still a hard failure,
not a warning: increase it to `N` and rebuild, or pass `--no-auto-stack-size`
to skip the check.

**`buffers leave only N contiguous bytes for the core's data sections, which
need M bytes` (warning).** The buffers were all placed, but the largest
surviving run is too small for the core's own sections. The free space is simply
too broken up. Options, cheapest first: reduce L1 buffer usage on the tile;
force tight packing on that tile with `allocation_scheme="basic-sequential"` (or
`--alloc-scheme=basic-sequential` design-wide), which packs from the bottom and
leaves one big run; or, if the total genuinely does not fit, move data off the
tile.
