<!---//===- core_data_memory.md ------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Core Data Memory

Every AIE compute tile has one small block of local data memory, 64 kB on npu2
for example. Three things share that block:

- the **stack**, at offset zero by default;
- the **`aie.buffer`s** that the buffer allocator places on the tile: the L1
  storage behind ObjectFifos, and any hand-declared buffer;
- the core's **own compiled sections** (`.data`, `.rodata`, `.bss`, and
  bank-pinned sections): the
  globals, the constants and the zero-initialized statics of the code that
  runs on the core, including its kernels.

The buffer allocator keeps the `aie.buffer`s clear of the stack reservation.
Ordinary static data needs one contiguous run that the buffers leave. On
Peano, bank-pinned static tables instead use separate bank-specific linker
regions; Chess uses its native storage constraints. Buffer and stack placement
therefore govern whether the core links.

Placement runs *after* each core is compiled, so how much of each bank that
core's own sections want is measured first and reserved. See
[The allocator reserves these banks for you](#the-allocator-reserves-these-banks-for-you).

This page covers the **stack** and the **core's own sections**: sizing,
bank placement, the attributes and flags that control them, and what to do
when a diagnostic fires.

One rule governs both checks: *the compiler measures and reports, you declare
and rebuild.* `aiecc` never writes `stack_size` or `data_size`. A value you set
explicitly stays as you wrote it (`data_size = 0` is legal; `stack_size` must
be positive). When the measured requirement exceeds the declared value, the
build reports the number to set and stops.

## The stack: `stack_size`

`stack_size` is a per-core attribute on `aie.core`. A core that leaves it
absent uses the target default from
`AIETargetModel::getDefaultCoreStackSize()`, currently 1024 bytes. IRON spells
it on the `Worker`:

```python
Worker(core_fn, [args], stack_size=4096)
```

The stack grows upward from its assigned address. A core whose frames exceed
the reservation can overwrite buffers or static data beyond it. `aiecc`
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
when `aiecc` cannot measure the core; see the stack-contribution overrides below.

### Stack placement: `stack_bank` and `stack_address`

These optional attributes on `aie.core` control placement, independently of
`stack_size`:

- **`stack_address`** is a byte offset within the tile's local data memory,
  not an absolute ELF address.
- **`stack_bank`** selects bank 0–3 (A–D) on AIE2/AIE2P. With no address, the
  allocator chooses an aligned free run in that bank and writes
  `stack_address`, accounting for fixed-address buffers first.
- With both attributes, the address must lie in the requested bank.

For example, on npu2, bank B starts at tile-relative offset `0x4000`:

```mlir
aie.core(%tile_0_2) { ... } {stack_size = 2048 : i32, stack_bank = 1 : i32, stack_address = 16384 : i32}
```

An explicitly placed stack must stay within local memory and not overlap a
buffer or static-data reservation. The address must satisfy the stack ABI
alignment: 32 bytes on AIE1/AIE2, 64 bytes on AIE2P/AIE2PS. Omitting both
attributes retains the legacy stack at offset zero.

A `stack_address` may cross bank boundaries, as the legacy stack always could.
What such a stack cannot do is claim a bank: `-aie-stack-addrspace` names one,
so aiecc leaves it at Peano's unrestricted default rather than promise the
bank-conflict model something untrue. `stack_bank` is the attribute that names
a bank, and it must hold the whole stack -- a `stack_bank` whose stack runs
past it is an error, because the two would disagree.

Moving within bank A works with both Peano and Chess. Moving to banks B–D
through `aiecc` requires Peano compilation and linking, with no separately
linked kernel inputs (`link_files`). Use LLVM IR kernels with
`link_with_mode = "merge"` instead: `aiecc` passes the selected stack address
space to Peano when compiling the combined core. Separately compiled kernels
may still assume bank A, and equivalent Chess assumptions cannot be verified,
so those combinations are rejected.

These placement attributes currently belong to the MLIR `aie.core` API;
IRON's `Worker` exposes `stack_size`, but not `stack_bank` or `stack_address`.

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

## The core's own sections: `data_size`

Ordinary, unpinned `.data`, `.rodata` and `.bss` do not go wherever there is
room. Peano's generated linker script grants them **one** contiguous `data`
region, so the number that matters is the largest single free run on the tile,
not the total free memory. Bank-pinned sections have separate regions,
described below.

`data_size` is a per-core attribute that reserves that run:

```mlir
aie.core(%tile_0_2) { ... } {data_size = 8192 : i32}
```

`aie-assign-buffer-addresses` turns the reservation into an `aie.buffer` marked
`core_data` and places it alongside the tile's other buffers:

```mlir
%core_data_0_2 = aie.buffer(%tile_0_2) {address = 49152 : i32, core_data, sym_name = "core_data_0_2"} : memref<8192xi8>
```

`aie-translate` emits that buffer's extent as the `data` MEMORY region of the
linker script. The buffer names no symbol, so nothing else refers to it.

Reserving costs nothing when the tile has room, and it improves placement. The
allocator packs the buffers around a declared reservation, and a packing that
leaves 8192 contiguous bytes often exists where the unconstrained placement
leaves two runs of 4096. A reservation the allocator cannot satisfy fails buffer
allocation and names the core.

On Peano, a core that leaves `data_size` absent normally gets an exact aligned
`core_data` reservation from the probe's `measured_data_size` and
`measured_data_alignment`, before buffer placement. No explicit `data_size` is
needed to make the allocator account for this ordinary static data.

Without that measurement or an explicit reservation, ordinary data uses the
largest aligned free run left by the stack and buffers. This fallback applies
when measurement is disabled or unavailable, including the Chess flow. On
Peano, any nonempty bank-pinned sections intersecting that run move the
ordinary-data start past them; the end stays fixed. This can leave unused gaps
rather than packing data into every hole.

The allocator caps the leftover run it aims for. Free space beyond the bytes
still to place serves nothing, so the ranking counts a run only up to that
amount. Within the cap it prefers layouts that spread buffers across banks,
which limits DMA contention.

### `measured_data_size`

`aiecc` counts the allocated `.data`, `.rodata` and `.bss` of each **linked core
ELF** and writes the total to `measured_data_size`:

```mlir
aie.core(%tile_0_2) { ... } {data_size = 8192 : i32, measured_data_size = 6144 : i32}
```

The count comes from the linked ELF because `--gc-sections` runs during the
link. A kernel header that defines a large lookup table no code reads adds to
the object files and drops out of the ELF. A count taken from the objects would
reject designs that fit.

Before placement, the probe measurement also includes padding between output
sections and records their maximum alignment as `measured_data_alignment`.
The allocator preserves this alignment for the core-data reservation, including
an explicitly declared `data_size`, so over-aligned globals do not consume
unreserved leading padding at the final link.
Bank-pinned reservations likewise retain their per-bank start alignment in
`measured_bank_alignments`; rounding only their sizes cannot cover that padding.

On Peano, this count excludes the separate `.aie.bank0`–`.aie.bank3` output
sections. It is not a total of all static storage on the tile: pinned sections
must fit their own regions, independently of `data_size`. Chess sections whose
names start with `.data`, `.rodata` or `.bss` are included in the count.

Pass `--get=measured_data_sizes.mlir` to dump the module carrying both measured
attributes (when both measurements are enabled). A `data_size` below
`measured_data_size` fails the build and reports the number to set.

## Bank-pinned static tables and LUT checks

For AIE2/AIE2P, include the installed `aie_bank_placement.h` in a kernel and
annotate tables with `AIE_BANK_A` through `AIE_BANK_D`. For example:

```cpp
#include "aie_bank_placement.h"

alignas(64) static const int table_ab[16] AIE_BANK_A = { /* ... */ };
alignas(64) static const int table_cd[16] AIE_BANK_B = { /* ... */ };
```

The table contents, sizes and alignment must still match the LUT operation.
These annotations place static storage; they do not move automatic,
stack-local arrays. The AIE2/AIE2P runtime exp and tanh table pairs use these
macros to select banks A and B.

| Feature | Peano | Chess |
| --- | --- | --- |
| `AIE_BANK_A`–`AIE_BANK_D` | Emit `.aie.bank0`–`.aie.bank3` sections; Peano does not honor `chess_storage`. | Use native `chess_storage(DM_bankA)`–`DM_bankD` constraints. |
| Bank-specific linker regions | One aligned contiguous reservation per measured bank, excluding buffers, the stack and ordinary-data reservations; falls back to a free run when unmeasured. | Not used; placement uses the Chess/BCF flow. |
| Default bank-placement check | Supported; reads object sections and linked ELF symbols. | Supported; recognizes native `DM_bankX` section names, without needing LLVM IR. |
| Opt-in `--check-lut-banks` | Supported for recognized gather patterns with readable LLVM IR. | Rejected with `--xchesscc` or `--xbridge`; Chess IR is incompatible with the analyzer. |

A Peano bank region never spills into a neighboring bank. An explicit
`data_size` reservation competes with pinned tables for memory; increasing it
can leave less space in the requested banks, not more. Without an explicit
`data_size`, the probe normally supplies the ordinary-data reservation. Only
the unmeasured fallback adjusts the leftover `data` region around pinned
sections as described above.

### The allocator reserves these banks for you

Nothing above needs declaring. `aiecc` compiles each core, links it once
against permissive regions to see how large its `.aie.bank<N>` sections
actually are, and records the answer in `measured_bank_sizes` on the
`aie.core`. Buffer placement then holds that much room in each of those banks
before it places anything unconstrained, so a buffer cannot take a bank a
kernel's tables need.

This is why placement runs after the core compile rather than before it: the
sections only exist once the object is built. A consequence worth knowing is
that `input_with_addresses.mlir` now costs a core compile. Use
`--get-input-with-symbols` for the same module before placement, which does
not.

The measurement is the linked size, so `--gc-sections` has already run and a
pinned table the kernel never reads reserves nothing. Reservations appear in a
tile's memory map as `(bank-pinned static data)`. Chess places bank-pinned
statics through its own storage constraints rather than linker regions, so none
of this applies there.

A table larger than its bank still cannot be placed, and is now reported while
reserving rather than by the linker:

```
error: 'aie.buffer' op this core's static data pinned to bank 1 requires 32768
bytes, which cannot fit in bank 1 (16384 bytes total)
```

The **default placement check** compares each unambiguous bank-annotated
definition with its linked address and complete nonzero symbol extent. It
reads native objects and static-archive members, including thin archives.
Absent or ambiguous symbols are skipped, and bank assumptions expressed only
by pointer casts inside a kernel are not visible to this check. It is not a
proof that every LUT pair occupies different banks.

The stronger **`--check-lut-banks`** check recovers pairs from LLVM IR and
requires the two tables to occupy different banks. Object-linked kernels need
readable embedded `.llvmbc`; merged kernels and generated core code use the
optimized core IR. The check can resolve parameters bound to bank-pinned MLIR
buffers on the core's own tile. Missing IR, unresolved operands or parameter
bindings, and stack-local tables fail rather than silently passing. Archive
inputs and prebuilt `elf_file` cores are unsupported by this opt-in check.
Native functions known to have been removed by linker garbage collection are
omitted; analysis stays conservative about inlined or renamed functions.

The option is off by default because preserving IR adds compilation cost:

- For source-backed kernels built through IRON, use
  `@iron.jit(aiecc_flags=["--check-lut-banks"])`.
- Direct `compile_mlir_module(..., options=["--check-lut-banks"],
  device=..., work_dir=...)` calls also retain IR when auto-building
  source-backed external kernels.
- When compiling a kernel separately with `compile_cxx_core_function`, request
  `embed_bitcode=True` and pass `--check-lut-banks` to the later `aiecc` call.
  Precompiled objects are not retroactively given IR.
- Participating Make recipes using `attach_bitcode` opt in with
  `AIE_CHECK_LUT_BANKS=1`; that variable does not modify arbitrary build
  recipes.

For LLVM IR kernel compilation, see the
[merge-mode example](../programming_examples/basic/inline_kernel/README.md).

## Cores built ahead of time

A core that carries an `elf_file` attribute comes linked, and its `.data`,
`.bss` and any `.aie.bank<N>` tables sit at the addresses that link chose.
`aiecc` reads those extents back out of the ELF and records them on the core as
`measured_data_ranges`, address/size pairs relative to the tile. Placement pins
a buffer over each one, so the tile's own buffers are kept clear of memory the
image already holds.

This is the same measurement a compiled core gets, read from the same sections;
only what is taken from it differs. A compiled core is probed for *sizes*,
because its addresses are chosen afterwards. A prebaked core is read for its
*addresses*, because they are already final.

`data_size` cannot express this: it is a size, and what a prebaked ELF occupies
is a specific range. Declaring the range yourself still works, and is worth
doing when the ELF is not where `aiecc` can read it:

```mlir
%prebaked = aie.buffer(%tile_0_3) {sym_name = "prebaked_data", address = 8192 : i32} : memref<4096xi8>
```

Either way the allocator treats the extent as occupied space, keeps every
buffer it places clear of it, reports a collision against it by name, and lists
it in the memory map when a tile runs out of room.

With `--xchesscc`/`--xbridge`, `aiecc` compiles and links an `elf_file` core, so
that core gets a `data` region like any other.

## Escape hatches

Separate flags disable measurements and checks, for debugging or a build that
has to skip them:

- **`--no-measure-stack-size`** drops the stack measurement and its check, so no
  `measured_stack_size` reaches the IR.
- **`--no-measure-data-size`** drops ordinary static-data measurement before
  placement and after linking, including its `data_size` check. Per-bank
  measurements and reservations remain enabled. Explicit `data_size`
  reservations and linker capacity limits still apply.
- **`--no-check-bank-placement`** drops the default annotation-placement
  check, but does not relax linker bank regions or disable an explicitly
  requested `--check-lut-banks`.

A design-wide stand-in for the built-in default covers any core that leaves
`stack_size` absent:

- **`--default-stack-size=<bytes>`** assumes this many bytes in place of
  `AIETargetModel::getDefaultCoreStackSize()` for any core without an explicit
  `stack_size`. The rest of the build then treats that core as if it declared
  `stack_size` explicitly, and the diagnostics call the value assumed. A core
  with an explicit `stack_size` keeps it.

## How placement chooses addresses

One allocator places every extent on a tile: the stack, each buffer, the core's
own sections, and the bank reservations described above. It is bank-aware
throughout — there is no second, bank-oblivious scheme to select or fall back
to, because "succeeding" by ignoring a `mem_bank` request is not success.

Extents are placed most-constrained first: those with an explicit `address`,
then those pinned to a bank, then the rest, largest first. Each candidate
address is ranked by

1. banks touched, fewest first — a spanned bank costs DMA bandwidth;
2. the largest free run left behind, biggest first, capped by the bytes still
   to place;
3. distance from a round-robin cursor, which spreads buffers over banks;
4. tightness of fit, then lowest address for determinism.

A buffer larger than one bank must span banks, and any unpinned buffer may span
if that is what fits; the memory map marks it `(straddles into bank N)`. A
`mem_bank` pin is a hard single-bank constraint, so a pinned buffer larger than
its bank is an error rather than a straddle — the pin exists to tell Peano's
bank-conflict model which bank the accesses hit, and a straddling buffer would
make that untrue.

**Placement backtracks.** Ranking alone is not enough: the placement that
touches the fewest banks can strand the free run a later buffer needs. When a
buffer has nowhere to go, the allocator undoes an earlier choice and tries that
buffer's next-best address. It explores in rank order, so the first arrangement
it tries is the one the ranking prefers and most designs never backtrack at all.

Backtracking only tries positions flush against what is already placed, so a
layout where a buffer must sit against one placed after it is out of its reach.
When the ranked search fails, an exhaustive one takes over: it slides every
buffer down into address order and chooses which buffer comes next, which
reaches every layout that exists. Most designs never get there, and those that
do only need a legal layout, so it does not rank.

The exhaustive search is bounded at 200000 placements per tile — packing around
fixed obstacles is NP-hard, and the bound is a count rather than a time limit
so builds stay reproducible. Within the bound, "no room" is a proof that no
layout exists; reaching it is reported (see the diagnostics below) rather than
passed off as one.

- **`Buffer(mem_bank=...)`** pins a buffer to a bank. The pin is honored or the
  build fails; it is never silently dropped.
- **`Buffer(address=...)`** pins an exact address, for a buffer an external ABI
  fixes — an RTP buffer a host writes at a known location, say. A pinned
  address is held only to the bus width, not the stricter vector alignment the
  allocator applies to addresses it chooses itself.

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
`core X needs space for up to N bytes of static data`.** The core's own
sections do not fit the run left for them, and the linker rather than the
allocator noticed. A core whose sections were measured usually fails earlier,
while placing, with the `could not be placed` error above; this is the residual
case — a `--xbridge` build, or a core whose measurement was skipped. `aiecc`
adds the linker's shortfall to the size of the region to state `N`. Set
`data_size = N` on the core, or `Worker(data_size=N)` in IRON, and rebuild.
Shrinking or moving buffers, or lowering `stack_size`, frees the bytes when `N`
does not fit.

**`will not fit in region 'program'`, followed by `this core's code exceeds
the tile's program memory`.** Program memory is fixed and the region covers all
of it, so only the code can shrink. Split the work across more cores, remove
unused kernels from `link_files`, or lower the optimization level.

**`will not fit in region 'bankN'` from the Peano linker.** Pinned sections
cannot fit the contiguous free run in that bank. Shrink or repin the tables,
or move buffers, the stack or an explicit `data_size` reservation out of it.
Increasing `data_size` does not enlarge a pinned region.

**A bank-placement violation or LUT-pair separation error.** Check the final
symbol addresses, their complete extents, and the requested banks. Put the two
tables in distinct banks using the portable annotations or bank-pinned MLIR
buffers. If the opt-in check cannot recover placement, supply readable kernel
IR and resolvable table bindings rather than treating the result as a
same-bank diagnosis.

**`requires a N-byte stack in bank B, but no contiguous aligned space remains
after address-pinned buffers` (error).** `stack_bank` names a bank whose free
space is already broken up by buffers pinned to exact addresses, which cannot
move. Repin those buffers, choose another bank, or give the stack an explicit
aligned `stack_address`. Moving outside bank A also requires the Peano
merge-mode restrictions described above.

**`data_size M is smaller than the N bytes this core's linked sections occupy`
(error).** `aiecc` measured the linked ELF and the reservation does not cover
it. Set `data_size = N` and rebuild.

**`could not be placed: <what> needs N bytes and this tile has no room left for
it` (error).** Everything the tile must hold does not fit. The message lists
the tile's memory map, showing the closest arrangement the allocator reached,
and `<what>` names the one extent left over — a buffer, `this core's data
sections (data_size)`, or `this core's static data pinned to bank B`. The bytes
may well exist while being too fragmented to use, so shrinking any extent can
help, not only the one named. Lower `data_size`, shrink or move buffers, or
lower `stack_size`.

**`the search hit its 200000-placement budget with arrangements still untried`
(note, attached to the error above).** The allocator gave up rather than proved
the design infeasible, so a layout may exist that it did not reach. This is
rare and worth reporting: please file an issue with the design. Pinning a
buffer or two with `mem_bank` cuts the search down and often gets a build
through in the meantime.

**`<what> requires N bytes, which cannot fit in bank B (C bytes total)`
(error).** A `mem_bank` pin, or a kernel's bank-pinned static data, asks for
more than a whole bank holds. Nothing about the rest of the design can fix
this: shrink the extent, or drop the pin and let it span banks.

**`<what> requires N bytes in bank B, but only M of C bytes are free there`
(error).** The bank would hold it on its own, but the stack, address-pinned
buffers, or another pin already in that bank leave too little. Repin to a
different bank, or move what is in the way.
