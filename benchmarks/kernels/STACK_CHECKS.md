<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Stack-usage checks for aie_kernels

A plan, not an implementation. Facts marked **[verified]** were read
from the named source; those marked **[confirm]** have not been checked
and come first when this is picked up.

## Why this matters on AIE

- **[verified]** `lib/Dialect/AIE/Transforms/AIEAssignBuffers.cpp`: the
  buffer allocator starts at `address = core.getStackSize()`; the stack
  owns `[0, stack_size)` of the tile's local data memory and every
  `aie.buffer` (all ObjectFifo elements) is placed above it.
- **[verified]** `lib/Targets/AIETargetLdScript.cpp`: the linker script
  emits `_sp_start_value_DM_stack` at the memory base and then
  `. += stack_size`. The stack pointer therefore starts at the *bottom*
  of the region and grows *upward*.
- Consequence: an overflow writes into the lowest-addressed buffer,
  which is typically the first ObjectFifo element. There is no MMU and
  no fault; the symptom is wrong output (or a hang if a lock word is
  hit). A Peano bump that changes inlining or spilling is the most
  likely way stack usage creeps.
- **[verified]** the linker script keeps a `.stack_sizes` section
  (`*(.stack_sizes)`), i.e. Peano emits LLVM's per-function stack frame
  sizes; `llvm-readobj --stack-sizes <elf>` prints them.
- **[confirm]** PR #3680 "[aiecc] Measure stack size after linking,
  counting the runtime entry frame" adds a post-link measurement in
  `aiecc`. Confirm: (a) is it reported per core in the log / cache dir /
  an `--get` artifact, (b) is it a call-graph worst-case depth or the
  largest single frame, (c) does it error or warn when it exceeds
  `stack_size`.
- **[confirm]** default `stack_size` on `aie.core` (believed 0x400) and
  how IRON exposes it (`Worker(..., stack_size=)` or `Program`).

## Tier 1: static, in the nightly

Add to `harness.measure_compile`, per core ELF in the cache dir:

```
stack_bytes_alloc   = aie.core stack_size          (from input_with_addresses.mlir)
stack_bytes_used    = aiecc post-link measurement  (preferred)
                    | worst-case path over .stack_sizes + call graph (fallback)
stack_headroom      = alloc - used
```

Emit `stack_bytes_used` and `stack_headroom` per kernel/case in
`bench.json` (unit: bytes). Regression rule: `stack_bytes_used` at 3 %
like the other size metrics; **hard gate** in `run.py`: any kernel with
`stack_headroom < max(256, 0.10 * alloc)` invalidates the run (exit 3),
and the canary check includes it.

Fallback computation if aiecc does not expose a number:
`llvm-readobj --stack-sizes --elf-output-style=JSON main_core_C_R.elf`
gives `{function -> frame bytes}`; `llvm-objdump -d` gives call edges;
worst-case depth = longest path from the entry (`core_C_R` → `main`)
through the call graph summing frame sizes, plus the runtime entry frame
(the "counting the runtime entry frame" part of #3680). Recursion or
indirect calls → report `unknown` and fail the gate, do not guess.

Also record `first_buffer_above_stack` (name + address from
`input_with_addresses.mlir`) in `meta.json` so an overflow report says
what would be corrupted.

## Tier 2: dynamic high-water mark (needs a small runtime change)

Static numbers miss `alloca`, recursion and anything the linker cannot
see. The standard embedded technique is stack painting:

1. Extend the AIE `crt0` / entry wrapper in `aie_runtime_lib` with an
   opt-in (compile flag, e.g. `-DAIE_STACK_PAINT`) that fills
   `[_sp_start_value_DM_stack, +stack_size)` with `0xAA` before calling
   `main`.
2. Add a helper `aie_stack_high_water()` (same lib) that scans from the
   top of the region downward for the first non-`0xAA` word and returns
   the byte offset.
3. The benchmark harness worker calls it after the kernel loop and
   writes the value to a 1-element `Out` ObjectFifo, so it rides out on
   the normal drain. This is why it belongs in the runtime lib rather
   than in the harness: the harness cannot address the stack region as an
   `aie.buffer` (the allocator starts above it).

Emit `stack_high_water_bytes`; gate as in tier 1. Painting costs one
memset per run so it runs in the correctness suite and in a separate,
untimed pass of the benchmark — never in the timed iterations.

## Precedent

Firmware and DSP targets do both: `-fstack-usage` / `-Wframe-larger-than`
in CI, and stack painting with a high-water-mark reader at runtime. An
AIE core's stack is the embedded case exactly: fixed, small, unprotected
and adjacent to live data.
