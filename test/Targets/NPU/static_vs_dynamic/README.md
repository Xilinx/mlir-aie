<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Static-vs-dynamic TXN equivalence harness

These tests prove milestone #3222's correctness spine: a compiled-once
`aie.runtime_sequence` produces the *same* TXN word stream whether its scalar
fields are baked-in constants or supplied as runtime arguments.

Each test lowers two sequences — one static (all constants), one dynamic (takes
an `i32` argument) — to terminal npu ops, then compares three word streams that
must be byte-identical:

1. **golden** — the production binary emitter,
   `aie-translate --aie-npu-to-binary -aie-output-binary=false` on the static
   sequence.
2. **`generate_txn_<static>()`** — the C++ TXN builder from
   `aie-translate --aie-npu-to-cpp`, same constants. Catches EmitC-codegen drift
   against the binary emitter.
3. **`generate_txn_<dynamic>(N)`** — the C++ builder from the dynamic sequence,
   invoked with the constant the static side bakes in. Catches runtime-argument
   substitution drift.

`Inputs/compare_main.cpp` reads the golden hex file and calls the two generated
functions (selected via `-DSTATIC_FN` / `-DDYN_FN`, argument via `-DARGVAL`,
header via `-DGEN_HDR`), reporting the first divergent word on mismatch.

Two DMA lowering paths are covered because they converge on the same terminal
ops but reach them differently:

- `memcpy_nd.mlir` — the `aiex.npu.dma_memcpy_nd` path (`--aie-dma-to-npu`).
- `dma_task.mlir` — the DMA-task path
  (`dma_configure_task_for` / `dma_start_task` / `dma_await_task`, carrying the
  SSA `bd_id`), which needs `--aie-substitute-shim-dma-allocations
  --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu` before the shared
  `--aie-dma-to-npu` step, and is the only path that exercises the `sync` op.

Both paths are covered on both device generations: `memcpy_nd.mlir` /
`dma_task.mlir` target npu2, and `memcpy_nd_npu1.mlir` / `dma_task_npu1.mlir`
target npu1. Device info is baked into the TXN header words, so equivalence is
checked per generation.

Between them the tests exercise every op the C++ TXN target supports: `write32`,
`maskwrite32`, `blockwrite`, `address_patch`, and `sync`.

Only `rtp_write` carries the runtime value: it is the most a runtime argument
can drive without pushing the BD onto the per-register `write32` path, which
would make the static and dynamic streams differ structurally rather than in
value. Genuinely runtime-valued DMA sizes/strides arrive with the Phase-2
dynamic BD-word encoder, which will extend this harness.

## Adding a new size

Add a static sequence that bakes the new value (e.g. `@memcpy_static_8192`, as
in `memcpy_nd.mlir`), then add two RUN lines: one that emits its golden with
`-aie-sequence-name=<that sequence>`, and one that compiles the comparator with
`-DSTATIC_FN=<that sequence>` and `-DDYN_FN` still the shared dynamic sequence,
passing the new value as `-DARGVAL=`. The dynamic sequence is reused unchanged.
No C++ changes are needed — `compare_main.cpp` is fully parameterized by `-D`
macros. (The static side needs its own sequence because the golden is the binary
emitter run on all-constant fields.)

## Adding a new DMA pattern

Add another `aiex.npu.dma_memcpy_nd` (or another BD in a `dma_configure_task_for`)
to *both* the static and dynamic sequences, keeping them structurally identical
apart from the runtime argument. Remember the `aie.dma_bd` length is the product
of the lowest three dimension sizes — the highest dimension is the BD repeat
count, not part of the transfer length.

## Adding a conditional (scf.if)

A rolled `scf.if` over a runtime `i1` is proven against TWO static oracles: the
taken branch (the transfer, through the static allocator) and the not-taken
branch (an empty sequence). `rolled_if.mlir` compiles all three and the
comparator (`Inputs/rolled_if_compare.cpp`) asserts
`generate_txn_*_rolled_if(true)` equals the taken oracle and
`...rolled_if(false)` the not-taken one. The byte-identical guarantee holds only
when the runtime predicate equals the baked constant AND the pool's pop order
matches the static allocation, so keep these single-BD.
