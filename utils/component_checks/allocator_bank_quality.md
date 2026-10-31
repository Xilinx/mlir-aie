<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Measuring the buffer allocator on real designs

Status: proposal (2026-10-29). Nothing here is implemented yet.

## Why

The nightly *Buffer allocator model-based tests* (`test/assign-buffer-addresses/allocator_properties.py`)
generate 200 designs from fixed seeds and check that `--aie-assign-buffer-addresses` places them
legally, completely and deterministically. Its one quality number, *bank pair sharing*, is
measured on those same generated designs. The inputs never change, so the number only moves
when the allocator's code does: it is a regression tripwire, and the dashboard's flat line is
expected. It does not tell us whether a real design (MobileNet, YOLO26n, Llama) loses time to
bank conflicts, which is the question that matters.

Two measurements would answer it: a cheap one at compile time, every night, and the real cost
on hardware.

## A. Bank sharing on real designs (compile time)

**What.** For each design the hardware nightly already compiles, count the pairs of buffers that
one kernel reads or writes together and that the allocator put in the same bank. Two such
buffers serialize the core's accesses; two buffers merely on the same tile do not.

**Where the data is.** aiecc already writes `input_with_addresses.mlir` into each build's
working directory (`--get-input-with-addresses`; `python/utils/compile/utils.py:1220`, which the
JIT's DMA-size validator and the trace parser already read). Every `aie.buffer` there carries
its `address` and `mem_bank`.

**Which buffers are read together.** In order of fidelity:

1. *v1, per tile:* every buffer on a core tile, as `bank_pair_sharing` already does for the
   generated designs (`allocator_properties.py:815`). Simple, but it overcounts: it includes
   ping and pong buffers that are never live at once.
2. *v2, per call:* the buffers passed together to one `func.call` in a core's body, after
   objectFIFO lowering: each call's operands, and for a ping-pong FIFO each of its buffers in
   turn. This is the set the kernel actually reads together.

**Metrics,** one row per design case (`mobilenet/seed=3/effort=default`, `yolo26n/640`,
`llama/persist_grow/N=16`):

| metric | meaning | threshold |
|---|---|---|
| `bank_shared_pairs` | co-accessed buffer pairs in one bank, summed over tiles (lower is better) | `abs: 1`, gated |
| `bank_pair_sharing_real` | the same normalized 0–1 as the synthetic test's (0 = the most even split the banks allow) | `pct: 5` |
| `worst_tile_shared_pairs` | the worst single tile's count, with the tile named in the range | `abs: 1` |

**Plumbing.**

- Move `bank_pair_sharing` out of the test into a module both use, e.g. a `bank_quality.py`
  beside `aiecc_stages.py`.
- Call it from `sa_placer_hw.py`, `yolo26n_hw.py` and `llama_chain_hw.py` after the cold
  compile, reading the build's `input_with_addresses.mlir` with the MLIR Python bindings.
- No extra hardware time: it is a read of a file the compile already wrote.
- On the dashboard, the Buffer allocator card on Compiler checks gets a per-design table and
  chart beside the synthetic one, and the Overview names a design whose shared pairs rise.

**Caveats.** DMAs also contend for banks, and this ignores them. The core's static data region
is excluded, as in the synthetic metric. A pair in one bank costs nothing if the kernel never
accesses both in the same cycle window: that is what B measures.

## B. Bank-conflict stalls on hardware

**What.** How many cycles the cores actually lose to bank conflicts, per design, at batch 1.

**How.** The AIE-ML memory module raises `CONFLICT_DM_BANK_0` to `CONFLICT_DM_BANK_3` on a
conflict in each data-memory bank. The core raises `MEMORY_STALL` while it waits. The trace
setup already includes all of them in its defaults for core memory and core tiles
(`python/utils/trace/setup.py:306`), so two routes are open:

1. **Trace (available today).** Run one design once more with trace on (those events on every
   core tile it uses), then count events per tile with `get_trace_summary.py`. Trace perturbs
   timing and its buffer limits run length, so this is a separate, short, untimed run, never
   the timed one.
2. **Performance counters (cheaper, new work).** Count `CONFLICT_DM_BANK_*` and `MEMORY_STALL`
   in each tile's counters over the whole timed run. The repo can only *reset* the counters
   today (`AIERT.cpp:1185`, `resetPerfCounters`); it would need to configure them on an event
   and read them back after the run. No trace buffer, no perturbation, whole-run totals.

**Metrics,** per design:

- `bank_conflict_cycles_per_image`: conflict events summed over tiles, per image.
- `memory_stall_pct`: memory-stall cycles as a share of core cycles.
- The worst tile, named with its conflicts.

Both are lower-is-better and machine-independent within one NPU generation (`across_parts`).

**Prove the measurement first.** Add a two-tile hardware test where a kernel reads two buffers
every cycle, built twice: with both pinned to one bank (`mem_bank`), and in two banks. The first
must count conflicts and the second none. Without that, a zero on the dashboard could just mean
the counters never armed.

## Rollout

1. **A, v1** for MobileNet, YOLO26n and Llama: no hardware cost; publish, and chart it on the
   allocator card.
2. **B by trace** for MobileNet (seed 3, default effort, batch 1) in the component nightly, with
   the two-bank proof test.
3. **A, v2** (per call). Then compare A with B over a few weeks. If A predicts B, A becomes the
   gated number and B a weekly check. If not, B is the truth, and A's model of "read together"
   needs work.
4. **Performance counters** for B, once 2 shows the events are worth tracking, so it can run
   every night on every design.

The synthetic test stays as it is: a regression tripwire for legality, completeness and
determinism. Its card should say so, so that a flat line reads as expected.
