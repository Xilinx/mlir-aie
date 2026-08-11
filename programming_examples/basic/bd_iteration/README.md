<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# BD Iteration

## The feature

On supported target devices, a buffer descriptor can carry three optional
iteration fields (`iteration_size`, `iteration_stride`, `iteration_current`).
When set, the DMA engine advances the BD's base address by `iteration_stride`
elements after each execution and wraps back to the start after `iteration_size`
executions (`iteration_current` is the starting step, default 0). So execution
`k` of one descriptor accesses
`base + ((iteration_current + k) mod iteration_size) * iteration_stride`.

This gives one BD a regular, per-execution address progression. Without this
functionality, the on-tile path expressed this with an N-deep BD chain
(one descriptor per offset, cycled by the channel).

## The program

A shim tile streams a 256-element buffer into a 256-element MemTile buffer in
four 64-element bites. The receiving side is a single self-chained `aie.dma_bd`
with `iteration_size=4`, `iteration_stride=64`:

```python
Bd(
    buffer=mem_buff,
    length=CHUNK,            # 64 elements per execution
    iteration_size=n_slots,  # wrap after n_slots executions
    iteration_stride=CHUNK,  # advance the base address by CHUNK each execution
    acquires=[Acquire(slot_credit, value=1, greater_equal=True)],
    releases=[Release(fill_count, value=1)],
    next="self",             # re-run to consume the whole stream
)
```

The BD executes four times; iteration advances its base one 64-element slot per
execution, so bite `k` lands at offset `k * 64`. A second MemTile BD reads the
buffer back to the host. The two BDs handshake through two locks: `slot_credit`
(one credit per bite, so the receive BD runs exactly four times) and
`fill_count` (posted per execution; the readback waits for all four before
draining, so it never reads a partial buffer).

The host check is exact against a slot-distinct input (bite `k` is
`(k+1)*1000 + [0..64)`), so a wrong placement is a value mismatch rather than a
coincidental match.

The `--slots` knob overrides `iteration_size` while the host check
stays fixed on the four-distinct-slot reference. `--slots 1` collapses the
cycle to a single slot, i.e., every execution writes offset 0, so the run fails.

## Run

```bash
python3 bd_iteration.py             # expected: verify PASSES
python3 bd_iteration.py --slots 1   # expected: verify FAILS
```
