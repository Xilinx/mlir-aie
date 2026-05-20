# m8 fusion + cascade + shared-L1 plan

## Problem

After Phase 3a-3f vectorization, m8 standalone is 623 ms — unchanged
from scalar despite all 8+ kernels (streamed and non-streamed) being
vec'd. m8 is the chain bottleneck (chain batched steady-state per-sample
≈ 560 ms; chain throughput 1.79 fps; m8 sets the floor).

m8 is **not compute-bound** on the inner kernels. The 623 ms is
bounded by some mix of:
- MemTile weight-stream DMA throughput (cv1, cv2 stream weights;
  inner pairs stream too in stage 5)
- Inter-tile fifo lock + DMA setup overhead (m8 has 8 compute tiles
  in a deep pipeline)
- Per-chunk weight-DMA per-call cost on cv2's chunked path (n_chunks
  small transfers)

We can't easily get per-tile profile data: `parse_trace` is too slow
on 1MB traces (didn't finish in 6+ min CPU), and per the existing
finding ("INSTR_EVENT_0/1 trace cycles understate compute ~200×")
event-delta cycles wouldn't be reliable anyway. So we work by
informed experiment, not by measurement.

## Tools available on AIE2P

1. **Shared L1 ObjectFifo (no via_DMA)** between vertically adjacent
   tiles in the same column. Adjacent tiles (col, row) and (col, row±1)
   share the 32 KB L1 SRAM; data placed there acts as a zero-copy fifo
   with lock-only synchronization. No DMA, no S2MM/MM2S channel cost.
2. **Cascade interconnect** — per-column int32 (or other accumulator)
   bus between adjacent cores. Carries mmul accumulator state directly,
   eliminating the int8 quant + L1 + lock round-trip when one tile's
   output is mathematically another tile's input (chained conv accum
   or partial sums).
3. **Tile fusion** — merge two adjacent stages' compute into one
   worker on one tile. Removes per-stage fifo + lock pair entirely.
   Bounded by L1 budget (weights + scratch must fit).

Mobilenet's `bottleneck/{regular,pipeline,cascade}.py` are three
placements of the same logical compute exploring these axes — worth
borrowing patterns from.

## m8 tile map (stage 5, per `placement.py` + `scripts/m8_stage.py`)

```
col:   4         5         6         7
row 5: cv1       —         —         —
row 4: cv2       cv3       pair1_cv2 —
row 3: —         m_0_split pair1_cv1 —
row 2: —         pair0_cv1 pair0_cv2 —
```

(- = unused; m8 occupies 8 compute tiles. m9 owns col 7. cv2 streams
weights from a memtile; cv1 streams weights from a memtile; pair0_cv1
+ pair0_cv2 + pair1_cv1 + pair1_cv2 use static weights in tile L1.)

**Vertically-adjacent pairs in m8** (candidates for shared-L1 or
cascade, no extra placement work):

| Pair | Logical relationship |
|---|---|
| cv1 (4,5) ↔ cv2 (4,4) | cv1 outputs `top` half + `bot` half; cv2 consumes both halves + m_0_inner output |
| m_0_split (5,3) ↔ pair0_cv1 (5,2) | split outputs `split_a` (goes to pair0_cv1) + `split_b` (goes to cv3) |
| cv3 (5,4) ↔ m_0_split (5,3) | m_0_split's `split_b` output → cv3 input |
| pair0_cv2 (6,2) ↔ pair1_cv1 (6,3) | pair0_cv2's output feeds pair1_cv1 |
| pair1_cv1 (6,3) ↔ pair1_cv2 (6,4) | pair1_cv1's output feeds pair1_cv2 |

**Horizontally-adjacent or non-adjacent** (no shared L1; would need
DMA regardless):

| Pair | Why DMA-only |
|---|---|
| m_0_split (5,3) ↔ pair0_cv2 (6,2) | non-adjacent (diagonal) |
| cv1 (4,5) → m_0_split (5,3) | different column |
| pair1_cv2 (6,4) → cv3 (5,4) | horizontal, no shared L1 |
| pair1_cv2 (6,4) → cv2 (4,4) | non-adjacent |

## Candidate experiments, ranked by ROI

### Experiment A — Shared-L1 for cv1 → cv2 `top` fifo

**Hypothesis:** the `top` fifo from cv1 (4,5) to cv2 (4,4) currently
uses DMA. The cv2 tile consumes it row-by-row alongside the streamed
cv2 weight. Switching `top` to a shared-L1 ObjectFifo (no `via_DMA`)
eliminates one DMA channel pair + memtile staging + per-row setup
between (4,5) and (4,4). Potential saving: 10-50 ms per dispatch
depending on how often the DMA serialized cv1/cv2 work.

**Steps:**
1. In `scripts/m8_stage.py` find the `top_fifo` allocation.
2. Set `via_DMA=False` (or remove the kwarg).
3. Verify the producer + consumer placements are adjacent (already
   are: (4,5) and (4,4)).
4. Build m8 standalone, run, measure.

**Bit-exactness verification:** `make BLOCK=m8 run_ort` must return
0 mismatches. The math doesn't change — only the data path does.

**Fail mitigation:** if the IRON shared-mem lowering hits the
"declared-but-unused via_DMA hangs the runtime" issue (Bug D from
placement.py), revert to `via_DMA=True` and consider whether the
shared-L1 placement is genuinely viable on Strix today.

### Experiment B — Shared-L1 for pair1_cv1 → pair1_cv2 inner pass

**Hypothesis:** pair1 cv1 (6,3) and cv2 (6,4) are vertically adjacent
in column 6. The intermediate fifo between them currently flows via
DMA. Switching to shared-L1 saves DMA per (oc_chunk, row) call.

**Steps:** same as Experiment A on the pair1 inner fifo.

**Estimated saving:** similar magnitude to A, applied to pair1.
Possibly stackable with A.

### Experiment C — Cascade pair0_cv1 → pair0_cv2 partial-sum

**Hypothesis:** pair0_cv2_skip's input is pair0_cv1's SiLU output.
The skip-add does `y * y_mult + cv2silu * cv2_mult` then SRS — a
true reduction. We could:
- Have pair0_cv1 leave the accumulator at int32 (skip SiLU + clamp)
- Cascade the int32 acc to pair0_cv2
- pair0_cv2 does its mmul, then ADDS the cascaded acc, applies the
  scaled skip, and emits int8

This eliminates pair0_cv1's SiLU LUT lookup + int8 store and pair0_cv2's
int8 load + int32 re-promote. Saves ~2× the SiLU + requant work for
the pair0 chain.

**Caveat:** changes both kernel signatures (pair0_cv1 must emit
accumulator instead of int8; pair0_cv2 must accept cascaded acc).
Needs IRON's cascade primitive (already used in mobilenet's
`bottleneck/cascade.py`).

**Estimated saving:** 30-80 ms if cv2 SRS+LUT was meaningful per-row.

### Experiment D — Fuse cv3 + m_0_split into one worker

**Hypothesis:** m_0_split (5,3) outputs `split_a` (→ pair0_cv1) and
`split_b` (→ cv3 (5,4)). The `split_b` → cv3 hop is currently DMA.
Fusing them: one worker on (5,3) computes split, immediately runs
cv3's 1x1 inline. Output: cv3's result + split_a.

**Caveat:** total L1 budget on (5,3) becomes split-weights +
cv3-weights + intermediate scratch + bias + LUT. Need to check it
fits. m8 cv1 already at ~57 KB on (4,5) per the m8 fix — be careful.

### Experiment E — Reduce cv2 weight-chunk granularity

**Hypothesis:** cv2 streams weights from memtile in `n_chunks`
chunks. Each chunk = one MemTile→L1 DMA + one kernel call. Per-chunk
overhead (BD setup + lock cycles) accumulates. If memtile L2 budget
allows, doubling chunk size = halving the per-call overhead.

**Steps:**
1. Find m8 cv2's `n_chunks` (likely `N_PAIR_CHUNKS` in m8_stage.py).
2. Try halving it. Verify L1 still fits.
3. Build + bench.

**Estimated saving:** depends on per-call DMA setup; could be
5-30 ms.

## Execution order

Pragmatic — cheapest first, each verifying bit-exact + measuring before
moving on:

1. **Experiment E** (n_chunks tweak) — 30 min, code-trivial, immediate measure
2. **Experiment A** (cv1↔cv2 top shared-L1) — 1-2 hr, one fifo flag flip + careful verify
3. **Experiment B** (pair1 inner shared-L1) — 1 hr, identical to A pattern
4. **Experiment C** (cascade pair0) — 4-8 hr, two kernel rewrites + IRON cascade wiring
5. **Experiment D** (fuse cv3 + m_0_split) — 4-8 hr, kernel rewrite + L1 budget check

Stop early if any one of these moves m8 standalone by >50 ms — that's
strong evidence about which bottleneck class is real, and we focus
follow-up there. Stop if NONE of A-E moves m8 by >20 ms — bottleneck
is likely something deeper (memtile DMA arbitration?), needs better
profiling tooling.

## Bit-exactness invariants (every experiment)

1. `make BLOCK=m8 run_ort` — 0 mismatches
2. `make chain run_chain` — 0 mismatches end-to-end
3. Per-block bench (`/tmp/bench_per_block.py --block m8`) — record
   new median NPU compute time

## Expected outcome envelope

- **If only Experiment E helps**: m8 was per-chunk-DMA-bound. Tune
  chunk size for best fps. Chain settles ~480 ms/sample → ~2.1 fps.
- **If A+B help**: shared-L1 saves DMA overhead. Chain settles
  ~450 ms/sample → ~2.2 fps.
- **If C/D help**: fusion saves per-stage overhead. Could drop m8 to
  ~400 ms, chain to ~3 fps.
- **If NONE help meaningfully**: bottleneck is memtile-DMA arbitration
  (multiple parallel weight streams + activation paths competing for
  memtile DMA channels). Fix would be redistributing streams across
  different memtiles, which is Phase 5 (placement rework) territory.
