# m8 megakernel: design plan

## Goal

Collapse m8's 8-tile pipeline into a single megakernel running on one
tile per sample, then (later) replicate the megakernel tile K-way for
K-way batch throughput. Per-sample wall-time becomes
**compute + weight-DMA + I/O-DMA**, with NO inter-tile fifo crossings
and NO fill/drain overhead.

## L1 budget (the constraint)

AIE2P compute tile = **64 KB L1 SRAM** total for program memory +
data buffers + fifo slots. Subtract ~8 KB for program/stack overhead
=> ~56 KB usable for data.

### Weight inventory (m8 total)

| Layer | Shape | Bytes |
|---|---|---:|
| cv1 (1×1) | 256 × 256 | 65 536 |
| cv2 (1×1) | 192 × 256 (3×64×256) | 49 152 |
| m_0_split cv1 (1×1) | 128 × 64 | 8 192 |
| m_0_split cv2 (1×1) | 128 × 64 | 8 192 |
| m_0_cv3 (1×1) | 128 × 128 | 16 384 |
| pair0_cv1 (3×3) | 64 × 64 × 9 | 36 864 |
| pair0_cv2 (3×3) | 64 × 64 × 9 | 36 864 |
| pair1_cv1 (3×3) | 64 × 64 × 9 | 36 864 |
| pair1_cv2 (3×3) | 64 × 64 × 9 | 36 864 |
| **Total weights** | | **~295 KB** |

**Static-on-tile is impossible** (295 >> 56). All large weight sets must
be **streamed from memtile** per sample.

### Intermediate activation scratch (per-sample, fully materialized)

| Tensor | Shape | Bytes |
|---|---|---:|
| cv1 top  | 16 × 16 × 128 | 4 096 |
| cv1 bot  | 16 × 16 × 128 | 4 096 |
| split_a  | 16 × 16 × 64  | 2 048 |
| split_b  | 16 × 16 × 64  | 2 048 |
| pair0_out| 16 × 16 × 64  | 2 048 |
| pair1_out| 16 × 16 × 64  | 2 048 |
| cv3_out  | 16 × 16 × 128 | 4 096 |
| **Total scratch** | | **~20 KB** |

### I/O buffer budget

- act_in: 1 row × 256 ch × 16 rows = 4 KB (or batch one row at a time)
- act_out: 1 row × 256 ch = 256 B per row (4 KB if buffered)
- Weight stream chunks: 4 KB × 2 concurrent streams = 8 KB
- LUTs: 8 × 256 B = 2 KB
- Biases: 9 × small = <2 KB

**Estimated total L1 occupancy: ~20 (scratch) + 8 (streams) + 8 (I/O) + 4 (LUT+bias) = 40 KB. Fits.**

## Weight streaming strategy

Tile has only **2 S2MM channels** per direction, so at most 2 active
weight streams concurrently. Approach: **serialize** the per-sample
compute into sub-passes, each pass loads its weights then computes,
then releases:

```
PASS 1 (cv1 + m_0_split):
  - stream cv1 wts (64 KB), compute cv1 for all 16 rows -> top/bot scratch
  - stream m_0_split wts (16 KB), compute split for all 16 rows -> split_a/b scratch
PASS 2 (pair0 + pair1):
  - stream pair0_cv1 wts (36 KB), compute pair0_cv1 -> pair0_mid scratch
  - stream pair0_cv2 wts (36 KB), compute pair0_cv2 -> pair0_out scratch
  - stream pair1_cv1 wts (36 KB), compute pair1_cv1 -> pair1_mid scratch
  - stream pair1_cv2 wts (36 KB), compute pair1_cv2 -> pair1_out scratch
PASS 3 (cv3 + cv2):
  - stream cv3 wts (16 KB), compute cv3 over split_b + pair1_out -> cv3_out scratch
  - stream cv2 wts (48 KB), compute cv2 over top + bot + cv3_out -> act_out
```

Sub-passes serialize compute (no overlap with weight DMA *within* a
sample), but that's fine — the whole thing still runs in
**~5-10 ms** estimated, vs current 623 ms.

### Weight DMA cost estimate

Total weight bytes per sample = 295 KB.
Memtile DMA bandwidth ~ 500 MB/s effective on AIE2P.
295 KB / 500 MB/s = **~590 us per sample for weight DMA alone**.

### Compute cost estimate

24M MACs / 32 MAC/cycle / 1 GHz = **~750 us per sample for compute**.

### Total estimated per-sample time

~590 us (weight DMA) + ~750 us (compute) + ~100 us (I/O DMA) +
overhead = **~1.5-3 ms per sample on 1 tile**.

**vs current 623 ms = 200-400x speedup for m8.**

If this lands, m8 is no longer the chain bottleneck — m6 (280 ms) and
m0 (56 ms) take over and need similar treatment.

## Replication (Phase 2, if needed)

If 1-tile megakernel hits the estimate, single-frame latency is already
near peak. For batched throughput, replicate to K tiles, each handling
one sample, weights memtile-broadcast.

L1 budget per tile = unchanged (~40 KB).
Memtile broadcast: 1 weight DMA serves all K tiles in parallel
=> per-K-sample DMA cost unchanged (~590 us for K samples), but
per-sample throughput = K / 1.5ms = **~K × 666 sample/sec**.

For K=4 tiles: 2.6 K samples/sec = batched fps absurdly high. Chain
would be limited by other blocks long before this.

## Bit-exactness gate

Sub-passes compute the same math as the unfused versions, just on
different data layouts. Bit-exact required at every step:
1. `make BLOCK=m8 run_ort` -> 0 mismatches
2. `make chain run_chain` -> 0 mismatches

## Open risks

1. **Program memory budget**: a single kernel function doing all 9
   sub-ops may exceed the tile's I-cache. Mitigation: factor each
   sub-op into a callable helper; verify .o text-size < some threshold
   (~16 KB program memory typical).

2. **Streamed-weight kernel API**: existing IRON `StaticWeightStream`
   helper handles one weight set per call. The megakernel needs
   multiple sequential streamed-weight inputs. Either:
   - (a) wrap each sub-pass in its own kernel call, each with one
     StaticWeightStream input. The IRON worker body runs them in
     sequence, passing scratch through Buffers. This is the cleanest
     wiring and the chosen approach.
   - (b) make a single kernel call with 6 streamed-weight inputs +
     6 acquire/release loops internal to the kernel. Probably hits
     IRON-side restrictions on multi-stream kernels.

3. **Memtile contention**: streaming 6 weight sets through one tile's
   2 S2MM channels means 3 sequential transfers per channel per sample.
   Memtile can handle it but BD count may grow.

## Sub-task order (next sessions)

1. **L1 audit**: run the m8_model.py-style analysis with the proposed
   megakernel-tile budget; confirm under 56 KB usable.
2. **Kernel scaffold**: write `kernels/yolo_m8_megakernel_passN_vec.cc`
   files - one per sub-pass (3 files total). Each takes 1 streamed
   weight input + scratch buffers + LUTs + biases.
3. **IRON wiring**: rewrite `scripts/m8_stage.py` to use a single
   megakernel tile (likely (5,3) or (5,4)) with 3 worker calls in
   sequence, scratch buffers shared between them, and 6
   StaticWeightStreams (3 active per pass).
4. **Single-tile validation**: build standalone, run bit-exact, measure.
5. **Spatial replication**: only if single-tile is not already fast
   enough — likely won't be needed for m8 itself.
