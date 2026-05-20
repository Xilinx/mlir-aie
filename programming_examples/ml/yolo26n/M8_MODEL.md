# m8 static-MLIR analysis and redesign

## Why this exists

After full kernel vectorization (Phase 3a-3f), m8 standalone is 623 ms.
Vectorizing every inner kernel moved the chain throughput zero, so the
bottleneck is structural, not compute. We can't easily trace m8 on HW
(`parse_trace` doesn't scale on 1 MB traces; cycle-event deltas are
unreliable per prior findings). So we **model m8 from the static MLIR**
instead — what tiles, what fifos, what depths, what compute load.

## What the static MLIR shows

### Tile layout (8 compute tiles, 4 memtiles)

```
col:     4              5                6              7
row 5:   cv1 (4,5)      —                —              —
row 4:   cv2 (4,4)      cv3 (5,4)        pair1_cv2 (6,4) —
row 3:   —              m_0_split (5,3)  pair1_cv1 (6,3) —
row 2:   —              pair0_cv1 (5,2)  pair0_cv2 (6,2) —
row 1:   mem (4,1)      mem (5,1)        mem (6,1)      —    + mem (3,1)
row 0:   —              —                —              —    + shim (0,0), (7,0)
```

Memtiles: 4 active, each feeding one compute tile's streamed weights:
| Memtile | Streams to | Carrying |
|---|---|---|
| (3,1) | cv2 (4,4) | cv2_concat3 weight chunks |
| (4,1) | cv1 (4,5) | cv1_split_streamed weight chunks |
| (5,1) | pair0_cv1 (5,2) | pair0_cv1 weight chunks |
| (6,1) | pair0_cv2 (6,2) | pair0_cv2 weight chunks |

pair1_cv1 and pair1_cv2 use STATIC weights on their compute tiles (36.5 KB
each on (6,3) and (6,4)).

### Per-tile L1 utilization (sum of buffers + fifo elements)

| Tile | bufs KB | fifos KB | total KB | headroom |
|---|---:|---:|---:|---|
| (4,5) cv1 | 9.2 | 48.0 | **57.2** | 6.8 |
| (4,4) cv2 | 13.2 | 44.0 | **57.2** | 6.8 |
| (5,2) pair0_cv1 | 18.5 | 23.0 | 41.5 | 22.5 |
| (5,3) m_0_split | 17.0 | 27.0 | 44.0 | 20.0 |
| (5,4) cv3 | 16.8 | 22.0 | 38.8 | 25.2 |
| (6,2) pair0_cv2 | 18.5 | 24.0 | 42.5 | 21.5 |
| (6,3) pair1_cv1 | 36.5 | 24.0 | **60.5** | 3.5 |
| (6,4) pair1_cv2 | 36.5 | 22.0 | **58.5** | 5.5 |

Tiles at the L1 ceiling: (4,5), (4,4), (6,3), (6,4). The streamed
tiles ((5,2), (5,3), (5,4), (6,2)) have headroom because their
weights are streamed not static.

### ObjectFifo connectivity

14 fifos between compute tiles. All but one (`m8_cv3_to_cv2`) use
shared L1 (no `via_DMA`) — that's a known win already. The cv3 → cv2
hop is horizontal (5,4)→(4,4) so it has no choice but DMA.

Notable deep fifos (consume L1 + indicate large temporal misalignment):
| Fifo | Producer → Consumer | Depth | Element B | Total KB |
|---|---|---:|---:|---:|
| of5 | m_0_split → cv3 | 16 | 1024 | 16 |
| of7 | pair0_cv1 → pair0_cv2 | 16 | 1024 | 16 |
| of10 | pair1_cv1 → pair1_cv2 | 16 | 1024 | 16 |

Each "depth=16" fifo holds the producer's full row-output until the
consumer is ready. This is the symptom of a pipeline where one stage
runs far ahead of a downstream stage.

### Compute budget

Per dispatch (one 16×16×256 input sample):
| Kernel | MACs/dispatch |
|---|---:|
| cv1 (1×1 256→128) | 16×16×128×256 = 8.4 M |
| cv2 (1×1 192→256) | 16×16×256×192 = 12.6 M |
| cv3 (1×1 64→64) | 16×16×64×64 = 1.0 M |
| m_0_split (1×1 64→32+32) | 16×16×64×2 = 1.0 M |
| pair0/pair1 cv1+cv2 (3×3 32→16/16→32) | 16×16×16×9×32 × 4 = 1.2 M |
| **Total m8 compute** | **~24 M MACs** |

AIE2P `mmul<4,8,8,int8,int8>` peak: 32 MACs/cycle, 8 cycles/call → **32 MAC/cycle/tile**.
On 1 tile: 24 M / 32 = **750 K cycles = 0.6 ms** at peak utilization.
On 8 tiles parallel (ideal): 0.6 / 8 = **0.075 ms**.

**Actual m8: 623 ms.** Utilization = **0.6 / 623 ≈ 0.1%.**

## Diagnosis

**m8 is bound by pipeline traversal, not compute.** The 8-tile pipeline
needs to fill (each tile gets its first row of data) then drain (last
tile emits its last row) for each dispatch. With deep fifos (depth 16
on three fifos) the temporal misalignment between producer and
consumer stages is huge — m_0_split, pair0_cv1, pair1_cv1 all run
**16 rows ahead** of their consumers cv3 and pair0_cv2/pair1_cv2.

Per-row time = 623 / 16 = **39 ms/row**. At 32 MAC/cycle peak this
should be sub-millisecond. The remaining 38+ ms/row is lock
acquire/release latency, memtile DMA setup per chunk, and back-pressure
serialization across 8 pipeline stages.

The standard trick of "process N samples concurrently to fill the
pipeline" — i.e. batching — would amortize the fill/drain. We see
this empirically: batched N=15 chain steady-state is 560 ms/sample
vs 1990 ms single-frame chain. m8's contribution shrinks in batched
mode but is still the floor.

## Why "just vectorize more" can't fix this

Even if we got every kernel call to **0 cycles**, m8 would still
take ~half of its current time — the pipeline structure forces
sequential row propagation through tiles, with fifo-lock waits at
every boundary. The per-tile compute is already a tiny fraction of
per-tile wall time.

The fix has to be **structural**: fewer pipeline stages, less back-
pressure, or fewer fifo-lock crossings per row.

## Redesign options

### Option 1 — Fuse pair0_cv1+cv2 onto one tile, same for pair1 (8 → 6 tiles)

Each "pair" today uses 2 tiles (cv1 → cv2_skip). Both 3×3 stride-1
convs operate on the same row data; cv2's input is cv1's output. Fuse
into one worker: per row, do cv1's compute into a local scratch,
immediately do cv2_skip's compute consuming the scratch, emit the
final SiLU+skip output. Removes the cv1↔cv2 fifo entirely.

**L1 budget on the fused tile** (rough): cv1 weights (static 36.5 KB
for pair1) + cv2 weights (similar) + scratch (1 row of intermediate)
+ input row fifos + output row fifo. ~90 KB — **EXCEEDS 64 KB**.
Probably won't fit static-weight-for-both. Workaround: stream one of
the two weight sets (memtile DMA per row), keep the other static.
~40 + 8 + 16 = 64 KB, tight but feasible.

**Expected gain**: each fused pair saves 1 pipeline stage. 8 → 6
tiles ≈ 25% fewer stages ≈ 25% less fill/drain ≈ **~470 ms?** (rough)

### Option 2 — Fuse cv3 + m_0_split (8 → 7 tiles)

m_0_split outputs `split_a` (→ pair0_cv1) and `split_b` (→ cv3). cv3
consumes `split_b` + pair1_cv2's output. If m_0_split and cv3 are
fused on (5,3): per row, do split's compute, immediately stream
`split_b` into cv3's input scratch (no fifo). cv3 still needs to
wait for pair1_cv2's output via the existing of11 fifo.

**L1 budget**: m_0_split weights + cv3 weights + scratch. m_0_split
weights are small (2x 2KB chunks). cv3 weights ~4KB. Add scratch.
Fits easily in (5,3)'s current 44 KB / 64 KB.

**Expected gain**: removes of5 (depth=16 × 1024B = **16 KB freed**)
+ one pipeline stage. Maybe 50-100 ms.

### Option 3 — Aggressive: m8 on 4 tiles instead of 8 (half pipeline depth)

Per the user's "no optimization is too hard if it gets us to 60 fps":
fuse pairs (option 1), fuse cv3+m_0_split (option 2), and additionally
fuse cv1 + m_0_split (both 1×1, consume same input + slice it). Final
m8: 4 tiles total — cv1+split combined, pair0 fused, pair1 fused,
cv2+cv3 fused.

**L1 budget per fused tile**:
- cv1+m_0_split: cv1 streamed wts + split static wts (4 KB) + scratch + I/O. ~30 KB OK.
- pair0/pair1 fused: ~64 KB tight as analyzed in option 1.
- cv2+cv3 fused: cv2 streamed wts + cv3 wts (4 KB) + I/O fifos. ~30 KB OK.

**Expected gain**: 4-stage pipeline = ~half the fill/drain of 8-stage =
**~300 ms?** Plus removes ~3 fifo-lock crossings per row.

### Option 4 — Radical: m8 on a single tile (no pipelining)

Run all m8 compute on one tile, all weights via memtile streaming.
The kernel call becomes monolithic: input row in → output row out
with all internal computation hidden. No inter-tile fifo crossings
at all.

**L1 budget**: need all weights' streamed chunks at once + accumulators
+ scratch. Roughly 64 KB likely fits if every weight is small-chunk
streamed.

**Compute**: 24M MACs on 1 tile at 32 MAC/cycle = 0.6 ms compute. Plus
per-row memtile DMA setup for the streamed weights. Even with 100×
overhead, that's ~60 ms = **>10× faster than current 623 ms**.

**Risk**: programming complexity is HIGH (single kernel doing 8
logical stages with internal state); kernel size may exceed program
memory budget. But if it fits, it's the killer fix.

### Option 5 — Multi-sample batched inside m8

The current design processes ONE sample at a time through the m8
pipeline. The batched N=15 chain dispatch is sequential at the chain
level — each sample fully flows through before the next starts (or
at most they overlap by one pipeline depth).

If we reorganize so that each tile in m8 processes ROWS FROM
MULTIPLE SAMPLES IN AN INTERLEAVED FASHION, the per-sample fill/drain
amortizes across N samples. Effectively the same as deeper pipelining
but explicit.

**Risk**: significant runtime + IRON rewrite; pipelining over samples
needs careful fifo depth management.

## Recommended attack order

1. **Option 2 first** (fuse cv3 + m_0_split on (5,3)): cheapest. Bounded
   L1 risk. Estimated 50-100 ms gain on m8. Validates the fusion
   approach before deeper investment.
2. **Option 1 next** (pair fusion): if option 2 confirms gain comes
   from pipeline-depth reduction, pair fusion doubles the gain (two
   stages removed instead of one). L1 budget tight; may need
   one-streamed-one-static weight strategy per pair.
3. **Option 3 third** (4-tile m8): combines 1+2 plus cv1+split fusion.
4. **Option 4 (single tile)** only if 1-3 don't get us to ≤100 ms m8.
   Big rewrite but biggest potential gain.

Bit-exactness gate at every step (`make BLOCK=m8 run_ort` and
`make chain run_chain` must pass).

## What this means for chain throughput

If m8 drops 623 → ~200 ms via fusion:
- New chain bottleneck: m10 (14 ms, vec) → likely **m0 (56 ms)** or
  the chain steady-state floor from fill/drain on the rest of the chain.
- Estimated chain batched: 200 ms / sample = **5 fps**.

If m8 drops to ~50 ms (option 3 or 4):
- New bottleneck: probably m6 (280 ms scalar non-compute) — would need
  m6 to get the same structural treatment.
- After both m6 and m8 fixed: chain bound by m3/m5 (124-127 ms vec
  chunked) or m0 (56 ms).
- Estimated chain batched: ~100 ms / sample = **10 fps**.

Path to 60 fps from there requires Phases 5-8 of VECTORIZATION_PLAN.md:
- Per-block tile parallelism (split a hot block across more tiles)
- Cross-block fusion (m_N's last stage fuses with m_{N+1}'s first)
- Batched pipelining tuning

So m8 redesign is necessary but not sufficient. Cumulative 60 fps
remains the multi-week effort estimated in the original plan.
