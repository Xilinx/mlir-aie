# Path to 60 fps on Strix Point

## Target and headroom

**Goal:** 60 fps end-to-end on NPU2 Strix Point for the yolo26n-cls INT8
chain, batched (steady-state per-sample latency ≤ 16.7 ms).

**Today:** 0.57 fps batched (N=15 dispatch) / 0.33 fps single-frame.
Chain bit-exact NPU == ORT for all blocks. Bottleneck is m3 (1,675 ms
standalone, the slowest single block). Every kernel is scalar AIE2P C.

**Theoretical ceiling on Strix:** 32 tiles × 32 INT8 MACs/cycle × 1.25 GHz =
1.28 TOPS sustainable on `aie::mmul<...,int8,int8>`. YOLO26n compute load
per sample is ~1.03 GMACs (sum of `tile_layout.py`'s per-block GMAC ratings
÷ batch=4). Floor: **0.8 ms/sample = 1,250 fps theoretical**. At 50%
realistic efficiency: **~625 fps achievable**. 60 fps = 16.7 ms/sample is
**~25× below** the ceiling — comfortably reachable with kernel-level
vectorization + supporting design changes.

**The gap is not subtle:** 0.57 fps to 60 fps is **~105×**. Current
underperformance is because **no kernel uses vector intrinsics at all** —
we're running scalar C loops on hardware whose entire reason for existing
is `aie::mmul`. There's no incremental tuning that gets us to 60 fps;
this requires a real rewrite of the compute pipeline.

## Levers, multiplicative

| # | Lever | Mechanism | Realistic gain |
|---|---|---|---|
| 1 | **Vectorize hot kernels** (conv2dk3 family: m1/m3/m5/m7 = 52% of standalone compute) | `aie::mmul<4, 8, 8, int8, int8>` on ic-tile-major activations + chess pipelining hints + 3-vector preload pattern (`shift()` / `extract_v32int8`) | 15-25× per kernel |
| 2 | **Activation layout: ic-tile-major end-to-end** | `act[ic_tile][row][col][ic_inner]` instead of HWC. Required for (1) to hit full speed; no permute in tree converts HWC vector loads into mmul-shaped operands cleanly. | Enabler for (1); also unlocks (3), (5) |
| 3 | **Vectorize c3k2 + m9 PSA + m10 head kernels** | Same mmul-based pattern for the remaining 48% of compute. `aie_kernels/aie2p/conv2dk1_i8.cc` (already vectorized) + `aie_kernels/aie2p/mm.cc` (gemm) as templates for 1×1 + GEMM. | 10-20× per kernel |
| 4 | **Per-block tile parallelism** | Several blocks use 1 tile today (m1, m3, m5, m7, m10). Splitting work across 2-4 tiles per block scales linearly with no per-tile complexity change. | 2-4× per such block where free tiles exist (requires (6) to free tiles) |
| 5 | **Memtile-DMA-based transposes** | Layout conversions (HWC ↔ ic-tile-major) done by configuring multi-dim BD strides on memtile DMA, zero compute-tile cycles | Saves the 10-20% otherwise eaten by compute-tile transpose workers |
| 6 | **Block fusion** | Adjacent blocks that fit on shared/adjacent tiles fuse into a single worker (single fifo lock per row instead of per-block lock pairs) | 1.5-2× on fusion-eligible block pairs |
| 7 | **Cascade flows + shared memory** | AIE cascade interconnect carries int32 mmul accumulators between vertically-adjacent tiles in the same column, bypassing L1 + lock + DMA + int8 requant round-trip. Vertically-adjacent compute tiles also share 32 KB L1 directly — zero-copy fifos between them. | 1.5-3× on cascade- or shared-mem-eligible chains. **First-class lever, applied from Phase 1 onward, not deferred.** |
| 8 | **Batched pipelining tuning** | Current batched gain is only 1.7× over single-frame (0.33→0.57 fps), indicating per-sample fill/drain dominates. Deeper fifos + multi-task BDs + larger N | 1.5-2× additional |

**Multiplicative envelope** of (1) × (3) × (4) × (8) ≈ 15 × 15 × 2 × 2 = 900×.
Realistic capture: 100-200×. That's the 60-120 fps range.

The plan below is a roadmap; each phase ships independently with bit-exact
verification gates.

## Realistic timeline

Roughly **6-10 weeks of focused engineering**, with continuous bit-exact
verification:

| Phase | Goal | Duration |
|---|---|---|
| 0 | Spike: memtile-DMA-based HWC ↔ ic-tile-major transpose | 1-2 days |
| 1 | Pilot: m3 (chunked conv2dk3) vectorized + ic-tile-major + bit-exact | 3-5 days |
| 2 | Extend kernel family: m1, m5, m7 vectorized + bit-exact | 2-3 days |
| 3 | c3k2 inner + outer kernels: m2, m4, m6, m8 vectorized | 5-7 days |
| 4 | m10 head: GAP + GEMM + softmax vectorized | 2-3 days |
| 5 | Per-block tile reassignment + placement rework | 3-5 days |
| 6 | Block fusion + cascade flows where applicable | 5-10 days |
| 7 | Batched pipelining tuning | 2-3 days |
| 8 | Debug, regression sweep, integration polish | ~30% overhead on phases 1-7 |

Per-phase deliverables include: code, bit-exact test (`make run_ort` and
`make run_chain`), per-block bench measurement, accuracy bench every
~3 phases (cheap sanity check that accuracy is preserved end-to-end).

## Phase 0 — Memtile-DMA transpose spike

**Question:** can a single memtile DMA with multi-dim stride configuration
convert HWC → ic-tile-major in one shot, no compute-tile cycles?

If yes, every block migration in Phases 1-4 becomes structurally cheap —
add a transpose-in / transpose-out via memtile BD config, no new workers
needed. Tile budget unaffected.

If no, we fall back to compute-tile transpose workers, which cost cycles
AND tiles (chain is at 32/32 today — need to free tiles first via
fusion).

**Test design:** small standalone IRON example that takes a `(H, W, C)`
HWC int8 buffer from DRAM, configures a memtile BD chain with strides
`[H*W*8, W*8, 8, 1]` and sizes `[C/8, H, W, 8]` to emit ic-tile-major
into another DRAM buffer. Verify the output bytes match a Python
reference reshape.

**Pass criterion:** byte-exact transpose with measurable DMA throughput
(≥ 5 GB/s sustained, well above the 100 MB/s we actually need per block).

**Fail mitigation:** compute-tile transpose worker as fallback. Costs
~10-15% per migrated block on per-row time + 1 tile per transpose
direction. Phases 1-4 still proceed, but Phase 5 (tile reassignment)
becomes higher priority to free tiles.

This spike happens in this session — it gates the whole strategy.

## Cascade + shared memory as a first-class design lever

Strix AIE2P offers two zero-overhead inter-tile communication paths
between vertically adjacent compute tiles in the same column:

1. **Shared L1 memory**: tiles (col, row) and (col, row±1) share the
   32 KB L1 SRAM in that column-pair. A buffer placed in shared L1 acts
   as a zero-copy fifo between adjacent tiles — no DMA, no S2MM/MM2S
   channel cost, lock-only synchronization. IRON's `ObjectFifo` already
   uses shared L1 when the producer + consumer placements are adjacent;
   the design's existing per-block builders rely on this where possible.

2. **Cascade interconnect**: a per-column int32 (or other accumulator
   type) bus carrying mmul accumulator state directly between vertically
   adjacent tile cores. Eliminates the entire int8 requant + DMA + L1
   buffer + lock roundtrip when one tile's output is mathematically
   another tile's input (e.g., the second half of a convolution
   accumulator, or chained 1×1 + 3×3 partial sums).

These should be **applied from Phase 1**, not deferred to a separate
fusion phase:

- **Phase 1 (m3 standalone)**: place the HWC → ic-tile-major transpose
  worker (compute-tile fallback if Phase 0 memtile spike fails)
  vertically adjacent to the conv worker. Communicate via shared L1
  ObjectFifo, not DMA.
- **Phase 3 (c3k2 inner pipeline)**: the c3k2_small `cv1_split → m0_cv1
  → m0_cv2_skip → cv2_concat3` chain is 4 sequential conv stages. Place
  them in a single column with cv1_split at the bottom, cv2_concat3 at
  the top. Use cascade flows for cv1_split → m0_cv1 (their accumulators
  share the int32 representation pre-SRS). Same for c3k2_heavy.
- **Phase 6 (cross-block fusion)**: extend the same principle to cross
  the m_N → m_{N+1} boundary. E.g., m1's output ic-tile-major fifo to
  m2/cv1_split: place m1 and m2/cv1_split in the same column, share L1.

Mobilenet's `bottleneck/{regular,pipeline,cascade}.py` are three
placements of the same logical compute exploring these axes — worth
borrowing patterns from.

## Phase 1 — Pilot: m3 vectorized

m3 is the chain bottleneck (1,675 ms standalone). Vectorizing m3 alone
moves the chain to m5-bound (~1,663 ms) — small win. But m3 also exercises
every pattern we need for Phase 2: chunked weight streaming,
OIYXI8O8 weights, stride-2, bias-as-accumulator-init, banker SRS, SiLU
LUT.

**Tasks:**
1. Rewrite `yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked.cc` for
   ic-tile-major input layout. Template-specialized for m3's shape
   (`in_c=64, out_c=64, in_w=128, stride=2`).
2. Inner loop pattern (port of mlir-air `14_conv2d_i8_extern_vec`):
   - 3-vector preload of 192 bytes covering 3 input rows × N pixels at
     one ic_tile
   - `mmul<4, 8, 8, int8, int8>` with `shift(A0, A1, byte_offset)` +
     `extract_v32int8` for kx walking
   - Stride-2 modification: byte offsets become `16` (2 pixels) instead
     of `8` (1 pixel); output write stride doubles
   - `chess_prepare_for_pipelining` + `chess_loop_range(min, max)` on
     the inner (kx, ic_tile, ky) loop
   - `__restrict` on all pointers
3. Bias + SRS + clamp + LUT epilogue: load 32 int32 bias into the
   accumulator via `from_vector` initialization, do `acc.to_vector<int8>(rs)`
   for SRS + clamp in one op, then `aie::lookup` (or scalar fallback) for
   SiLU.
4. Memtile-DMA transpose (Phase 0 output) wired into the m3 builder for
   HWC ↔ ic-tile-major at block boundaries.
5. **Bit-exact verify**: `make BLOCK=m3 run_ort` → 0 mismatches.
6. **Per-block bench**: target 150-300 ms (10× over scalar 1,675 ms).
7. **Chain bit-exact**: `make chain run_chain` still passes 0 mismatches.

## Phase 2 — Extend kernel family

m1 (non-chunked), m5, m7 (chunked) reuse the Phase 1 kernel. The
`-DKERNEL_SUFFIX=_mN` mangling already handles per-block compilation;
each block just needs its transpose-in/out memtile-DMA wiring updated
and its per-block bench re-measured.

After Phase 2: chain steady-state should be m2-bound (~1,094 ms / 1 fps)
or m6-bound (~628 ms / 1.6 fps) depending on whether m2 was still HWC
internally. Real progress emerges here.

## Phase 3 — c3k2 (m2, m4, m6, m8)

Four distinct kernel patterns:
- `cv1_split` (1×1 conv, channel chunking into halves): port from
  `aie_kernels/aie2p/conv2dk1_i8.cc` directly; thin epilogue wrapper.
- `m0_cv1` (3×3 stride-1 inner conv): reuse Phase 1 kernel with
  stride=1 specialization.
- `m0_cv2_skip` (3×3 stride-1 + skip-add): Phase 1 kernel with a
  vectorized skip-add added to the accumulator before SRS.
- `cv2_concat3` / `cv3_concat2` (1×1 + 3-input channel concat):
  conv2dk1 vec + a multi-input gather. Concat is just a different
  output write pattern in ic-tile-major layout — should be cheap.

m6 + m8 add the c3k2_heavy variants (split inner pairs, streamed
weights). The streamed-weights path requires extra care because the
weight buffer doesn't fully live in tile L1 — load patterns change.

Per-block bit-exact verification at each.

## Phase 4 — m10 head

- `conv2dk1_silu_xy_pool`: 1×1 conv folded with spatial GAP. Vectorize
  the 1×1 via conv2dk1_i8 pattern, then `aie::reduce_add` over the
  spatial dim.
- `linear_gemm`: int8 GEMM. Port from `aie_kernels/aie2p/mm.cc`.
- `softmax`: 2 elements only (binary classifier). Currently 748 ms which
  is huge for 2 elements — the softmax is probably running a generic
  N-class path. Specialize for N=2 or vectorize the LUT-driven exp.

m9 PSA is excluded from the vectorization roadmap (150 ms standalone,
< 2% of chain time). It can stay scalar without harming the 60 fps
target. The PSA in-kernel batch loop noted in the README is a separate
exploration.

## Phase 5 — Per-block tile parallelism

After Phase 4, the per-block compute times are roughly all in the
100-200 ms range. The chain throughput is now bounded by the worst
block (likely a c3k2 block due to its multi-stage 1×1+3×3+1×1+concat).
Splitting that block across more tiles (e.g., row-parallel: rows 0-7
on tile A, rows 8-15 on tile B) gives proportional speedup if tile
budget allows.

Tile budget today: 32/32. Phases 1-4 should free 2-4 tiles via fusion
(m10 already fused; PSA might fuse further if we don't grow it). Use
the freed tiles to widen the bottleneck blocks.

Goal: bring per-block max down to ~50 ms → ~20 fps chain.

## Phase 6 — Block fusion + cascade

Adjacent blocks where output of N == input of N+1 in shape and
distribution can fuse:
- A single worker computes m_N's last stage AND m_{N+1}'s first stage
  in one row pass, with the intermediate activation kept in tile L1
  (no fifo lock, no DMA, no memtile round-trip).
- Where compute is partial-sum-accumulating (conv → conv-with-add),
  cascade flows can hand the int32 accumulator directly to the next
  tile, skipping the int8 quantization round-trip.

Candidate fusion pairs:
- m1 (single conv) + m2/cv1_split: m1 output IS m2 cv1 input; fuse.
- m3 + m4/cv1_split: same.
- m5 + m6/cv1_split: same.
- m7 + m8/cv1_split: same.
- m2/cv2_concat3 + m3: m2 output is m3 input; fuse final 1×1 with m3 stride-2.

Each fusion ~1.5-2× per fused pair. Several fusions stack
multiplicatively to the extent they're orthogonal.

Goal: another ~2× → 40-50 fps.

## Phase 7 — Batched pipelining tuning

Current ratio (single-frame 3.07s vs batched-per-sample 1.77s) shows
1.74× from batching — pure pipeline fill/drain. With deeper fifos +
proper multi-task BDs, this should push closer to 2-3× over
single-frame steady state.

After Phases 1-6 collapse single-frame to ~30 ms, batched should hit
the 16.7 ms target.

## Phase 8 — Integration polish

Continuous overhead across all phases:
- Bit-exact regression suite (per-block + chain) run after every commit.
- Accuracy bench (5000 images) run after every 3 phases.
- Trace analysis (per-block, since the 31-tile cap blocks full-chain
  trace) to find unexpected stalls.
- Documentation updates as the design shifts.

## File-level impact summary

### New files
- `kernels/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_vec.cc` (Phase 1)
- `kernels/yolo_c3k2_small_*_vec.cc` (Phase 3, 4 kernels)
- `kernels/yolo_c3k2_heavy_*_vec.cc` (Phase 3, 4 kernels)
- `kernels/yolo_m10_*_vec.cc` (Phase 4, 3 kernels)
- `kernels/yolo_m0_conv2dk3_silu_bias_vec.cc` (Phase 2, if m0 needs special handling for in_c=3 padded to 8)

### Modified
- `aie2_yolo_per_block.py`: every `_build_*` function updated for
  ic-tile-major fifo element types + memtile-DMA-transpose wiring
- `placement.py`: tile reassignments (Phase 5) + fusion-aware placement (Phase 6)
- `scripts/m8_stage.py` and `scripts/m9_stage.py` (m9 only minimally touched)
- `Makefile`: add new kernel files
- `aie2_yolo_iron_partial.py`: any chain-level wiring changes for fused boundaries
- `aie2_yolo_iron.py`: same
- `README.md`: performance section updates as numbers move

### Likely unchanged
- `gen_yolo_data.py`, `gen_yolo_silu_luts.py`: weight + LUT formats stay
  the same; ic-tile-major is an activation-layout choice, not a weight one
- `models/*.onnx`, `models/*.pt`: source quantized model unchanged
- `test_block_ort.py`, `test_chain_ort.py`: host shim still presents HWC at
  the boundaries; m0 input + m10 output do the final HWC ↔ ic-tile-major
  conversion (probably memtile-DMA-based)
- `notebooks/*`: training + quantization pipeline unchanged
- `yolo_spec.py`: logical network spec unchanged

## Bit-exactness invariants

Every phase ends with:
1. **`make BLOCK=<changed> run_ort`** = 0 mismatches per changed block
2. **`make chain run_chain`** = 0 mismatches end-to-end
3. **Per-block bench** numbers recorded in a `PERF.md` rolling table

Periodically (every 3 phases or before a session ends):
4. **Accuracy bench** (`/tmp/eval_npu_accuracy.py`) sanity-check that
   classification accuracy is preserved (bit-exact at each block
   guarantees this mathematically; the bench is a cheap belt+suspenders
   check)

## Open questions

1. **Memtile-DMA transpose** (Phase 0): can a single memtile BD chain
   express HWC ↔ ic-tile-major via multi-dim strides? This determines
   the cost model for the rest of the plan. Spike in this session.
2. **Tile budget elasticity**: chain is at 32/32 today. Can we free
   2-4 tiles via fusion within Phases 1-4, or does Phase 5 (tile
   reassignment) need to happen earlier to make room for vectorized
   blocks?
3. **AIE2P chess intrinsic stability**: the `chess_prepare_for_pipelining`
   + `chess_loop_range` hints we want are AIE2-tested; AIE2P codegen
   may differ. Phase 1 will reveal.
4. **Vectorized SiLU lookup**: AIE has `aie::lookup` for elementwise
   table-driven LUT. Does it handle 256-entry int8 LUTs at 4-element-
   wide vector parallelism efficiently on AIE2P? Phase 1 spike.
5. **Cascade interconnect** (Phase 6): need to verify which adjacent
   block placements are cascade-eligible on Strix. AIE2P cascade
   topology might restrict some fusion options.

## Parallelization & team strategy

This work can be partially parallelized once Phase 1 succeeds (i.e., the
kernel template + ic-tile-major pattern + bit-exactness flow are proven).

**Parallel-friendly phases:**
- Phase 3 (4 distinct c3k2 kernel files — `cv1_split`, `m0_cv1`,
  `m0_cv2_skip`, `cv2_concat3`): one agent per kernel, each in their own
  git worktree. Each agent writes + builds + does MLIR-gen + scalar-emulated
  bit-exact spike. Integration agent merges + does HW bit-exact verify
  sequentially.
- Phase 4 (m10 head): 3 kernels (`conv2dk1_silu_xy_pool`, `linear_gemm`,
  `softmax`), can be parallel with each other.
- Phase 0 spike + Phase 1 pilot can run in parallel (different files,
  different concerns).

**Strictly serial phases:**
- Phase 1 itself (the pilot — needs single focused author + iterative
  HW debug; team-of-three on the SAME kernel adds merge pain).
- Phase 5 (tile reassignment) blocks Phase 6 (fusion). Both are global
  design decisions that don't decompose into parallel kernel work.
- HW bit-exactness verification across all phases — single Strix box,
  serial NPU dispatches.

**Team pattern** (if pursued):
1. Lead agent (architect): owns `aie2_yolo_per_block.py`, `placement.py`,
   `VECTORIZATION_PLAN.md`, and integration. Reviews all kernel agents'
   work before merging.
2. Kernel agents (1 per kernel file in flight): own a single `.cc` file
   each, work in their own git worktree, MLIR-gen spike locally.
3. Verification agent: serializes HW bit-exactness checks after lead
   merges. Owns the bench harness + accuracy bench.

Lead agent maintains a `STATUS.md` for who's owning what at any moment
to avoid two agents touching the same file.

**Honest take**: a team helps Phase 3 a lot (4 parallel kernel files →
maybe 2-3× faster) but doesn't help Phase 0, Phase 1, Phase 5, or
Phase 6 much. Use the team when the rate-limiting step is "write more
kernel code" — most of the rest of the time, single focused work is
faster than coordination overhead.

## Status tracking

This document is updated at the end of each phase with measured per-phase
fps and any deviations from the plan. The git history is the audit trail.
