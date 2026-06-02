# YOLO26n-cls (INT8) on AI Engine (IRON)

End-to-end YOLO26n classification inference on the AMD Strix Point NPU2
(8-column × 6-row array: row 0 shim DMA, row 1 memtile, rows 2–5 the
4 compute-tile rows = 32 compute tiles total), implemented in the
high-level IRON Python API. All 11 blocks (`m0..m10`) run depth-first
across the array, keeping intermediate activations on-chip, and emit
classification probabilities **bit-exactly matching the Quark XINT8 ONNX
reference**.

The full int8 quantization recipe lives in [`notebooks/`](notebooks): a
FP32 YOLO26n-cls PyTorch checkpoint → FP32 ONNX → AMD Quark XINT8 PTQ →
the deployment ONNX shipped in [`models/`](models).

## Quickstart

```bash
# 1. Activate the standard mlir-aie environment.
source /opt/xilinx/xrt/setup.sh
source ../../../ironenv/bin/activate
source ../../../utils/env_setup.sh ../../../install

# 1b. Install the ML extras (onnx + onnxruntime + torch). The standard
#     mlir-aie env install only pulls requirements.txt + requirements_dev.txt;
#     `make data` and the ORT bit-exact tests need this file too.
pip install -r ../../../python/requirements_ml.txt

# 2. List available targets.
make help

# 3. Extract per-block weight / bias / SiLU LUT bins from the shipped XINT8 ONNX.
make data

# 4. Bit-exact-check a single block on the NPU.
make BLOCK=m8 run_ort

# 5. Bit-exact-check the full chain on the NPU.
make run_chain                          # N=1, single-sample latency mode
CHAIN_N_SAMPLES=15 make run_chain       # batched throughput mode

# 6. Time it.
make BLOCK=m8 time                      # per-block (median ms + fps)
make time_chain                         # chain N=1
CHAIN_N_SAMPLES=15 make time_chain      # chain N=15
```

`make data` is idempotent; the resulting `data/` tree is ~2 MB of
per-layer bins (gitignored).

## Performance (Strix HW, bit-exact vs ORT)

All measurements taken with the NPU in turbo mode
(`sudo xrt-smi configure --pmode turbo`).

**Chain — full sweep, post-audit.** Every prefix m0..mN measured at
N=1/4/15 via `CHAIN_BLOCKS=… make time_chain`. ms columns are per-sample
(N=1 == single-dispatch wall, N>1 == per-sample throughput); fps =
1000 / per-sample-ms. Full chain m0..m10 N=15 is bit-exact vs ORT.

| Chain | N=1 ms | N=1 fps | N=4 ms | N=4 fps | N=15 ms | N=15 fps |
|---|---:|---:|---:|---:|---:|---:|
| m0                     | 11.13 | 89.81 | 11.06 | 90.45 | 11.01 | 90.81 |
| m0..m1                 | 11.20 | 89.29 | 11.07 | 90.33 | 11.02 | 90.78 |
| m0..m2                 | 13.19 | 75.81 | 12.82 | 78.01 | 12.66 | 78.99 |
| m0..m3                 | 13.27 | 75.34 | 12.78 | 78.23 | 12.66 | 78.96 |
| m0..m4                 | 13.95 | 71.70 | 12.96 | 77.17 | 12.71 | 78.68 |
| m0..m5                 | 14.10 | 70.91 | 12.99 | 77.01 | 12.72 | 78.59 |
| m0..m6                 | 16.02 | 62.44 | 13.48 | 74.18 | 12.85 | 77.83 |
| m0..m7                 | 16.27 | 61.45 | 13.53 | 73.92 | 12.87 | 77.72 |
| m0..m8                 | 22.94 | 43.59 | 16.90 | 59.17 | 15.43 | 64.80 |
| m0..m9                 | 26.80 | 37.32 | 17.87 | 55.95 | 15.69 | 63.73 |
| **m0..m10 (FULL)**     | **32.17** | **31.08** | **19.22** | **52.03** | **16.05** | **62.32** |

Every chain prefix through m0..m7 sits above 70 fps at N=15. The full
m0..m10 chain crosses 60 fps at N=15 (62.32 fps). m6 was hanging at
runtime before the upstream IRON ObjectFifo depth-collapse fix
(PR #3096) was cherry-picked into this branch — multi-consumer fanout
fifos were silently being sized to ping-pong (2 buffers) instead of
the declared depth, deadlocking c3k2_heavy's bot_fifo broadcast.

**After m8 4-tile + m9-cv2 relocation** (chain default, see "Sprint 3"
below): full m0..m10 chain N=15 = **15.17 ms / 65.94 fps**, N=1 = 36.91
ms. m8 4-tile reduces standalone m8 from 16.32 ms → 14.97 ms; the bulk
of m8's 16-row pipeline is still spent on the two inner-pair tiles
(B + C carry ~9.82 ms of the 14.97 ms wall per per-tile ablation), so
a real per-pair-conv split into 6 tiles is the next lever.

**After m8 cv1→cv2 mmul-layout pre-pack** (Sprint 4, see below):
full m0..m10 chain N=15 = **13.63 ms / 73.35 fps**, m8 standalone
4-tile = **11.44 ms / 87.4 fps**. The pair_cv1 → pair_cv2 inter-kernel
buffer (`p0_mid`/`p1_mid`) now uses mmul-ready `(ic_t, x_block, p*8+chan)`
layout: producer cv1 writes its mmul output via one `aie::store_v<64>`;
consumer cv2 reads via one `aie::load_v<64>` for kx=1 and 2 vlda +
`aie::shuffle_down` for kx=0/2. No more scalar `a_buf` pack between
the vec input gather and the mmul call. Bit-exact preserved on all 15
chain samples.

**After Sprint 4 extended across all m8 internal buffers**
(Stages 1a-1d, see below): full m0..m10 chain N=15 = **13.48 ms /
74.18 fps**, **m8 standalone 4-tile = 4.91 ms / 203.8 fps** (−67%
vs README, +205% fps). Every internal m8 producer-consumer buffer
(front → back, m_0_split → back, pair_cv1 ↔ pair_cv2 internal,
cv2_skip → pair_cv1 cross-pair, back cv3 ↔ cv2 internal scratch)
now uses the same mmul-packed `(ic_t, x_block, p*8+chan)` layout.
At N=15 chain is now at 99.9% of the m8-noop'd ceiling — m8 is
no longer the chain gate.

**After Sprint 5 (m6 c3k2_heavy non-streamed kernels)**:
m6 standalone = **4.61 ms / 216.9 fps** (was 9.23 ms fresh measure,
12.04 README baseline). Heavy-specific kernels (m_0_split,
inner_pair_cv1, inner_pair_cv2_skip, cv3_concat2) all switched to
the same mmul-packed layout. cv1_split + cv2_concat3 deferred to
Sprint 6 (shared with m2/m4). Chain N=15 barely moved (13.45 ms /
74.32 fps) because m2/m4 became the next gate.

**After Sprint 6 (c3k2_small family + m6 boundaries)**:
chain N=15 = **11.79 ms / 84.82 fps**. Standalone wins: m2 = 7.27 /
137.6 fps (−44%); m4 = 3.84 / 260.1 fps (−70%); m6 = 2.39 / 419.1 fps
(−80% from original 12.04 ms / 33.1 ms wall). Four c3k2_small files
(cv1_split, m0_cv1, m0_cv2_skip, cv2_concat3) switched to 4-pixel-
block packed layout (32-byte blocks). m6's heavy boundaries
(m_0_split input read, cv3_concat2 output write) updated to match.
Chain co-gate moves to m0 (11.12 ms standalone) — next sprint
target.

**Per-block standalone wall time on NPU**, median of n=20 (turbo).
All rows reflect the post-audit state; "was X" values in the rightmost
column are the README snapshot from prior to the kernel-by-kernel audit.

| Block | Topology | Median (ms) | fps | |
|---:|---|---:|---:|:--|
| m0  | conv_stride stem               | **11.12** | **89.9**  | ✓ (was 76.4; Sprint 7 candidate) |
| m1  | conv_stride                    | **4.82**  | **207.5** | ✓ (was 116.1) |
| m2  | c3k2_small                     | **7.27**  | **137.6** | ✓ (was 12.98 before Sprint 6 mmul pre-pack) |
| m3  | conv_stride (chunked)          | **5.09**  | **196.5** | ✓ (was 124.9) |
| m4  | c3k2_small                     | **3.84**  | **260.1** | ✓ (was 12.58 before Sprint 6) |
| m5  | conv_stride (chunked)          | **5.40**  | **185.2** | ✓ (was 139.5) |
| m6  | c3k2_heavy                     | **2.39**  | **419.1** | ✓ (was 12.04 before Sprints 5+6; -80% / +400% fps) |
| m7  | conv_stride (chunked)          | **3.88**  | **257.7** | ✓ (was 206.8) |
| m8  | c3k2_heavy (2-tile megakernel) | **16.32** | **61.3**  | ✓ (M8_TILES=2; default for per-block standalone) |
| m8  | c3k2_heavy (4-tile megakernel) | **4.39**  | **227.6** | ✓ (M8_TILES=4; default in chain) — was 14.97 / 66.8 fps before Sprint 4. Sprint 4 + stages 1a-1e converted every internal m8 buffer (front↔back, m_0_split, pair_cv1↔pair_cv2, back internal scratch, cv1↔m_0_split) to mmul-packed layout. Final sweep clean of `__divsi3` and per-iter branches in mac bodies. |
| m9 stage 1 (cv1 only)            | PSA cv1 |  1.78 | 561.8 | |
| m9 stage 10 (full PSA block)     | PSA full | **4.95** | **202.2** | ✓ (was 195.9 → qkv SHAPES_ARE_CONST) |
| m10 (classifier head)            | conv2dk1+GAP+linear+softmax | **7.69** | **130.1** | ✓ |


m9 is supported standalone via `M9_STAGE=N make BLOCK=m9 run_ort` —
stage 1 compares against `/model.9/cv1/act` post-SiLU; stage 10 (default)
compares against `/model.9/cv2/act` post-SiLU. Intermediate stages emit
PSA-internal tensors with no clean ORT analog; verify those via
`make run_chain`. m10 (head) still standalone-unsupported; covered by
the chain test.

### Exemplar kernel playbook

Every active kernel in this design was audited against a 10-item
optimization playbook. Items applied where structurally compatible:

1. **Constexpr trip counts** for the kernel's known call site (eliminates
   `__divsi3` / `__divsf3` software-math calls in the elf; folds address
   arithmetic into immediates). Verified by `llvm-nm | grep __div`.
2. **Bias-init the mmul accumulator** (8 int32 biases -> 32-wide
   `acc<acc32>` via `concat(b8,b8) -> concat -> from_vector`). Replaces
   per-element scalar bias add.
3. **Vec `to_vector<int8>(rs)` SRS** with `set_rounding(conv_even)` to
   match the scalar `banker_srs` reference. Replaces scalar SRS+clamp tail.
4. **`aie::inv(float)`** for unavoidable fp reciprocals (HW intrinsic;
   single op vs `__divsf3` ~80 cycles). Used in m9 attn_score and m10
   softmax.
5. **Vec mac/mul** in inner loops (mmul.mac or aie::mac).
6. **mmul<8,8,8>** over `<4,8,8>` only where it pays - confirmed on
   pair_cv1 (m8); regresses on 1x1 and stride-2 stem (m0 tried + reverted).
7. **Hoisted vec input gather** for kernels with KW x KH > 1 mac groups
   per pix load (3x3 stride-1 only - stride-2 has non-contiguous cols,
   1x1 doesn't amortize).
8. **`AIE_LOOP_RANGE` + `AIE_LOOP_UNROLL_FULL` hints** on every
   non-trivial loop (requires `aie_kernel_utils.h`).
9. **Compile-time shape macros** (`SHAPES_ARE_CONST` pattern) for
   kernels with multiple call-site shapes. Now wired on every kernel
   that benefits from it: c3k2_small {m2, m4}; chunked stride-2 {m3,
   m5, m7}; c3k2_heavy m6 (cv1_split, cv2_concat3, inner_pair_cv1,
   inner_pair_cv2_skip, m_0_split, cv3_concat2); m9 qkv. Kernels that
   were already constexpr-hardcoded under a different convention
   (proj_skip_row, pe_add_row, m10 trio) provide the same effect.
10. **Bank-aware memory** (address-space-qualified pointer types per
    `example.h` `IFM_DM_BANK` pattern) - **not yet applied**;
    cross-cutting work that requires IRON-side
    `Buffer(..., placement=bank_N)` API + kernel-side
    `__attribute__((address_space(N)))` aliases.

**Known peano AIE2P codegen bug.** `accum<acc32, 16>::to_vector<int8>(shift)`
and `accum<acc32, 32>::to_vector<int8>(shift)` crash in
`getCombinedOpcodeUNPACKLoad` during InstructionSelect when the acc
comes from an `aie::mul`/`aie::mac` chain (not from `mmul.mac`).
64-wide acc survives. Workaround: keep the affected tail scalar, or
widen the acc to 64 via concat-with-zeros. Affects m9 attn_score,
proj_skip_row, ffn_1_skip_row, pe_add_row (all use scalar tail for
the second SRS).

### Biggest per-block wins from the audit

| Block | Pre-audit fps | Post-audit fps | Driver |
|---|---:|---:|---|
| m9 stage 10 |  91.6 | 202.2 | `pe_add_row` vec dw3x3; then qkv SHAPES_ARE_CONST |
| m1          | 116.1 | 206.6 | `conv2dk3_stride2` bias-init + vec SRS |
| m3          | 124.9 | 195.1 | `conv2dk3_stride2_chunked` bias-init + vec SRS |
| m5          | 139.5 | 185.1 | same |
| m7          | 206.8 | 253.7 | same |
| m6          |  66.9 |  83.1 | c3k2_heavy bias-init + vec SRS; then SHAPES_ARE_CONST on all 6 m6 kernels |
| m0          |  76.4 |  88.3 | stem bias-init + vec SRS |
| m4          |  75.2 |  79.5 | c3k2_small (4 shared kernels) bias-init |
| m2          |  69.7 |  77.0 | same |

m8 (61.3 fps) and m10 (130.1 fps) were already deep-opt'd in an earlier
m8/m9/m10 sweep that preceded the kernel-by-kernel audit.

### Sprint 3: chain-context per-block ablation

Re-ran the per-block NOOP ablation at N=15 with a full `make clean` between
iters (the first attempt accidentally chained NOOP flags across iters
because Make tracks mtime not flag changes — the true per-block
contributions are what's recorded here). Each row = noop one block,
re-time the chain:

| Block | Δ vs baseline (16.03 ms) | Standalone | In-chain contribution |
|---|---:|---:|---|
| m0  | −0.01 | 11.32 | hidden behind m8 wall |
| m1  |  0.00 | 4.84  | hidden |
| m2  | −0.03 | 12.98 | hidden, but **co-gates** at 13 ms once m8 drops |
| m3  |  0.00 | 5.13  | hidden |
| m4  | small | 12.58 | co-gates with m2/m6 |
| m5  | −0.04 | 5.40  | hidden |
| m6  | small | 12.04 | co-gates |
| m7  | −0.01 | 3.94  | hidden |
| m8  | **−2.57** | 16.32 | **primary gate** |
| m9  | −0.10 | 4.95  | pipelines well — almost free in-chain |
| m10 | −0.35 | 7.69  | also pipelines well |

The combo "isolate m8" run (noop m9+m10): chain throughput = 15.03 ms,
matching m8 standalone's 16.32 ms within noise. So m8 is the dominant
gate. After m8 drops, the cluster max(m2, m4, m6) ≈ 13 ms becomes the
next wall.

### m8 4-tile per-tile ablation

`M8_TILES=4` baseline standalone = 14.98 ms. NOOP each tile's kernels
in turn (compute swap; locks + DMA still run, so the cross-tile pipeline
shape is unchanged):

| NOOP | ms | Exposes |
|---|---:|---|
| baseline 4t          | 14.98 | full pipeline |
| noop A (front)       | 12.66 | B + C + D path |
| noop D (back)        | 12.15 | A + B + C path |
| noop A + D           | 9.82  | B + C alone |
| **noop B + C (inner pairs)** | **6.91** | **A + D alone — gives the headroom number** |
| noop all 4           | 0.46  | pure DMA / pipeline floor |

Reading: the inner pairs on B (pair0_cv1 + pair0_cv2) and C (pair1_cv1
+ pair1_cv2) carry the bulk of m8's wall. The 4-tile design grouped
2 ops per tile on B and C; halving each into its own tile (real 6-tile
design) should drop B+C from 9.82 → ~5 ms and the m8 wall from 15 → ~7
ms, lifting the chain past 100 fps.

### Real 6-tile m8 — built standalone bit-exact, but no perf win

`scripts/m8_megakernel_6tile.py` implements the per-pair-conv split
described above (one inner conv per tile, snake-shaped A→B1→B2→C1→C2→D
pipeline). Standalone bit-exact via `M8_TILES=6 make BLOCK=m8 run_ort`
at **14.67 ms** (vs 4-tile's 15.08 ms — basically identical). **Chain
context: M8_TILES=6 currently can't be used** — its memtile placements
(2,1) and (7,1) conflict with m9 which holds (2,1) for ffn weights and
(7,1) for cv1 weights. So the chain default is `M8_TILES=4`.

Sprint 3 deep-dive on 6-tile cv1↔cv2 serialization (extensive):

- **Pipeline plumbing is fine** — `noop_all` (every compute kernel
  returns early) drains the 16-row × 6-stage pipeline in 0.45 ms.
- **cv1 stages alone or cv2 stages alone pipeline correctly** —
  `noop_cv2` = 10.75 ms = A+D (6.92) + max-stage cv1 (3.83). Same for
  cv2 alone.
- **With both active, cv1 + cv2 SUM rather than max per-pair** —
  baseline 14.66 ≈ 6.92 + 3.83 + 3.88. The pair stages on the same
  pair (B1↔B2 or C1↔C2) do not overlap, only pair0↔pair1 pipelines.

Hypotheses tested and FALSIFIED:

| Hypothesis | Test | Result |
|---|---|---|
| OF depth too shallow on `pair_mid_xt` | bumped depth 4 → 6 → 10 → 12 | no change |
| scalar `sa_mid → sk_r` memcpy is the bottleneck | removed the memcpy | only 0.16 ms savings |
| memtile DMA serializes cv1+cv2 weight streams | moved cv2 to a different memtile | no help |
| L1 bank contention between B1 producer writes + B2 consumer reads | `via_DMA=True` on `pair_mid_xt` (separate buffer pools) | no help |
| Producer/consumer different banks via depth modulo | varied depth 4..12 | flat scan, no signal |
| Peano software-pipelining differs across modes | static asm inspection — no `loop` / `.zol` markers in either | not the cause |
| `dynamic_objfifo_lowering=True` overhead in inner loop | local 3-row scratch cache → peek-1 on cross-tile OF (bit-exact) | same perf as peek-3 (14.87 ms vs 14.66) |

A `peek-1` mode test (acquire 1 per iter, pass `pm, pm, pm` to kernel
— breaks bit-exact) gave 6.60 ms; this initially looked like proof of
peek-3 lockstep, but was actually a compiler artifact: with the same
SSA pointer used 3× as kernel arg, Peano dead-load-eliminated 2/3 of
the input loads.

The actual cause of per-pair B1+B2 serialization is **not identified**.
The 6-tile scaffold is kept in `scripts/m8_megakernel_6tile.py` for
future revival; the m9 placement reshuffle from this sprint (m9-cv2 →
(4,5)) is retained because it's also needed for any future m8 widening.

### Sprint 4: mmul A-layout pre-pack on cv1↔cv2 inter-kernel buffer

Objdump-driven diagnosis on the m8 inner-pair kernels found a hidden
per-mmul cost the playbook's "hoisted vec pixel gather" (item #7) had
not eliminated. After loading 8 pixels × 64 channels as 8 `vector<int8,
64>` registers, the code still had to scalar-extract the 8-byte slice
for each (ic_t, p) into a stack `a_buf[64]` to feed the mmul A vec:

```cpp
// pix[p] already in vec registers (good)
for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
  alignas(64) int8_t a_buf[64];                 // scratch
  for (int p = 0; p < MMUL_M; ++p)
    for (int b = 0; b < 8; ++b)
      a_buf[p * 8 + b] = pix[p][ic_t * 8 + b];  // <-- 64 vextract.8 per mmul
  aie::vector<int8, 64> in_a = aie::load_v<64>(a_buf);
  // ...
}
```

Inner loop body of the cv2 SHAPES_ARE_CONST mac loop was ~32 cycles
per mmul vs ~6-8 cycles ideal (4-5× off peak). Root cause: aie-api
forbids `aie::concat` of sub-128-bit vectors, so a 64-byte mmul A vec
can't be built from 8-byte `aie::load_v<8>` chunks at strided pointers
— the scalar fallback is the only way when input layout is (W, C)
raster. (Same shape of bug on the 1×1 kernels at ~12% peak.)

**Fix:** change the inter-kernel buffer layout. Producer cv1's mmul
output is *already* in `(p*8+chan)` byte order from
`acc.to_vector<int8>(rs)`, so writing it as ONE `aie::store_v<64>` at
offset `(chunk_oc_t_full, x_tile)` in the `p0_mid`/`p1_mid` row is
free. Consumer cv2 then reads the A vec with:

- **kx=1**: single `aie::load_v<64>(base + ic_t*stride + x_tile*64)` — direct.
- **kx=0/2**: pixels straddle blocks. Load 2 adjacent blocks (vec<int8, 64>
  each), `aie::concat` to vec<int8, 128>, `aie::shuffle_down(combined, 56 or 8)`,
  `.extract<64>(0)`. Out-of-bounds blocks zero-filled.

Buffer size unchanged (1024 B/row); only byte-level interpretation
changed. Bit-exact preserved on all 15 chain samples.

| Metric                       | Before  | After   | Δ              |
|------------------------------|--------:|--------:|---------------:|
| m8 standalone (M8_TILES=4)   | 14.97 ms / 66.8 fps  | **11.44 / 87.4** | **−23.6%** |
| Chain m0..m10 N=15           | 15.17 ms / 65.91 fps | **13.63 / 73.35** | **+7.4 fps** |
| Chain N=15, m8 noop'd ceiling| 13.47 ms / 74.26 fps | (unchanged)       | within 1.2% |

The standalone delta (3.5 ms) exceeded the byte-count projection
(~1 ms) — removing the scalar pack also lets Peano schedule the
surrounding loop more tightly than a pure instruction count predicts.

The kernel files touched are `kernels/yolo_c3k2_heavy_inner_pair_cv1_streamed_vec.cc`
(output write loop) and `kernels/yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_vec.cc`
(`load_a_mmul_kx` helper + input read). The shared symbol means both
pair0 (tile B) and pair1 (tile C) get the optimization in lockstep.

### Sprint 4 follow-on: extend the mmul pre-pack across all m8 buffers

Cascading the same producer-vec-store / consumer-vec-load pattern to
every remaining internal m8 buffer, in four ship-as-you-go stages:

| Stage | Touched | m8 standalone | Δ |
|-------|---------|---:|---:|
| (Sprint 4 above) | pair_cv1 → pair_cv2 (p0_mid, p1_mid) | 11.44 ms | (−23.6% from 14.97) |
| **1a** | front cv1 (top, bot_to_cv2) → back cv2 | **10.48 ms** | −0.96 ms |
| **1b** | m_0_split's split_b → back cv3 | **9.77 ms** | −0.71 ms |
| **1c** | back internal cv3 → cv2 scratch handoff | **8.88 ms** | −0.89 ms |
| **1d** | pair_cv2_skip output / pair_cv1 input / m_0_split's split_a / back cv3 inner1 / cv2_skip's skip-add path (coordinated) | **4.91 ms** | **−3.97 ms** |
| **1e** | cv1's s_bot → m_0_split's in_bot (last internal buffer) | **4.39 ms** | −0.52 ms |

Why 1d was so much larger: it eliminated the scalar pack on **both
3×3 inner_pair kernels** (the dominant cost in the B+C tiles, ~9.8
ms of the original wall), and crucially also caught the **skip-add**
path inside cv2_skip — `skip_v = aie::load_v<64>(skip_row + ...)`
instead of 64 scalar gathers per acc. The first attempt at 1d had
NPU != ORT because the orchestration's byte-wise `sk_r[x,0,kk] =
sa_mid[x,0,kk]` copy preserves whatever format the producer wrote,
so the consumer-side read must match. (Pattern saved as
`feedback_mmul_layout_prepack` + `feedback_no_branch_in_mac_loop`
in memory.)

Mixed-source kernels (back's cv2 reads in_top + in_bot packed,
in_m0 raster initially) use **split per-source loops** in the inner
IC dimension, never a per-iter `if` inside the mac body — that
branch defeats Peano's VLIW pipelining of `vlda + vldb + vmac` and
cost ~7% of the standalone gain on Stage 1b until refactored.

| Chain | After Sprint 3 | Sprint 4 | After Stage 1e | After Sprints 5+6 (current) | Δ vs Sprint 3 |
|------:|---:|---:|---:|---:|---:|
| N=1   | 36.91 ms / 27.1 fps | 31.05 / 32.21 | 25.47 / 39.27 | **22.66 / 44.13** | **−38.6% / +62.8%** |
| N=4   | 19.22 ms / 52.03    | (not measured)| 15.83 / 63.18 | **13.91 / 71.88** | **−27.6% / +38.2%** |
| N=15  | 15.17 ms / 65.94    | 13.63 / 73.35 | 13.48 / 74.20 | **11.79 / 84.82** | **−22.3% / +28.6%** |

At N=15 we're at **99.9% of the m8-noop'd ceiling** (74.26 fps) —
further m8 work helps N=1/N=4 single-dispatch latency but not
steady-state throughput. The next chain-level lever moves to
m2/m4/m6 (now the co-gate at ~13.5 ms).

### Next remaining levers
1. **Extend mmul A-layout pre-pack** (Sprint 4 pattern) to the remaining
   m8 producer-consumer pairs and then across the network. In m8: front
   cv1 (1×1, ~12% peak) reading from tile A's m_0_split; back cv3+cv2
   (1×1, ~12% peak) reading from tile A's top/bot/m0 + tile C's
   inner_1_out; cv2_skip's output → next pair's cv1 input. After m8 is
   exhausted, the same pattern applies to m6 inner_pair_* (same kernel
   family) and the m2/m4 c3k2_small kernels.
2. **m2 / m4 OC-split to 5 tiles each** — only meaningful AFTER m8
   drops, since today both are hidden behind m8. After step 1, they
   become the co-gate at ~13 ms and a 5-tile OC-split should drop each
   to ~7-8 ms.
3. **Winograd F(2,3) on 3×3 stride-1** - potential 2.25× MAC reduction
   on the dominant kernel family (m6 inner_pair_*, m8 pair_cv1). INT8
   bit-exactness against the Quark ONNX reference needs an empirical
   numerical-drift study first; if it diverges, bfp16 Winograd via
   the ATB-style framework is the fallback.
4. **ATB-style L1 asymmetry on m9 GEMMs** (paper:
   [arXiv:2511.16041](https://arxiv.org/abs/2511.16041), reference
   example at `programming_examples/ml/block_datatypes/gemm_asymmetric_tile_buffering/`).
   Demonstrated 31.3 TFLOPS on the same Strix HX 370 silicon — large
   headroom on m9 qkv/proj/ffn0/ffn1. Chess-only today; porting the
   pattern to peano is research-grade work.
5. **Bank-aware memory** - playbook item #10. Cross-cutting; needs an
   IRON-side `Buffer(..., placement=bank_N)` API hook + kernel-side
   address-space-qualified pointer types modeled after `example.h`'s
   `IFM_DM_BANK` / `WT_DM_BANK` / `OFM_DM_BANK` aliases.
6. **Softmax exp2** - replace the scalar fp32 LUT in
   `yolo_m9_attn_score_fused_vec.cc` with hardware
   `aie::exp2<bfloat16>` (see `aie_kernels/aie2p/softmax.cc`).
   Bit-exactness risk; would be tested separately.
7. **Push the peano backend bug fix upstream** for the
   `getCombinedOpcodeUNPACKLoad` crash on
   `accum<acc32, {16,32}>::to_vector<int8>(shift)` following
   `aie::mul/aie::mac`. Unblocks 4 m9 kernels' vec skip-add tails.

The previous "next remaining levers #1" was `mmul<8,8,8>` on m6
inner_pair_cv1/cv2_skip — that's already in place under the
SHAPES_ARE_CONST=1 path triggered by `M6_PAIR_SHAPE` (and similarly
under `M8_PAIR_SHAPE` for the streamed m8 variants). Both pair kernels
on both blocks already select `MMUL_T = MMUL8x8x8` at compile time.

**`aie::parallel_lookup` investigation** (deferred). PL on AIE2P peano
compiles + runs correctly, but `lut<4>` is
fundamentally a linear-approximation API (offset + slope * frac), not
a plain-byte lookup. For yolo's INT8 SiLU LUT use case, effective
throughput is ~equal to or slower than scalar `silu_lut[srs+128]`.
PL becomes interesting again only if SiLU is re-derived as a
piecewise linear approximation.

## Make targets

Run `make help` for the canonical list. Highlights:

| Target | What it does |
|---|---|
| `make data` | ONNX → `data/{weights,bias,silu_lut}.bin` + manifest |
| `make BLOCK=mN` | Build per-block xclbin + insts |
| `make BLOCK=mN run_ort` | Build + run + bit-exact compare vs ORT |
| `make BLOCK=mN time` | Build + time (mean/median/min/max ms, fps) |
| `make chain` | Build chain xclbin + insts (defaults to N=1) |
| `make run_chain` | Build + run + bit-exact compare vs ORT (every sample) |
| `make time_chain` | Build + time (latency or per-sample throughput) |
| `make trace` | Build with HW packet trace, run 1 iter, parse summary |
| `make clean` | Wipe `build/` entirely |
| `make clean_chain` | Wipe only the chain xclbin/MLIR (cheap) |

**Environment knobs:**

- `BLOCK=mN` — selects the per-block target (default `m0`).
- `CHAIN_N_SAMPLES=N` — number of samples per chain dispatch. The Makefile
  tracks this in a stamp file and forces the chain MLIR to rebuild
  whenever the value changes, so `run_chain` and `time_chain` always
  match the xclbin's compiled-in `N`.
- `M9_STAGE=N` — for `BLOCK=m9`, selects which staged build of m9 to
  produce (1 = cv1 only, 10 = full PSA block; default 10). Same stamp-
  file rebuild trigger as `CHAIN_N_SAMPLES`. `M9_CHAIN_STAGE` is
  accepted as a legacy alias. Only stages 1 and 10 are wired through
  `test_block_ort.py` (others lack a clean single-tensor ORT analog).
- `M8_TILES={2,4}` — selects m8 megakernel layout. Chain default is 4
  (set by the Makefile, scoped to the chain MLIR build; stamp-tracked
  so toggling rebuilds). Per-block standalone (`make BLOCK=m8`) default
  is 2 — pass `M8_TILES=4 make BLOCK=m8` to switch. 2-tile = 16.32 ms
  standalone, 4-tile = 14.97 ms standalone. A real 6-tile is in
  development (see "Sprint 3").
- `TIME_ARGS="--n-warmup 5 --n-iters 100"` — passed through to the timing
  harnesses.
- `TRACE_SIZE_PER_WORKER=N`, `TRACE_EVENTS=A,B,C` — bake AIE2P packet
  tracing into per-block MLIR. The `make trace` target sets these and
  handles the build/run/parse round-trip via `scripts/trace_m8.sh`.

**Note on incremental builds:** the Makefile correctly rebuilds on `.cc`
or `.py` source changes (mtime-tracked). If you edit the `Makefile` itself
(e.g., changing `PEANOWRAP2P_FLAGS`) or change a `TRACE_*` env var,
existing `.o` / `.mlir` files won't pick up the change automatically — run
`make clean` first. `CHAIN_N_SAMPLES` is handled automatically.

## Tracing

`scripts/trace_m8.sh` (via `make trace`) builds the m8 megakernel with
HW packet trace ops, runs one iteration, and parses the trace. Output:

- `build/trace_m8.txt` — raw trace packets (hex)
- `build/trace_m8.json` — parsed events
- stdout — per-tile cycle summary from `python/utils/trace/get_trace_summary.py`

Pick the events you want via `TRACE_EVENTS`:

```bash
# Default (8-event packed set: INSTR_EVENT_0/1 + stalls + vector + ...)
make trace

# Stall attribution — what's the core blocked on?
TRACE_EVENTS=INSTR_EVENT_0,INSTR_EVENT_1,LOCK_STALL,STREAM_STALL,MEMORY_STALL \
    make trace

# Vector activity — what fraction of cycles are vmac?
TRACE_EVENTS=INSTR_EVENT_0,INSTR_EVENT_1,ACTIVE,INSTR_VECTOR \
    make trace

# Larger trace BO (default 32 KB / worker)
TRACE_SIZE_PER_WORKER=65536 make trace
```

Event names come from `aie.utils.trace.events.CoreEventAIE2P`. The
chain-side equivalent uses `aie2_yolo_iron_partial.py`'s `TRACE_BLOCKS`
env to restrict tracing to specific blocks (the AIE2 trace packet-ID
space caps at 31 simultaneously-traced workers, so the full 26-tile chain
needs front-half/back-half splits).

## Design overview

The 11 blocks map to these kernel families:

| Block | Topology | Kernel family | Tiles |
|---|---|---|---:|
| m0 | conv_stride (stem) | `yolo_m0_conv2dk3_silu_bias` (OIYX raw, in_c=3 padded to 8) | 1 |
| m1 | conv_stride | `yolo_conv2dk3_stride2_silu_bias_oiyxi8o8` (shared, runtime shapes) | 1 |
| m3, m5, m7 | conv_stride | `yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked` (per-block, weight-streamed) | 1 each |
| m2, m4 | c3k2_small | `yolo_c3k2_small_{cv1_split, m0_cv1, m0_cv2_skip, cv2_concat3}` | 3 each |
| m6 | c3k2_heavy | `yolo_c3k2_heavy_{m_0_split, inner_pair_cv1, inner_pair_cv2_skip, cv3_concat2}` + `c3k2_small_{cv1_split, cv2_concat3}` | 5 |
| **m8** | **c3k2_heavy 4-tile megakernel** (set `M8_TILES=2` for the smaller 2-tile variant) | `yolo_m8_front_cv1_split_fused` + `yolo_m8_back_cv3_cv2_fused` + `yolo_c3k2_heavy_inner_pair_{cv1,cv2_skip}_streamed` | **4** |
| m9 | PSA (attention + FFN) | 11 kernels: `yolo_m9_{cv1_split, qkv, qk_pack, attn_score_fused, v_pack, sv_row, pe_add_row, proj_skip_row, ffn_0_silu_row, ffn_1_skip_row, cv2_concat2_streamed}` (qk_row + attn_scale + softmax_row collapsed into `attn_score_fused_vec.cc`; sv_row + sv_row_acc share the same `.o` with two extern C entries) | 7 |
| m10 | head | `yolo_m10_{conv2dk1_silu_xy_pool, linear_gemm, softmax}` (fused onto 1 tile) | 1 |

Total: **28 of 32** compute tiles used.

### m8 — megakernel (4-tile default; 2-tile variant available)

The most compute-dense block. `M8_TILES` selects the variant (default 4,
applies to both chain and per-block standalone):

- **4-tile (default)** — [`scripts/m8_megakernel_4tile.py`](scripts/m8_megakernel_4tile.py).
  Four compute workers (`worker_a/b/c/d` on tiles (5,3), (5,4), and two
  delegates). Each pair kernel gets its own dedicated worker, targeting
  ~5 ms standalone.
- **2-tile** (`M8_TILES=2`) — [`scripts/m8_megakernel_2tile.py`](scripts/m8_megakernel_2tile.py).
  Two compute tiles, each running a single Worker that fuses three
  c3k2_heavy sub-operations into one C kernel call per chunk:
  - **Tile A (5,3)** — `k_m8_front` (cv1 + m_0_split) + `k_pair_cv1` +
    `k_pair_cv2` for pair0
  - **Tile B (5,4)** — `k_pair_cv1` + `k_pair_cv2` for pair1 +
    `k_m8_back` (cv3 + cv2)
- **6-tile** (`M8_TILES=6`) — compiles standalone but blocked in chain by
  memtile conflicts with m9 ((2,1) + (7,1) overlap). Standalone-only
  until placement is reworked.

Cross-tile data lives in shared L1 ObjectFifos (no DMA hop). Weights for
the big convs (cv1, cv2, pair0, pair1) are streamed from memtiles via
`StaticWeightStream`; the small m_0_split and cv3 weights are static on
the compute tiles.

### m9 — staged PSA

[`scripts/m9_stage.py`](scripts/m9_stage.py) implements the PSA block
across 5 compute tiles + 2 memtiles in 10 stages (each stage adds one
attention sub-kernel). The chain uses stage 10 = full m9. Lower stages
exist for chain-bisect debugging via `M9_CHAIN_STAGE=N`.

### Kernel symbol mangling

Several blocks share a kernel template (e.g. c3k2_small for m2/m4,
c3k2_heavy for m6, chunked conv_stride for m3/m5/m7) but with different
runtime shapes / weight sizes. To let them coexist in one MLIR module the
Makefile compiles each block's variant with `-DKERNEL_SUFFIX=_mN`, which
suffixes every exported symbol — so `yolo_c3k2_small_cv1_split_vec.cc`
produces `..._m2.o` and `..._m4.o` with non-colliding `func.func` names.
The IRON builder passes the matching suffixed name into `Kernel(...)`.

### Kernel design notes (chunked stride-2 conv)

The shared
[`yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_vec.cc`](kernels/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_vec.cc)
that backs m3/m5/m7 demonstrates several patterns worth carrying to other
peano-compiled AIE2P kernels in this repo:

**Compile-time shape specialization.** The kernel requires
`-DYOLO_IN_W=… -DYOLO_IN_C=… -DYOLO_OUT_C=… -DYOLO_INTERIOR_MODE=…` per
block (set in the Makefile's `CHUNKED_SHAPE_$(blk)` lookup). These become
`constexpr int` constants at file scope; the runtime ABI args are kept
in the C signature but the function body reads the constexprs. This is
the single biggest perf lever (typically 1.5-3× on top of vector loops)
because peano can:
- fold address math (`* in_c`, `* out_c`, `* (ic_tiles*kH*kW*64)`) into
  shifts and immediates, eliminating runtime scalar muls,
- dead-strip the unused `if constexpr (...)` branch (in this kernel:
  the unused `INTERIOR_MODE` variant), and
- emit tighter pipelined loops with known trip counts.

**`AIE_LOOP_RANGE` everywhere.** Peano needs `[[clang::loop_min/max_iteration_count]]`
to pipeline (and unroll) loops with runtime-derived bounds. Wrap each
non-fully-unrolled loop with `AIE_LOOP_RANGE(min, max)`; use
`AIE_LOOP_UNROLL_FULL` on small fixed-bound loops (e.g. `kx` over 3).
Note that `AIE_PREPARE_FOR_PIPELINING` is a no-op under peano (it only
lowers for chess) — see `aie_kernels/aie_kernel_utils.h`.

**3-way split for edge vs interior.** The bounds-check branch needed for
left/right border columns blocks software pipelining if it sits inside
the hot loop. Splitting the `x_tile` loop into left-edge / interior /
right-edge nests (with the edges' bodies *inlined*, not factored into
lambdas — peano doesn't reliably inline them) lets the interior become
fully straight-line vector ops.

**OC×2 and 2X×2OC accumulator fold.** The mmul-throughput-hiding pattern
from `example.h`: carry multiple accumulators so each weight load is
reused across multiple X positions and each input gather is reused across
multiple OC banks. `YOLO_INTERIOR_MODE=2` (4 accs: 2 X × 2 OC) for blocks
with enough x_tiles to amortize the pair setup (m3, m5); `MODE=1` (2 accs:
single-X × 2 OC) for small-x_tiles blocks (m7) where the pair setup
overhead would dominate.

**64-bit word-copy gather instead of scalar byte loop.** The 4×8-byte
input gather for one mmul is built via 4 aligned `uint64_t` reads into a
stack `a_buf`, then a single `aie::load_v<32>`. Peano hoists this into
register-to-register `vpush.hi.64` in the steady-state pipelined inner
loop, eliminating the stack round-trip in the hot path.

### Bit-exactness

The integration test of record is `test_chain_ort.py`: same seeded RGB
input fed to both (a) the NPU chain xclbin and (b) ONNX Runtime on the
XINT8 ONNX, with hard element-wise INT8 equality on the final softmax
probs. For `CHAIN_N_SAMPLES=N>1`, every output slice is compared
independently — all N samples must match the single reference.

The conv_stride and c3k2 blocks (m0..m8) each pass `test_block_ort.py
--block mN` bit-exact standalone (see the `run_ort` target). m9 (PSA)
is supported in two stages: `M9_STAGE=1 make BLOCK=m9 run_ort` compares
NPU vs ORT for `/model.9/cv1/act` (cv1 only); `M9_STAGE=10 make BLOCK=m9
run_ort` (default) compares for `/model.9/cv2/act` (full m9). Other m9
stages and m10 (head) aren't supported standalone — covered by the
end-to-end chain test.

[`run_strix_makefile.lit`](run_strix_makefile.lit) exercises both paths
on NPU2 hardware.

## The int8 recipe

Three stages: train float, quantize to int8, materialize per-block
kernel inputs.

### 1. Train (upstream)

YOLO26n-cls is a 1.5M-parameter classifier from Ultralytics. The FP32
checkpoint shipped here (`models/phase1_25k_acc0.9424.pt`, 90.0% test /
94.2% val accuracy on a person/no-person dataset) is the source. See
[`notebooks/yoloexploration.ipynb`](notebooks/yoloexploration.ipynb) for
loading, inference, and COCO validation.

### 2. Quantize (Quark XINT8 PTQ)

[`notebooks/quark_quantization.ipynb`](notebooks/quark_quantization.ipynb)
walks through:

1. Export FP32 PyTorch → FP32 ONNX at fixed `(batch=4, 3, 512, 512)` (Ryzen
   AI requires static shapes).
2. Sanity-check FP32 ONNX vs the PyTorch baseline — catches preprocessing
   bugs before quantization muddies the signal.
3. Build an `onnxruntime.quantization.CalibrationDataReader` from 500
   in-distribution images.
4. Run Quark static PTQ with the **XINT8** spec for both activations and
   weights, **AdaRound** for weight rounding refinement, and **exclude the
   final Softmax** from quantization (empirically: quantizing the final
   Softmax causes a sign-flip that drops accuracy to ~44%; keeping it
   FP32 recovers 89.5%).
5. Evaluate the XINT8 ONNX on the held-out test set, compare to FP32.

Output: `phase1_25k_xint8_acc0.8968.onnx` (1.7 MB, 89.68% test
accuracy, ~0.3 pp from FP32 — within typical static-PTQ overhead). Shipped
under [`models/`](models) and consumed by `gen_yolo_data.py` /
`gen_yolo_silu_luts.py`.

### 3. Extract per-block kernel inputs

[`gen_yolo_data.py`](gen_yolo_data.py) walks the Quark XINT8 ONNX and, for
every Conv / MatMul / Gemm node, emits:

- INT8 weight bins, reordered from ONNX `OIYX` to the `OIYXI8O8` layout
  the AIE2P kernels expect (m0 keeps `OIYX` because `in_c=3` is not
  8-aligned and gets host-side padded to 8 instead).
- INT32 bias bins, promoted from INT8 by `bias_pre_shift` so the runtime
  kernel can use them as the accumulator init when `weight_index == 0`
  (folds the bias-add into the conv's first MAC — no separate bias stage).
- A per-op `right_shift = log2(scale_in) + log2(scale_w) − log2(scale_out)`
  for the int32→int8 banker-rounded requant epilogue.
- `data/manifest.json` indexing it all by layer name.

[`gen_yolo_silu_luts.py`](gen_yolo_silu_luts.py) handles the SiLU
non-linearity: the Quark XINT8 graph encodes SiLU as
`Conv → QL → DQ → HardSigmoid → Mul(const) → QL → DQ → Mul(linear) → QL`.
Per Conv we extract the three QuantizeLinear scales + HardSigmoid params
+ constant multiplier, then sample the closed-form HardSiLU over all 256
INT8 inputs to build a 256-byte LUT (`data/model.N/.../silu_lut.bin`).
At runtime each kernel's epilogue is one LUT lookup:
`out_i8 = silu_lut[clamp(banker_srs(acc, rs)) + 128]`.

## Repo layout

```
yolo26n/
├── README.md                          # this file
├── Makefile                           # build + run + time + trace targets (make help)
├── aie2_yolo_iron_partial.py          # chain orchestrator (CHAIN_BLOCKS / CHAIN_N_SAMPLES env)
├── aie2_yolo_per_block.py             # per-block builders (the meat)
├── yolo_spec.py                       # declarative network spec (shapes, ops)
├── placement.py                       # PLACEMENT[block_name] → tile coords + design rules
├── tile_layout.py                     # ASCII tile-map + per-block sizing
├── lowlevel_dma.py                    # StaticWeightStream helper for chunked weight DMA
├── kernels/                           # 30 .cc + 2 .h AIE2P kernel files. Every active kernel has been audited against the playbook in `## Performance` (constexpr trip counts, bias-init mmul, vec to_vector<int8>(rs) SRS, AIE_LOOP_RANGE hints, shape macros). The pack utilities (yolo_m9_{qkv,qk,v}_pack.cc) are scalar strided-copy transposes — no mmul, no vec primitive applies.
├── scripts/
│   ├── m8_megakernel_2tile.py         # m8 2-tile megakernel (the only m8 path)
│   ├── m9_stage.py                    # m9 PSA staged builder (stages 1..10)
│   ├── time_block.py                  # per-block timing harness
│   ├── time_chain.py                  # chain timing harness
│   └── trace_m8.sh                    # m8 HW packet-trace driver
├── gen_yolo_data.py                   # XINT8 ONNX → data/manifest.json + per-Conv bins
├── gen_yolo_silu_luts.py              # XINT8 ONNX → per-Conv 256-byte SiLU LUTs
├── data/                              # gitignored; regenerated by `make data`
├── test_block_ort.py                  # per-block (m0..m8) NPU vs ORT bit-exact host driver
├── test_chain_ort.py                  # full chain NPU vs ORT (N=1 or N>1) bit-exact host driver
├── notebooks/
│   ├── quark_quantization.ipynb       # full int8 recipe (FP32 → XINT8)
│   └── yoloexploration.ipynb          # upstream YOLO26n context + COCO val
├── models/
│   ├── phase1_25k_acc0.9424.pt        # FP32 PyTorch source (3.1 MB)
│   └── phase1_25k_xint8_acc0.8968.onnx # Quark XINT8 deployment artifact (1.7 MB)
└── run_strix_makefile.lit             # lit test (m0 per-block + full chain end-to-end)
```

## Contributing & debugging

### Where to start if you're new

1. **Run something on HW first.** `make BLOCK=m6 run_ort` is the simplest
   end-to-end (small block, vectorized, bit-exact passes in under a minute).
   If that works, your environment is set up.
2. **Read `placement.py`** — the header lists the per-block tile budget,
   and the **"DESIGN RULES"** comment block (lines 33–~150) captures the
   hard-won lessons (deadlock patterns, via_DMA gotchas, kernel-arg
   stride bugs, etc.). Worth reading once cover-to-cover.
3. **Run `python3 tile_layout.py`** for an ASCII map of the 32 compute
   tiles + which block sits where.
4. **Pick a per-block builder in `aie2_yolo_per_block.py`** and read its
   preceding block-comment header — those describe each block's
   architecture, fan-out, and kernel inventory.

### Adding an optimization to an existing block

1. **Baseline.** `make BLOCK=mN time TIME_ARGS="--n-warmup 5 --n-iters 100"`
   and write down the median ms.
2. **Bit-exact gate.** `make BLOCK=mN run_ort` must pass after every
   step (m9/m10 only via `make run_chain`).
3. **Trace if needed.** `make trace` builds the m8 megakernel with HW
   packet trace ops and prints the per-tile cycle summary. For other
   blocks, copy `scripts/trace_m8.sh` and adapt; the per-block trace
   plumbing in `aie2_yolo_per_block.py` already honors
   `TRACE_SIZE_PER_WORKER` and `TRACE_EVENTS`.
4. **Iterate.** `make clean` is rarely needed for `.cc`/`.py` edits;
   Make's mtime tracking is correct for those. Run `make clean` if you
   edit the Makefile itself or change a `TRACE_*` env var.

### Adding a new kernel

1. Drop a `_vec.cc` under `kernels/` (vectorized AIE2P API style — see
   `yolo_m8_back_cv3_cv2_fused_vec.cc` for a non-trivial example with
   chunked-OC, fused requant, SiLU LUT, and skip-add).
2. Wire a build rule in the **kernel build rules** section of the
   Makefile (search for "c3k2_heavy streamed inner-pair kernels" for the
   `$(foreach blk, ... $(eval ...))` pattern). Add the `.o` to
   `KERNEL_OBJS`.
3. Declare it as a `Kernel(...)` in the relevant builder
   (`aie2_yolo_per_block.py` or one of `scripts/*.py`) with arg types
   matching the C signature exactly.
4. Call it from a `Worker` body. Pass `aie::int8_t*` args as IRON
   `B._i8(shape)` types; scalar ints as `np.int32` (or `np.uint32` if
   the kernel wants to divide / shift on them — see below).

### Block ablation (bottleneck attribution)

When the chain is slower than the per-block sum suggests it should be (or
faster — both happen in pipelined chains), per-block standalone timing
can mislead. Use the `NOOP_BLOCK` Make env var to compile any subset of
blocks' kernels with `-DNOOP_KERNEL`. Their `extern "C"` entries
early-return, so the chain's DMA / lock / placement pattern is unchanged
but those blocks contribute ~0 ms of compute. Output is garbage —
bit-exactness fails by design; the signal is chain wall-clock.

```bash
# Ablation: how much does m9 contribute to the chain?
rm -f build/yolo_m9_*.o && make clean_chain
CHAIN_N_SAMPLES=15 NOOP_BLOCK=m9 make time_chain
# Historical (pre-m9-1x1-vec arc): 317 ms baseline → 57 ms noop'd ⇒ m9 = 260 ms / 82%
# Today (post-cv2 vec):              75 ms baseline → ? ms noop'd  ⇒ rerun to attribute the new remainder

# Combine: noop multiple blocks (space-separated, quoted)
rm -f build/yolo_m9_*.o build/yolo_m10_*.o && make clean_chain
CHAIN_N_SAMPLES=15 NOOP_BLOCK="m9 m10" make time_chain
```

Notes:

- All 11 blocks (m0..m10) are supported.
- The `-DNOOP_KERNEL` flag only affects compile of that block's kernel
  `.o` files. Make won't rebuild a `.o` when only `NOOP_BLOCK` changes
  (mtime tracking misses it), so manually `rm` the affected `.o` plus
  `make clean_chain` between runs. For unsure cases, just `make clean`.
- The harness adds zero overhead when `NOOP_BLOCK` is unset — the
  `#ifdef NOOP_KERNEL return;` guards compile out completely.
- Output garbage from noop'd blocks doesn't break the chain — downstream
  blocks just consume junk inputs. Chain DMA scheduling is unaffected.

**Finer-grained per-kernel ablation inside a c3k2 block** — when
NOOP_BLOCK shows a c3k2_small / c3k2_heavy block as the bottleneck,
pass `NOOP_KERNEL_FILES="<obj-stem> ..."` to noop only specific
kernels within it. Useful for picking which tile to deep-opt first
without rebuilding the whole block:

```bash
# Standalone ablation: which tile in m2 dominates?
rm -f build/yolo_c3k2_small_*_m2.o build/final_m2.xclbin build/insts_m2.bin
NOOP_KERNEL_FILES="yolo_c3k2_small_cv2_concat3_m2" make BLOCK=m2 time
# m2 baseline 27.75 ms → noop'd 18.70 ms ⇒ cv2_concat3 = 9.05 ms

# Multiple kernels (space-separated, quoted)
rm -f build/yolo_c3k2_small_*_m2.o build/final_m2.xclbin build/insts_m2.bin
NOOP_KERNEL_FILES="yolo_c3k2_small_m0_cv1_m2 yolo_c3k2_small_m0_cv2_skip_m2" \
  make BLOCK=m2 time
```

This is how the m2 deep-opt arc found cv2_concat3 as the first bottleneck
(9 ms), then m0_cv2_skip after the cv2 vec epilogue (7 ms surfaced post-cv2),
then cv1_split (4.5 ms surfaced post-m0_cv2_skip). Per-kernel attribution
between c3k2 sub-kernels is wired into the eval rules for
`yolo_c3k2_small_{cv1_split,m0_cv1,m0_cv2_skip,cv2_concat3}_{m2,m4,m6}.o`.

This is how the m9-is-the-bottleneck finding was discovered: the
optimization arc that took m3/m5/m7 from 186 ms → 19 ms standalone moved
the chain by only ~14 ms, and ablating each block individually (m0
alone, m1 alone, the 6 conv_stride blocks together) showed each
contributed <1 ms. Only m9 ablation moved the chain meaningfully. The
same harness then surfaced cv2 as the m9 critical-path kernel during
the m9 1×1 vec arc.

### Common debugging patterns

- **Build hangs, no error**: check the build log for **L1 overflow**
  warnings (e.g. `Failed to allocate buffer "X" with size: NNNN bytes`,
  `Bank-aware allocation failed`). The xclbin may still build but you
  used too much L1; reduce per-tile buffer depth or stream weights from
  the memtile via `StaticWeightStream`.
- **Bit-exact failure with a partial set of mismatches**: usually a
  kernel signature mismatch (IRON Kernel arg types vs C function args),
  or a stride mistake in a memref index. The kernel signature must match
  positionally — IRON doesn't type-check across the C boundary.
- **`make` builds something stale**: if you've edited the Makefile or
  changed a `TRACE_*` env var, run `make clean`. `CHAIN_N_SAMPLES`
  changes are handled automatically by a stamp file.
- **Runtime timeout (`ERT_CMD_STATE_TIMEOUT`)**: a deadlock. Likely
  causes — see `placement.py` DESIGN RULES section. Most common is a
  cons-side `acquire(N)` with N>1 emitting an `AcquireGE,N` that blocks
  the chain ahead of it. Smallest acquire first, or bump fifo depths.
- **`__divsi3` in a kernel `.o`** (visible via `llvm-readobj -r build/foo.o`):
  Peano can't prove signed-int division by a power of 2 is shift-eligible.
  Cast the dividend to `(uint32_t)` and the divisor to `Nu` — `/8`
  becomes `>> 3`. (Cleanup-only, not a major perf lever.)
- **Looking at the compiled `.o` directly**: `llvm-objdump -d build/foo.o`
  to see instructions; `grep -E '\bv[a-z]+\b'` for vector ops; the
  presence of `vmac`, `vldb`, `vmov` means you got vectorization. If
  the kernel was supposed to vectorize and only has scalar `mac/mov/add`,
  check the surrounding loop for branches that might be inhibiting the
  vectorizer.

### Adding a new block builder

1. Add a `_build_mN(...)` function in `aie2_yolo_per_block.py` following
   the pattern of an existing block of similar topology (conv_stride →
   `_build_conv_stride_block`, c3k2 → `_build_c3k2_small/heavy`,
   attention → `_build_psa` or the staged variant in `scripts/m9_stage.py`).
2. Register it in the `_BUILDERS` dict near the bottom of
   `aie2_yolo_per_block.py`.
3. Add the block to `yolo_spec.NETWORK` so the layer shapes / kernel
   parameter inference works.
4. Add a `PLACEMENT[block_name]` entry (or delegate to a staged builder
   that hardcodes its own coords, like `m8_megakernel_2tile.py`).

## Known limitations and follow-ups

- **PSA m9 multi-sample**: end-to-end multi-sample dispatch already
  works via `CHAIN_N_SAMPLES=N` (HW BD-chain replays the per-sample
  transfer N times in one XRT call). What's *not* done: pushing the batch
  dimension into the PSA m9 kernel signatures so the attention matmuls
  compute on `(B, nh, S, d)` natively instead of N sequential
  `(nh, S, d)` tensors. Whether it would pay off depends on the current
  per-sample breakdown — m9 isn't the top contributor today.
- **Trace flow tile cap**: AIE2 trace packet IDs cap simultaneously-
  traced workers at 31 — the full chain (26 compute tiles) fits, but
  larger designs may need front-half / back-half traces (the
  `TRACE_BLOCKS=m0,m1,...` env in `aie2_yolo_iron_partial.py` handles
  this).
- **m8 megakernel scalar skip-row copy**: the worker body has a small
  IRON-emitted `for x: for kk: sk_r[x,0,kk] = sa_mid[x,0,kk]` to forward
  one row from split_a to pair0_skip (and same for pair1). The cost is
  small relative to the conv but folding it into `k_pair_cv1` as a second
  vectorized output would be cleaner. See the TODO in
  `scripts/m8_megakernel_2tile.py:_do_p0c1`.

## References

- [`programming_examples/ml/mobilenet/`](../mobilenet) — the reference
  IRON design this example parallels in structure
- [AMD Quark documentation](https://quark.docs.amd.com/) — the int8
  quantization library used by `notebooks/quark_quantization.ipynb`
- [Ryzen AI Software Stack](https://ryzenai.docs.amd.com/) — the Vitis
  AI Execution Provider path for running the same XINT8 ONNX on Ryzen AI
  through ONNX Runtime
- [Ultralytics YOLO26](https://docs.ultralytics.com/) — upstream model
