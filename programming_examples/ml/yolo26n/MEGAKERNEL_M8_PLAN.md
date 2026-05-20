# m8 megakernel: v2 design plan

## Goal

Replace m8's 8-tile pipeline (623 ms, 0.1% utilization) with a
**single-tile megakernel** running all 9 sub-ops sequentially on (5,3),
using vertically-adjacent neighbors (5,2) and (5,4) to host large
static weights via `delegate_tile`. Eliminates **all** inter-tile
fifos and pipeline fill/drain.

## Template: mobilenet `bottleneck/regular.py`

Direct precedent at `programming_examples/ml/mobilenet/bottleneck/regular.py:474-492`:

```python
def _of(ch, depth):
    return ObjectFifo(
        _u8((in_w, 1, ch)),
        depth=depth,
        disable_synchronization=True,  # one core, sequential order = implicit sync
        delegate_tile=alloc_tile,       # spill to chosen tile's L1
    )
```

Single Worker with self-loop ObjectFifos providing sliding 3-row
windows for the 3×3 convs. We copy this skeleton and substitute m8's
sub-op composition.

## Why we don't write a new .cc kernel

Every m8 sub-op already has a **per-row vec kernel** that takes 1-3
input rows and produces 1 output row. We compose them inside the
megakernel Worker as sequential `k_*(...)` calls — no new C code.

| Sub-op | Existing kernel | Source variant |
|---|---|---|
| cv1 (1×1, 256→256 split) | `yolo_c3k2_small_cv1_split_streamed_silu_bias_i8_i8` | streamed |
| m_0_split (1×1 ×2) | `yolo_c3k2_heavy_m_0_split_silu_bias_i8_i8` | static |
| pair0_cv1 (3×3) | `yolo_c3k2_heavy_inner_pair_cv1_conv2dk3_silu_bias_i8_i8` | static (used for pair1 today) |
| pair0_cv2 (3×3+skip) | `yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_silu_bias_i8_i8` | streamed |
| pair1_cv1 (3×3) | `yolo_c3k2_heavy_inner_pair_cv1_streamed_conv2dk3_silu_bias_i8_i8` | streamed |
| pair1_cv2 (3×3+skip) | `yolo_c3k2_heavy_inner_pair_cv2_skip_silu_bias_i8_i8` | static |
| cv3 (1×1 concat2) | `yolo_c3k2_heavy_cv3_concat2_silu_bias_i8_i8` | static |
| cv2 (1×1 concat3) | `yolo_c3k2_small_cv2_concat3_streamed_silu_bias_i8_i8` | streamed |

All vectorized (mmul<4,8,8>) and bit-exact today. Megakernel = Python
orchestration; no kernel-C rewrites.

## L1 budget (corrected)

AIE2P compute tile = 64 KB total. Subtract ~8 KB program + stack.
**Usable per-tile: ~56 KB.**

### Weight inventory

| Layer | Bytes |
|---|---:|
| cv1 (256→256, 1×1) | 65 536 |
| cv2 (384→256, 1×1, 3-concat input) | 98 304 |
| m_0_split cv1 (128→64) | 8 192 |
| m_0_split cv2 (128→64) | 8 192 |
| m_0_cv3 (128→128) | 16 384 |
| pair0_cv1 (64→64, 3×3) | 36 864 |
| pair0_cv2 (64→64, 3×3) | 36 864 |
| pair1_cv1 (64→64, 3×3) | 36 864 |
| pair1_cv2 (64→64, 3×3) | 36 864 |
| **Total** | **~344 KB** |

### Allocation across (5,2) / (5,3) / (5,4)

Two of the four pair convs go static on neighbors; the other two stream
from memtile alongside cv1 and cv2.

| Tile | Static weights | KB |
|---|---|---:|
| (5,2) south neighbor | pair0_cv1 (36) + m_0_split cv1 (8) + m_0_split cv2 (8) | 52 |
| (5,4) north neighbor | pair1_cv2 (36) + m_0_cv3 (16) | 52 |
| (5,3) compute | sliding-window OFs (~32) + I/O fifos (~8) + active stream chunks (~8) + LUTs+biases (~6) | ~54 |
| Streamed from memtile per sample | cv1 (64) + pair0_cv2 (36) + pair1_cv1 (36) + cv2 (96) | 232 (DMA, not L1) |

Each tile under the 56 KB ceiling. Neighbor cores stay idle — their
DMEMs are pure storage that (5,3)'s kernel reads via shared L1.

### Sliding-window OF sizes

| OF (self-loop on (5,3), delegate to (5,2/3/4) as needed) | depth | size |
|---|---:|---:|
| top (cv1 → cv2) | 2 | 4 KB |
| bot_to_cv2 (cv1 → cv2) | 2 | 4 KB |
| bot (cv1 → m_0_split) | 2 | 4 KB |
| split_a (m_0_split → pair0_cv1) | 3 | 3 KB |
| split_b (m_0_split → cv3) | 2 | 2 KB |
| pair0_mid (pair0_cv1 → pair0_cv2) | 3 | 3 KB |
| pair0_skip (pair0_cv1 → pair0_cv2) | in_h | 16 KB ← biggest |
| pair0_out / inner_0_out (pair0_cv2 → pair1_cv1) | 3 | 3 KB |
| pair1_mid (pair1_cv1 → pair1_cv2) | 3 | 3 KB |
| pair1_skip (pair1_cv1 → pair1_cv2) | in_h | 16 KB |
| pair1_out (pair1_cv2 → cv3) | 2 | 2 KB |
| cv3_out (cv3 → cv2) | 2 | 2 KB |

Two `pair*_skip` fifos at in_h=16 depth are the big ones (each 16 KB).
Both can be delegated to neighbor tiles to keep (5,3)'s budget under
control. Probably:
- pair0_skip → delegated to (5,2)
- pair1_skip → delegated to (5,4)

Then (5,3)'s sliding-window OF total is ~24 KB.

## Per-sample wall-time estimate

- Compute: 24M MACs / 32 MAC·cycle / 1 GHz = **750 us** at peak vec; realistic ~1.5 ms at 50% utilization
- Weight DMA: 232 KB / 500 MB/s = **464 us** (or worse under memtile contention — see memtile_bw_bench)
- I/O DMA: ~100 us

**Estimated per-sample: 1.5-3 ms** vs current 623 ms = **200-400x speedup**.

For 60 fps target (16.7 ms/sample budget across 11 blocks), m8's
1.5-3 ms is well under fair-share. The chain bottleneck would move to
m6 (280 ms) — same megakernel treatment can be applied.

## Open risks

1. **Program memory budget (16 KB)** on (5,3) — 9 sub-ops worth of
   kernel code may overflow. The orchestration is Python-side so it's
   not in tile progmem. Each `k_*` is a separate .o file linked to its
   call site; total text on the tile is the sum of the called kernel
   .o text sizes. If it overflows, fallback: split into 2 megakernel
   tiles (e.g., (5,3) cv1+split+pair0, (5,4) pair1+cv3+cv2) connected
   by a single inter-tile fifo — 2-stage pipeline.

2. **Memtile DMA throughput under contention** — answered by the
   running `scripts/memtile_bw_bench.py` agent. If sustained < 200 MB/s,
   the 464 us DMA estimate becomes ~1 ms and we need to consider
   keeping more weights static (drop cv1 streaming, place those weights
   on a memtile-staging buffer that loads once at design init).

3. **IRON `delegate_tile` correctness** for cross-tile static Buffers —
   verified by mobilenet using the same pattern. No risk.

4. **Compile time** — 9 kernels linked into one worker may slow build.
   Acceptable; this is build-time only.

## Bit-exactness gate

1. `M8_MEGAKERNEL=1 make BLOCK=m8 run_ort` → 0 mismatches
2. `M8_MEGAKERNEL=1 make chain run_chain` → 0 mismatches

## Implementation plan

1. **Write `scripts/m8_megakernel.py`** — new builder mirroring mobilenet
   `regular.py`. Single Worker on (5,3), all OFs self-loop with
   delegate to (5,2)/(5,4)/(5,3), static weight Buffers on neighbors,
   StaticWeightStream for the 4 streamed weight sets, sequential
   per-row kernel calls inside `for _ in range_(in_h)`.
2. **Hook into `scripts/m8_stage.py`** — at the top of `build()`, if
   `M8_MEGAKERNEL` env is set, delegate to `m8_megakernel.build()`
   and return. Leaves the 8-tile path intact as the fallback.
3. **Build + bit-exact verify** — `make BLOCK=m8` standalone first;
   then chain.
4. **Measure** — record m8 standalone latency, chain throughput.
5. **Stretch goals if perf disappoints**:
   - Profile which sub-op dominates; cascade-split that one only
   - Apply same megakernel pattern to m6 (next chain bottleneck)
