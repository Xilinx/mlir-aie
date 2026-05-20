# m8 megakernel: design plan (v3 — post-Path-X findings)

## TL;DR for the next session

m8 single-tile megakernel is achievable but requires writing fused C
kernels (Python-side IRON orchestration blows the 16 KB program memory
cap on its own). Plan: 3 fused C kernels + simplified Worker.

Status: **m8_back done** (cv3+cv2 fused, 4.9 KB .o); m8_front and m8_pair
remaining; Worker rewrite is the final step.

## What we proved out

### Constraints walked (all fit on 3 vertically-adjacent tiles)

| Constraint | Limit per tile | Solution |
|---|---:|---|
| L1 SRAM | 64 KB | `delegate_tile` on every sliding-window OF + cv2 stream recv_buf, spread across (5,2)/(5,3)/(5,4) |
| DMA input channels | 2 | `StaticWeightStream(compute_placement=neighbor)` lands recv_bufs on (5,2)/(5,4); ping-pong of pair0_cv1+cv2 and pair1_cv1+cv2 (same chunk size, both 18 KB) collapses 6 streams → 4 |
| Program memory | 16 KB | Fuse multi-op orchestration into single C kernels (the open work) |

### Verified IRON capabilities (saved to memory)

- ✅ `StaticWeightStream(compute_placement=neighbor)` — recv_buf lives on a non-Worker tile, Worker reads via shared L1. Used for ws_cv1/ws_pair0 on (5,2), ws_pair1/ws_cv2 on (5,4).
- ✅ `ObjectFifo(delegate_tile=neighbor, disable_synchronization=True)` — self-loop sliding-window OFs spilled to neighbor L1. Matches mobilenet `bottleneck/regular.py:474-492`.
- ✅ Same-pair ping-pong via `StaticWeightStream(ping_pong_buf=...)` — two streams on one channel when chunk sizes match.

### Known IRON limitations (also saved to memory)

- ❌ `Buffer(tile=neighbor)` passed to a different-tile Worker — rejected at construction. Use lockless OF + init_values workaround, OR (for weight data) StaticWeightStream's placed-dialect path.
- ❌ IRON-Python helper functions cost ~5 KB ELF text each after lowering. Single-tile orchestration tops out around 3 sub-ops worth of helpers before busting 16 KB program memory.

## Final design — 1 tile compute, 3 fused C kernels

```
Tile A (5,3) megakernel Worker calls 3 fused kernels per row:
  k_m8_front(...)   →  cv1 + m_0_split            (chunks N_CV1_CHUNKS=8)
  k_m8_pair(...)    →  pair_cv1 + pair_cv2 + skip (used twice: pair0, pair1)
  k_m8_back(...)    →  cv3 + cv2                  (chunks N_CV2_CHUNKS=8)
```

### Weight & I/O layout (verified to fit)

| Tile | Resources |
|---|---|
| (5,3) compute | act_in OF (ch 0), m_0_split static (16 KB), m_0_cv3 static (16 KB), small sliding OFs (split_a/p0_mid/p0_skip on south, others on north via delegate), all biases + LUTs |
| (5,2) south | ws_cv1 recv (ch 0, 8 KB), ws_pair0 ping-pong recv (ch 1, 18 KB), delegated OFs: top/bot/bot_to_cv2/split_b/p0_*  |
| (5,4) north | ws_pair1 ping-pong recv (ch 0, 18 KB), ws_cv2 recv (ch 1, 12 KB), delegated OFs: bot_to_cv2/cv3_out/inner_*/p1_* |

Three memtiles supply weight streams: (5,1)→ws_cv1, (4,1)→ws_pair0, (6,1)→ws_pair1, (3,1)→ws_cv2.

## Fused kernel design notes

### m8_back (DONE — kernels/yolo_m8_back_cv3_cv2_fused_vec.cc)

Combines cv3 (1×1 concat2, 128 oc) + cv2 (1×1 concat3, 256 oc chunked) per call.
- On cv2 chunk_idx == 0: recompute cv3 → file-scope static `s_cv3_out[16 × 128]`
- Always: compute this cv2 chunk using `s_cv3_out` as the m0 input
- Compiles to 2864 bytes text + 2048 bytes bss

### m8_front (TODO — similar pattern to m8_back)

cv1 (1×1 split, 256 oc chunked, first-half→top, second-half→bot/bot_to_cv2) + m_0_split (1×1×2, bot→split_a + split_b).
- Static scratch needed: `s_bot[16 × 128]` (2 KB) holds bot during cv1 chunk loop
- On cv1 chunk_idx == 0: clear s_bot
- Always: do this cv1 chunk, writing to top OR (s_bot AND bot_to_cv2)
- On cv1 chunk_idx == N_CV1_CHUNKS-1: bot is complete, run m_0_split using s_bot → split_a + split_b

Estimated kernel size: ~3-4 KB text + 2 KB bss.

### m8_pair (TODO — hardest; sliding window across calls)

pair_cv1 (3×3, 64→64) + pair_cv2 (3×3 + skip, 64→64). Used for both pair0 (with pair0 weights) and pair1 (with pair1 weights — same kernel symbol).

Sliding window challenge: pair_cv2(row r) needs pair_cv1(rows r-1, r, r+1). Within ONE kernel call (per row), we only have pair_cv1's row r output. So pair_cv2 can only run when we've accumulated 3 rows of pair_cv1 output.

Two approaches:
1. **Per-call computes pair_cv1 only**; pair_cv2 runs separately on a different call (defeats fusion purpose for pair_cv2).
2. **Two static scratch arrays** hold the rolling 3 rows of pair_cv1 output AND pair_cv1 skip rows. Per call:
   - Always: compute pair_cv1(current row), write to scratch slot (current_iter % 3)
   - If iter ≥ 2: compute pair_cv2 using scratch rows (current-2, current-1, current) + skip
   - Border handling: rows 0, in_h-1 replicate via static-array index management

Approach 2 is the right one — keeps everything in one kernel call per row, manages sliding internally. Estimated 4-5 KB text + 6 KB bss (pair0_mid 3×rows + pair0_skip + pair1_mid + pair1_skip, but only one set live at a time since pair0 and pair1 calls are sequential).

Actually simpler: m8_pair uses a `static int s_iter_idx` counter to track its own row position. Borders by checking `s_iter_idx == 0` or `s_iter_idx == in_h - 1`. Tile holds 3 + 3 + 3 + 3 = 12 KB of scratch across both pair calls (since calls are sequential, scratch could share).

### Worker rewrite

After all 3 fused kernels exist, m8_megakernel.py's Worker body simplifies from 6 helpers + preamble/postamble (~37 KB ELF) to 3 kernel calls in a single `for _ in range_(in_h)` loop (~10-15 KB ELF expected). Borders handled inside kernels.

## Next-session starting point

```bash
# Pick up here:
git log --oneline -5     # confirms the m8_back commit landed
ls programming_examples/ml/yolo26n/kernels/yolo_m8_back_cv3_cv2_fused_vec.cc

# Next two kernels to write (in order):
# 1. kernels/yolo_m8_front_cv1_split_fused_vec.cc — easier, mirrors m8_back pattern
# 2. kernels/yolo_m8_pair_two_3x3_skip_fused_vec.cc — sliding window in static scratch

# Then rewire:
# 3. scripts/m8_megakernel.py — replace 6-helper Worker with 3-helper Worker
# 4. M8_MEGAKERNEL=1 make BLOCK=m8 run_ort   (bit-exact gate)
```

## What this means for chain perf

If m8 megakernel lands at the estimated ~3 ms/sample (compute + DMA on 1
tile, no fill/drain), m8 drops from 623 ms to ~3 ms — a 200× win. The
chain bottleneck moves to m6 (280 ms) or the rest of the structural
floor. **Same megakernel pattern (3 fused C kernels per tile) applies
to m6.**

Target chain throughput after both m6 and m8 megakerneled: low tens of
ms per sample → potentially 60+ fps batched.
