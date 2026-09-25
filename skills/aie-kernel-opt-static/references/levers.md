<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Static signals, levers, and how well they predict hardware

This file answers one question: **which static signal predicts a hardware
win, and how reliably?** Every row comes from a case where a static signal
was recorded and the same change was then measured on hardware (traced core
cycles per call on AIE2P, npu2, unless marked otherwise). Misses are listed
next to the hits.

Nothing in this file is a speedup claim for *your* kernel. The HW numbers are
precedents, measured by `aie-kernel-opt-hw` on other kernels. A candidate you
produce here is unconfirmed until that skill measures it.

## Signal → HW outcome

| # | Static signal (before → after) | Kernel, change | HW outcome (cycles per call) | Verdict |
|---|---|---|---|---|
| S01 | Unroll screen: loop `byte_count` flat from ×1 to ×4, no new `[sp, #` | `add`, `mul`, `leaky_relu`, `gelu`, `UNROLL(4)` (L07) | 390 → 150, 454 → 142, 298 → 86, 594 → 318 | hit, 4 of 4 |
| S02 | `__divsf3`, `__floatsisf` gone from the `libcalls` row; the loop now has an II | `rms_norm` (L02) | 1597 → 438 | hit |
| S03 | `__divsi3` gone; unsigned trip count | `axpy` (L11) | 317 → 178 | hit |
| S04 | `[sp, #` traffic around a counter-indexed accumulator array gone | `conv2dk1_i8`, `UNROLL_FULL` (L01) | 7623 → 504 | hit |
| S05 | Per-call prediction `bundles + 5 + (trips-1) × II` from the object | `zero` cursors (L06); `mm_bfp_mixed` (L10) | predicted ~75 / ~130, measured 78 / 134; predicted 1348 / 1312, measured 1349 / 1313 | hit, including magnitude |
| S06 | `unpipelined_loops` 6 → 2 with the inner II unchanged at 35 | `fused_mm` z loop `UNROLL(2)` (L10) | k_step 216 → 193 | hit: an outer loop that stops paying entry/exit can win with no II change |
| S07 | Epilogue II30 per 16 lanes → II19 per 64 | `fused_mm` epilogue (L03, L04, L06) | chunk 126 → 19 | hit |
| S08 | II67 per 128 values → II17 per 64, `[sp]` refs 26 → 4 | `q4nx_dequant` (L19) | 4719 → 2245 (2.10x) | hit; the per-value II ratio (-49%) matched HW (-52%) |
| S09 | Final II 17 → 16 on a 128-trip loop; pre-RA II 16 → 11/12; `ns` rises | `q4nx_dequant` stage cap 5 (L17) | 2245 → 2115 (-5.8%) | hit **only if scaled by trips**: 128 × 1 cycle predicts ~2117. The pre-RA II over-promised. A 1-cycle II move looks like nothing and was worth 130 cycles |
| S10 | II125 → 37, frame 0x840 → 0x700 | prefill `fv`, paired accumulator (L18) | 2034 → 1265 (-35%) | hit in direction; the II ratio (-70%) over-predicts because the call does other work |
| S11 | `[sp]` refs 22 → 10, `.text` -32..-80 B, II unchanged at the mv-slot bound | `mm` cursors (L06) | int16 -4.2%, bf16 -0.8% | hit in direction; small where the loop sits at its bound |
| S12 | Compiler reports the K unroll worse (II35 → 156, spills) | bf16 / int16 `mm` (X24) | unchanged | hit: predicted no gain, got none |
| S13 | Compiler reports the z loop at II119 after an i-loop full unroll | `fused_mm` (X40) | k_step 216 → 287 (`aie-kernel-opt-hw` X12) | hit: a compiler-reported loss was a HW loss |
| S14 | II33 → 18 on the edited function | gelu in-place variant | 594 → 594 (`aie-kernel-opt-hw` X01) | **miss**: the benchmark never calls that symbol. Resolve the called symbol first |
| S15 | II37 → 31 from an opaque `add_2d` pointer bump that keeps S loads in the loop | prefill `fv` | 1265 → 1374-1576, slower (`aie-kernel-opt-hw` X13) | **miss**: a better II, a slower kernel |
| S16 | Object `.text` halved by making a helper `static` | any (X39) | 0 B change in the core ELF | **miss** on size: measure the entry symbol, not the object |
| S17 | Per-row unpipelined `reduce_add` (II125 at K=64) replaced by a 4-row group body at II56 / II90 (14 / 22.5 per row); max loop II *rose* from II10; `.text` +48..576 B; `unpipelined_loops` unchanged at 12 | bf16 `mv`, four rows per transposed tree (L20) | 1154 → 788 (-31.7%); amd/IRON's attention shape (4x64, `vec_size` 32) 159 → 110. Ablation had priced the reduction at 520 of 1154 | hit **only when II is normalized per unit of work**: the raw max II went up. Divide by rows (or values) per iteration before you class a change `reject` |
| S18 | Mac loop II9 → II7 from per-row cursors, K=2048 | bf16 `mv` 4x2048 (L20) | 420 → 408 (-2.9%) | hit in direction, **miss in magnitude**: rows 4 KB apart likely contend for banks, which remarks can't see |
| S19 | `[sp` refs 73 → 4 (the rest prologue/epilogue), `unpipelined_loops` 2 → 0, group loop II58; max II *rose* from 15 (per k step) to 58 (per 2x2 group) | `mm_bfp` one A, one B and one C stream (L19) | 4243-4529 → 951 (4.5x); odd-K 32x24x48 850-894 → 269. II58 × 16 groups + 23 cycles of call and prologue = 951, the traced value | hit, including magnitude. The strongest spill-count row: the in-loop `[sp` count predicted the win; the raw max II pointed the other way |
| S20 | Max II 13 → 48 on `partial_softmax`: the 8-row fold group has MII 38, over `SwpMaxMii` 27, so only the postpipeliner runs it (47 bundles per 8 rows). `.text.partial_softmax` 4496 → 6880 B, frame 64 → 192 B, object `unpipelined_loops` unchanged at 74 | `mha` flash-decode softmax, all rows per pass and eight rows per fold tree (L20) | full block 8171 → 1696 (4.82x), causal diagonal 10912 → 2162 (5.05x), padded 5107 / 6710 → 1448 / 1740. The `mha` bench cases stayed at 10001 and 9913: none calls `partial_softmax` | hit **only when normalized per row**, as S17. A loop the pipeliner gave up on (P13) was still the win |
| S21 | `noinline` straight-line body 489 → 423 bundles per 32 lanes; `[sp` 62 → 67, frame 768 → 832 B; the calling loop's II unchanged at 13 | `exp2f_vec` (AIE2): the final `p * 2^k`, an emulated f32 multiply, becomes an integer add into p's exponent field | 16077 → 13997 (-12.9%); predicted 32 calls × (423 + 13) = 13952 | hit, including magnitude: for a `noinline` body, callee bundles × calls predicts the traced number |
| S22 | Body 423 → 346 bundles per 32 lanes; `[sp` 67 → 150, frame 832 → 1792 B, over the 1024 B default | `exp2f_vec` (AIE2): Horner's five dependent emulated f32 multiplies → Estrin's three (L09); contract `stack_bytes` 2048 on AIE2 | 13997 → 11981 (-14.4%); predicted 11488 | hit, magnitude 4% short: the added spills stall beyond their bundles. Issuing `f^4` after the three linear terms, same 346 bundles, measured 12205 → 11981 |
| S23 | Frame 416 → 0 B; loop II tanh 70 → 28 (NS 2), sigmoid 84 → 62, silu 104 per 32 lanes → 189 per 64 (LLVM now unrolls it 4x, not 2x), gelu and swiglu unchanged at 183 and 139 | `getTanhBf16` (AIE2, npu1): `aie::linear_approx`, whose scratchpad member sent the object to the stack on every call, written out as an integer clamp, `load_lut_2x_float` and one `mac_elem_16_2`; bit-identical on all 65536 bf16 inputs | tanh 2243 → 911, sigmoid 2691 → 1991, silu 3330 → 3031, silu llama-prefill 13372 → 12134; gelu 5859 → 5865, swiglu 4449 → 4455 | hit, including magnitude: trips × II gives 2240 → 896, 2688 → 1984, 3328 → 3024. A raw II that *rose* (silu) was a win once divided by the lanes per trip |
| S24 | Loop II84 per 16 vectors (4 chains, unrolled 4x more by LLVM: MII 32 > `SwpMaxMii` 27, postpipeliner only, no overlap) → II1 per vector, NS 14; `.text` 368 → 304 B | `add` (AIE2, npu1): `__restrict` on the pointer params, one chain per iteration under `AIE_LOOP_NO_UNROLL`, and b added as b × 1 in a `vmac.f` so only a needs the a-port-only `vlda.conv`. `begin_restrict_vector` alone does not give LLVM noalias | 342 → 78 (4.4x); `add_sized` 2048 678 → 142, 256 83 → 30; bit-identical on all 7 data cases. `aie::add` in the same loop (two `vlda.conv`, II2) 141; `__restrict` alone 197 | hit, including magnitude: 63 trips at II1 plus prologue and epilogue predicted ~83 |
| S25 | Loop II76 per 16 vectors → II1 per vector, NS 14; `.text` 368 → 304 B | `mul` (AIE2, npu1): as S24, `__restrict` plus one `aie::mul` chain per iteration under `AIE_LOOP_NO_UNROLL` | 310 → 78 (4.0x); `mul_sized` 4096 1222 → 270, 2048 614 → 142, 256 87 → 30; bit-identical on all 7 data cases. `__restrict` alone (4 chains, II16 per 16 vectors) matched it at 1024-4096 and gave 39 at 256 | hit: the missing `__restrict`, not the chain count, was worth almost all of it |
| S26 | Frame 1408 → 0 B, `unpipelined_loops` 16 → 1, ic loop II61 → II34 (NS 2, prologue 36, epilogue 10) for 24 `vmac`s; program memory 2816 → 800 B | `conv2dk3` (AIE2, npu1): the separate left-border, middle and right-border passes over 4-pixel blocks become one pass per output row with 8 accumulators; the kx = 0 and 2 windows are `shuffle_down_fill` (one `vshift`) of 64 B registers, and the borders are the edge blocks' zero fill. One template serves int8 and uint8 | int8 99136 → 6985 (14.2x), uint8 29127 → 6978 (4.2x); bit-identical. Two passes of 4 accumulators: II19 + II16 per ic, no better; weight-major mac order: II44, 256 B of spills | hit, including magnitude: (36 + 7 × 34 + 10) per ky × 3 ky × 8 oc = 6816 predicted. The remaining gap to the resource MII (~24) is register pressure: regalloc reports loop spill/reload copies |
| S27 | Loops II64 (mul) and II60 (add) per 4 vectors → II1 per vector, NS 14; `.text` 480 → 640 B | `mul_add` (AIE2, npu1): as S24 and S25. The base already had `__restrict` locals copied from the parameters and still did not pipeline; the change puts `__restrict` on the parameters | mul 1042 → 85 (12.3x), add 969 → 85 (11.4x); bit-identical on all 5 data cases | hit, including magnitude: 63 trips at II1 plus about 20 cycles outside the loop |
| S28 | Loop II18 per 2 × 32 lanes, NS 2 (index addressing `a + i`, unrolled 2x, "Unable to find schedule") → II2 per 32 lanes, NS 6: two 256-bit stores per 32 lanes is the store-port bound | `relu` (AIE2, npu1): walk `v32bfloat16` pointers instead of indexing, under `AIE_LOOP_NO_UNROLL` | 294 → 75 (3.9x); bit-identical on all 5 data cases | hit, including magnitude: 32 trips × II2 + 11 |
| S29 | Loop II15 per 64 lanes, not ZOL → II1 per 16 lanes, NS 14, for rows of 256 or more; II4 per 16 lanes below that | `axpy` (AIE2, npu1): y loaded straight into the accumulator (a-port `vlda.conv`), x on the b port, one `vmac.f` per 16 lanes. A runtime trip count needs `AIE_LOOP_MIN_ITERATION_COUNT(16)` before the 14-stage schedule is used | 269 → 87 (3.1x). Bit-identical except on subnormal data, where 104874 / 262144 flushed zeros change sign (the reference's sign matches in 7225 vs 3823). Rejected: one loop with `MIN_ITERATION_COUNT(4)`, II4 per 16; four chains per trip, II10 per 64, not ZOL | hit, including magnitude: 63 trips at II1 plus 24 |
| S30 | Loop II24 per 8 × 16 lanes, not ZOL → II2 per 16 lanes, NS 5 | `convert_copy` (AIE2, npu1): one chain under `AIE_LOOP_NO_UNROLL` and `AIE_LOOP_MIN_ITERATION_COUNT(8)`. The f32 vector loads straight into the accumulator, and only the a port loads accumulators, so II2 is the port bound | 207 → 144 (1.4x); bit-identical on all 5 data cases | hit, including magnitude: 64 trips × II2 + 16 |
| S31 | Loop II bf16_exp 53 → 39 (NS 2, prepipelined), softmax exp loop 58 → 47; bf16_exp `missing_bank_loads` 1 → 0 | `getExpBf16` (AIE2, npu1): `aie::parallel_lookup` written out as `vfloor` byte offsets masked to the table and two `load_lut_2x_int8`. The lookup's accumulator shift needed crRnd = floor, and saving, setting and restoring it on every call kept the loops from pipelining; `vfloor` floors regardless. Bit-identical on all 65536 bf16 inputs | bf16_exp 3400 → 2497, softmax 5657 → 4952, softmax llama-prefill 10905 → 9496 | hit, including magnitude: trips × ΔII gives 896, 704, 1408 against 903, 705, 1409 |
| S32 | Frame 1408 → 32 B; 8 macs per ic8 at II29 with the accumulators on the stack → 16 macs (both input halves) at II16, NS 2; the residual epilogue, a serial chain at II40 per 4 pixels, → one pass at II20 per 8 blocks | `conv2dk1_skip` (AIE2, npu1): accumulators unrolled into registers; the int8 conv result parks in the output buffer and one pipelined pass adds the residual (`vlda.ups` + `vmac` by 1 + `vst.srs`), with `__restrict` on `output` and `skip` | uint8 and int8 residual 19585 → 1331 (14.7x); bit-identical. Residual in the per-block epilogue: 2025. Final pass unrolled 4x (LLVM's choice) 1464; `ups` + `vadd` at unroll 8: II27; skip as `vmul` by 1: II26 | hit: the mac-bound floor is 1024 (one `vmac` per cycle). The pass without `__restrict` ran load-after-store serial, II61 per 4 blocks |
| S33 | Loop II39 NS 2 → II15 NS 3 (pipeliner MII 23 → 13); one 64 B spill reloaded on the idle a port | `bf16_exp` (AIE2, npu1): the loop rotated by one, so the table reads of vector i+1 are issued before the store of vector i. The LUT loads carry no alias information against the output, so `__restrict` does not help: each iteration's gathers waited on the previous store, a store-to-load recurrence through the whole exp chain. Restrict iterators alone: II26 | 2497 → 1057 (2.4x); bit-identical on all 65536 bf16 inputs | hit, including magnitude: 63 trips × II15 + prologue. Rotating by two: II20; two vectors per iteration: II37 per 32 lanes |
| S34 | `unpipelined_loops` 6 → 9, `non_zol_loops` 6 → 8, frame 32 → 544 B (out_split 96 → 704), `pm_bytes` 1312 → 3696; the one new pipelined loop at II36, NS 1 | `bn_conv2dk3_dw`, `bn_conv2dk3_dw_out_split` (AIE2, npu1): a vector path beside the scalar one. 32 lanes of uint8 × int8 `aie::mul`/`mac` hold 4 pixels × 8 channels. Stride 1 gets its neighbours from `shuffle_up_fill`/`shuffle_down_fill`, stride 2 from `filter_even`/`filter_odd`, and a border row from zeroed weights. Stores to 32-byte aligned addresses must be `store_v`: `store_unaligned_v` read-modify-writes the whole enclosing 64-byte window, up to 32 bytes past the vector | 270416 → 4646, stride 2 322779 → 3656 and 370209 → 9080, 401847 → 8407, out_split 506897 → 12053; bit-identical. Unaligned stores everywhere (unsafe): out_split 9647 | miss as a predictor: every counter got worse while cycles fell 40-90x. The scalar fallback stays in the object, and the 9 weight vectors each hold a 512-bit x register (spills) |
| S35 | `unpipelined_loops` 4 → 6, `pm_bytes` 816 → 4400, frame 64 → 96 B; 7 new mac loops at II9-13, NS 1, for 1-4 mmuls | `bn_conv2dk1_relu` (AIE2, npu1): `aie::mmul<4,8,8,int8,int8>` over [C/8][W][8] rows, 4 accumulators of 4 pixels per weight block, round half to even and saturate. When the width is not a multiple of 4, the last chunk starts at width - 4 and overlaps the one before it | 480059 → 2386, 262667 → 1861, 521656 → 5596, 341312 → 2721; bit-identical | miss as a predictor, as S34. At 4.2-9 cycles per mmul the loops are short (input_channels / 8 - 1 trips) and single-stage |
| S36 | Max loop II8 → II1 NS 8, exp loop II47 → II19 NS 3, scale loop II16 → II2 NS 7; `pm_bytes` 976 → 2496, frame 64 B | `softmax` (AIE2, npu1): the exp loop rotated by one as in S33, restrict iterators, and `AIE_LOOP_MIN_ITERATION_COUNT` on all three loops. The count is a template parameter: tiles of 144 elements and up (8 exp trips) take the pipelined instance, smaller ones an instance at count 1. Without a count the loops do not pipeline; at count 2 the exp loop reaches II25, max II5, scale II7 | 1024: 4952 → 1845 (2.7x); 2048 llama-prefill: 9496 → 3306 (2.9x); bit-identical at 32, 128, 160 and 1024 | hit, including magnitude: trips × II plus the base's ~470 fixed cycles predicts 1916 and 3324. `__divsf3` for the one reciprocal remains |
| S37 | Loop II68, not pipelined → II48, NS 3 (pipeliner II36; x-register spills add 12); frame 0 → 64 B | `rgba2hue` (AIE2, npu1): restrict parameters, the loop kept rolled under `AIE_LOOP_NO_UNROLL` with `AIE_LOOP_MIN_ITERATION_COUNT(4)`. A row under 4 vectors takes a second instance of the function without the count. The two loops must be in separate noinline functions: two copies of the body in one function (if/else or loop versioning) find no schedule | 4142 → 2954 (1.4x); bit-identical on all 5 data cases | hit, including magnitude: 60 trips × II48 plus the base's ~62 fixed cycles predicts 2942 |
| S38 | Squares loop II16 → II9 per 4 chunks, NS 2; scale loop II24 → II3 per 32 lanes, NS 6; libcalls `__divsf3 __floatsisf __mulsf3` → `__udivsi3` once plus `__mulsf3` in the tail; `pm_bytes` 1456 → 2192, frame 96 B | `rms_norm` (AIE2, npu1): `mac_elem_16_2` on bf16 x against itself for the squares (exact products, one mac per 32 lanes), four accumulators for the mac latency. 1/rms is computed in bf16 macs (q31 1/cols, Newton step, exact residual correction) and held as a [hi \| lo] bf16 pair, so the scale is one mac. `AIE_LOOP_MIN_ITERATION_COUNT(8)`, with plain loops for rows under 8 chunks: at count 2 the loops cap at NS 2 (II5 squares, II8 scale) and run at 754 | 1024: 2224 → 429 (5.2x); all outputs correctly rounded (base 71–78%) | hit in direction. Magnitude is under-predicted: trips × II gives 168 cycles of 429, and the rest is the udiv, the reductions and the reciprocal. Remarks misrank close variants: II5 per pair on split pointers ran 463, II6 on one pointer 447 |
| S39 | Frame 2048 → 288 B, `pm_bytes` 3392 → 2208, libcall `__modsi3` gone; three inner mac loops at II29 with the accumulators on the stack → one ic loop at II16 (16 macs, both input halves) and the skip loop at II9 (8 macs), both NS 2 with ZOL | `conv2dk1_skip_init` (AIE2, npu1): `conv2dk1_skip`'s recipe (S32) plus `AIE_LOOP_MIN_ITERATION_COUNT(2)` on the runtime-trip ic loops as a template parameter; loops of one trip take an instance at count 1. The residual epilogue runs each step (`srs`, `ups`, `mac` by 1, store) across all 8 accumulators before the next | uint8 16222 → 1349, int8 residual 16233 → 1348 (12x); bit-identical. Count 1 everywhere (skip loop II15, no ZOL): 1741. Count 2, one serial epilogue chain: 1529. First ic step peeled as a `mul`: 1477 | hit in direction; trips × II gives 800 of 1349, the rest is the NS 2 prologues and the epilogue. Stores at `out_ptr + x8 * 32` in the unrolled epilogue miscompile (post-increment `vst.srs` on a copied pointer, 9 of 14 gate tests fail); walking the pointer is correct |
| S40 | Loop II28, not pipelined → II5, NS 6 | `rgba2gray` (AIE2, npu1): restrict parameters, the rounding term seeds the accumulator (three chained `mac`s instead of a `mul` and three `mac`s), the loop kept rolled under `AIE_LOOP_NO_UNROLL` with `AIE_LOOP_MIN_ITERATION_COUNT(6)`; rows under 6 vectors take a plain loop in the same function (II24). The count with the original `mul` start: II12. Restrict alone: II26 | 1707 → 337 (5.1x); bit-identical on all 5 data cases | hit, including magnitude: 60 trips × II5 plus ~30 fixed cycles predicts 330 |
| S41 | bf16 K loop II31, NS 2, 4-8 trips per j → j body II105 per 32 macs (K 32) and II175 per 64 (K 64), frame 64 → 224 / 352 B; int8 j body II84 per 32 macs, frame 160 B; looped bf16 K (K ≥ 128) II31 → II18 | `mm` (AIE2, npu1): K unrolled fully into the j body when dim_k / s ≤ 8 (bf16) or ≤ 4 (int8), with each A row block read through one `add_2d_byte` cursor that wraps to the block start after dim_k / s steps. The wrap is opaque to LICM, so A is not hoisted out of j and spilled. B and the other A rows are constant offsets from single cursors | bf16 3307 → 1881 (1.76x), b_col_maj K 64 4922 → 2969, int8_int32 1985 → 1449, int16_int32 (cursor only, K loop kept) 3681 → 3625; bit-identical. Unroll without the cursor: bf16 1945, b_col_maj 3377 with a 1120 B frame. Cursor alone: 2235 / 3309 | hit, including magnitude: 16 j bodies × II105 = 1680 of 1881, × II175 = 2800 of 2969, × II84 = 1344 of 1449. S12's "unroll is worse" was the LICM spill, not the unroll |
| S42 | Loop II14 per 16 px, NS 1 → II4 per 32 px, NS 4 | `gray2rgba` (AIE2, npu1): restrict parameters; the 16-pixel broadcast-then-`bor` body replaced by 32 pixels through three `shuffle`s (`INTLV_lo_8o16` of y with itself and with 255, then `INTLV_lo/hi_16o32` of the two) and two 64-byte stores; the loop under `AIE_LOOP_NO_UNROLL` with `AIE_LOOP_MIN_ITERATION_COUNT(4)`, a plain loop (II15) for rows under 4 steps and the old body for a 16-pixel tail. Restrict alone: II14. Restrict and the count on the old body: II4 per 16 px | 1706 → 274 (6.2x); bit-identical on all 5 data cases | hit, including magnitude: 60 trips × II4 plus ~30 fixed cycles predicts 270 |
| S43 | Frame 1408 → 288 B, `pm_bytes` 1088 → 464; the mac loop over the 8 accumulators, which found no schedule, → an ic loop of 8 macs at II9, NS 3 with ZOL | `conv2dk1` (AIE2, npu1): as S32, one template for the int8 and uint8 builds with the accumulators unrolled into registers. The block's activations come through two pointers, one per half block, so both load units carry them, and the ic loop is kept rolled under `AIE_LOOP_NO_UNROLL`. One pointer, LLVM's unroll by 2: II23 per two steps, 863. One pointer, rolled: II11, 958. Two oc/8 groups × 16 pixels per block (4 loads feed 8 macs): II9, 793, but oc must be a multiple of 16. The ic loop fully unrolled: 781 with a 1728 B frame | int8 and uint8 9968 → 801 (12.4x); bit-identical | hit in direction; trips × II gives 576 of 801, the rest is the NS 3 fill and drain and the stores of each block. The mac-bound floor is 512 |
| S44 | Middle loop II22, NS 1 → II6, NS 4 | `filter2d` (AIE2, npu1): restrict parameters and the middle-of-line loop kept rolled under `AIE_LOOP_NO_UNROLL` with `AIE_LOOP_MIN_ITERATION_COUNT(4)`; rows under 4 middle vectors take a plain loop in the same function (II20). Restrict alone: II22. Rolled without the count: II20. Count 8: II6, no better | 1362 → 444 (3.1x); bit-identical on all 5 data cases | hit, including magnitude: 58 trips × II6 plus base's 86 cycles outside the loop predicts 434 |
| S45 | Statistics loop II14 per 16 lanes, not pipelined → II4 per 64, NS 3; normalize loop II38 per 16 lanes, not pipelined → II3 per 32, NS 6; libcalls `__divsf3 __floatunsisf __mulsf3` → `__udivsi3` once; `pm_bytes` 1200 → 2400, frame 224 → 96 B | `layer_norm` (AIE2, npu1): S38's recipe with a sum beside the squares: `mac_elem_16_2` against ones and against x itself, two accumulator pairs. The variance is (n s2 − s1²) / n² from bf16 limbs in one f32 accumulator, and 1/sqrt is held as a [hi \| lo] pair, so normalize is one mac per 16 lanes with c = −mean · inv_std as the addend. The q31 1/cols divide runs before any vector state is live: after the statistics loop it spills them, frame 416 B. One normalize step per trip: II3 per 16 lanes | 1024: 4683 → 503 (9.3x); correctly rounded 99.96% (base 54–65%) | hit in direction. As S38, magnitude is under-predicted: trips × II gives 160 of 503, the rest is the udiv, the two reductions and the reciprocal |
| S46 | Straight-line code 160 → 145 bundles, frame 224 → 32 B | `dwconv1d_channels_last` (AIE2, npu1): the fully unrolled channel-group loop regrouped tap-major over four 32-lane groups at a time (four accumulators, each still summing its taps in order), so the ten tap pointers are each walked once per four groups. Two groups: 175 bundles. Eight groups: 142 bundles, 160 B frame | 155 → 127 (1.22x); eight groups 155 → 132; bit-identical on all 5 data cases, clamped and not | hit in direction; bundles 160 → 145 predicts about 10%, measured 18%. The 40 macs over 80 loads bound it at 40 cycles plus the call |
| S47 | Two-trip i loop II10, NS 2 inside a two-trip j loop → one k-step body at II64 per 32 macs, NS 1, ZOL; `pm_bytes` 1632 → 1280, frame 2080 → 2048 B, `[sp]` references 22 → 2 | `fused_mm` (AIE2, npu1): in the bf16 `mm_fused_mmul_2x2`, i and j unrolled fully when colA ≤ 4 and colB ≤ 4, so each k step is straight code and the next block's C loads overlap the current macs. j alone: about 28 bundles per 8 macs; i alone: II26 per 8 macs | 32x32x16x4: 1177 → 691 (1.70x); bit-identical | hit in direction; 4 k steps × II64 = 256 of 691; the rest is the accumulator init and the drain, both unchanged |
| S48 | `unpipelined_loops` 4 → 6, `non_zol_loops` 3 → 15, `pm_bytes` 832 → 4640, frame 64 → 128 B; 7 new mac loops at II9-13, NS 1 | `bn_conv2dk1_i8` (AIE2, npu1): S35's block with uint8 activations, `aie::mmul<4,8,8,uint8,int8>` and int8 outputs. The same block written once as a template over element types, with the load and store passed in as lambdas, ran up to 1% slower here, 3-9% slower for `bn_conv2dk1_relu` and 2-28% slower for `bn_conv2dk1_skip` | 349269 → 1904, 413726 → 1254, 396338 → 5107, 295741 → 5730; bit-identical | miss as a predictor, as S34 |
| S49 | `unpipelined_loops` 4 → 6, `non_zol_loops` 3 → 15, `pm_bytes` 896 → 5552, frame 64 → 160 B; 7 new mac loops at II9-13, NS 1 | `bn_conv2dk1_skip` (AIE2, npu1): S48's block, then the skip added in 32-bit lanes as two multiplies by a vector of ones into one accumulator, requantized by `skip_scale` with round half to even and saturation. The lambda-generic block of S48 cost 2-28% here | 367491 → 2069, 377559 → 2072, 480604 → 1535, 301194 → 2316, 430301 → 4665; bit-identical | miss as a predictor, as S34 |
| S50 | Interleaved loop II42 per 64 elements, NS 1, no ZOL → II7 per 32, NS 4; two-halves loop II47 per 64, NS 1 → II7 per 32, NS 3; `pm_bytes` 544 → 736 | `rope` (AIE2, npu1): each rotated half is one `mul_elem_16_2` on [x_e \| x_o] × [cos \| −sin] or [x_o \| x_e] × [cos \| sin] (lane i gets a[i] b[i] + a[i+16] b[i+16]) instead of a zero-padded 16-lane `aie::mul` and `mac`/`msc`. x and the table are deinterleaved with one `T16_16x2` shuffle each and the result re-interleaved with `T16_2x16`; −sin is a uint16 add of 0x8000 to the high lanes. `AIE_LOOP_MIN_ITERATION_COUNT(4)`, plain loops (II26, II21) under 4 steps | 1024: 688 → 261 (2.6x); llama 64 two-halves (plain loop, 2 steps): 73 → 68; bit-identical | hit, including magnitude: 32 trips × II7 = 224 plus base's ~16 fixed cycles predicts 240 of 261; 2 × II21 plus base's 26 predicts 68 |
| S51 | `sv_row_block` loop II149 per tile pair, NS 1 → II36 per pair, NS 2; its 2272 B frame gone (inlined, `prefill_fv_step` frame 64 B); round stack 2432 → 1152 B | `prefill_fv` (AIE2, npu1): the aie2p form, whose operands AIE2 builds by emulating `vextbcst` (`extract<8>` + `concat` ×4) around 16 elementwise 64-lane macs per pair, → one `aie::mmul<8, 8, 8, bf16, bf16>` per 8x8 output tile, two tiles per trip, y loaded one pair ahead. One tile per trip with the y prefetch (II24 per tile): 1560 / 3189. One tile per trip, no prefetch: II29 per tile. Indexed addressing: II34. aie2p keeps its form | 512: 5311 → 1462 (3.6x), 256: 11019 → 3077 (3.6x); 174 / 611 f32 words differ, since the mmul sums in another order: worst 0.12 of the f32 order bound (base 0.15) | hit, including magnitude: 32 pairs × II36 = 1152 of 1462, 4 × 16 × II36 = 2304 of 3077 |
| S52 | Loop II139 per 32 lanes → II76 per 32 (SMS); frame 0 → 640 B | `swiglu` (AIE2, npu1): the tanh gathers are ordered against every load and store, so each lookup waited for the last store and each load for the lookup. Two vectors per trip, the next trip's three inputs loaded before the lookups (the last trip reloads its own), both stores after both lookups. Four per trip: II123 per 64, but a 1376 B frame fails the 1024 B stack check. Loads inside the trip: the vector loops stay rolled; unrolled by pragma, II207 per 64 at 1088 B | 1024: 4455 → 2564 (1.74x); bit-identical over all 65536 inputs | hit, including magnitude: 32 trips × II76 = 2432 of 2564 |
| S53 | ic loop II23 per two steps (LLVM's unroll by 2), NS 2 → II9 per step, NS 3 with ZOL; frame 0 → 288 B, `pm_bytes` 432 → 464 | `conv2dk1_i8` (AIE2, npu1): S43's loop under `#if __AIE_ARCH__ == 20`: the block's activations through two pointers, one per half block, and the ic loop kept rolled under `AIE_LOOP_NO_UNROLL`. The accumulators were already unrolled into registers | 871 → 801 (1.09x); bit-identical on HW; AIE2P `.text` byte-identical | hit in direction; trips × II gives 576 of 801, as S43 |
| S54 | `unpipelined_loops` 5 → 9, `non_zol_loops` 3 → 14, `pm_bytes` 1648 → 6064, frame 160 → 544 B; libcalls unchanged (all in the scalar fallback) | `bn_conv2dk1_relu_xy_pool_padded` (AIE2, npu1): S35's block per output channel block into a stack row, summed in uint16 lanes; the float average by 49 as an integer multiply-shift, `(acc * 42799) >> 21`, with the scalar's rounding on the remainder. Dropping the redundant quotient correction cut 6167 → 5584 | 445905 → 5584; bit-identical | miss as a predictor, as S34 |
| S55 | Loop II19 per two 32-value blocks, NS 2 (4 `vmul.f` + 4 `vmsc.f`) → II8, NS 3 (4 `vmac.f` + 2 `vmul.f`); group 64: II38 per group → II6; group 96: II24 per group → II16 per two groups | `expand` (AIE2, npu1): a block's 32 biased values are one `vmac.f` operand straight from the byte interleave, the scale sits in one half of the other operand, and the accumulator starts at −128 × scale, so one mac per 16 outputs. The scale vectors are built once per group, and odd-block groups go two at a time so every 64-nibble load stays 32-byte aligned. Scales rebuilt per block pair: II10 (group 64), II34 not ZOL (group 96) | 324 → 161 (2.0x); bit-identical at group 32, 64 and 96 | hit, including magnitude: 16 trips × II8 = 128 of 161. II8 is the 8 `vsel`/`vbcst`/`vshuffle` per trip, which issue one per cycle |
| S56 | 32x32: loop II25 per 8 columns of 16 rows (16 `vshuffle` + 8 `vextbcst`), two blocks × 4 trips → straight code, frame 0; 32x128: II25 per 8 columns of 16 rows → II12 per 8 columns of 32 rows, NS 2 | `mv` int16 (AIE2, npu1): a 32-lane load of the word-transposed A holds 16 rows of one column pair, so it is macced elementwise against that column pair of b (one int32 broadcast) and the even and odd lanes are added once per row block, instead of being split per column. Two row blocks share each b load, K is unrolled fully up to 32 columns. Cursors that put A on both load ports: 92 unrolled but II17 looped | 32x32: 120 → 95 (1.26x); bit-identical | hit in direction. The K ≥ 40 loop (II25 → II12 per 16 rows twice) is gated but has no npu1 case to time |
| S57 | K 9: loop II54 per two 16-lane blocks, NS 1 → II19, NS 2; frame 576 → 32 B. K 17: II104 → II40, frame 1760 → 416 B | `dwconv1d_channels_first` (AIE2, npu1): no bf16 sliding multiply on AIE2, so `sliding_mul` spent a zero-padded `vmac.f` and two shuffle-slot ops per tap and block. One `shuffle_down_fill` of a 48-sample window now serves a tap for two blocks (its low half the first block's, its high half the second's), and the low (or high) halves of two taps' shifts fill one `vmac.f`: K − 1 shifts, K + 1 merges and K + 1 macs per two blocks. The pair loop is a noinline template instance under `AIE_LOOP_MIN_ITERATION_COUNT(2)`, with a plain instance for rows under 4 blocks. Both loops in one function: the pipelined one 3 cycles slower | 1842 → 666 (2.8x), `dwconv1d` 1842 → 666; bit-identical on all 5 data cases, K 9 and K 1 | hit, including magnitude: 32 trips × II19 = 608 of 666. The 8 `vshift` and 10 `vshuffle` per trip never share a bundle, so II18 is the bound |
| S58 | Loop II68 per 4 × 32 pixels, NS 1 (LLVM's unroll by 4: MII 32 > `SwpMaxMii` 27) → II8 per 4 × 32, NS 3 | `add_weighted` (AIE2, npu1): `__restrict` on the pointer parameters, nothing else. Also kept rolled under `AIE_LOOP_NO_UNROLL`: II2 per 32 pixels, NS 8, 135. Rolled without `__restrict`: II17 | 1026 → 137 (7.5x); bit-identical on all 5 data cases | hit, including magnitude: 15 trips × II8 = 120 of 137. II2 per 32 pixels is the two `vldb.unpack` loads, both on the b port |
| S59 | ReLU loop II23 per 16 lanes, NS 1 (`aie::max` on f32) and identity loop II9, not ZOL → one loop II5, NS 3, ZOL; `unpipelined_loops` 4 → 2, `pm_bytes` 1216 → 1312 | `mm_activation_epilogue` (AIE2, npu1): identity and ReLU as one integer select on the bit pattern, +0 where x − 1 is below a threshold: −inf's pattern for ReLU (zeroes the negative floats but −0 and −NaN, as `aie::max` does), INT32_MIN for identity. Without the −1: II4, but −0 becomes +0. With a second compare to keep −0: II7 | identity 582 → 338, relu 1490 → 338 (4.4x); bit-identical on all 130 gate runs and 16 special patterns | hit, including magnitude: 64 trips × II5 = 320 of 338 |
| S60 | Loop II14 per 4 × 16 lanes → II22 per 8, both post-pipelined at NS 2 (the pipeliner fails on every form: unsupported structure unrolled, no schedule rolled); `pm_bytes` 272 → 432 | `leaky_relu` (AIE2, npu1): unroll 8 instead of 4 under `#if __AIE_ARCH__ == 20`, the same 128 elements per pass as aie2p's 4 × 32. Unroll 16: II44 per 16. Rolled with `AIE_LOOP_MIN_ITERATION_COUNT`: II11 per vector | 1024: 234 → 188 (1.24x); bit-identical over all 65536 inputs at 3 tile sizes | hit, including magnitude: 8 passes × II22 = 176 of 188 |
| S61 | Three loops II77 to II143 per 16 lanes, no ZOL (aie_api's bf16 emulation of each f32 multiply) → pass 1 II9 per 4 chunks, NS 2; pass 2 II19 per 2, NS 2; f32 pass 3 II11, NS 4; affine pass 3 II27, NS 3; libcalls `__divsf3 __mulsf3` → `__udivsi3` once; frame 576 → 64 B (f32), 320 → 128 B (affine) | `layer_norm_f32`, `layer_norm_affine_cast` (AIE2, npu1, from `aie2p/layer_norm.cc`): S45's recipe carried to f32. x − mean and 1/sqrt as three bf16 limbs, each output the limb products with i + j ≤ 2 in `mac_elem_16_2`; the affine cast takes n and gamma to three limbs each. Mean and variance / n by a Q31 reciprocal, divided before any vector state is live. Two limbs throughout: 1333 and 4451 but max \|error\| 3.5e-5 (base 1.5e-6); two-limb affine: 2919 but 27–33 of 16384 words no longer correctly rounded | f32 6611 → 1962 (3.4x), affine 13064 → 3015 (4.3x); f32 max \|error\| 2.3e-6 → 7.6e-7 (worst of seeds 0–2), affine 1–2 of 16384 words differ, more correctly rounded than base | hit in direction: trips × II = 1456 of 1962 and 2480 of 3015; the rest is the two divides, the reductions and 1/sqrt |
| S62 | uint8, per 64 pixels, all NS 1 → pipelined: trunc II12 → II2, binary II14 → II3, to-zero II14 → II4; the binary selects no longer outlined | `threshold` (AIE2, npu1): `__restrict` on the two pointers, and the constants as whole vectors (`zeros`, `broadcast`) rather than elements of one vector. `__restrict` alone: the same IIs, 469 → 140 | 469 → 132 (3.6x); bit-identical on all 5 data cases, 4 full-row and 5 one-vector cases | hit, including magnitude: 30 trips × II3 = 90 of 132 |
| S63 | Loop II13 per vector, NS 1 (uint8, int16), II26 (int32) → II3, NS 5 | `bitwise_and` (AIE2, npu1): `__restrict` on the three pointers, and the loop kept rolled under `AIE_LOOP_NO_UNROLL`. `__restrict` alone: the same uint8 and int16 objects, but int32 unrolls by 2 at II9 per two vectors | 396 → 101 (3.9x); bit-identical on all 5 data cases | hit, including magnitude: 30 trips × II3 = 90 of 101 |
| S64 | SiLU loop II83, GELU loop II77, both NS 1 → II39 and II46, NS 2; `unpipelined_loops` 2 → 1, `pm_bytes` 1312 → 2720, frame 0 → 128 B | `mm_activation_epilogue` (AIE2, npu1): the table-read tanh's loads carry no alias information, so each store is ordered before the next vector's table reads and the one-vector loop never pipelines, even with an II hint (67, 79). Each result is now stored a trip late, under `AIE_TRY_INITIATION_INTERVAL(46)`; without the hint, 66 and 78. Computing 2 or 4 vectors before storing any: 3652 and 3520 | silu 5346 → 2567 (2.1x), gelu 4966 → 3012 (1.65x); bit-identical on all 130 gate runs | hit, including magnitude: 63 trips × II39 = 2457 of 2567, × II46 = 2898 of 3012 |
| S65 | Loop II13 per vector, NS 1 (uint8, int16), II26 (int32) → II3, NS 5 | `bitwise_or` (AIE2, npu1): S63 on the OR kernel. `__restrict` alone: the same uint8 and int16 objects, but int32 unrolls by 2 at II9 per two vectors | 396 → 101 (3.9x); bit-identical on all 5 data cases | hit, including magnitude: 30 trips × II3 = 90 of 101 |
| S66 | `unpipelined_loops` 9 → 13, `non_zol_loops` 11 → 20, `pm_bytes` 2032 → 5808, frame 96 → 864 B; the general path's channel loops II10-17, its row loops II31-54; the 8-channel path, the one measured, has no pipelined loop | `bn_conv2dk3` (AIE2, npu1): stride-2 3x3 as `aie::mmul<4,8,8>` per 4 output pixels; `filter_even`/`filter_odd` split the input into centre and right taps and the right tap shifted up by one pixel is the left. At 8 input channels one pass per output channel block walks the row two 4-pixel chunks at a time, carrying each row's odd pixels between chunks, with a dropped border row given zero weights. The general path used at 8 channels: 7093; 4 chunks per step: 4755 | 405416 → 1618 (250x); bit-identical | miss as a predictor, as S34 |
| S67 | Loop II183 per 32 lanes → II116 per 64, both NS 1; `pm_bytes` 752 → 992, frame 0 → 832 B | `gelu` (AIE2, npu1): the tanh table reads are ordered against every load and store, so the one-vector loop runs load, lookup and store in series. `lut_map_bf16` (AIE2 `lut_based_ops.h`) takes 4 vectors per trip, loads the next trip's input ahead through a clamped pointer, and stores after all 4 lookups. Loads inside the trip instead: II128; clamping the index in bf16: II115, but worse for sigmoid and silu | 5865 → 1981 (3.0x); bit-identical on all 65536 inputs at 7 tile sizes | hit, including magnitude: 16 trips × II116 = 1856 of 1981 |
| S68 | Loop II189 → II83 per 64 elements, both NS 1; `pm_bytes` 752 → 704, frame 0 → 704 B | `silu` (AIE2, npu1): S67's `lut_map_bf16` around the same tanh-and-mac body. Loads inside the trip instead: II89; clamping the index in bf16: II87, frame 800 B | 1024: 3031 → 1434 (2.1x), 4096: 12133 → 5763; bit-identical on all 65536 inputs at 6 tile sizes | hit, including magnitude: 16 trips × II83 = 1328 of 1434 |
| S69 | Loop II15 per 4 vectors, NS 3 (unrolled by 4, one serial chain) → II2 per vector, NS 5 | `reduce_add` (AIE2, npu1): a walked `v16int32` pointer instead of `in + i`, and the loop kept rolled under `AIE_LOOP_NO_UNROLL`. `in + i` rolled: II3; the walked pointer unrolled by 4 under `AIE_PREPARE_FOR_PIPELINING`: II8 per 4, 146 | 249 → 147 (1.7x); bit-identical | hit, including magnitude: 64 trips × II2 = 128 of 147 |
| S70 | Loop II15 per 4 vectors, NS 3 → II2 per vector, NS 5 | `reduce_min` (AIE2, npu1): S69. `in + i` rolled: II5; unrolled by 4: II8 per 4, 146 | 251 → 147 (1.7x); bit-identical | hit, including magnitude: 64 trips × II2 = 128 of 147 |
| S71 | int32 loop II15 per 4 vectors, NS 3; bfloat16 II9 per 2, NS 2 → both II2 per vector, NS 5 | `reduce_max` (AIE2, npu1): S69 with `aie::load_v` from the walked pointer. `in + i` rolled: II5; unrolled by 4: int32 154, bfloat16 135 | int32 259 → 155 (1.7x), bfloat16 171 → 90 (1.9x); bit-identical | hit, including magnitude: 64 trips × II2 = 128 of 155, 32 × II2 = 64 of 90 |
| S72 | `unpipelined_loops` 4 → 5, `non_zol_loops` 3 → 5, `pm_bytes` 832 → 1520, frame 64 → 96 B; the new loop II10 per 32 input channels, not ZOL | `bn_fc_relu_ui16_pad` (AIE2, npu1): the single-pixel FC as `aie::mmul<2,8,8,uint16,int8>`: 16 inputs as a 2 x 8 matrix against the next 8 weight rows, twice, and only row 0 of one product and row 1 of the other kept. Four accumulators, 32 input channels per trip. Two accumulators (16 per trip): 922; six or eight: the stack goes over the 1024 B reserved | 49002 → 459 (107x); MobileNet's FC1 (960 of 1280 rows) 36842 → 359; bit-identical | hit, including magnitude: 40 trips × II10 = 400 of 459 |
| S73 | int16 j body II51 per 32 macs, NS 1, frame 32 B → II89 per 64 macs, NS 1, frame 288 B (accumulator halves through the stack); `pm_bytes` 544 → 1024 | `mm` int16_int32 (AIE2, npu1): `matmul_vectorized_4x2_mmul` instead of 2x2 when m % 16 == 0, K unrolled up to 16 steps. Eight C tiles per j, so the C loads, upshifts and stores per mac halve. Macs ordered B-major: II89, the same. K unrolled only up to 8: 32x64x32 1393 against 1351 | 64x32x64 3626 → 3219 (1.13x), 64x16x64 2513 → 2025, 32x64x32 1493 → 1351; bit-identical | hit, including magnitude: 32 j bodies × II89 = 2848 of 3219 (base 64 × II51 = 3264 of 3626) |
| S74 | Pixel-pair loop II9 for 4 `vmac`s and 4 `vshuffle`s, NS 3, once per oc/8 group → II9 for 8 `vmac`s, NS 3, once per two groups; `pm_bytes` 368 → 352, frame 32 B in both | `conv2dk14` (AIE2, npu1): `aie::mmul<8,8,8,uint8,int8>` is two native 4x8x8 macs, the second on a `vshuffle` of the upper 256 bits. `mmul<4,8,8>` on 256-bit activation loads needs no shuffle, and two oc/8 groups share each activation load (8 accumulators); an odd last group is computed twice and stored once. One group, 4 accumulators: LLVM's unroll by 2 II12 per two steps, rolled II8 (static). One group, two pixel-parity accumulator sets: II9 per two steps, 1117. 512-bit activation loads split with `extract`: II14 (static) | 1985 → 990 (2.0x); bit-identical | hit, including magnitude: 98 trips × II9 = 882 of 990. The mac-bound floor is 784 |
| S75 | SiLU loop II39 → II36, NS 2; `pm_bytes` 2720 → 2624 | `mm_activation_epilogue` (AIE2, npu1): SiLU's tanh reads hi / 2, exact in bf16, instead of bf16(hi / 2 + lo / 2); the two differ only on a rounding tie. One mac and its conversion leave the chain; the model and its exact gate follow | 2567 → 2372 (1.08x); 14 of 769792 gate words differ, all ties; mean error against float64 unchanged | hit, including magnitude: 63 trips × 3 cycles = 189 of 195 |
| S76 | `pm_bytes` 6064 → 5696; loops, frame and libcalls unchanged (the scalar finalize was straight-line code, fully unrolled) | `bn_conv2dk1_relu_xy_pool_padded` (AIE2, npu1): S54's per-output average in 8 vector lanes. `(acc * 42799) >> 21` as `acc << 16` in acc64 plus a mac by −22737, shifted out with floor rounding; the remainder and the tie test as vector compares and a select | 5584 → 4494 (−20%); bit-identical | miss as a predictor: no loop or libcall metric moved |
| S77 | Loop II28 per 32 → II44 per 64, both NS 2 (post-pipelined); `pm_bytes` 400 → 736, frame 0 → 256 B. With S67's prefetch: II47, frame 480 B | `tanh` (AIE2, npu1): S67's `lut_map_bf16` with its new `Prefetch` off: with only the lookup in the body, the trip-ahead load costs more than it hides. Two or eight vectors per trip: II73 per 32, II118 per 128 | 911 → 727 (1.25x); with the prefetch 812; bit-identical on all 65536 inputs at 2 tile sizes | hit, including magnitude: 16 trips × II44 = 704 of 727; II47 → II44 predicted the prefetch's 85-cycle loss in direction only |
| S78 | Loop II62 per 32 → II77 per 64, both NS 1; `pm_bytes` 416 → 640, frame 64 → 448 B. With S67's prefetch: the same II77, frame 640 B | `sigmoid` (AIE2, npu1): S77, around tanh's single mac. Eight vectors per trip: frame 1152 B, over the 1024 reserved | 1991 → 1259 (1.58x); with the prefetch 1265; bit-identical on all 65536 inputs at 3 tile sizes | hit, including magnitude: 16 trips × II77 = 1232 of 1259 |
| S79 | Loop II116 → II85 per 64, both NS 1; `pm_bytes` 992 → 784, frame 832 → 736 B | `gelu` (AIE2, npu1): eight bf16-rounded ops around the lookup as five: u = x (c + d x²) with c preloaded in the fp32 accumulator, and 0.5x + 0.5x tanh(u) as one mac on the fp32 0.5x. Judged against a float64 reference, not a kernel model, so the rounding change is allowed and measured: max \|error\| 0.009302 → 0.008875 over \|x\| ≤ 16, 641 outputs closer and 100 further | 1981 → 1439 (1.38x); 1410 of 65536 words change, no tolerance failures | hit, including magnitude: 16 trips × II85 = 1360 of 1439 |

### Confidence classes

Every candidate gets exactly one class, from the rows above:

| Class | When | Record behind it |
|---|---|---|
| **strong** | The signal removes a libcall, removes spill or stack traffic in the hot loop, makes a hot loop pipeline (or drops `unpipelined_loops` on the called path), or passes the unroll screen | S01-S04, S06, S08, S19: every recorded case won on HW |
| **likely** | The hot loop's II or bundles-per-element drops on the called symbol, and the change doesn't defeat an optimization to get there | S05, S07-S11, S17, S18, S20: always the right direction; magnitude from the per-call prediction, not the II ratio |
| **experiment** | No precedent row; the II drop comes from hiding work from LLVM (opaque pointers, `volatile`, LICM blockers); the loop sits at a resource bound; or the called symbol isn't resolved | S14, S15: the two recorded misses |
| **reject** | The compiler reports it worse: II up, new spills, stack over the declared budget, or `pass_failed` drops the pragma | S12, S13: both HW checks agreed with the compiler |

Rules for using the classes:

- Predict magnitude with `bundles + 5 + (trips-1) × II` per call
  (`static-checks.md` §Predict), never with the II ratio alone (S09, S10).
- A 1-cycle II change on a long trip count is not noise. Scale it by the trip
  count before you dismiss it (S09).
- Compare II per unit of work (per row, per value, per tile), not per loop.
  A body that does four rows at II56 beats one row at II10 plus a II125
  reduction (S17), and an 8-row group at 47 bundles beat per-row calls
  with a max II of 13 (S20).
- `likely` and `strong` are statements about the record, not about your
  kernel. The report still says "candidate, HW unconfirmed".

## Levers ranked by best measured delta

A lever is listed only if it produced a measured hardware win on a real
kernel. "Class" is the class a candidate using it gets **when its Check
moves**. If the Check doesn't move, the candidate is `reject`.

| ID | Lever | Best HW delta | Class |
|---|---|---|---|
| L01 | Fully unroll loops that index register arrays | `conv2dk1_i8` 7623 → 504 (-93.4%) | strong |
| L06 | Walking `__restrict` cursors | `convert_copy` 652 → 88 (combined); `zero<float,4096>` 519 → 262 (isolated) | likely |
| L20 | Batch horizontal reductions: several rows through one transposed tree | `mha` `partial_softmax` 8171 → 1696 (4.82x, combined); bf16 `mv` 1154 → 788 | likely (normalize II per row, S17, S20) |
| L19 | One bfp16 stream per operand per loop | `mm_bfp` 4243-4529 → 951 (4.5x, combined); `q4nx_dequant` 2.10x | strong (spills removed, S08, S19) |
| L05 | Fold constants into one mac | `sigmoid` 498 → 118 (-76.3%) | likely |
| L02 | No soft-float or 64-bit libcalls in hot loops | `rms_norm` 1597 → 438 (-72.6%) | strong |
| L07 | `AIE_LOOP_UNROLL(4)` on latency-bound bodies | `leaky_relu` 298 → 86 (-71.1%) | strong, if the screen passes |
| L03 | Avoid f32 vector multiply: split the f32 operand only | matmul epilogue silu 8098 → 3213 (-60.3%) | likely |
| L09 | Split dependency chains | int16 `mv` 291 → 127 (-56.4%) | likely |
| L13 | `UNROLL_FULL`, not `RANGE`, on a loop that switches on its counter | 5.31 → 2.84 ms (-47%, internal model) | likely |
| L18 | Pair two 8x8 output tiles on one 64-lane accumulator | prefill `fv` 2034-2082 → 1265-1393 (-35%) | likely |
| L10 | Make the hot loop innermost and single-block | int8 `mm` 993 → 737 (-25.8%) | strong when a loop newly pipelines; otherwise likely |
| L15 | Producer writes mmul-A order | -23.6% on one block (internal model) | likely |
| L12 | Wide stores, never byte loops | bfp16 shuffle 10.4x per call (rep slope) | strong |
| L04 | Run emulated f32 chains 32 lanes wide | `bf16_exp` 32482 → 15425 (2.11x, combined) | likely |
| L08 | Use the full register width | `rope` 1932 → 298 (combined) | likely |
| L11 | Unsigned counted trip | `axpy` 317 → 178 (-43.8%) | strong |
| L14 | Vector int8 epilogue | -42% on one block (internal model) | likely |
| L16 | Pure strided copy → DMA transform | +12% (internal model) | hand off to `aie-dataflow-opt` |
| L17 | Raise the pipeliner stage cap on a long dependency chain | `q4nx_dequant` 2245-2265 → 2115-2116 (-5.8%) | likely (scale by trips, S09) |

"Internal model" rows are wall-clock or end-to-end numbers from a quantized
model outside this repository. The number is real, but you can't reproduce it
from this repository.

Measured flat: bf16 `mm` (5761 cycles) and int16 `mm` (2017) were unchanged
by L10's K unroll. L06 cursors later moved bf16 `mm` only 5761 → 5713
(-0.8%), because its k loop is at the mv-slot bound (§Bounds). The bf16 GEMV
(1154 cycles) got worse under six kernel-local tweaks (`aie-kernel-opt-hw`
X02-X04) before L20 restructured its reduction.

Each lever below gives:
- **When**: the static signal (remarks rows and meta, `llvm-objdump`).
- **Do**: the change.
- **Check**: what must move in the static output. If it doesn't, the
  candidate is `reject`.
- **HW**: the precedent. "Combined" means the commit applied more than one
  lever, so the number isn't isolated.

Lever IDs follow the shared lessons catalog, so the programming guide and
both kernel skills cite the same numbers.

---

## L01 Fully unroll loops that index register arrays

- **When:** a short fixed-count loop indexes an array of accumulators or
  vectors with its counter (`acc[i]`, `strip[j]`), or calls
  `insert`/`extract`/`set` with it. `llvm-objdump` shows `[sp, #...]`
  loads/stores around the accumulators, and the enclosing loop is in
  `unpipelined_loops` or has a high II.
- **Do:** put `AIE_LOOP_UNROLL_FULL` before the loop. Each index becomes a
  constant, and the array stays in registers.
- **Check:** the stack traffic is gone, and the enclosing loop now has an II.
- **HW:** `conv2dk1_i8/2048x8` 7623 → 504 (isolated). `transpose`
  subtile=8, wall clock 177.2 → 72.0 µs uint32 and 162.7 → 71.3 µs bf16
  (combined with L10 and cursors).

## L02 No soft-float or 64-bit libcalls in hot loops

- **When:** the remarks `libcalls` row lists `__mulsf3`, `__divsf3`,
  `__floatsisf`, `__floatunsisf`, `__ltsf2`, `__gtsf2`, `__muldi3`,
  `__divsi3`, or any `df` helper. The full list is in `traps.md` P01.
- **Do:** use these replacements:
  - compares: `aie::max`/`aie::min`
  - divides: `aie::inv` times a multiply
  - int → float: `aie::to_float`
  - index math: integers, not `double`
  - a per-chunk horizontal max: reduce element-wise across chunks, and reduce
    horizontally once at the end
- **Check:** the `libcalls` row no longer lists the helper, and the loop now has an
  II.
- **HW:**
  - `rms_norm/1024` 1597 → 438 (isolated)
  - with L03: `layer_norm` 2421 → 581; `layer_norm_f32` 13639 → 6094;
    `layer_norm_affine_cast` 12771 → 10631
  - with reduce-once: `softmax/1024x16` 1428 → 993

## L03 Avoid the f32 vector multiply (AIE2P)

- **When:** `aie::mul` or `aie::mac` on `vector<float,N>` in the hot loop, or
  `aie::min`/`aie::max` on it. AIE2P has no f32 vector multiplier, so each one is a bf16 split emulation of
  about 41-60 bundles per vector (`traps.md` P02).
- **Do**, in order:
  1. Skip multiplies by a known 1 or 0 (for example an identity affine).
  2. If one operand is exact in bf16 (a tanh or sigmoid result, or a constant
     chosen to be exact), split only the f32 operand into bf16 limbs and mac
     each limb against the exact operand:
     - Peel `x0 = bf16(x)`, `x1 = bf16(x - x0)`, `x2 = bf16(x - x0 - x1)`, with
       each residual formed in the f32 accumulator (`aie::msc`).
     - **Use three limbs.** 3 × 8 ≥ 24 significand bits, so the product is
       exact. Two limbs change output bits (`aie-kernel-opt-hw` X06).
     - For a compile-time constant, split it at compile time and
       `static_assert` that the limbs sum to it.
  3. AIE2P has no native f32 min/max either, so an f32 clamp is emulated.
     Clamp the rounded bf16 result against bounds rounded with the same mode
     (`conv_even`). Rounding is monotone and leaves representable values
     fixed, so this is exact for finite inputs.
- **Check:** no `vector<float>` multiply is left in the loop, and the II drops.
  Bit-identity is proven on hardware by `aie-kernel-opt-hw` (raw-word diff);
  flag it in the candidate report.
- **HW:**
  - `mm_activation_epilogue/silu` 8098 → 3213 (isolated). Cursors alone on
    the rows without an f32 multiply: relu 1234 → 1106, identity 652 → 518.
  - bf16 clamp, with L06 cursors and L04 32 lanes: `fused_mm` epilogue chunk
    126 → 21, identity per call 1920 → 1084.5 (combined). As landed with
    L07's `UNROLL(2)` on top (commit 23da3300557): chunk 126 → 19 (18-20). Raw output bit-identical, 0 of 327,680 words in 10
    configurations. A mutation that rounded the bounds with `floor` changed
    45,159 words and still passed the 0.04 tolerance (`aie-kernel-opt-hw` X11).

## L04 Run emulated f32 chains 32 lanes wide (AIE2P)

- **When:** an emulated f32 chain (exp, poly) steps 16 lanes. Emulated f32
  math is 32 lanes wide, so a 16-lane `aie::mul` pays for 32 lanes and uses 16.
  Vector `to_fixed`/`to_float` inside the chain means SRS/UPS work plus
  mode-register writes.
- **Do:** step 32 lanes. Replace the vector `to_fixed`/`to_float` floor with
  the magic-number floor `x + 1.5*2^23`.
- **Check:** bundles per element drop. For a `noinline` straight-line body the
  remarks report the *calling* loop's II, so count the bundles yourself.
- **HW:** `bf16_exp/1024` 32482 → 15425 (2.11x) and `exp2f_vec/1024`
  31558 → 14125 (2.23x), combined. Output is bit-identical except that NaN is
  now a quiet NaN. The `fused_mm` epilogue at 32 lanes is in L03's HW list
  (combined). Keep 16 lanes where the LUT tanh is 16-lane (AIE2).

## L05 Fold constants into one mac

- **When:** `(tanh+1)/2`, or constant scales applied as separate multiplies
  and adds.
- **Do:** write `0.5*tanh + 0.5` as one mac into an accumulator preloaded with
  0.5, and fold constant scales together. Power-of-two scaling commutes with
  rounding, so this is bit-exact.
- **Check:** one fewer vector op per element; the II drops.
- **HW:**
  - `sigmoid/1024` 498 → 118 (isolated); LUT build 2435 → 1806
  - with L08: `swiglu` 3335 → 856; LUT build 5124 → 3010
  - with L07: `silu` 1255 → 211; LUT build with no unroll 2691 → 2466

## L06 Walking `__restrict` cursors

- **When:** addressing as `base[i*stride]`. Peano doesn't strength-reduce it to
  post-increment, so the loop body shows address arithmetic. Or the in/out
  pointers lack `__restrict`.
- **Do:** advance a `T *__restrict` cursor, and mark non-aliasing in/out
  pointers `__restrict`. Don't use `__restrict` for in-place kernels, where
  the streams alias.
  For multi-dimensional walks, advance with the `add_2d_byte` /
  `add_3d_byte` intrinsics (unqualified; see
  `aie_kernels/generic/q4nx_dequant.cc`) so the scalar ALU carries no address
  math. Scalar counters in their place cost II17 → 22 (X43).
- **Check:** the II drops (`zero`: 2 → 1 bundle per store), and `[sp, #...]`
  references in the body fall.
- **HW:**
  - `zero<float,4096>` 519 → 262, `<bf16,4096>` 263 → 134, `<u8,4608>`
    150 → 78, fused 32x16 tile 71 → 37 (isolated). The predictions 147 → ~75
    and 391 → ~130 were written down before the run.
  - `mm` B and C cursors across j (isolated; commit 245d0134847): int16
    `c_col_maj` 2873 → 2753 (-4.2%), `b_col_maj+c_col_maj` 3161 → 3041
    (-3.8%), bf16 5761 → 5713 (-0.8%), `mha` 10089 → 10001. Raw output
    bit-identical across 15 cases; `[sp]` references 22 → 10.
  - `expand/576` 839 → 162 (with an exponent-trick convert)
  - `convert_copy/1024` 652 → 88 (with unroll)
  - `q4nx_dequant`: all four cursors on `add_3d_byte`/`add_2d_byte`, part of
    L19 (combined).

## L07 `AIE_LOOP_UNROLL(4)` on latency-bound bodies

- **When:** the pipelined body is latency-bound, with many empty bundles at a
  fixed II. Screen for it (`static-checks.md` §Unroll screen): if the loop's
  `byte_count` stays flat from ×1 to ×4, the extra iterations fill stall slots
  you already pay for.
- **Do:** `AIE_LOOP_UNROLL(4)`. Add a case that reaches the remainder
  (`static-checks.md` §Remainder case).
- **Don't:** apply it to load-, resource- or register-bound bodies (X20-X23).
  Stop when the body grows or spills appear.
- **Check:** II per element drops, and there's no new `[sp, #...]` traffic.
- **HW** (isolated): `leaky_relu` 298 → 86, `mul` 454 → 142, `add` 390 → 150,
  `gelu` 594 → 318.
- **HW, `AIE_LOOP_UNROLL(2)`** on the `fused_mm` epilogue chunk loop (isolated,
  per call): gelu 2582 → 1919.2 (-25.7%), silu 2294.75 → 1767, sigmoid
  2070.75 → 1646.2, identity 991.5 → 976. Chunk cycles gelu 172 → 89, silu
  136 → 70, sigmoid 108 → 55. `UNROLL_FULL` on the same loop grew `.text`
  in every configuration (X33).

## L08 Use the full register width

- **When:** the loop steps 16 lanes on 8/16-bit data or bf16 arithmetic.
- **Do:** step 64 lanes for 8/16-bit data and 32 lanes for bf16 arithmetic.
  Template the lane count on the architecture.
- **Check:** the trip count halves (or quarters) at a similar II.
- **HW:** `rope/1024` 1932 → 298, with L06 cursors; `swiglu` with L05.

## L09 Split dependency chains

- **When:** one long mac chain per block, or lane broadcasts spilled and
  reloaded per row block (`[sp, #...]` in the body).
- **Do:** use two independent accumulators and add them at the end, or block
  the rows so one broadcast feeds several accumulators. Stop before spills
  appear.
- **Check:** the II drops, and the spills are gone.
- **HW:** `mv/32x32 int16` 291 → 127 (two row blocks per broadcast, spills
  gone). `dwconv1d` 2810 → 1505 (two `sliding_mul` chains, unrolled
  `taps.set`, count-down cursors; combined). `exp2f_vec` on AIE2 13997 →
  11981 from an Estrin polynomial, although stack traffic doubled (S22).

## L10 Make the hot loop innermost and single-block

- **When:** Peano pipelines only innermost single-basic-block loops. The
  signal is the loop you care about in `unpipelined_loops` because it
  encloses a short loop, or outer loops that pay prologue and epilogue per
  trip.
- **Do:**
  - Fully unroll a short K reduction into its parent.
  - Fold nested tile loops into one counter.
  - Hoist a per-tile `if` out of a mac loop by splitting it into straight
    loops.
  - Opt in per dtype. The same unroll left bf16 and int16 unchanged on HW,
    and the compiler reports it worse for them (X24).
- **Check:** the parent loop now has an II. Estimate
  `bundles + 5 + (trips-1)×II`, which matched HW within one cycle for
  `mm_bfp_mixed`.
- **HW:**
  - `mm/64x32x64 int8` 993 → 737 (isolated; bf16 and int16 unchanged)
  - `mm_bfp/64x64x64/mixed` 1349 → 1313 and non-square 64x32x32 589 → 545,
    bit-identical over 360448 raw words
  - `AIE_LOOP_UNROLL(2)` on an unpipelined outer loop (`fused_mm`'s bf16 z
    loop around an II35 mac loop), so two trips' C loads and stores overlap:
    k_step 216 → 192-194, identity per call 1084.5 → 991.5 (-8.6%, isolated;
    commit f32e367c609). `UNROLL(2)`, not `FULL`, keeps code bounded for
    larger row counts; full unrolls of z and i spilled or got slower (`aie-kernel-opt-hw` X12).
  - branch split: 10.48 → 9.77 ms end-to-end (internal model)

## L11 Unsigned counted trip

- **When:** a signed trip count, or a signed divide or shift by 2^k (it
  doesn't lower to a shift).
- **Do:** compute the trip count up front as unsigned.
  `AIE_LOOP_MIN_ITERATION_COUNT(1)` was part of this win, but it cost the
  zero-overhead loop in two other kernels (X37). Re-check `non_zol_loops`
  every time you add it.
- **Check:** no `__divsi3`; the II drops; `non_zol_loops` is unchanged.
- **HW:** `axpy/1024` 317 → 178.

## L12 Wide stores, never byte loops

- **When:** an `lda.s8`/`st.s8` loop. Peano won't pipeline it (II72 at any
  unroll, X36).
- **Do:** copy with `uint64_t`/`uint32_t`, or merge byte blocks into 32 B
  vector stores. Align both ends.
- **Check:** the loop pipelines, with 2 memory ops per 8 B instead of 16.
- **HW:**
  - bfp16 shuffle 27.5 → 2.6 µs per call (10.4x), unshuffle 28.7 → 3.2 µs
    (9.1x), by rep slope; 183.0 → 80.7 µs wall
  - explicit `uint64_t` copy: +14% fps (internal model)

## L13 `UNROLL_FULL`, not `RANGE`, on a loop that switches on its counter

- **When:** a small loop whose body branches on the loop variable (for
  example a 3-tap `kx` loop calling a helper that switches on `kx`).
- **Do:** `AIE_LOOP_UNROLL_FULL`. `AIE_LOOP_RANGE` is only a trip-count hint
  and leaves the branch in place.
- **HW:** `RANGE(3,3)` → `UNROLL_FULL` on `kx`: 5.31 → 2.84 ms (internal
  model).

## L14 Vector int8 epilogue (AIE2P)

- **When:** a scalar `acc + bias → SRS → clamp` tail per output element.
- **Do:** add the bias on an int32 vector, then call
  `acc.to_vector<int8>(shift)` under `aie::rounding_mode::conv_even`. That is
  bit-exact with scalar banker's-rounding SRS.
- **HW:** -42% cumulative on one block, 20-25% per kernel (internal model).

## L15 Producer writes mmul-A order (AIE2P)

- **When:** a consumer builds the mmul A operand by scalar gather, and you
  own both ends.
- **Why the gather exists:** `aie::concat` won't join vectors narrower than
  128 bits, so a wide operand can't be assembled from strided narrow loads,
  and the compiler falls back to a scalar byte copy into a stack buffer.
- **Do:** have the producer store in mmul-A order: an mmul result's
  `acc.to_vector<int8>(shift)` is already in the byte order the next mmul
  wants as A, so it is one vector store. The consumer then does one aligned
  `vlda`, or builds a shifted window from two aligned blocks:
  ```cpp
  auto combined = aie::concat(lo_block, hi_block);   // >= 128 bits each side
  return aie::shuffle_down(combined, shift_amt).template extract<N>(0);
  ```
  Both ends must agree on the stride.
- **HW:** -23.6% on one block (internal model).

## L16 Pure strided copy → DMA transform

- **When:** the kernel body is only a rearrangement (a stride-2 deinterleave
  or a transpose).
- **Do:** move it into a memtile `dims_to_stream` transform. The element must
  be ≥ 512 B, and int8 vector loads need a 32 B-aligned start. Dataflow
  placement belongs to `aie-dataflow-opt`.
- **HW:** stride-2 deinterleave, +12% (internal model).
- **Example:** `programming_examples/basic/transposes/transposes.py` shows
  both ends: `--strategy dma` (pure DMA, 4-byte elements only) and
  `--strategy combined` (a shim DMA block reshuffle plus a small kernel).

## L17 Raise the pipeliner stage cap on a long dependency chain

- **When:** the hot loop is pipelined, its remarks `ns` is 3 (the default
  cap), and the body is one long latency chain with no dominant step: ablating
  any single step moves the II by at most a couple of cycles. `q4nx_dequant`'s
  chain was about 48 cycles (load → shuffle → unpack → to_float → mac → srs →
  transpose → store) at II17.
- **Do:** add `-mllvm --aie-pipeliner-max-stagecount=N` (N = 4 or 5) to that
  kernel's factory `compile_flags` only, never globally:
  ```python
  compile_flags=[...] + ["-mllvm", "--aie-pipeliner-max-stagecount=5"],
  ```
- **Check:** statically first. Run the remarks on the arm (they compile with
  the factory's `compile_flags`) and confirm the loop's `ns` rises. The final
  `ii` can barely move, because the postpipeliner can eat the gain:
  `q4nx_dequant`'s pre-RA II went 16 → 11/12 but the final II only 17 → 16.
  A small static move is not a reason to skip the HW A/B; the HW win was
  larger than the static II suggested. Also check `pm_bytes` and the
  `stack_bytes` row against the contract.
- **HW** (isolated; commit c3ca24f4d88), per call:

  | case | default (N=3) | N=4 | N=5 |
  |---|---:|---:|---:|
  | `q4nx_dequant/5120x4` | 2245-2265 | 2119-2122 | 2115-2116 (-5.8%) |
  | `1536x2`, group 24 | 712-715 | 682-684 | 673-677 (-5.4%) |
  | `512x2`, group 8 | 205 | 201 | 201 (-2.0%) |
  | `.text` B | 688/672/672 | 800/784/800 | 912/896/800 |

  N=4 gives most of the win. N=3 compiles to the default code. Kernel frame
  unchanged (64 B); byte-exact gate passes. Noise was about 20 cycles on
  5120x4 and about 1 on the others.

## L18 Pair two 8x8 output tiles on one 64-lane accumulator (AIE2P, bf16)

- **When:** Y += S·V built from several `aie::mmul<8,8,8>` accumulators, with
  a high II and a large stack frame. Prefill `attn_fv` ran four mmul
  accumulators with volatile y loads at II125 (head dim 512) and II97 (256),
  frames 0x840/0xC80.
- **Do:**
  - Keep two neighbouring 8x8 output tiles on one 64-lane `accfloat`
    accumulator (rows 0-3 of both tiles in the low half, rows 4-7 in the high
    half), so one native `vmac.f` per k advances both tiles.
  - Build the S-side operand once per row block and reuse it for every pair.
  - Load the next pair's y before this pair's store. Otherwise the pipeliner
    can't separate the load from the store (a may-alias loop-carried chain,
    RecMII about 35) and serializes them. Clamp the last iteration's pointer
    to itself so nothing reads past y.
  - Keep each output's ascending-k fp32 sum so the result stays bit-identical.
- **Check:** the loop's II and frame drop (II125/97 → 37, frame → 0x700).
  More tiles per accumulator group crashed Peano (`traps.md` P06).
- **HW** (commit 4dde796eff1): `prefill_fv` head dim 512 2034-2082 →
  1265-1393 (-35%); head dim 256 3199-3274 → 2707-2963 (-12%). Full prefill
  round -17% / -15% at 512 and -5% at 256. Raw output bit-identical over 4
  geometries. Variants that lost on HW are `aie-kernel-opt-hw` X13 (static row S15).

## L19 One bfp16 stream per operand per loop (AIE2P)

- **When:** a loop keeps more bfp16 streams live than AIE2P has state
  registers, and the stream state spills every step:
  - output: two `block_vector_output_buffer_stream`s share the single `sf`
    register, so `llvm-objdump` shows `vlda sfl/sfh` and `vst sfl/sfh` to
    `[sp, #...]` around every `vst.push...bfp16`;
  - input: there are two `lf` FIFO registers. `mm_bfp` built four A/B input
    streams plus two C-in and two C-out streams per 2x2 group, and the
    FIFO state and address scalars spilled on every k step (73 `[sp`
    references).
- **Do:** order the output so it is contiguous and write it through one
  stream. In `q4nx_dequant` this came with: one 8x8 tile per iteration with a
  single 64-lane mac (L08), a shuffle mode read from a `constexpr` table so
  both halves share one body, the (n8, step) loop flattened into one counter
  (L10), and every cursor on `add_3d_byte`/`add_2d_byte` (L06).
- **Check:** the `sf` spills are gone; body `[sp]` references drop (26 → 4);
  the II per value drops (II67 per 128 → II17 per 64).
- **HW** (combined; commit 00a21467744), per call: `q4nx_dequant/5120x4`
  4719-4764 → 2245-2265 (2.10x), `1536x2` 1489 → 712-715 (2.09x), `512x2`
  322-329 → 205 (1.59x). Byte-exact under `Tolerance.exact`.
- **Do (`mm_bfp`, input side):** one A and one B stream per 2x2 group that
  hop between the group's two rows with `pop_seek`, and one C output stream
  for the whole call. Loop once over the (rows/2)·(cols/2) groups with the k
  loop fully unrolled. Read the next group's C before writing this group's,
  so the loads overlap the mac tail. Keep the mac order per output tile.
  Two blocks per row between seeks (`pop`, then `pop_seek`) halve the FIFO
  fills, **but only for an even block count**: see `traps.md` P14 and keep a
  seek-after-every-pop loop for odd K.
- **Check:** in-loop `[sp` references gone (73 → 4, all prologue/epilogue);
  `unpipelined_loops` 2 → 0. Normalize the II per 2x2 group (S17): 15 per
  k step became 58 per group.
- **HW** (combined; commit a558bba4c10), per call:
  `mm_bfp/64x64x64x16/bfp16ebs8` 4243-4529 → 951 (4.5x), `64x64x64x4`
  4246-4529 → 951, odd-K `32x24x48x4` 850-894 → 269 (3.2x). Raw HW output
  byte-identical on 33 of 33 comparable outputs (12 shapes × 3 patterns).
  The base kernel's 4243-4529 spread was a period-6 external stall that its
  stack traffic amplified to about 285 cycles; with the spills gone the same
  stall costs 25-27 cycles in one call of six.

## L20 Batch horizontal reductions (AIE2P, bf16)

- **When:** each row ends in its own `reduce_add` or `reduce_max` (or
  another horizontal tree), the remarks meta shows that tree as an
  unpipelined loop or a long straight-line chain (II125 at K=64 in `mv`),
  and the row loop never overlaps it. On hardware the reduction was priced by ablation at 520 of
  1154 cycles (`aie-kernel-opt-hw` §Bounds found by ablation).
- **Do:**
  1. Mac four rows against each b chunk, so one load of b feeds four macs.
  2. Pack the four accumulators (quarter-permuting concat/add, no shuffles)
     so quarter q holds row q's 16 partials, then fold all four rows with one
     `interleave_unzip` + `add` tree instead of four `reduce_add`s.
  3. Finish group g's tree in iteration g+1, after group g+1's macs issue.
     When the mac loop is fully unrolled (chunks ≤ 4), carry the whole
     accumulator. Otherwise carry only the folded `vector<float,16>`:
     carrying the 2048-bit accumulator across the mac loop spilled (stack
     704 B at K=2048).
  4. Ride-alongs: a cursor per row (one pointer post-incremented for all 8
     loads chained them at II9; per-row cursors give II7), and peel the first
     chunk as `aie::mul` so the zero accumulator isn't reloaded from `[sp]`.
  5. Keep the original single-row code for the m % 4 tail, and add a case
     for the mac-loop path (`mv/10x512x4/.../edge-mac-loop`).
- **Check:** the per-row `reduce_add` loops are gone; the group body's II
  **per row** is below the old row loop plus its reduction (II56 / 4 rows at
  K=64); the frame stays within the declared stack. 8 or 16 rows per group
  overflowed the 1024 B Worker stack (1728 / 3328 B).
- **HW** (combined; commit 2da9f2d0d71), per call: `mv/32x256` 1154 → 788
  (-31.7%), 4x64 at `vec_size` 32 (amd/IRON's attention shape) 159 → 110, 4x128 149 → 125, 6-row tail 287 → 273, 10x512
  551 → 518, 4x2048 420 → 408. Raw output bit-identical on 10 shapes: the
  transposed tree pairs lanes exactly as `reduce_add` does.
- **Second kernel, `mha` `partial_softmax`** (flash-attention decode,
  commit f0212ca5c58). The base called a per-row helper for each query row:
  a rounding-mode swap, two passes too short to pipeline and two five-step
  reductions per row.
  - **Do:** run each pass over all rows as one pipelined loop, and fold the
    row max and the row sum eight rows at a time with `interleave_unzip`,
    which pairs each row's halves the way `reduce_max`/`reduce_add` do. Keep
    the old path for shapes and scales the fast path doesn't cover.
  - **Check:** max II rises 13 → 48 and that loop is over `SwpMaxMii`
    (`traps.md` P13); per row it is about 6 bundles. The frame is 192 B. A
    whole-block stack scratch variant ran 3964 but needed 3 KB, over the
    0xD00 Worker stack.
  - **HW** (combined), per call: full 64x64 block 8171 → 1696 (4.82x), causal
    diagonal 10912 → 2162 (5.05x), padded 5107 / 6710 → 1448 / 1740. Raw
    output O bit-identical (0 diffs at `s_eff` 256 and 165).
  - Steps measured on HW one at a time (full / diagonal block):
    - the batched passes and folds alone: 8171 → 2562 / 10912 → 3271;
    - cursors (L06): 2562 → 2327, 1908 → 1861;
    - max before scaling (exact for a positive, finite scale because
      rounding is monotone; other scales take the old path): 2327 → 2151;
    - `AIE_LOOP_MIN_ITERATION_COUNT(2)` on the group loops, which always run
      at least 2 groups, so the postpipeliner overlaps them: 1861 → 1720;
    - `aie::msc(a*s, m, 1)` instead of broadcasting `m`: 1720 → 1694;
    - `UNROLL(4)` on the unpipelinable mask pass (L07): diagonal 2241 → 2162.
- **Rows per group** are bounded by the stack: 4 for `mv` (8 overflowed
  1 KB), 8 for `partial_softmax` (192 B frame).


---

## Compiler reports, not HW (X20-X46)

These variants were dropped because the remarks or the object showed them
worse, so they never reached the NPU. They are **not** hardware numbers. Cite
them as "the compiler reports". A candidate matching a row here is class
`reject`: in the two cases later checked on hardware (S12, S13), the compiler
was right.

| ID | Change | Kernel | Compiler reports |
|---|---|---|---|
| X20 | `UNROLL(2)` | swiglu | spills, 288 → 576 B |
| X21 | `UNROLL(8)` | silu / gelu | II93 / II133 |
| X22 | ×4 on the LUT tanh path | silu LUT build | II77×32 → II269×8: 13% for +848 B. Gate unrolls on `ACTIVATIONS_NATIVE_TANH` |
| X23 | ×4 on a resource-bound body | tanh | flat at II4 |
| X24 | `UNROLL_FULL` on K | bf16 / int16 `mm` | II35 → 156 / spills |
| X25 | `AIE_PREPARE_FOR_POSTPIPELINING` | swiglu | II49 (it disables pipelining, `traps.md` P04) |
| X26 | `UNROLL(2)` | rope | helper goes out of line, II25 |
| X27 | Unconditional unroll | `zero` | 176 → 560 B |
| X28 | `UNROLL_FULL` on the column loop | transpose | 2304 B |
| X29 | 1x4 / 1x1 accumulator expansion | mha | II164 / II26 × 64 pairs ≈ 13312 |
| X30 | 4 accumulators at head dim 256 | flash prefill | 16 × II266 = 4256 vs 3104 |
| X31 | Factoring the polynomial | gelu | II69 (the chain gets deeper) |
| X32 | 3 mac chains / bias moved to the end | dwconv | II31 / II32 vs 29 |
| X33 | `UNROLL_FULL` | fused epilogue | `.text` up in all 5 configurations |
| X34 | `UNROLL(2/4)` | `mm_bfp_mixed` | II16 / II44 vs 6 |
| X35 | Unroll 2 or 4 | cast | worse than 8 |
| X36 | Byte loops at any unroll | bfp shuffle | II72 |
| X37 | `AIE_LOOP_MIN_ITERATION_COUNT` | matmul epilogue, dwconv | lost the zero-overhead loop |
| X38 | `__builtin_memcpy(d,s,8)`, or a byte loop at -O2 | copy | 16 memory ops vs 2 |
| X39 | Making a helper `static` to shrink `.text` | any | 2x smaller object, 0 B in the ELF |
| X40 | Full unroll of the i loop | `mm_fused` | z loop II119 (llvm-aie#1066) |
| X41 | 32 lanes but keeping `to_fixed/to_float`; inlining the poly; interleaving 2-4 vectors; `UNROLL(2/4)` | `bf16_exp` / `exp2f_vec` | II466/427; 0xac0 B frame; frames 0x5c0 / 0x12c0 |
| X42 | K full unroll / unroll 2 / peel k=0 / A1 via `vlda.conv` / rotating operands into the store tail / 1x2 expansion | `mm_bfp_mixed` | tile II84 + spills / II16-17 / 88 per tile / II9 + 4 spills / 91-114 per tile / half the bound |
| X43 | Scalar counters instead of `add_3d_byte` cursors; byte or delta offset tables; 32-lane `to_float` | `q4nx_dequant` | II22 / II25 / II21 / II20 vs 17 per 64 values |
| X44 | bf16 `unroll_k` (K folded into j) / peel the first k-step / int16 1x2 expansion / one 64-lane accumulator for a pair / int16 accumulate in acc32 | `mm` | LICM hoists A patterns into a ~3.5 KB frame (over the 1 KB Worker stack) / II110 with 68 `[sp` refs / II38 per j ≈ 2432 vs 2017 / 6 mv ops per substep vs 4 / not bit-identical |
| X45 | `UNROLL(2)` on the paired fv loop; LICM blockers (select, `j/64`, `ptr>>31`) | prefill `fv` | II92 per 2 pairs (46 vs 37 per pair); folded by LLVM, frame 0xF40 over the 2304 B budget |
| X46 | One flat row loop with both reductions / per-row loop `UNROLL(4)`, `UNROLL(8)` / one fused group loop / scale-then-max in the group loop / 16-lane partials with a wider fold / max folded from 64-lane rows / sum fused into the l loop / exp in two 32-lane halves / mask from a 65-entry uint32 table / mask `UNROLL(8)` | `mha` `partial_softmax` | II106-113 per row / 63.5 cycles per row, spills / II139 / in-loop spills, II37-41 / l loop unpipelined, II56 vs 35 / 78 bundles vs 57 / 0x740 B frame + spills / 36 bundles per 2 rows vs 14 / 19 bundles vs 16 / loses the zero-overhead loop, +160 B |

## Bounds (NO-CHANGE)

When the II already equals the resource or latency bound, the report is
NO-CHANGE with the bound. That is a valid result. `mm_bfp_mixed`'s k loop
sits at II6, which is the `vmac.f` acc→acc latency. AIE2P has 5 accumulator registers (dm0-dm4), so a fifth chain is
impossible. Only the loop-nest overhead moved on hardware (L10).

bf16 `mm`'s k loop is the other measured bound. The bf16 emulation needs 34
mv-slot ops per step, so MII is 34 and the achieved II35 is 97% slot
efficiency. Peano's software pipeliner gives up above MII 27 (`SwpMaxMii` in
the remarks meta `schedule_notes`), so only the postpipeliner runs there.
Cursors moved it -0.8% (L06); report NO-CHANGE for further kernel-local work.

The bf16 GEMV's horizontal reduction looked like a bound after X02-X04,
but it was a per-row latency chain, and L20 batched it. A transposed-A
layout for the bf16 GEMV has never been measured.
