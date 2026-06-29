# yolo26n kernels — index

30 `.cc` + 3 `.h` AIE2P kernel files, grouped here by the network block they
serve. The directory is kept flat (one filesystem path per build rule in the
Makefile, no per-subdir `-I` plumbing); this README is the index.

Naming convention: `yolo_<group>_<op>[_<variant>][_vec].cc`. The `_vec` suffix
marks files that use the AIE2P vectorized API (`aie::mmul`, `aie::vector`,
SRS, LUT lookup); files without it are scalar strided-copy helpers where no
vec primitive applies.

Every active kernel has been audited against the playbook in
`../README.md#performance` — constexpr trip counts, bias-init mmul, vec
`to_vector<int8>(rs)` SRS, `AIE_LOOP_RANGE` hints, shape macros.

## Shared headers

| File | Role |
|---|---|
| `yolo_kernel_common.h` | Header-only helpers (every helper is `static __attribute__((always_inline)) inline`). Shared by multiple groups. |
| `yolo_m0_conv2dk3_silu_bias.h` | m0-specific signatures (forked from `aie_kernels/aie2/bottleneck/bn_conv2dk3.cc`). |
| `yolo_conv2dk3_stride2_silu_bias_oiyxi8o8.h` | OIYXI8O8 stride-2 conv signature (shared by m1/m3/m5/m7). |

All kernels also include `../../../../aie_kernels/aie_kernel_utils.h` from
the upstream kernels tree.

## m0 — stem (3×3 stride-2, 512×512×8 → 256×256×16)

| File | Notes |
|---|---|
| `yolo_m0_conv2dk3_silu_bias_vec.cc` | Deep-opt vectorized stem. Host-pre-packed mmul.B weights — no per-row pack on the hot path. |

## m1 / m3 / m5 / m7 — stride-2 OIYXI8O8 conv family

Same kernel binary used for all four blocks; differentiated at link time.

| File | Notes |
|---|---|
| `yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_vec.cc` | Vectorized 3×3 stride-2 INT8 conv with OIYXI8O8 weights. |
| `yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_vec.cc` | Chunked-OC variant (same math, smaller L1 footprint). |

## c3k2_small — m2 / m4 (1×1 + 3×3 1-step blocks)

| File | Notes |
|---|---|
| `yolo_c3k2_small_cv1_split_vec.cc` | 1×1 conv with channel-wise output split. |
| `yolo_c3k2_small_m0_cv1_vec.cc` | 3×3 stride-1 conv (OIYXI8O8). |
| `yolo_c3k2_small_m0_cv2_skip_vec.cc` | 3×3 stride-1 conv + SiLU LUT + int8-saturating skip-add. |
| `yolo_c3k2_small_cv2_concat3_vec.cc` | 1×1 conv on 3 concatenated input rows + SiLU LUT. |

## c3k2_heavy — m6 (inner-pair pattern + cv3 merge)

| File | Notes |
|---|---|
| `yolo_c3k2_heavy_m_0_split_vec.cc` | 1×1 with two parallel branches sharing shape (independent weights/biases/LUTs/rs). |
| `yolo_c3k2_heavy_inner_pair_cv1_vec.cc` | 3×3 stride-1 OIYXI8O8 conv. |
| `yolo_c3k2_heavy_inner_pair_cv1_streamed_vec.cc` | Same inner pattern, streamed-OC variant. |
| `yolo_c3k2_heavy_inner_pair_cv2_skip_vec.cc` | 3×3 stride-1 + SiLU LUT + cross-scale skip-add. |
| `yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_vec.cc` | Streamed-OC variant of the above. |
| `yolo_c3k2_heavy_cv3_concat2_vec.cc` | 1×1 merge for the cv3 concat path. |

## m8 — megakernel front + back (PSA precursor)

Two fused kernels are pasted into 2/4/6 compute tiles depending on `M8_TILES`
(see `../scripts/m8_megakernel_*tile.py`).

| File | Notes |
|---|---|
| `yolo_m8_front_cv1_split_fused_vec.cc` | Called per (row, cv1_chunk_idx) — cv1 + m_0_split fused. |
| `yolo_m8_back_cv3_cv2_fused_vec.cc` | Called per (row, cv2_chunk_idx) — cv3 + cv2 fused (cv3 recomputed on chunk 0). |

## m9 — PSA attention block

10 kernels covering qkv projection, attention scores, softmax-row, sv
multiplication, FFN0/FFN1, output projection + skip, position embed add,
and the pack/transpose helpers.

| File | Notes |
|---|---|
| `yolo_m9_cv1_split_vec.cc` | Chunked-OC variant of m9 cv1 1×1 INT8 conv. |
| `yolo_m9_qkv_vec.cc` | 1×1 INT8 conv 128 → 256 (bias-init only, no activation). |
| `yolo_m9_pack.cc` | Scalar strided-copy transpose. Single .cc compiled as `qkv_pack`, `qk_pack`, `v_pack` via `-D`. |
| `yolo_m9_attn_score_fused_vec.cc` | Per (head, query_idx) attention-score + softmax fused. |
| `yolo_m9_sv_row_vec.cc` | Per (head, n) sv-product row. |
| `yolo_m9_pe_add_row.cc` | Position-embed add row (V held as two per-head buffers on sv tile). |
| `yolo_m9_proj_skip_row_vec.cc` | Fused attn/proj 1×1 + cross-scale skip-add. |
| `yolo_m9_ffn_0_silu_row_vec.cc` | 1×1 INT8 conv 128 → 256 + bias + SiLU (chunked-OC streamed). |
| `yolo_m9_ffn_1_skip_row_vec.cc` | 1×1 INT8 conv 256 → 128 + plain (same-scale) skip-add. |
| `yolo_m9_cv2_concat2_streamed_vec.cc` | Streamed cv2 1×1 conv (final m9 mixing). |

## m10 — classifier head (1×1 + GAP + linear + softmax)

| File | Notes |
|---|---|
| `yolo_m10_conv2dk1_silu_xy_pool_vec.cc` | Fused 1×1 256 → 1280 + HardSiLU + xy-pool (GAP). |
| `yolo_m10_concat_pool.cc` | Merges two pool halves before the final Gemm (scalar, vectorized 32 B copies). |
| `yolo_m10_linear_gemm.cc` | 1280 → 2 Linear (raw row-major weight layout, no OIYXI8O8). |
| `yolo_m10_softmax.cc` | Same algorithm as `m9_softmax_row` but on flat `(n_classes,)`. |
