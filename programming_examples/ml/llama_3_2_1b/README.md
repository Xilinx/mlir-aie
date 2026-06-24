# Llama 3.2 1B (INT8) on AI Engine (IRON) — WIP

End-to-end INT8 Llama 3.2 1B decode on the AMD Strix Point NPU2
(AIE2P), implemented in the high-level IRON Python API and structured
to mirror `programming_examples/ml/yolo26n/`. All 16 decoder layers
will run via a single reused worker set with per-layer INT8 weights
streamed from DRAM, plus on-device sampling. Validation target: bit-exact
against the `cautious-eureka` numpy reference oracle.

The dataflow design + per-channel INT8 quant recipe were developed and
simulation-validated in the [cautious-eureka](../../../../) repo
(`npu2/llama_layer_ref.py` is the oracle, `npu2/aie2_llama_iron.py` is
the design source-of-truth). This example is the hardware bring-up.

## Status

| Phase | State |
|---|---|
| 0. Scaffolding (`llama_spec.py`, `placement.py`, env)         | done |
| 1. Dataflow stubs — all topology validated on hardware         | done |
| 2. All 7 real kernels — BIT-EXACT on Strix Halo NPU             | done |
| 3. **Single decoder layer integration — BIT-EXACT end-to-end**  | **done** |
| 4. 16-layer decode chain via `build_decode_design` (small sizes) | pending |
| 5. KV cache append (re-add k_proj/v_proj/rope_k)               | pending |
| 6. ATB-tiled gemm rewrite (M>1, chunked K/N)                   | pending |
| 7. Scale decode to production sizes (D=2048, HD=8192)          | pending |
| 8. Prefill overlay (PREFILL_PLACEMENT + full-causal-softmax)   | pending |
| 9. End-to-end generation (prefill → decode loop → sample)      | pending |

**Prefill is in scope** for this PR: without it, the example can't
actually run text generation (decode needs a prefilled KV cache).
The ATB-tiled gemm rewrite (Phase 6) does double duty — it enables
both production-size decode and prefill, so the two phases share most
of the kernel work.

### Phase 2 real kernels (bit-exact)

Every kernel computes byte-identical output to a numpy reference that
mirrors the same arithmetic (LUT lookups for transcendentals; software
invsqrt; explicit fp32 chains where bf16 multiplication would diverge).

| `make` target          | Kernel              | Tile       | Mismatches |
|---                     |---                  |---         |--- |
| `run_rmsnorm_int8`     | rmsnorm + per-element gamma | (5, 4) | **0 / 2048** |
| `run_gemm_srs`         | int8×int8→int32 + bias + SRS (GEMV) | (0, 2) | **0 / 64** |
| `run_rope`             | half-split RoPE (Llama-3 layout) | (4, 4) | **0 / 512** |
| `run_silu_mul`         | LUT-based SiLU × up | (4, 5) | **0 / 8192** |
| `run_flowkv`           | qk → sv pair (full-softmax)         | (0, 4) → (0, 5) | **0 / 64** |
| `run_sample`           | greedy argmax       | (5, 5)     | **0 / V** |
| `run_rmsnorm`          | bf16 RMSNorm (warm-up)              | (5, 4) | within bf16 ULP |

Key bit-exactness techniques (yolo m9/m10 patterns):
- **LUT-based transcendentals.** Both kernel and reference index the
  SAME precomputed LUT (silu, exp). Build-time codegen emits the LUT
  header from gen_silu_lut.py / gen_exp_lut.py; the test re-runs the
  same Python to reproduce the LUT in numpy.
- **Software invsqrt** (Quake-III + 2 Newton-Raphson). Pure IEEE
  fp32 ops; replaces `aie::invsqrt` which is a HW approximation that
  doesn't match numpy at the last bit.
- **Exact int-domain accumulators** where possible (rmsnorm sum-of-
  squares, gemm dot product). Eliminates fp accumulation-order
  ambiguity.
- **fp32 internal chains** in glue kernels — gamma/probs loaded as
  bf16, cast once to fp32, all multiplies in fp32. bf16 vec multiply
  paths are a follow-up optimization (production wants bf16-internal
  for throughput; v0 prefers correctness).
- **Exact `1.0f / sum`** instead of `aie::inv` (HW reciprocal).

### On-chip sampler + embed gather in one table stream

`run_topk_sample_probe` (`kernels/llama_topk_sample.cc`) is the primitive that
lets decode close the token→next-embedding feedback fully on-device in a **single
pass** over the 262 MB tied embed table. The lm_head must stream that whole table
every token to produce logits; rather than stream it a *second* time to fetch the
sampled token's embedding row (two redundant 262 MB shim streams saturate the
shim DMA and waste ~25 % of the per-token bandwidth — the decode bottleneck), the
sampler keeps a **resident top-k set** of `{logit, global index, embed_sc, embed
row}` as the table flows by once. After the stream it samples over that set →
emits the token id **and** the next token's requantised embedding seed
(`int8[D] + fp32 scale`), with no second pass and one shim channel. Validated
bit-exact (token + full embed seed) for greedy and multinomial-top-k across
vocab/k/seed sweeps.

**Sampling semantics — clean top-k renormalisation.** The softmax + inverse-CDF
draw run over **exactly the k surviving tokens**. This is standard top-k
sampling and is the only semantics expressible in a single resident-set pass.
(The earlier resident/streamed samplers, `llama_sample.cc` /
`llama_sample_streamed.cc`, instead masked filtered logits in place over all V;
at the real vocab V=128256 that masked tail carries the large majority of the
softmax mass and biases the draw — clean renormalisation is the correct fix.)

**Limitations (by design):**
- **`top_k = 0` (full-vocab multinomial) is not supported** in the one-stream
  path. Any token could be drawn, so no bounded resident set is guaranteed to
  hold the winner's embedding row, and the gather would need a second selective
  pass over the table. Greedy (`temperature ≤ 0`) and top-k / top-p sampling
  cover the functional bar; production sampling effectively always uses a
  top-k/top-p cutoff. Full-vocab multinomial would require either streaming the
  table twice or routing the sampled logits back to the host.
- **Resident-row capacity.** The selected rows live on the compute tile, so
  `k × D` bytes must fit alongside the streamed-table fifo in the 64 KB tile
  data memory. At the production `D = 2048` this fits up to `k ≈ 8` on the
  compute tile; larger `k` requires holding the top-k rows in a memtile
  (512 KB) — a placement detail handled when the primitive is fused into the
  chain, not an algorithmic limit.

### Persistent on-device autoregressive loop (capstone)

`LLAMA_CHAIN_ONESTREAM=1` fuses the one-stream sampler+gather into the full
N-layer chain (`run_chain_onestream_mh`): one dispatch maps an input embedding to
`(token, next-token embed seed)` entirely on-device — final_norm → fused lm_head
GEMM + top-k insert + finalize over a **single** 262 MB table pass, emitting both
the sampled token and the requantised next embedding with no DDR logits scratch
and no second gather stream.

`LLAMA_CHAIN_PERSIST=1` (`run_chain_persist_mh`, `PT=` tokens per dispatch) closes
the loop: the sampled token's embed seed feeds back to the chain **on-chip** (via
a depth-2 self-feedback fifo + a seed-mux that picks the host seed for token 0 and
the on-chip feedback for tokens 1..PT-1), so the device generates `PT` tokens in
one dispatch with **zero host involvement between tokens** — the host only streams
weights and drains the `PT` token/seed records. Validated greedy bit-exact against
a numpy autoregressive oracle (PT=2, PT=4) at the real `V=128256`.

**Limitation — KV cache (increment 1).** The persistent loop currently holds the
KV cache at a **fixed position**: the host re-streams the same (pristine) KV to
the device for each of the `PT` tokens, so every step attends over identical KV.
This proves the device-originated control loop (token + seed never leave the chip
across a token boundary) but is **not yet a growing cache** — a true decode must
append each generated token's K/V and advance the position. Making the KV cache
resident in memtiles and growing across the `PT` tokens (with position and cos/sin
auto-advancing on-chip) is the next increment; until then `PT` tokens share one
cache state rather than accumulating context.

## Dataflow stubs (Phase 1)

Every placement pattern the real design needs has been validated
**bit-exact on hardware** by a small stub kernel that runs in the right
tile(s) and the right ObjectFifo topology. Each stub uses simple
math (passthrough / xor / int8 add) so the host can compute the
expected output explicitly. Building real kernels from here on is a
focused kernel-level task — the dataflow risk is retired.

| Target | Validates | CTs |
|---|---|---|
| `make run_gemm_pt`      | 1-CT packed weight-stream payload (act + weights blob) | 1 |
| `make run_gemm_pt_col`  | 2-CT col: activation broadcast + memtile split + memtile join | 2 |
| `make run_gemm_pt_proj` | Full decode-overlay projection: 16 CTs (rows 2–3 × cols 0–7), consolidated runtime args + TAP-based per-col dispatch | 16 |
| `make run_flowkv_pair`  | CT0→CT1 vertical neighbor stream (attn pair0: col 0, rows 4↔5) | 2 |
| `make run_glue`         | 2-input fanin convergence at one CT (rmsnorm+residual / silu+mul shape) | 1 |
| `make run_layer_pt`     | **Full 16-worker single-decoder-layer integration** at real dimensions (D=2048, QD=2048, KVD=512, HD=8192). 3-way h1 broadcast, residual hold on x_in + x1, FlowKV pair, 2-input adds — all together. | 16 |

Each stub kernel pairs with a Python test that bit-exact-compares the
NPU output against the traced-by-hand expected result. The
`layer_pt` test, for instance, expects `out == 6 * x_in (mod 256)`
(see [`test_layer_pt.py`](test_layer_pt.py) for the trace).

## Quickstart

```bash
# 1. Env (one-time)
source /opt/xilinx/xrt/setup.sh
source ../../../utils/quick_setup.sh       # creates ironenv via wheels
pip install --upgrade cmake                # ironenv cmake (needs >= 3.30)
pip install -r ../../../python/requirements_ml.txt

# 2. Point at weights (only needed for Phase 5+ layer-level tests
#    against the cautious-eureka oracle; the stubs need nothing here)
export LLAMA_3_2_1B_WEIGHTS=/path/to/llama-3.2-1b   # dir with model.safetensors

# 3. List targets
make help    # or just open the Makefile
```

## Files

**Design / placement (ports of cautious-eureka)**
- `llama_spec.py` — algorithm + shapes (one decoder layer parameterized by M).
  Self-validates param counts on import.
- `placement.py` — physical tile placement for both decode and prefill
  overlays (`DECODE_PLACEMENT`, `PREFILL_PLACEMENT`). The ONLY place tile
  coordinates appear. `render_diagram` prints the 28/32-tile decode layout.

**Dataflow stubs (Phase 1)**
- `aie2_gemm_int8_srs.py` / `test_gemm_int8_srs_pt.py` — 1-CT frame
- `aie2_gemm_int8_srs_col.py` / `test_gemm_int8_srs_pt_col.py` — 2-CT col
- `aie2_gemm_int8_srs_proj.py` / `test_gemm_int8_srs_pt_proj.py` — 16-CT projection
- `aie2_flowkv_pair.py` / `test_flowkv_pair_pt.py` — qk→sv neighbor stream
- `aie2_glue.py` / `test_glue_pt.py` — 2-input fanin
- `aie2_layer.py` / `test_layer_pt.py` — full single-layer integration

**Stub kernels** (`kernels/`)
- `llama_gemm_int8_srs_pt.cc` — act passthrough, ignores weight blob
- `llama_flowkv_pt.cc` — bitwise-inverts input (qk and sv share semantics)
- `llama_glue_pt.cc` — xor of two inputs
- `llama_layer_pt.cc` — shape-specific copy / tile / add / first-of-two-inputs
  symbols for every call site in the full-layer stub

## Hardware constraints learned during bring-up

These are real and apply to every new IRON design on AIE2P. Apply
preemptively when sketching new designs to skip a build cycle:

1. **Compute tile DMA budget: 2 in + 2 out per CT.** Hardware constraint.
   Surfaces as `tile requires N input/M output DMA channels, but only
   2 input/2 output available`. → Pack per-call constants (weights +
   bias + scale) into one `ObjectFifo` as a packed payload. This is the
   pattern cautious-eureka's `StaticWeightStream` already uses.

2. **`Worker(tile=Tile(col, row))` is required for multi-column designs.**
   Without it the auto-placer piles fifos onto col 0 and hits
   `'aie.tile' op number of output DMA channel exceeded!` on the shim.
   Per-col layouts (like the 8-col decode projection) must pin each
   worker to its column.

3. **`DefaultNPURuntime.run_test` segfaults past ~5 XRT kernel args.**
   Keep runtime args to ~3 big consolidated buffers (e.g. one act, one
   packed weights, one packed outputs); dispatch per-tile via
   `TensorAccessPattern(tensor_dims=[total], offset=c*slice,
   sizes=[1,1,1,slice], strides=[0,0,0,1])` in `rt.fill` / `rt.drain`.
   Matches the `whole_array_iron` consolidated-args style.

4. **One `Kernel` object per unique C symbol — not per call site.**
   `Kernel(symbol, ...)` emits a fresh MLIR func decl every time;
   constructing two `Kernel` objects for the same symbol (even with
   different arg types) collides at `Program.resolve_program()`. Share
   one `Kernel` across workers that have the same signature; use a
   different C symbol per shape when signatures differ.

## Phase 3: full single-layer integration

`make run_layer_real` runs **all 7 real kernels chained end-to-end on
one decoder layer** and bit-exact compares against a numpy reference
that composes the per-kernel reference functions:

```
x_in
 │
 ├─ rmsnorm1 ─ h1 ─ q_proj ─ qf ─ rope ─ qr ─┐
 │                                           ├─ flowkv (qk → sv) ─ af ─ o_proj ─ op ─┐
 │           (k_proj/v_proj/rope_k deferred) │                                       │
 └────────────────────── residual ─────────────────────────────────────── add1 ─ x1 ─┤
                                                                                    │
                                                                                    └─→
x1
 │
 ├─ rmsnorm2 ─ h2 ─ gate_proj ─ gf ─┐
 │              ─ up_proj ────── uf ─┴ silu_mul ─ sf ─ down_proj ─ df ─┐
 └──────────────────── residual ──────────────────────────────── add2 ─┴ layer_out
```

13 workers, 4 consolidated runtime buffers (`x_in`, `weights_blob`,
`kv_cache`, `layer_out`), per-kernel weight streams sourced via
`TensorAccessPattern` taps. Validation at first-bring-up sizes
(D=64, HD=256, head_dim=64, T=16):

```
layer_real NPU vs chained-numpy: D=64  mismatches=0/64  max|diff|=0
BIT-EXACT PASS  (full single-layer end-to-end)
```

**Bit-exact because every kernel was already 0-diff in isolation**;
chaining them and the numpy references reproduces the kernel chain
byte-for-byte.

v0 simplifications (each is a focused follow-up):
- `k_proj`/`v_proj`/`rope_k` dropped — current-token K/V would normally
  rope (K only) and append to the cache; v0 supplies the whole cache
  from a runtime arg
- Small sizes (D=64 vs 2048, HD=256 vs 8192) — production scale-up
  is just changing the `static constexpr int kD/kHD` in the kernel
  .cc files + the corresponding constants in `aie2_layer_real.py`
- Uniform per-tensor scales across all calls — production would use
  per-call calibrated scales

## What still needs work

Per-kernel follow-ups (all are perf, not correctness):
- vectorize the rmsnorm pass-2 / silu / rope / flowkv inner loops
  (currently scalar to keep order deterministic)
- per-channel weight scales on gemm_int8_srs (currently uses a
  single per-tensor right_shift)
- prefill (M > 1) variant of gemm with `aie::mmul<8,8,8,int8,int8>`
- flowkv → chunked online-softmax (the real "flowkv" name) for KV
  cache scalability
- sample → full temperature + top-k + multinomial (needs a PRNG)
- shard the gemm across the 16 projection tiles (dataflow already
  proven by `gemm_pt_proj`)

Integration roadmap (in dependency order):

1. **16-layer decode chain at small sizes.** Reuse the same compute
   tiles across all 16 layers; per-layer weights streamed in
   sequence from a multi-layer weights blob. Tests the tile-reuse
   pattern without changing data sizes; ~1–2 hours.
2. **KV cache append.** Re-add `k_proj`/`v_proj`/`rope_k` and write the
   current token's K/V back to the cache. Decode becomes self-
   contained (no host-supplied cache).
3. **ATB-tiled gemm rewrite.** Currently the gemm has K and N hardcoded
   in the .cc and loops them in a single call. For production sizes
   (wq=2048×2048=4MB, wg=2048×8192=16MB) we need chunked K and chunked
   N — read one K-block at a time, accumulate into an N-block
   accumulator. This is the bulk of the remaining kernel work and
   unblocks both Phase 7 and Phase 8.
4. **Scale decode to production sizes.** D=2048, QD=2048, KVD=512,
   HD=8192. Most of the design parameters update mechanically once
   the ATB gemm is in place.
5. **Prefill overlay.** Different placement (`PREFILL_PLACEMENT`: 24
   projection tiles + 2 attention pairs vs decode's 16+4). Different
   attention path (full causal softmax barrier instead of FlowKV
   chunked online softmax — yolo's m9 attention adapts directly).
   Variable M (128/512/2048) with bin-and-pad.
6. **End-to-end generation.** Prefill the prompt → KV cache filled
   → decode loop calls the chain once per generated token → sample.
   Swap random weights for real Llama 3.2 1B safetensors via
   `gen_llama_data.py` (reads `$LLAMA_3_2_1B_WEIGHTS`).
