<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Llama 3.2 1B (INT8) decode on AI Engine (IRON)

A fully on-NPU INT8 decode of Llama 3.2 1B on the AMD Strix Halo NPU2 (AIE2P),
written in the high-level IRON Python API. One `aiecc`-built xclbin runs the whole
per-token decode on the AI Engine array — all 16 transformer layers, attention with
a growing KV cache, the tied 262 MB lm_head, the sampler, and the token→embedding
feedback — with the host doing only weight I/O. The W8A8-dynamic INT8 recipe tracks
the reference bf16 HF model (75 % greedy top-1 on hard one-word prompts, 94–100 %
on prose); the device is bit-exact to the numpy INT8 oracle, so it adds no error
beyond quantization.

> Status: the decode path is complete and validated on hardware. RMSNorm,
> SiLU×up, RoPE, FlowKV attention, and the top-k sampler's hot path are
> vectorized (2.21 → 4.71 tok/s at N=16). Prefill (batched M>1), the unplaced
> layout, and further per-kernel tuning are the open follow-ups — see
> [What's next](#whats-next).

---

## 1. Decode design

### The autoregressive loop, on-chip

Decode generates one token at a time: `token → embedding → 16 layers → lm_head →
sample → next token`. The whole loop runs on the AI Engine array within a single
device dispatch; the only thing crossing the host boundary per token is the weight
stream from DDR (fundamental — a 1 GB model can't live on-chip) and, at the end,
the generated token ids.

```
                       ┌──────────────── one device dispatch, PT tokens ────────────────┐
  host: xin(token0) ──►│ seed-mux ─► router ─► [ rmsnorm ─ q/k/v ─ rope ─ KV-append ─    │
                       │     ▲                   flowkv-attn ─ o_proj ─ +res ─ rmsnorm ─  │
                       │     │                   gate/up ─ silu ─ down ─ +res ] ×16 ─►    │
                       │     │                 final_norm ─► lm_head GEMM + top-k insert  │
                       │     │                            ─► sample ─► (token, embed seed)│
                       │     └──────────── embed seed feeds back on-chip ────────────────┤
  host: weights (DDR) ─────────────────────────► streamed per token ─────────────────────►
  host: tokens out  ◄──────────────────────────── PT packed records ◄─────────────────────┘
```

Key design pieces (each validated bit-exact in isolation before integration):

- **One-stream sampler + embed gather** (`kernels/llama_topk_sample.cc`). The
  lm_head must stream the full 262 MB tied embed table every token. Rather than
  stream it twice (once for logits, once to fetch the sampled token's embedding),
  the sampler keeps a **resident top-k set** of `{logit, index, embed_sc, embed
  row}` as the table flows by **once**, then emits both the token id and the
  next-token embedding seed. No second pass, one shim channel.

- **On-chip token feedback** (`LLAMA_CHAIN_PERSIST=1`). The sampled token's embed
  seed feeds back to layer 0 **on-chip** via a depth-2 self-feedback fifo + a
  seed-mux (host seed for token 0, on-chip feedback for tokens 1..PT-1). The host
  does zero compute and zero token handling between tokens.

- **Growing KV cache** (`LLAMA_CHAIN_PERSIST_GROW=1`). Each token appends its K/V
  on-chip at an advancing position (the append writes slot `T_used` and bumps it,
  so **position advances on-device**; the host never computes a position) and
  attention widens to include the new slot. Rope advances with position too. This
  is real autoregressive decode — context accumulates.

- **Resident KV cache** (`LLAMA_CHAIN_PERSIST_RESIDENT=1`). The KV body lives in
  worker-local buffers on the attention tiles, seeded once on token 0 then
  read-modify-written in place — **zero per-token KV DMA, host streams only
  weights**. Fits a compute tile at small `N_LAYERS` (32 KB/head at N=2); the full
  16-layer cache (256 KB/head) needs a memtile (the one remaining residency item).

### Single dispatch, single xclbin, fully on-NPU

For a given config, decode is one `aiecc`-built xclbin and one device dispatch that
generates `PT` tokens. Every transformer op, the lm_head, the sampler, and the
token feedback run as AI Engine kernels. The host builds the static instruction
stream and supplies/reads DDR buffers; it performs no per-token compute. The only
runtime-dynamic value in the loop — the sampled token — stays on-chip via the
feedback fifo, which is what makes it a genuine autoregressive loop rather than a
static replay.

### INT8 recipe (W8A8-dynamic)

- **Weights:** symmetric per-output-channel INT8 (`gen_llama_data.py`), streamed
  from DDR per token.
- **Activations:** per-token dynamic INT8 (absmax) at each matmul input; the
  residual and KV carriers also carry per-token scales (the self-calibration work
  — an earlier int8-everywhere version with a static residual scale collapsed to
  0/20, fixed by dynamic per-token scales).
- **Glue in bf16/fp32:** rmsnorm, rope, softmax, silu use full-precision internal
  chains with LUTs for transcendentals (so the kernels are bit-exact to numpy).

---

## 2. Performance

Measured on hardware (`--time` on the persist tests; device clock via
`result.npu_time`). Per-token = dispatch time / PT.

| Config | per-token | tok/s | bytes/token | DMA floor | ratio |
|---|---|---|---|---|---|
| **N=16 (full model)** | **212 ms** | **4.71** | 1253 MB | 10.45 ms | **20× over floor** |
| N=2 (bring-up) | 69 ms | 14.48 | 389 MB | 3.24 ms | 21× over floor |

RMSNorm, SiLU×up, RoPE, FlowKV attention, and the top-k sampler's hot insert
path are now vectorized (`aie::mac`/`aie::mul` over 64-lane int8/fp32 groups,
scalar only where AIE2P has no vector form — LUT gathers and the final
round-to-i8); N=16 moved from 452 ms/token (2.21 tok/s) to 212 ms/token, a
~2.1× speedup, with ratio-over-DMA-floor dropping from 43× to 20×.

**Decode is still compute-bound, not DMA-bound** — the harness's own
compute/DMA-floor ratio still reads "COMPUTE-bound (vectorization is the
lever)" at both configs, just a smaller gap than before. The weight prefetch
(depth-2 per-layer weight fifos stream layer L+1 while layer L computes) is
already hiding the DMA; the remaining wall is kernel compute. The ratio still
worsens with depth (more layers per token), which is why N=16 is further from
the floor than N=2.

**Vectorization remains the lever for the rest of the gap.** For reference,
FastFlowLM (tuned, vectorized, decode-on-NPU) reaches 60–89 tok/s on this
model — vectorizing the five kernels above roughly halved the gap (from
~30–40× to ~13–19×); closing the rest likely needs the unplaced layout (Phase
7, better placement can shrink dispatch overhead) and further per-kernel
tuning (loop pipelining, the LUT-gather techniques `flm_gemma4`'s
bank-replication trick points at) rather than a new architectural change.

> Decode controls **TPS** (sustained generation throughput). Prefill controls
> **TTFT** (time to first token) and is a separate, not-yet-built phase.

---

## 3. Accuracy / quality

The device is **bit-exact to the numpy INT8 oracle** (every kernel validated
0-diff in isolation). The remaining question is whether the INT8 **recipe** tracks
the real model. We compare against the true HF Llama 3.2 1B **bf16** weights
(loaded full-precision from the safetensors — no torch/transformers needed).

Evidence chain: `device == numpy-int8` (bit-exact) → `numpy-int8 == NPU silicon`
(identical 15/20 below) → `int8 == HF bf16` (next).

### Next-token greedy agreement, NPU vs HF bf16 (20 hard prompts, N=16)

The N=16 decode chain on silicon, scored against the true bf16 model:

| Prompt | HF bf16 | NPU int8 | |
|---|---|---|---|
| The capital of France is | ` Paris` | ` Paris` | ✓ |
| The capital of Italy is | ` Rome` | ` Rome` | ✓ |
| The capital of Japan is | ` Tokyo` | ` Tokyo` | ✓ |
| The capital of Germany is | ` Berlin` | ` Berlin` | ✓ |
| The author of Hamlet is | ` not` | ` a` | ✗ |
| The largest planet in our solar system is | ` called` | ` the` | ✗ |
| The chemical symbol for water is | ` H` | ` H` | ✓ |
| The first president of the United States was | ` George` | ` George` | ✓ |
| The opposite of black is | ` white` | ` white` | ✓ |
| The number of legs on a spider is | `:\n` | `\n` | ✗ |
| Two plus two equals | ` four` | ` four` | ✓ |
| The sun rises in the | ` east` | ` east` | ✓ |
| Romeo and Juliet was written by | ` William` | ` William` | ✓ |
| Mount Everest is in the country of | ` Nepal` | ` Nepal` | ✓ |
| The Eiffel Tower is located in | ` the` | ` the` | ✓ |
| The boiling point of water in Celsius is | ` ` | ` ` | ✓ |
| The smallest prime number is | ` ` | ` ` | ✓ |
| Shakespeare wrote in the | ` ` | ` ` | ✓ |
| The currency of Japan is the | ` yen` | ` Yen` | ✗ |
| The Great Wall is in | ` China` | ` the` | ✗ |

**NPU vs HF bf16: 15/20 = 75 % top-1** — identical to the numpy INT8 recipe's
15/20, confirming the device adds no error beyond quantization. The 5 misses are
benign near-ties (` Yen`/` yen` casing; ` the`/` China`; ` a`/` not`), not
garbage — the INT8 logits cluster the right answers at the top.

### Perplexity / agreement on prose (N=16)

`accuracy_vs_hf.py` (numpy INT8 vs true bf16, teacher-forced on running text):

| Metric | Value |
|---|---|
| next-token top-1 agreement | 94–100 % |
| int8-argmax in bf16 top-5 | 100 % |
| perplexity ratio (int8 / bf16) | 1.02–1.08× |

Sub-10 % perplexity degradation and 94–100 % greedy agreement is solid for INT8
PTQ; the recipe faithfully tracks the real model on natural text. (Hard one-word
prompts are the harder case — hence 75 % there.)

---

## 4. How to reproduce

### Environment (one-time)

```bash
cd <mlir-aie>
source /opt/xilinx/xrt/setup.sh
source programming_examples/ml/llama_3_2_1b/ironenv/bin/activate
source utils/env_setup.sh install/
cd programming_examples/ml/llama_3_2_1b
```

The gated `meta-llama/Llama-3.2-1B` checkpoint is found where amd/IRON's examples
find it: `model.safetensors` and `tokenizer.model` in
`$IRON_EXAMPLE_WEIGHTS_DIR/llama3.2-1b/` (`IRON_EXAMPLE_WEIGHTS_DIR` defaults to
`/srv`); `--weights`/`--tokenizer` (`--hf-weights` for the bf16 reference)
override it. The INT8 weights live in `data/`, generated from it by
`gen_llama_data.py`. The bf16 reference reads the safetensors directly.

Each test compiles the design it checks with `@iron.jit` (cached under
`NPU_CACHE_HOME`, so a rerun reuses it), or runs a prebuilt one given
`--xclbin`/`--instr`. Each design's own CLI (`python aie2_*.py ...
--xclbin-path X --insts-path Y`) compiles it ahead of time.

> **Always set `LLAMA_CHAIN_N=16`** for any accuracy/timing/baseline run.
> `N_LAYERS` defaults to 2 (a fast bring-up toy); a 2-layer run silently produces
> meaningless quality numbers.

### Accuracy (no device — pure numpy)

```bash
# INT8 recipe vs true HF bf16: top-1/top-5 agreement + perplexity on prose
python accuracy_vs_hf.py --max-tokens 128

# 20-prompt greedy bench, INT8 recipe vs true HF bf16
LLAMA_CHAIN_N=16 python bench_quality_mh.py --bf16-ref --residual-dyn --attn-lut
```

### Accuracy on silicon (NPU vs HF bf16)

```bash
# compile the N=16 decode chain, then score the device vs bf16
LLAMA_CHAIN_N=16 python bench_quality_mh.py --device --self-prefill --bf16-ref
```

### Full persistent decode (on-chip loop + growing KV) + timing

```bash
# the full 16-layer autoregressive decode (PT tokens / dispatch)
LLAMA_CHAIN_N=16 python test_chain_persist_grow_mh.py

# per-token TPS at full config (reuses the compiled design)
LLAMA_CHAIN_N=16 python test_chain_persist_grow_mh.py --time
```

### Resident-KV decode (host streams only weights, N=2)

```bash
python test_chain_persist_resident_mh.py
```

### Primitive / kernel validation (bit-exact, isolation)

```bash
python test_topk_sample_probe.py    # one-stream sampler + embed gather
python test_chain_onestream_mh.py   # fused chain + lm_head + sampler
python test_rmsnorm_int8.py         # also test_rmsnorm_int8_dyn, test_rope_int8
python test_silu_mul_int8.py --selfcal   # and test_flowkv.py --dyn, ...
```

---

## Configuration flags (env)

The chain tests read these; each test picks its own mode.

| Flag | Meaning |
|---|---|
| `LLAMA_CHAIN_N` | number of decoder layers (**set 16** for real runs; default 2) |
| `LLAMA_CHAIN_PT` | tokens generated per dispatch (persistent loop; default 4) |
| `LLAMA_CHAIN_T` | KV cache length (default 128) |
| `LLAMA_CHAIN_ONESTREAM_KSET` | the one-stream sampler's resident top-k set size (default 8) |
| `LLAMA_SAMPLE_TEMP` / `_TOPK` / `_SEED` | sampler params (temp≤0 = greedy) |

The chain modes are `build()`'s CompileTime flags in `aie2_chain_dynscale_mh.py`
(`--sample`, `--onestream`, `--persist`, `--persist-grow`, `--persist-resident`
on its CLI). Each `PERSIST*` flag is a strict superset of the one above; the
default (all off) chain stays byte-identical.

---

## Hardware constraints (AIE2P / IRON)

Real constraints learned during bring-up — apply preemptively to skip build cycles:

1. **Compute-tile DMA budget: 2 in + 2 out per tile.** Pack per-call constants
   (weights + bias + scale) into one `ObjectFifo` payload.
2. **`Worker(tile=Tile(col, row))` required for multi-column designs** — the
   auto-placer otherwise piles fifos on col 0 and overflows the shim.
3. **`run_test` segfaults past ~5 XRT kernel args.** Keep to a few consolidated
   buffers and dispatch sub-regions via `TensorAccessPattern` taps. (This is why
   the persistent loop packs token+seed into one output buffer.)
4. **One `Kernel` object per unique C symbol.** Share it across same-signature
   workers; use a distinct symbol per shape.
5. **L1 is 64 KB/tile.** Resident KV (32 KB/head at N=2) fits only with a reduced
   worker stack; the full 16-layer cache (256 KB/head) needs a memtile.
6. **Peano AIE2P codegen quirks** (see `kernels/`): `1.0f/x` is a ~10-bit HW
   reciprocal (use the NR `sw_recip`); fp32 stack arrays kept across loops can be
   corrupted (stash int LUT indices instead); `G_ROTL` fails to legalize (use a
   64-bit concat-shift rotate).

Bit-exactness techniques: LUT-based transcendentals (kernel and numpy index the
same generated LUT), software invsqrt (Quake-III + NR), exact int-domain
accumulators, and full-precision fp32 glue chains.

---

## What's next

- **Unplaced layout** — `placement.py` and the chain builder's explicit
  `Tile(...)` calls still hand-assign all 32 compute tiles; letting the
  placer derive layout from the dataflow graph (as mobilenet and the other
  current examples do) is the remaining architectural change, and may shrink
  dispatch overhead further.
- **Further per-kernel tuning** — decode is still 20× compute-bound at N=16
  after vectorizing (down from 43×). Loop pipelining hints and LUT-gather
  techniques (e.g. `flm_gemma4`'s bank-replicated gather) are the likely next
  gains; a per-stage NOOP ablation (mirrors `yolo26n/scripts/ablate_chain.sh`)
  would pinpoint which kernel now dominates.
- **Memtile-resident KV at N=16** — the last residency step (256 KB/head → memtile;
  the host-ferried growing mode already scales to N=16 today).
- **Prefill overlay** — batched M>1 attention + GEMM for TTFT (a separate phase;
  decode is the focus here).
