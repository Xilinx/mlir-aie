<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Llama 3.2 1B (INT8) decode on AI Engine (IRON)

An INT8 decode of Llama 3.2 1B on an AMD NPU2 (AIE2P), written in the IRON
Python API and compiled with `@iron.jit` to one full ELF. Each dispatch runs the
whole per-token decode on the AI Engine array for 1 to 4 tokens, chosen per
dispatch, each a prompt token or a generated one: all 16 transformer layers, attention with a growing KV cache, the tied 262 MB lm_head, the sampler,
and the token→embedding feedback. The W8A8-dynamic INT8 recipe tracks the
reference bf16 HF model (75 % greedy top-1 on hard one-word prompts, 94–100 % on
prose); the device is bit-exact to the numpy INT8 oracle, so it adds no error
beyond quantization.

> Status: decode is validated on hardware at 47.9 ms/token (20.9 tok/s) at
> N=16 at short context and 389 ms/token at 128k. Prefill runs on the device
> token by token (`generate.py`: prompt in, text out); batched prefill is
> next — see [On the device, and on the host](#on-the-device-and-on-the-host)
> and [What's next](#whats-next).

---

## 1. Decode design

### The autoregressive loop, on the device

Decode generates one token at a time: `token → embedding → 16 layers → lm_head →
sample → next token`. The whole loop runs on the AI Engine array within a single
device dispatch (`aie2_decode.py`), for up to 4 tokens; what crosses the DDR
boundary per token is the weight stream (fundamental — a 1 GB model can't live
on-chip), the live KV blocks, and the activations staged between the GEMV array
and the cores around it.

```
                    ┌──────────────── one device dispatch, P ≤ 4 tokens ────────────────┐
 DDR stage: seed ──►│ [ rmsnorm ─ qkv ─ rope + KV row ─ attention ─ o + residual ─      │
                    │   rmsnorm ─ gate, up ─ silu ─ down + residual ] × N ─ final norm ─ │
                    │   lm_head + top-k insert ─ sample ─► token, embed seed ─┐         │
                    │   ▲                                                      │         │
                    │   └────────── next token's seed, through the stage ──────┘         │
 DDR: weights ─────►│ 8 lanes into the GEMV array's 8 cores; the lm_head on one lane    │
 DDR: KV cache ◄───►│ each token's live blocks in, its new row out                      │
 host: records ◄────│ P packed [seed | scale | token]                                   │
                    └───────────────────────────────────────────────────────────────────┘
```

Key design pieces (each validated bit-exact in isolation before integration):

- **Shared 8-lane GEMV array** (`aie2_gemv_array.py`). Eight cores run every
  projection as units on one broadcast activation: qkv (fused, each core giving
  its KV group's chunk), o and down (fused with the residual add), gate and up.
  The layers' weights stream over eight shim lanes, one per core.

- **One-stream sampler + embed gather** (`kernels/llama_topk_sample.cc`). The
  lm_head must stream the full 262 MB tied embed table every token. Rather than
  stream it twice (once for logits, once to fetch the sampled token's embedding),
  the sampler keeps a **resident top-k set** of `{logit, index, embed_sc, embed
  row}` as the table flows by **once**, then emits both the token id and the
  next-token embedding seed. No second pass, one shim channel. Its temperature,
  top-k and seed are set per dispatch.

- **Token feedback.** The sampled token's embed seed goes to the seed slot of
  the `stage` buffer in DDR, where the next token's first RMSNorm reads it, in
  the same dispatch or the next; the host writes it only for the first token.
  (DDR, not core memory: after the driver's 5 s autosuspend the cores come back
  reset.)

- **Growing KV cache.** The cache lives in DDR, int32
  `[layer][half][block][head][64 rows]`, a row being one token's quantized K and V
  and their scales, for up to `T` tokens. Each layer, a token's attention reads
  only the live blocks of its 8 KV heads, one linear transfer per half; two cores
  per KV head fold them block by block, alternating (`llama_attn_block.cc`, an
  exact online softmax), and the token's own new row joins its last block on chip
  and is the one row written back. Rope advances with position too.

### On the device, and on the host

Every transformer op, the lm_head, the sampler, the token feedback and the KV
append run as AI Engine kernels, and one compiled design serves every dispatch.
The host (`DispatchSession` in `llama_npu.py`) keeps one hardware context and,
per dispatch, writes the design's ScratchpadParameters (`DecodeShape.params` in
`llama_decode.py`): how many tokens it generates, which every
worker reads; the position of the dispatch's first token, which the attention
workers read; its row in the RoPE table the weight blob carries, built once
with Llama 3's rope scaling; per token slot, whether it runs (a slot that does
not run moves nothing), how many KV blocks to read and where its row goes; and
the sampler's settings. What the host still does, each to move to the device in
the stages of the one-design restructure:

- **Prefill** runs on the device token by token: a prompt token is a token slot
  whose input is the seed the lm_head core makes of the token's embed row, which
  it fetches from the lm_head's weights (`gat{t}`, `pg{t}`, `rows`), rather than
  the previous token's sample, and the prompt tokens before the last skip the
  lm_head and sampler (`lm{t}`); the host only says which tokens are the
  prompt's. Batched prefill, many prompt rows a pass, is the next stage.
- **Tokenization**, which stays on the host.

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

Measured on NPU2 (Strix) with `test_chain_persist_grow_mh.py --time`
(`taskset -c 2-4`, median of 10 timed dispatches after warmup); per-token =
dispatch time / tokens per dispatch. N=16, 4 tokens a dispatch unless noted:

| Context (position of the first token) | per-token | tok/s |
|---|---|---|
| **64** | **47.9 ms** | **20.9** |
| 4096 | 60.1 ms | 16.6 |
| 32768 | 132.9 ms | 7.5 |
| 65536 | 218.2 ms | 4.6 |
| 131068 | 388.8 ms | 2.6 |

N=16 went from 452 ms/token to 212 ms by vectorizing the kernels, then to
91.2 ms by moving the remaining scalar fp32 epilogues onto vectors (bit-exact),
taking the lm_head's top-k insert off soft float, and double-buffering the
up-projection weights, then to 86.8 ms by reading only the live KV blocks,
then to 48.6 ms by running every projection on the shared 8-lane GEMV array
(`aie2_decode.py`, now on aie::mmul, `kernels/llama_gemv_mma.cc`). Two cores
attend each KV head's blocks, alternating, and the vectorized block kernel
(`llama_attn_block.cc`, 3.76x per block) took 32k from 362 to 133 ms/token.

**What bounds it now.** The layers' weights stream over the array's eight
lanes (about 52 GB/s), but the lm_head's 262 MB still stream over one lane
(about 19 ms a token), and attention adds about 2.6 ms per 1k tokens of
context.

**Tokens per dispatch.** amdxdna times a dispatch out after about 5 s. The host
picks each dispatch's token count from its first position
(`DecodeShape.passes_for` in `llama_decode.py`): as many of the 4 as fit a 4 s
budget by a cost model fit to the table above (19 ms of lm_head a token, plus
per layer 1.80 ms and 0.163 ms per 1k rows of context). At 16 layers that is
4 tokens all the way to 128k (1.56 s a dispatch there).
The array's activations and outputs pass through a DDR `stage` buffer
between it and the cores around it (see `aie2_decode.py`). FastFlowLM reaches
60–89 tok/s on this model.

> Decode controls **TPS** (sustained generation throughput). Prefill controls
> **TTFT** (time to first token). `generate.py` at N=16, token by token, the
> prompt's tokens before its last skipping the lm_head: 6 prompt tokens, TTFT
> 190 ms (31.6 prompt tok/s); 27, 782 ms (34.5); then 20.9 tok/s.

---

## 3. Accuracy / quality

The device is **bit-exact to the numpy INT8 oracle** (every kernel validated
0-diff in isolation). The remaining question is whether the INT8 **recipe** tracks
the real model. We compare against the true HF Llama 3.2 1B **bf16** weights
(loaded full-precision from the safetensors — no torch/transformers needed).

Evidence chain: `device == numpy-int8` (bit-exact) → `numpy-int8 == NPU silicon`
(identical 15/20 below, old chain) → `int8 == HF bf16` (next).

### Next-token greedy agreement on 20 hard prompts (N=16)

`bench_quality_mh.py`. The bf16 column and the old chain's column are from an
earlier run, of the old chain design (the bf16 checkpoint is not on the box the
current decode was measured on, so they are not re-measured); the last column is
the current decode (`aie2_decode.py`), prefilling each prompt on the device:

| Prompt | HF bf16 (old run) | NPU, old chain vs bf16 | NPU, aie2_decode |
|---|---|---|---|
| The capital of France is | ` Paris` | ` Paris` ✓ | ` Paris` |
| The capital of Italy is | ` Rome` | ` Rome` ✓ | ` Rome` |
| The capital of Japan is | ` Tokyo` | ` Tokyo` ✓ | ` Tokyo` |
| The capital of Germany is | ` Berlin` | ` Berlin` ✓ | ` Berlin` |
| The author of Hamlet is | ` not` | ` a` ✗ | ` a` |
| The largest planet in our solar system is | ` called` | ` the` ✗ | ` the` |
| The chemical symbol for water is | ` H` | ` H` ✓ | ` H` |
| The first president of the United States was | ` George` | ` George` ✓ | ` George` |
| The opposite of black is | ` white` | ` white` ✓ | ` white` |
| The number of legs on a spider is | `:\n` | `\n` ✗ | ` a` |
| Two plus two equals | ` four` | ` four` ✓ | ` four` |
| The sun rises in the | ` east` | ` east` ✓ | ` east` |
| Romeo and Juliet was written by | ` William` | ` William` ✓ | ` William` |
| Mount Everest is in the country of | ` Nepal` | ` Nepal` ✓ | ` Nepal` |
| The Eiffel Tower is located in | ` the` | ` the` ✓ | ` the` |
| The boiling point of water in Celsius is | ` ` | ` ` ✓ | ` ` |
| The smallest prime number is | ` ` | ` ` ✓ | ` ` |
| Shakespeare wrote in the | ` ` | ` ` ✓ | ` ` |
| The currency of Japan is the | ` yen` | ` Yen` ✗ | ` Yen` |
| The Great Wall is in | ` China` | ` the` ✗ | ` China` |

**Old chain vs HF bf16: 15/20 = 75 % top-1**, identical to the numpy INT8
recipe's 15/20; its 5 misses are benign near-ties (` Yen`/` yen` casing;
` the`/` China`; ` a`/` not`), not garbage. **aie2_decode vs the INT8 recipe's
reference forward (`numpy_llama_ref.forward_full`): 16/20 = 80 %**; it differs
from the old chain on two prompts (` China` for the Great Wall, ` a` for the
spider), its prompt inputs now being the device's own embed seeds. Its bf16 score
awaits the checkpoint.

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
source ironenv/bin/activate
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
`NPU_CACHE_HOME`, so a rerun reuses it). The decode test,
`test_chain_persist_grow_mh.py`, runs a full ELF (or a prebuilt one, `--elf`,
from `python aie2_decode.py ... --full-elf-path X`); the others run
an xclbin (or a prebuilt one, `--xclbin`/`--instr`, from a design CLI's
`--xclbin-path X --insts-path Y`).

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

### On silicon: text, and accuracy vs HF bf16

Tokenizing needs tiktoken: `pip install tiktoken`.

```bash
# prompt in, text out, with TTFT and tok/s (--check scores each token against
# numpy_llama_ref's argmax, teacher-forced)
python generate.py --prompt "The capital of France is" --max-new 32

# prefill and predict each prompt on the NPU, vs bf16
LLAMA_CHAIN_N=16 python bench_quality_mh.py --device --bf16-ref
```

### Decode + timing

```bash
# the full 16-layer autoregressive decode (4 tokens a dispatch; --passes picks 1-4)
LLAMA_CHAIN_N=16 python test_chain_persist_grow_mh.py

# a 6-token prompt (seeded random ids) before generating
LLAMA_CHAIN_N=16 python test_chain_persist_grow_mh.py --prompt-tokens 6

# sampled instead of greedy (the sampler's settings are per dispatch)
LLAMA_CHAIN_N=16 python test_chain_persist_grow_mh.py --sample-temp 0.7 --sample-topk 4

# per-token TPS at full config (reuses the compiled design)
LLAMA_CHAIN_N=16 python test_chain_persist_grow_mh.py --time

# a generation loop: 10 dispatches carrying the KV cache and seed, with the
# NPU left idle (and suspended) between them
LLAMA_CHAIN_N=16 python test_chain_persist_grow_mh.py --dispatches 10 --idle-s 6
```

### Primitive / kernel validation (bit-exact, isolation)

```bash
python test_topk_sample_probe.py    # one-stream sampler + embed gather
python test_gemv_array.py           # the 8-lane GEMV array
python test_layer_mh.py             # one decoder layer
python test_rmsnorm_int8.py         # also test_rmsnorm_int8_dyn, test_rope_int8
python test_silu_mul_int8.py --selfcal   # and test_flowkv.py --dyn, ...
```

---

## Configuration

The decode test and `bench_quality_mh.py` read these from the environment and
pass them to the design as CompileTime values.

| Variable | Meaning |
|---|---|
| `LLAMA_CHAIN_N` | number of decoder layers (**set 16** for real runs; default 2) |
| `LLAMA_CHAIN_T` | KV cache length (default 128) |
| `LLAMA_CHAIN_ONESTREAM_KSET` | the one-stream sampler's resident top-k set size (default 8) |

The rest of `build()` in `aie2_decode.py` is on its CLI too: `--p-max`, the
tokens a dispatch can generate (default 4). `test_compile_time_keys.py` checks
that each CompileTime value is part of the JIT cache key. Per dispatch, as
ScratchpadParameters, go the tokens it generates (`--passes`, default by
`passes_for`), the first token's position (`--start-pos`) and the sampler's
temperature, top-k and seed (`--sample-temp`, `--sample-topk`, `--sample-seed`;
temperature ≤ 0 is greedy, top-k 0 the whole set).

---

## Hardware constraints (AIE2P / IRON)

Real constraints learned during bring-up — apply preemptively to skip build cycles:

1. **Compute-tile DMA budget: 2 in + 2 out per tile.** Pack per-call constants
   (weights + bias + scale) into one `ObjectFifo` payload.
2. **`Worker(tile=Tile(col, row))` required for multi-column designs** — the
   auto-placer otherwise piles fifos on col 0 and overflows the shim.
3. **`run_test` segfaults past ~5 XRT kernel args.** Keep to a few consolidated
   buffers and dispatch sub-regions via `TensorAccessPattern` taps. (This is why
   the decode packs each token's seed and id into one record buffer.)
4. **One `Kernel` object per unique C symbol.** Share it across same-signature
   workers; use a distinct symbol per shape.
5. **L1 is 64 KB/tile.** The KV cache lives in DDR; an attention core holds a
   block of 64 rows at a time.
6. **Peano AIE2P codegen quirks** (see `kernels/`): `1.0f/x` is a ~10-bit HW
   reciprocal (use the NR `sw_recip`); fp32 stack arrays kept across loops can be
   corrupted (stash int LUT indices instead); `G_ROTL` fails to legalize (use a
   64-bit concat-shift rotate).

Bit-exactness techniques: LUT-based transcendentals (kernel and numpy index the
same generated LUT), software invsqrt (Quake-III + NR), exact int-domain
accumulators, and full-precision fp32 glue chains.

---

## What's next

One compiled design for decode, token-by-token prefill and batched prefill, with
the per-call values (tokens per dispatch, position, live context, the sampler's
settings) as ScratchpadParameters:

- **The lm_head over more lanes** (it streams over one, about 19 ms a token),
  and the glue fused into the GEMV array's epilogues.
- **Batched prefill**, many prompt rows a pass through the GEMV array, for TTFT.
