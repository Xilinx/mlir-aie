<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Llama 3.2 1B (INT8) decode on AI Engine (IRON)

An INT8 decode of Llama 3.2 1B on an AMD NPU2 (AIE2P), written in the IRON
Python API and compiled with `@iron.jit` to one full ELF. Each dispatch runs the
whole per-token decode on the AI Engine array for 1 to 4 tokens, chosen per
dispatch, or a batch of up to 4 prompt tokens together: all 16 transformer
layers, attention with a growing KV cache, the tied 262 MB lm_head, the sampler,
and the token→embedding feedback. The W8A8-dynamic INT8 recipe tracks the
reference bf16 HF model (75 % greedy top-1 on hard one-word prompts, 94–100 % on
prose); the device is bit-exact to its numpy oracle (`llama_oracle.py`), which
follows the INT8 recipe up to the recipe's approximations (see Accuracy).

> Status: decode is validated on hardware at 47.5 ms/token (21.1 tok/s) at
> N=16 at short context and 389 ms/token at 128k. Prefill runs on the device
> in batches of up to 4 prompt tokens (`generate.py`: prompt in, text out) —
> see [On the device, and on the host](#on-the-device-and-on-the-host) and
> [What's next](#whats-next).

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

Key design pieces:

- **Shared 8-lane GEMV array** (`aie2_gemv_array.py`). Eight cores run every
  projection as units on one broadcast activation: qkv (fused, each core giving
  its KV group's chunk), o and down (fused with the residual add), gate and up.
  The layers' weights stream over eight shim lanes, one per core, on
  `aie::mmul<8, 8, 8>` (`kernels/llama_gemv_mma.cc`): a prefill batch's up to 4
  activations share one pass of the weights.

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
  and is the one row written back. A batch's queries read each block once, the
  batch's rows joining its block in turn so each query sees its own prefix. Rope
  advances with position too.

### On the device, and on the host

Every transformer op, the lm_head, the sampler, the token feedback and the KV
append run as AI Engine kernels, and one compiled design serves every dispatch.
The host (`DispatchSession` in `llama_npu.py`) keeps one hardware context and,
per dispatch, writes the design's ScratchpadParameters (`DecodeShape.params` in
`llama_decode.py`): how many passes it runs and how many rows the first one
has (one, or a prompt batch of up to 4), which every worker reads; the position
of the dispatch's first token, which the attention workers read; its row in the
RoPE table the weight blob carries, built once with Llama 3's rope scaling; per
pass, whether it runs (a pass that does not run moves nothing), how many KV
blocks to read and where its rows go; and the sampler's settings. What the host still does, each to move to the device in
the stages of the one-design restructure:

- **Prefill** runs on the device in batches: a dispatch's one pass takes up to
  4 prompt tokens as its rows, within one KV block, each row's input being the
  seed the lm_head core makes of the token's embed row, which it fetches from
  the lm_head's weights (`gat{m}`, `pg{m}`, `rows`). The GEMV array runs each
  projection once for all the rows; the norms, RoPE, attention and SiLU take
  them a row at a time. Only the prompt's last batch runs the lm_head and
  sampler (`lm{t}`), for its last row; the host only splits the prompt into
  batches (`DecodeShape.batches`).
- **Tokenization**, which stays on the host.

### INT8 recipe (W8A8-dynamic)

- **Weights:** symmetric per-output-channel INT8 (`gen_llama_data.py`), streamed
  from DDR per token.
- **Activations:** per-token dynamic INT8 (absmax) at each matmul input; the
  residual and KV carriers also carry per-token scales (a static residual scale
  scores 0/20 on the greedy prompts below).
- **Glue in fp32:** rmsnorm, rope, softmax and silu, each written so the numpy
  oracle reproduces it exactly: SiLU and the sampler's exp read generated
  tables, attention's exp is a fixed-point polynomial, the inverse square root
  a Quake seed and two Newton steps, and reciprocals round as IEEE division.

---

## 2. Performance

Measured on NPU2 (Strix) with `test_chain_persist_grow_mh.py --n-layers 16
--time --n-warmup 10 --n-iters 20` under `taskset`, the median dispatch over
the tokens it runs (4 a dispatch), after the run checks that dispatch against
the oracle:

| Context (position of the first token) | per-token | tok/s |
|---|---|---|
| **64** | **47.5 ms** | **21.1** |
| 4096 | 60.1 ms | 16.6 |
| 32768 | 132.9 ms | 7.5 |
| 65536 | 218.2 ms | 4.6 |
| 131068 | 388.8 ms | 2.6 |

The longer contexts are from an earlier build of this design and are not
re-measured. Every projection runs on a shared 8-lane GEMV array
(`kernels/llama_gemv_mma.cc`), and two cores attend each KV head's blocks,
alternating (`llama_attn_block.cc`).

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
> **TTFT** (time to first token). `generate.py` at N=16, the prompt in batches
> of up to 4 tokens, only the last batch running the lm_head: 6 prompt tokens,
> TTFT 117 ms (51 prompt tok/s), then 21.2 tok/s.

---

## 3. Accuracy / quality

The device is **bit-exact to its numpy oracle** (`llama_oracle.py`; the
decode test checks every token, KV row and record against it). The oracle
follows the INT8 recipe (`numpy_llama_ref.py`) but not bit for bit: its inverse
square root and attention exp are approximations, so teacher-forced on "The
capital of France is" the device picks the recipe's argmax on 27 of 32 tokens.
The question below is whether the INT8 recipe tracks the real model. We compare against the true HF Llama 3.2 1B **bf16** weights
(loaded full-precision from the safetensors — no torch/transformers needed).



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

The decode test, `test_chain_persist_grow_mh.py`, compiles `aie2_decode.py`
with `@iron.jit` to a full ELF (cached under `NPU_CACHE_HOME`, so a rerun
reuses it), or runs a prebuilt one (`--elf`, from
`python aie2_decode.py ... --full-elf-path X`).

> **Pass `--n-layers 16`** to the decode test for any accuracy or timing run:
> it defaults to 2 layers (a fast bring-up check), whose numbers mean nothing
> for the model. `bench_quality_mh.py` defaults to the model's 16.

### Accuracy (no device — pure numpy)

```bash
# INT8 recipe vs true HF bf16: top-1/top-5 agreement + perplexity on prose
python accuracy_vs_hf.py --max-tokens 128

# 20-prompt greedy bench, INT8 recipe vs true HF bf16
python bench_quality_mh.py --bf16-ref --residual-dyn --attn-flowkv
```

### On silicon: text, and accuracy vs HF bf16

Tokenizing needs tiktoken: `pip install tiktoken`.

```bash
# prompt in, text out, with TTFT and tok/s (--check scores each token against
# numpy_llama_ref's argmax, teacher-forced)
python generate.py --prompt "The capital of France is" --max-new 32

# prefill and predict each prompt on the NPU, vs bf16
python bench_quality_mh.py --device --bf16-ref
```

### Decode + timing

```bash
# the full 16-layer autoregressive decode (4 tokens a dispatch; --passes picks 1-4)
python test_chain_persist_grow_mh.py --n-layers 16

# a 6-token prompt (seeded random ids) before generating, in batches of up to
# 4; --vs-batch 1 also runs it a token per dispatch and compares the two
python test_chain_persist_grow_mh.py --n-layers 16 --prompt-tokens 6 --vs-batch 1

# sampled instead of greedy (the sampler's settings are per dispatch)
python test_chain_persist_grow_mh.py --n-layers 16 --sample-temp 0.7 --sample-topk 4

# per-token TPS at full config, after checking the timed dispatch
python test_chain_persist_grow_mh.py --n-layers 16 --time --n-warmup 10 --n-iters 20

# a generation loop: 10 dispatches carrying the KV cache and seed, with the
# NPU left idle (and suspended) between them
python test_chain_persist_grow_mh.py --n-layers 16 --dispatches 10 --idle-s 6
```

---

## Configuration

The decode test takes these flags and passes them to the design as
CompileTime values; `bench_quality_mh.py` takes the first two.

| Flag | Meaning |
|---|---|
| `--n-layers` | decoder layers (**16** for real runs; the test's default is 2) |
| `--t` | KV cache length (default 128) |
| `--kset` | the sampler's resident top-k set size (default 8) |

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
3. **One `Kernel` object per unique C symbol.** Share it across same-signature
   workers; use a distinct symbol per shape.
4. **L1 is 64 KB/tile.** The KV cache lives in DDR; an attention core holds a
   block of 64 rows at a time.
5. **Peano AIE2P:** `G_ROTL` fails to legalize (use a 64-bit concat-shift
   rotate).

---

## What's next

One compiled design serves decode and batched prefill, with the per-call
values (passes, a batch's rows, position, live context, the sampler's
settings) as ScratchpadParameters. Next:

- **The lm_head over more lanes** (it streams over one, about 19 ms a token),
  and the glue fused into the GEMV array's epilogues.
- **Larger prefill batches**: the mma tile has 8 activation slots and the
  design fills 4, its per-row parameters (`pg{m}`, `gat{m}`) sharing the
  32-parameter budget with the passes'.
