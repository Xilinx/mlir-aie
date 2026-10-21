<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Llama 3.2 1B (INT8) decode on AI Engine (IRON)

An INT8 decode of Llama 3.2 1B on an AMD NPU2 (AIE2P), written in the IRON
Python API and compiled with `@iron.jit` to one full ELF. Each dispatch runs the
whole per-token decode on the AI Engine array for 1 to 4 tokens, chosen per
dispatch, or a batch of up to 4 prompt tokens together: all 16 transformer
layers, attention with a growing KV cache, the tied 262 MB lm_head, the sampler,
and the token→embedding feedback. The device is bit-exact to its numpy oracle
(`llama_oracle.py`), and against the bf16 Hugging Face model that arithmetic
agrees on 84–92 % of next tokens at a mean KL of 0.024–0.049 (see Accuracy).

This example started from PR #3161, @hunhoffe's Llama 3.2 1B bring-up, whose
kernels, recipe and multi-head attention layer it carries forward into one
compiled design.

> Status: on NPU2 (Strix), decode runs at 44.8 ms/token (22.3 tok/s) at
> position 64 and 475 ms/token (2.1 tok/s) at 128k, every token checked
> against the oracle. Prefill runs on the device in batches of up to 4 prompt
> tokens (`generate.py`: prompt in, text out): a 6-token prompt's first token
> takes 124.6 ms.

---

## 1. Decode design

### The autoregressive loop, on the device

Decode generates one token at a time: `token → embedding → 16 layers → lm_head →
sample → next token`. The whole loop runs on the AI Engine array within a single
device dispatch (`aie2_decode.py`), for up to 4 tokens; what crosses the DDR
boundary per token is the weight stream (a 1 GB model cannot live on-chip), the
live KV blocks, and the activations staged between the GEMV array and the cores
around it.

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
  its KV group's chunk), o and down (whose outputs a glue core adds to the fp32
  residual), gate and up.
  The layers' weights stream over eight shim lanes, one per core, on
  `aie::mmul<8, 8, 8>` (`kernels/llama_gemv_mma.cc`): a prefill batch's up to 4
  activations share one pass of the weights.

- **One-stream sampler + embed gather** (`kernels/llama_topk_sample.cc`). The
  lm_head must stream the full 262 MB tied embed table every token. Rather than
  stream it twice (once for logits, once to fetch the sampled token's embedding),
  the sampler keeps a **resident top-k set** of `{logit, index, embed_sc, embed
  row}` as the table flows by **once**, then emits both the token id and the
  next-token embedding seed. No second pass, one shim channel. Its temperature,
  top-k and seed are set per dispatch; its xoshiro state carries across tokens
  in the stage buffer.

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
  advances with position too. Since attention reads only the live prefix, one
  build at a large `T` serves every context.

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
blocks to read and where its rows go; and the sampler's settings.

Prefill runs on the device in batches: a dispatch's one pass takes up to 4
prompt tokens as its rows, within one KV block, each row's input being the seed
the lm_head core makes of the token's embed row, which it fetches from the
lm_head's weights (`gat{m}`, `pg{m}`, `rows`). The GEMV array runs each
projection once for all the rows; the norms, RoPE, attention and SiLU take them
a row at a time. Only the prompt's last batch runs the lm_head and sampler
(`lm{t}`), for its last row. The host's part is tokenization and splitting the
prompt into batches (`DecodeShape.batches`).

### INT8 recipe (W8A8-dynamic)

- **Weights:** symmetric per-output-channel INT8 (`gen_llama_data.py`), streamed
  from DDR per token.
- **Activations:** per-token dynamic INT8 (absmax) at each matmul input; the KV
  carriers also carry per-token scales.
- **Residual stream in fp32:** each add is fp32, and each norm reads the
  residual requantized per token. An int8 residual would let BOS's massive
  activation (|x| ~ 146) crush the token's other channels.
- **Gate, up and Q in fp32:** SiLU runs on the fp32 gate (`exp_nonpos` and a
  Newton reciprocal) times the fp32 up, and Q is roped in fp32 before its
  per-head quantization.
- **Glue in fp32:** rmsnorm, rope, softmax and silu, each written so the numpy
  oracle reproduces it exactly: attention's and the sampler's exp is
  `exp_nonpos`, the inverse square root a Quake seed and two Newton steps, and
  reciprocals round as IEEE division.
- **The norm's input and the o projection's quantization are now as close to
  the recipe as the design gets:** RMSNorm reads the fp32 residual directly
  (no int8 requant on the way in), and the attention output is quantized once
  for the o projection, not per head then again on the concatenated row. The
  device's remaining gap to the recipe is the quantized weights and
  activations everywhere else, which both recipe and device share.

---

## 2. Performance

Measured on Strix (`test_chain_persist_grow_mh.py --save-run`/`--from-run`,
so the oracle check and the timing are separate runs), `taskset -c 2-23`,
fastest of 20 after 10 warm-ups, one 128k-row-KV-cache build timed at each
context:

| Context (position of the first token) | per-token | tok/s |
|---|---|---|
| **64** | **44.83 ms** | **22.31** |
| 1024 | 51.14 ms | 19.55 |
| 2048 | 52.27 ms | 19.13 |
| 4096 | 60.39 ms | 16.56 |
| 8192 | 73.58 ms | 13.59 |
| 16384 | 101.13 ms | 9.89 |
| 32768 | 153.23 ms | 6.53 |
| 65536 | 260.48 ms | 3.84 |
| 131068 | 475.27 ms | 2.10 |

The RMSNorm and attention-output fixes (below) were measured at 64 and 128k
only, interleaved before/after each other with the exactness check run
offline so it can't skew the timing; they removed a quantize-dequantize step
each, so the rows in between are carried from the pre-fix sweep and are not
expected to move.

**What bounds it.** The layers' weights stream over the array's eight lanes
(about 52 GB/s), the lm_head's 262 MB over one lane (about 19 ms a token), and
attention adds about 3.3 ms per 1k tokens of context. That is up from about
2.5 ms/1k measured just before the Oct-20 restructure (#3947), whose
score-to-probability step moved from an int8-quantized exp lookup table to
the oracle's exact polynomial (`vec_exp_nonpos`, called 16 times a 64-row
block): the per-context cost of the bit-exactness the checks above need,
not reverted.

**Tokens per dispatch.** amdxdna times a dispatch out after about 5 s. The host
picks each dispatch's token count from its first position
(`DecodeShape.passes_for` in `llama_decode.py`): as many of the 4 as fit a 4 s
budget by a cost model (19 ms of lm_head a token, plus per layer 1.80 ms and
0.163 ms per 1k rows of context). At 16 layers that is 4 tokens all the way to
128k, where a dispatch measures 1.91 s.
The array's activations and outputs pass through a DDR `stage` buffer
between it and the cores around it (see `aie2_decode.py`). For comparison,
FastFlowLM v0.9.30 reports 64.5 tok/s at 1k and 13.6 tok/s at 128k for this
model on a Ryzen AI 7 350 (Krackan)
([benchmarks](https://fastflowlm.com/docs/benchmarks/llama3_results/)).

> Decode controls **TPS** (sustained generation throughput). Prefill controls
> **TTFT** (time to first token). `generate.py` at N=16, from before
> tokenization to the first generated token, the median of 5 requests on a
> 6-token prompt: TTFT 124.6 ms (119.4 ms of prefill dispatches, 5.2 ms on the
> host), then 22.1 tok/s.

By prompt length (`generate.py --prompt-tokens`, median of 5), prefill runs
one dispatch of up to 4 tokens at a time, each a full N=16-layer pass, so
TTFT scales with the number of dispatches rather than with one big batched
op:

| Prompt length | TTFT |
|---|---|
| 512 | 8.16 s |
| 1024 | 17.08 s |
| 2048 | 37.05 s |

For comparison, FastFlowLM reports 2136 tok/s prefill at 2k
([benchmarks](https://fastflowlm.com/docs/benchmarks/llama3_results/)), about
960 ms for a 2048-token prompt; larger prefill batches (What's next) are the
lever to close this gap, not a change to TTFT's measurement.

---

## 3. Accuracy / quality

The device is **bit-exact to its numpy oracle** (`llama_oracle.py`): the
decode test checks every token, KV row, sampler state and final residual
against it, and a greedy token must be the oracle's argmax. So scoring the
oracle against the bf16 Hugging Face model scores the device. The INT8 recipe
(`numpy_llama_ref.py`) is shown beside it for scale: the recipe's gap to bf16
is the quantization's own cost, and the device now tracks the recipe closely
(RMSNorm and the attention output no longer add their own quantization
steps). The reference is the true Llama 3.2 1B **bf16** checkpoint, loaded
from the safetensors (no torch or transformers).

### KL and top-1 against bf16

`test_llama.py -m extensive`'s three prompts (the opening of *King Lear*,
teacher-forced), mean KL(bf16 ‖ model) of the next-token distributions:

| Prompt | tokens | device KL | device top-1 | recipe KL | recipe top-1 |
|---|---|---|---|---|---|
| small | 25 | 0.0492 | 84.0 % | 0.0409 | 88.0 % |
| medium | 61 | 0.0311 | 90.2 % | 0.0301 | 88.5 % |
| large | 105 | 0.0242 | 92.4 % | 0.0228 | 91.4 % |

On running prose (`accuracy_vs_hf.py`'s default sample, 47 positions):

| Metric | device | recipe |
|---|---|---|
| next-token top-1 agreement | 87.2 % | 89.4 % |
| argmax in bf16's top 5 | 100 % | 100 % |
| mean KL(bf16 ‖ model) | 0.0182 | 0.0159 |
| perplexity ratio (model / bf16) | 0.975× | 0.987× |

### Next-token greedy agreement on 20 hard prompts (N=16)

`bench_quality_mh.py --path numpy --bf16-ref` (the device's arithmetic,
prefilling each prompt as the device does), **20/20 = 100 %** against bf16:

| Prompt | bf16 | device |
|---|---|---|
| The capital of France is | ` Paris` | ` Paris` |
| The capital of Italy is | ` Rome` | ` Rome` |
| The capital of Japan is | ` Tokyo` | ` Tokyo` |
| The capital of Germany is | ` Berlin` | ` Berlin` |
| The author of Hamlet is | ` a` | ` a` |
| The largest planet in our solar system is | ` a` | ` a` |
| The chemical symbol for water is | ` H` | ` H` |
| The first president of the United States was | ` George` | ` George` |
| The opposite of black is | ` white` | ` white` |
| The number of legs on a spider is | `:\n` | `:\n` |
| Two plus two equals | ` four` | ` four` |
| The sun rises in the | ` east` | ` east` |
| Romeo and Juliet was written by | ` William` | ` William` |
| Mount Everest is in the country of | ` Nepal` | ` Nepal` |
| The Eiffel Tower is located in | ` the` | ` the` |
| The boiling point of water in Celsius is | ` ` | ` ` |
| The smallest prime number is | ` ` | ` ` |
| Shakespeare wrote in the | ` ` | ` ` |
| The currency of Japan is the | ` Japanese` | ` Japanese` |
| The Great Wall is in | ` the` | ` the` |

---

## 4. How to reproduce

### Environment (one-time)

```bash
cd <mlir-aie>
source /opt/xilinx/xrt/setup.sh
source ironenv/bin/activate
source utils/env_setup.sh install/
pip install tiktoken          # the tokenizer generate.py and the benches use
cd programming_examples/ml/llama_3_2_1b
```

The gated `meta-llama/Llama-3.2-1B` checkpoint is found where amd/IRON's examples
find it: `model.safetensors` and `tokenizer.model` in
`$IRON_EXAMPLE_WEIGHTS_DIR/llama3.2-1b/` (`IRON_EXAMPLE_WEIGHTS_DIR` defaults to
`/srv`); `--weights`/`--tokenizer` (`--hf-weights` for the bf16 reference)
override it. The INT8 weights live in `data/`, generated from it once:

```bash
python gen_llama_data.py      # writes data/: INT8 weights, scales, gammas, tokenizer
```

The bf16 reference reads the safetensors directly.

The decode test, `test_chain_persist_grow_mh.py`, compiles `aie2_decode.py`
with `@iron.jit` to a full ELF (cached under `NPU_CACHE_HOME`, so a rerun
reuses it), or runs a prebuilt one (`--elf`, from
`python aie2_decode.py ... --full-elf-path X`, which the same JIT key reuses).

`test_llama.py` is the pytest over it. Its default tier (what lit runs) is
the decode test's cases at 2 layers on synthetic weights. `-m extensive`
needs data/ and the bf16 safetensors: at 16 layers, on a small, a medium and
a large prompt (25, 61 and 105 tokens), the oracle's next-token distribution
must stay within a mean KL of the bf16 model's, and every token the device
generates must be the oracle's argmax.

```bash
python -m pytest test_llama.py -m "not extensive"
IRON_EXAMPLE_WEIGHTS_DIR=/path/to/models python -m pytest test_llama.py -m extensive
```

> **Pass `--n-layers 16`** to the decode test for any accuracy or timing run:
> it defaults to 2 layers (a fast bring-up check), whose numbers mean nothing
> for the model. `bench_quality_mh.py` defaults to the model's 16.

### Accuracy (no device: numpy)

```bash
# the device's arithmetic vs bf16 on prose: top-1/top-5, KL, perplexity
# (--model recipe scores the INT8 recipe instead)
python accuracy_vs_hf.py --max-tokens 128

# 20-prompt greedy bench, the device's arithmetic vs bf16
python bench_quality_mh.py --path numpy --bf16-ref
```

### On silicon: text, and accuracy vs bf16

```bash
# prompt in, text out, 5 requests timed (--repeats); --check fails unless each
# generated token is the oracle's argmax, and reports agreement with
# numpy_llama_ref
python generate.py --prompt "The capital of France is" --max-new 32 --check

# prefill and predict each prompt on the NPU, vs bf16
python bench_quality_mh.py --bf16-ref
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
taskset -c 2-23 python test_chain_persist_grow_mh.py --n-layers 16 --time

# the context sweep of section 2: one build at 128k, each context checked
# and timed (the oracle takes about 100 s a token at 128k)
python aie2_decode.py --n-layers 16 --t 131072 --full-elf-path build/X.elf
taskset -c 2-23 python test_chain_persist_grow_mh.py --n-layers 16 --t 131072 \
    --elf build/X.elf --contexts 64 1024 2048 4096 8192 16384 32768 65536 131068 --time

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
tokens a dispatch can generate (1 to 4, default 4). `test_compile_time_keys.py`
checks that each CompileTime value is part of the JIT cache key. Per dispatch,
as ScratchpadParameters, go the tokens it generates (`--passes`, default by
`passes_for`), the first token's position (`--start-pos`) and the sampler's
temperature, top-k and seed (`--sample-temp`, `--sample-topk`, `--sample-seed`;
temperature ≤ 0 is greedy, top-k 0 the whole set).

---

## Hardware constraints (AIE2P / IRON)

Constraints that shape the design:

1. **Compute-tile DMA budget: 2 in + 2 out per tile.** Pack per-call constants
   (weights + bias + scale) into one `ObjectFifo` payload.
2. **Placement.** The GEMV array's 8 cores are pinned
   (`Worker(tile=Tile(...))`, `CORE_TILES`) so each sits above its weight
   lane's shim column; the SA placer (`--placer=sa_placer`, its seed pinned)
   places the other cores, since the sequential placer leaves their flows
   unroutable along the shim row.
3. **One `Kernel` object per unique C symbol.** Share it across same-signature
   workers; use a distinct symbol per shape.
4. **L1 is 64 KB/tile.** The KV cache lives in DDR; an attention core holds a
   block of 64 rows at a time.
5. **32 ScratchpadParameters per design.** The per-pass and per-row
   parameters share them, which is what holds a dispatch at 4 passes.

---

## What's next

One compiled design serves decode and batched prefill, with the per-call
values (passes, a batch's rows, position, live context, the sampler's
settings) as ScratchpadParameters. Next:

- **The lm_head over more lanes** (it streams over one, about 19 ms a token),
  and the glue fused into the GEMV array's epilogues.
- **Larger prefill batches**: the mma tile has 8 activation slots and the
  design fills 4, its per-row parameters (`pg{m}`, `gat{m}`) sharing the
  32-parameter budget with the passes'. This is most of TTFT's gap to
  FastFlowLM above: one dispatch per 4 prompt tokens means a 2048-token
  prompt pays 512 dispatches' worth of per-dispatch overhead.
- **Long-context attention**: at 128k it takes about 90 % of a token's time.
