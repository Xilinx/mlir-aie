<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Llama 3.2 1B (INT8) decode on AI Engine (IRON)

An INT8 decode of Llama 3.2 1B on an AMD NPU2 (AIE2P), written in the IRON
Python API and compiled with `@iron.jit` to one full ELF. Each dispatch runs the
whole per-token decode on the AI Engine array for 1 to 4 tokens, chosen per
dispatch, or a batch of up to 4 prompt tokens together: all 16 transformer
layers, attention with a growing KV cache, the tied 262 MB lm_head, the sampler,
and the token→embedding feedback. The device is bit-exact to its numpy oracle
(`llama_oracle.py`); Accuracy scores that arithmetic against the bf16 Hugging
Face model.

> Status: on NPU2 (Strix), decode and prefill run on the device, prefill in
> batches of up to 4 prompt tokens (`generate.py`: prompt in, text out), every
> token checked against the oracle. Current timings are on the
> [dashboard](https://xilinx.github.io/mlir-aie/dashboard/).

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
 DDR: weights ─────►│ 8 lanes into the GEMV array's 8 cores; the lm_head on two lanes   │
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
  activations share one pass of the weights. Each lane is staged in its
  column's memtile and issued a layer ahead, so DDR keeps streaming while the
  array waits on the cores around it.

- **One-stream sampler + embed gather** (`kernels/llama_topk_sample.cc`). The
  lm_head must stream the full 262 MB tied embed table every token. Rather than
  stream it twice (once for logits, once to fetch the sampled token's embedding),
  the sampler keeps a **resident top-k set** of `{logit, index, embed_sc, embed
  row}` as the table flows by **once**, then emits both the token id and the
  next-token embedding seed. No second pass: the table streams once, over two
  shim lanes into the one core, a slot of each per call. Its temperature,
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
- **Norm input and attention output:** RMSNorm reads the fp32 residual
  directly, and the attention output is quantized once, for the o projection.

---

## 2. Performance

Per-token decode time at each context, time to first token by prompt
length, and the generation rate are measured every night on Strix and
charted on the [dashboard](https://xilinx.github.io/mlir-aie/dashboard/)
(`utils/component_checks/llama_chain_hw.py`; section 4 has the commands).

**What bounds it.** The layers' weights stream over the array's eight lanes
and the tied 262 MB lm_head over two, every token. Attention grows with the
live context: each token reads every live KV block, and its
score-to-probability step is the oracle's exact polynomial
(`vec_exp_nonpos`), so the device stays bit-exact.

**Tokens per dispatch.** amdxdna times a dispatch out after about 5 s. The host
picks each dispatch's token count from its first position
(`DecodeShape.passes_for` in `llama_decode.py`): as many of the 4 as fit a 4 s
budget by a cost model (10.7 ms of lm_head a token, plus per layer 1.80 ms and
0.163 ms per 1k rows of context). At 16 layers that is 4 tokens all the way to
128k. The array's activations and outputs pass through a DDR `stage` buffer
between it and the cores around it (see `aie2_decode.py`).

> Decode controls **TPS** (sustained generation throughput). Prefill controls
> **TTFT** (time to first token). Prefill runs one dispatch of up to 4 prompt
> tokens at a time, each a full 16-layer pass, so TTFT scales with the number
> of dispatches rather than with one big batched op; larger prefill batches
> (What's next) are the lever.

---

## 3. Accuracy / quality

The device is **bit-exact to its numpy oracle** (`llama_oracle.py`): the
decode test checks every token, KV row, sampler state and final residual
against it, and a greedy token must be the oracle's argmax. So scoring the
oracle against the bf16 Hugging Face model scores the device. The INT8 recipe
(`numpy_llama_ref.py`) is shown beside it for scale: the recipe's gap to bf16
is the quantization's own cost. The reference is the true Llama 3.2 1B **bf16** checkpoint, loaded
from the safetensors (no torch or transformers).

### KL and top-1 against bf16

On running prose (`accuracy_vs_hf.py`'s default sample), the nightly scores
the device's arithmetic against bf16 (next-token top-1 agreement and mean
KL(bf16 ‖ model)) and charts both on the [dashboard](https://xilinx.github.io/mlir-aie/dashboard/).
`accuracy_vs_hf.py --model recipe` scores the INT8 recipe the same way.

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

`run_llama.lit` runs the decode test's cases at 2 layers on synthetic
weights (`--synthetic-weights`): from an empty cache, across dispatches,
into the cache's last rows, split, sampled, and a prompt in batches. The
nightly component check runs all 16 real layers and scores the oracle
against bf16 (`accuracy_vs_hf.py`).

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
tokens a dispatch can generate (1 to 4, default 4). Per dispatch,
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

- **The lm_head over more lanes** (it streams over two),
  and the glue fused into the GEMV array's epilogues.
- **Larger prefill batches**: the mma tile has 8 activation slots and the
  design fills 4, its per-row parameters (`pg{m}`, `gat{m}`) sharing the
  32-parameter budget with the passes'. One dispatch per 4 prompt tokens
  means a 2048-token prompt pays 512 dispatches' worth of per-dispatch
  overhead.
- **Long-context attention**, which dominates a token's time at long
  context.
