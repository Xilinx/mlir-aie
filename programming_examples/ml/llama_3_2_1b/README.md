<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Llama 3.2 1B (INT8) decode on AI Engine (IRON)

An INT8 decode of Llama 3.2 1B on an AMD NPU2 (AIE2P), written in the IRON
Python API and compiled with `@iron.jit` to one full ELF. Each dispatch runs the
whole per-token decode on the AI Engine array for `PT` tokens: all 16 transformer
layers, attention with a growing KV cache, the tied 262 MB lm_head, the sampler,
and the token→embedding feedback. The W8A8-dynamic INT8 recipe tracks the
reference bf16 HF model (75 % greedy top-1 on hard one-word prompts, 94–100 % on
prose); the device is bit-exact to the numpy INT8 oracle, so it adds no error
beyond quantization.

> Status: decode is validated on hardware at 91.2 ms/token (10.97 tok/s) at
> N=16. Prefill, RoPE tables on the device, and a KV cache longer than 128 are
> the open stages — see [On the device, and on the host](#on-the-device-and-on-the-host)
> and [What's next](#whats-next).

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
  DDR: seed(token0) ──►│ seed-mux ─► router ─► [ rmsnorm ─ q/k/v ─ rope ─ KV-append ─    │
                       │     ▲                   flowkv-attn ─ o_proj ─ +res ─ rmsnorm ─  │
                       │     │                   gate/up ─ silu ─ down ─ +res ] ×16 ─►    │
                       │     │                 final_norm ─► lm_head GEMM + top-k insert  │
                       │     │                            ─► sample ─► (token, embed seed)│
                       │     └──────────── embed seed feeds back on-chip ────────────────┤
  host: weights (DDR) ─────────────────────────► streamed per token ─────────────────────►
  host: tokens out  ◄──────────────────────────── PT packed records ◄─────────────────────┘
  DDR: next seed    ◄──────────────────────────── last token's seed ◄──────────────────────┘
```

Key design pieces (each validated bit-exact in isolation before integration):

- **One-stream sampler + embed gather** (`kernels/llama_topk_sample.cc`). The
  lm_head must stream the full 262 MB tied embed table every token. Rather than
  stream it twice (once for logits, once to fetch the sampled token's embedding),
  the sampler keeps a **resident top-k set** of `{logit, index, embed_sc, embed
  row}` as the table flows by **once**, then emits both the token id and the
  next-token embedding seed. No second pass, one shim channel.

- **On-chip token feedback** (`--persist`). The sampled token's embed seed feeds
  back to layer 0 **on-chip** via a depth-2 self-feedback fifo + a seed-mux (the
  seed buffer for token 0, on-chip feedback for tokens 1..PT-1). The last token's
  seed goes back to the seed buffer in DDR, so the next dispatch continues from
  it; the host writes the seed buffer only for the first dispatch. (DDR, not core
  memory: after the driver's 5 s autosuspend the cores come back reset.)

- **Growing KV cache** (`--persist-grow`). Each token appends its K/V
  on-chip at an advancing position (the append writes slot `T_used` and bumps it,
  so **position advances on-device**; the host never computes a position) and
  attention widens to include the new slot. Rope advances with position too. This
  is real autoregressive decode — context accumulates.

- **Resident KV cache** (`--persist-resident`). The KV body lives in
  worker-local buffers on the attention tiles, seeded once on token 0 then
  read-modify-written in place — **zero per-token KV DMA, host streams only
  weights**. Fits a compute tile at small `N_LAYERS` (32 KB/head at N=2); the full
  16-layer cache (256 KB/head) needs a memtile (the one remaining residency item).

### On the device, and on the host

Every transformer op, the lm_head, the sampler, the token feedback and the KV
append run as AI Engine kernels, and one compiled design serves every dispatch.
The host (`DispatchSession` in `llama_npu.py`) keeps one hardware context and,
per dispatch, writes the design's ScratchpadParameters. What the host still does,
each to move to the device in the stages of the one-design restructure:

- **RoPE tables.** The test computes each position's cos/sin in numpy and writes
  them into the weight blob before each dispatch.
- **The KV half.** The KV buffer is two halves that tokens alternate between;
  the host passes the half a dispatch starts in (`kv_even`/`kv_odd`). A
  block-major cache appended in place replaces this.
- **Prefill.** There is none yet: the starting 64-token context is a fixture the
  numpy oracle builds.
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
(`taskset -c 2-4`, 5 warmup + 10 timed dispatches, median); per-token =
dispatch time / PT, PT=4.

| Config | per-token | tok/s | bytes/token |
|---|---|---|---|
| **N=16 (full model)** | **91.2 ms** | **10.97** | 1253 MB |
| N=2 (bring-up) | 30.3 ms | 33.0 | 389 MB |

N=16 went from 452 ms/token to 212 ms by vectorizing the kernels, then to
91.2 ms by moving the remaining scalar fp32 epilogues onto vectors (bit-exact),
taking the lm_head's top-k insert off soft float, and double-buffering the
up-projection weights.

**What bounds it now.** With every kernel replaced by a no-op, a token still
takes 69 ms at N=16: each weight stream has one shim lane (about 13.7 GB/s) and
only one stream runs at a time. The rest is compute the streams wait on. A
shared 8-lane GEMV array (`aie2_gemv_array.py`, measured at 52 GB/s) is the
next step; FastFlowLM reaches 60–89 tok/s on this model.

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
from `python aie2_chain_dynscale_mh.py ... --full-elf-path X`); the others run
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

# a generation loop: 10 dispatches carrying the KV cache and seed, with the
# NPU left idle (and suspended) between them
LLAMA_CHAIN_N=16 python test_chain_persist_grow_mh.py --dispatches 10 --idle-s 6
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

## Configuration

The chain tests read these from the environment and pass them to the design as
CompileTime values; each test picks its own mode.

| Variable | Meaning |
|---|---|
| `LLAMA_CHAIN_N` | number of decoder layers (**set 16** for real runs; default 2) |
| `LLAMA_CHAIN_PT` | tokens generated per dispatch (persistent loop; default 4) |
| `LLAMA_CHAIN_T` | KV cache length (default 128) |
| `LLAMA_CHAIN_ONESTREAM_KSET` | the one-stream sampler's resident top-k set size (default 8) |

Everything else that changes the build is a CompileTime value of `build()` in
`aie2_chain_dynscale_mh.py`, also on its CLI: the chain modes (`--sample`,
`--onestream`, `--persist`, `--persist-grow`, `--persist-resident`; each
`PERSIST*` flag a strict superset of the one above), the sampler
(`--sample-temp`, `--sample-topk`, `--sample-seed`; temp ≤ 0 is greedy) and
`--pingpong-depth`. `test_compile_time_keys.py` checks that each value is part
of the JIT cache key.

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
5. **L1 is 64 KB/tile.** Resident KV (32 KB/head at N=2) fits beside the worker
   stack aiecc measures; the full 16-layer cache (256 KB/head) needs a memtile.
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
the per-call values (tokens per pass, position, live context, layers) as
ScratchpadParameters:

- **RoPE on the device**: the table in the weight blob, indexed by position.
- **A long KV cache**: block-major in DDR up to 128k tokens, appended in place,
  with block-wise attention (`aie2_attn_block.py`).
- **The shared 8-lane GEMV array** for every projection and the lm_head, with
  the glue fused into its epilogues; the old chain modes retire with it.
- **Prefill** on the device, token by token and then batched, with TTFT and
  tok/s measured end to end.
