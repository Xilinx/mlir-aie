<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Python Kernel Library

Pre-built AIE kernel wrappers for common operations. These provide ready-to-use
`Worker`-compatible callables backed by optimized native AIE code. For the C++
kernel sources these wrap, see [C++ AIE kernels](aie_kernels.md).

## Element-wise operations

::: iron.kernels.eltwise
    options:
      show_root_heading: false

## Data movement

::: iron.kernels.datamovement
    options:
      show_root_heading: false

::: iron.kernels.zero
    options:
      show_root_heading: false

## Quantization

::: iron.kernels.quant
    options:
      show_root_heading: false

## Core state

::: iron.kernels.core
    options:
      show_root_heading: false

## Reduction

::: iron.kernels.reduce
    options:
      show_root_heading: false

## Linear algebra

::: iron.kernels.linalg
    options:
      show_root_heading: false

`fused_mm` computes one bf16 matrix-multiply tile on a single core, with an
f32 reduction and an optional activation and clamp fused into the drain.

::: iron.kernels.fused
    options:
      show_root_heading: false

## Convolution

::: iron.kernels.conv
    options:
      show_root_heading: false

## Activation functions

`gelu_erf` is GELU as defined, `x * Phi(x)` ([Hendrycks and Gimpel](https://arxiv.org/abs/1606.08415)), its erfc in the form of Numerical Recipes' `erfcc` (Press et al., 2nd ed., §6.2) with a refit polynomial.
`log10_clamp_f32` takes log-mel features as Hugging Face's [`WhisperFeatureExtractor`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py) does.

::: iron.kernels.activation
    options:
      show_root_heading: false

## Normalization

`rms_norm`, `rms_norm_eps`, `layer_norm`, and `layer_norm_eps` support both
aie2 and aie2p. They accept `tile_size` (default 1024) or its compatibility alias `cols`.
`rope` in Data movement has the same size API and supports both interleaved
and `two_halves=True` layouts. The transformer module re-exports the canonical
bf16 norm and RoPE factories and references; importing through either module
does not select a different implementation. Norm references accept `eps`.

::: iron.kernels.norm
    options:
      show_root_heading: false

## Transformer blocks

::: iron.kernels.transformer
    options:
      show_root_heading: false

## Vision

::: iron.kernels.vision
    options:
      show_root_heading: false

## Contracts and the generic design builder

Every factory the generic builder can build carries a `KernelContract` on the
returned `ExternalFunction` (`fn.contract`). The field-by-field account is in
[Kernel Library](../programming_guide/kernels_library.md#what-a-signature-cannot-say);
the reference below is generated from the dataclass, so it cannot drift from
it. Multi-dtype factories publish the combinations they support in
`kernels.DTYPES["<factory>"]`, and each factory's submodule defines its references as
`*_ref` functions (`eltwise.add_ref`, `eltwise.add_i8_ref`,
`reduce.reduce_max_ref`, ...), so host code never reimplements the math.

::: iron.kernels._common
    options:
      show_root_heading: false
      members:
        - KernelContract
        - TensorLayout
        - Param
        - Trace

`aie.iron.algorithms.KernelDesign` turns any contract-bearing kernel into a
design of one Worker, built on the same single-core pipeline as
`transform`, `for_each` and `reduce`. What a kernel can answer about itself
-- its
reference result, its safe input range, which arguments are parameters,
how to judge a device output -- lives on
[`ExternalFunction`][iron.ExternalFunction] instead, so bringing up a
kernel needs no test harness. See
[Kernel Library](../programming_guide/kernels_library.md#adding-a-kernel) for
the add-a-kernel procedure and the test tiers built on it.

Give the call count and guard setting once, on the builder:

```python
from aie.iron import kernels
from aie.iron.algorithms import KernelDesign

kd = KernelDesign(kernels.relu(tile_size=1024), calls=4, guard=True)
inputs = kd.sample_inputs()
outputs = kd.run(inputs)
assert kd.check(outputs, inputs)
```

`kd.design()` exposes the underlying JIT design, and `kd.as_mlir()` generates its
MLIR. `kd.host_args()`, `kd.output_size()` and `kd.upload(inputs)` share the same
configuration. `kd.cycles_per_call(inputs)` measures the declared trace intervals.
The exported `design(factory, **kwargs)` remains a compatibility wrapper.

::: iron.algorithms.kernel_design
    options:
      show_root_heading: false
      members:
        - design
        - KernelDesign
        - CallCycles
        - split_intervals


`test/python/npu/test_kernels_perf.py` times the library: correctness
first, then cycles, wall time and build size, gated by a device preflight
and a measurement-sanity test. It is an ordinary pytest module, so `-k`
selects cases and the session's exit status decides whether any numbers are
written. `--baseline-sources DIR` measures every selected case a second
time with its kernels from `DIR` and compares the two runs' raw output words
and cycles in `--perf-meta`. The timing helpers live here:

::: utils.benchmark
    options:
      show_root_heading: false
      members:
        - Stats
        - BenchmarkResult
        - Preflight
        - run_iters
        - run_interleaved
        - CompletionCores
        - cpu_wait_us
        - preflight
        - provenance
        - kernel_tree_digest

## Static checks

These compiler-remark checks are available on demand through
`python -m aie.utils.compile.remarks`; there is no static-check CI workflow.
The trace-marker check (`test/python/test_kernel_trace_markers.py`) runs in
lit on every PR, through `trace_markers` below.

::: utils.compile.remarks
    options:
      show_root_heading: false
      members:
        - LoopInfo
        - StaticReport
        - parse_yaml
        - parse_stderr
        - report_rows
        - workflow_annotations
        - compile_command
        - analyze
        - kernel_builds
        - trace_markers
        - trace_shape
        - entry_symbol

::: utils.compile.readobj
    options:
      show_root_heading: false
      members:
        - Linked
        - linked
        - parse_readobj

## Host-side helpers

`aie.utils.bfp` is the host side of the block-floating-point kernels: the
bfp16ebs8 codec and the tile shuffle the `mm_bfp` DMA layout needs. It is
the Python counterpart of `programming_examples/ml/block_datatypes/helper.h`,
which the examples' C++ hosts use; a Python host encodes and checks with
this module.

::: utils.bfp
    options:
      show_root_heading: false
      members:
        - encode
        - decode
        - quantize
        - shuffle

::: utils.verify
    options:
      show_root_heading: false
      members:
        - Tolerance
        - Verdict
        - compare
        - bf16_ulp_distance

## Sampling

`sample_select` and `sample_combine` draw the next token from a row of bf16
logits on the device: each `sample_select` core reduces one slice to its top
k, and one `sample_combine` core draws from those summaries. `sample_ref` is
the bit-exact host reference.

::: iron.kernels.sample
    options:
      show_root_heading: false

## Mixture of experts

`router_top_k` turns one `sample_select` summary into a router's `top_k`
expert ids and their softmax weights, deterministically. `router_top_k_ref`
is the bit-exact host reference.
`experts` runs one row through its chosen experts' gated feed-forward on one
core, from the row and one stream of weight tiles.

::: iron.kernels.moe
    options:
      show_root_heading: false

## State space models

`selective_scan` runs Mamba2's recurrence for one head over a block of rows,
and `gated_delta_rule` Gated DeltaNet's for one value head, each with its
float32 state carried from call to call.

- Mamba2: [Dao and Gu](https://arxiv.org/abs/2405.21060); the reference is `selective_state_update` in [state-spaces/mamba](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/selective_state_update.py), less its optional `silu(z)` gate.
- Gated DeltaNet: [Yang, Kautz and Hatamizadeh](https://arxiv.org/abs/2412.06464), on DeltaNet ([Yang et al.](https://arxiv.org/abs/2406.06484)); the references are fla-org's [`fused_recurrent_gated_delta_rule`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py) and Hugging Face's [Qwen3-Next](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py).
- Their exponentials, and `router_top_k`'s, reduce by ln 2 in two parts (Cody and Waite, *Software Manual for the Elementary Functions*, 1980).

::: iron.kernels.ssm
    options:
      show_root_heading: false

## FastFlowLM Gemma 4

::: iron.kernels.flm_gemma4
    options:
      show_root_heading: false
