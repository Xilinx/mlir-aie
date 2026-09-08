<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Kernel benchmarks and correctness suite

Nightly performance tracking and a full correctness sweep for the kernels
under `aie_kernels/`, driven through `aie.iron.kernels` and
`aie.utils.kernel_harness`; no programming example is involved.

Graphs (benchmark-action): `https://xilinx.github.io/mlir-aie/bench/npu2/`,
`/bench/npu1/`, `/bench/static/<target>/`.

## Where each fact lives

| Fact | Lives in | Read by |
| --- | --- | --- |
| argument roles, host reference, tolerance, ops per call | the factory's `KernelContract` (`python/iron/kernels/*.py`) | everything below |
| the single-Worker design that runs a kernel | `aie.utils.kernel_harness` | e2e tests, this suite, users |
| which shapes to time, which edge data to feed, the canary | `registry.py` (this directory) | `run.py`, `test_correctness.py`, `../static/run.py` |

Nothing here restates what a kernel computes. The harness refuses a
factory without a contract, and `test/python/test_kernel_contracts.py`
fails if a case names one.

## What is measured

| Metric | Source | Regression rule |
| --- | --- | --- |
| `cycles` | trace `INSTR_EVENT_0 → INSTR_EVENT_1` on the worker core, median over the run's invocations | 3 % (near-deterministic) |
| `cycles_per_kop` | cycles / (ops / 1000), ops from the contract's `ops_per_call` | 3 % |
| `npu_us`, `e2e_us` | `aie.utils.benchmark.run_iters`, median; `range` carries min/max/n | advisory |
| `compile_s` | wall time of a forced rebuild (`CallableDesign.compile` with explicit paths) | advisory |
| `xclbin_bytes`, `insts_bytes`, `core_elf_bytes` | the rebuilt artifacts | 3 % |

Correctness is checked **before** any timing, under the tolerance the
kernel declares. A kernel that produces wrong output is never timed and
invalidates the whole run (exit 3 → no JSON → nothing is recorded or
commented).

## Layout

```
benchmarks/static/     Peano remark checks (CPU-only); STATIC_CHECKS.md
benchmarks/kernels/
  STACK_CHECKS.md      plan for a stack-headroom gate
  registry.py          Case list: shapes, edge-data cases, canary   (edit this to add a case)
  harness.py           measurements on top of aie.utils.kernel_harness
  run.py               nightly driver: preflight → canary → run → gate → JSON
  _util.py             provenance, rows, preflight
  conftest.py          --seeds / --data-cases options for test_correctness
  test_correctness.py  full sweep: every case × data case × seed (device)
  test_designs_compile.py  every distinct design through aiecc to CDO (host, static workflow)
  test_run_driver.py   driver control flow with the hardware mocked (host)
.github/workflows/benchmarkKernels.yml
```

## Running locally

```bash
source utils/env_setup.sh
sudo xrt-smi configure -d <bdf> --pmode performance
pytest benchmarks/kernels/test_correctness.py -k eltwise --seeds 3    # on NPU
python -m benchmarks.kernels.run --out bench.json --meta meta.json --only '^add/'
```

## Adding a kernel

1. Give the factory a `KernelContract` (roles, numpy reference, tolerance
   with its evidence in `note`). `test/python/test_kernel_contracts.py`
   checks it against the real `arg_types()` and lowers a design on the
   host.
2. Add one line to `CASES` in `registry.py` with the shapes to time and
   the edge data it must survive. The e2e tier
   (`test/python/npu/test_kernels_e2e.py`) takes the same one-line entry.

Tolerance policy: the kernel owns its tolerance. Integers, selections and
lossless copies are exact; LUT approximations declare the `rtol=0.128`
their `*_ref` documents; a bound measured on a device says so in its
note. A factory that declares nothing gets
`Tolerance.default_for(out_dtype)`. Loosening happens in the factory,
next to the kernel, never in a suite default.

Data policy: integer inputs, random and edge, stay inside
`kernel_harness.input_limit`, which the contract's `acc_dtype` and
`reduction` fix (over the design's full `K` for a matmul), so an edge
case exercises the datapath rather than an overflow the source leaves
undefined. The output range bounds the inputs only under
`overflow="undefined"`: a kernel that declares `"saturate"` or `"wrap"`
is judged that way and gets full-range data (a requantising conv would
otherwise see values near zero). There is no int16 `scale` overflow case
for the same reason: `scale.cc` does not say what an overflow does, so
the judge refuses to grade one.

## Test tiers (each fact asserted once)

| Tier | What | Where | When |
| --- | --- | --- | --- |
| host | contract vs. factory, design lowers to MLIR | `test/python/test_kernel_contracts.py` | every PR (lit) |
| host, compile | every distinct design through aiecc to CDO (tile sets fit core memory, DMA patterns are expressible) | `test_designs_compile.py` | static workflow |
| device, smoke | one case per kernel via the harness | `test/python/npu/test_kernels_e2e.py` | every PR on the NPU runners |
| device, full | every case × data case × seed | `test_correctness.py` | nightly, as this suite's gate |
| host, static | Peano remarks per kernel | `benchmarks/static` | nightly + kernel/toolchain PRs |

## CI behaviour

Nightly on `bench`-labelled runners; data goes to `gh-pages:bench/<npu>/`,
outside mike's versioned docs directories. Peano-bump PRs compare against
the cached nightly baseline. A failed preflight or an invalid result set
means a red run with an artifact: no data, no comment.

- Correctness, canary, preflight: run invalid, nothing published.
- `cycles`, `*_bytes`: alert at 3 %. `II`, `unpipelined_loops`,
  `non_zol_loops`, `missing_bank_loads`, `pass_failed_warnings`: any
  increase.
- `npu_us`, `e2e_us`, `compile_s`: advisory.
- Nothing gates a pull request. A bump PR gets one action comment only
  if a hard-threshold series regressed; the static checker annotates
  dropped pragmas and compile failures on the PR's files instead.

## Kernels the harness cannot run

Five factories carry a contract whose `unsupported` field says why the
single-Worker harness cannot drive them; their references still say what
they compute, and `design()` refuses them with that reason:

| Factory | Why |
| --- | --- |
| `cascade_mm` | partial sums travel over the cascade stream, which is not an argument |
| `mm_bfp_shuffle` | a bfp16ebs8 tile through a plain fifo, which the harness samples only as a matmul operand |
| bf16 `mv` | its signature leads with runtime `m` / `row_offset` scalars; the matvec design drives the int16 `(A, b, c)` form |
| `mha` | a multi-core attention dataflow with a running softmax |
| `bn_conv2dk1_relu_xy_pool_padded` | accumulates across calls through its output, one row per `y_index`; the design hands the kernel a fresh output tile each call |

Five `bn_*` factories have no contract at all: the cascade halves
`bn_conv2dk1_partial_put_i8`, `bn_conv2dk1_partial_get_relu_i8`,
`bn_conv2dk3_dw_out_split`, `bn_conv2dk1_input_split_partial_put_ui8`
and `bn_conv2dk1_input_split_partial_skip_get`. A PUT kernel has no
output argument, so a contract would describe half a computation; the
semantics belong to the two-tile pair, which is a design.
`test_contract_coverage_is_explicit` pins this list.

## Open until a device run

- The power-mode field name in `xrt-smi examine --report platform` on
  Linux (`_util.preflight`).
- `CANARY_CYCLE_BAND`, from the first observed passthrough cycles.
- Subnormal and NaN / inf data reach only the kernels whose contract
  declares `subnormals` / `nonfinite` (the eltwise, axpy, mul_add,
  leaky_relu and transpose kernels). The LUT activations, norms,
  reductions and `convert_copy` leave both unspecified until a device
  run shows what the core does; declaring them on the contract widens
  the registry's data by itself.
- The sweep has not run on hardware yet. The harness designs lower to
  MLIR on the host for every case, and the tolerances are the ones the
  previous hand-written e2e tests passed with, so the expected failure
  mode is a layout or padding detail in a design, not a tolerance.
