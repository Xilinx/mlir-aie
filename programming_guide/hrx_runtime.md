<!---//===- hrx_runtime.md ----------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# HRX Runtime (amdxdna)

The **HRX** backend is an opt-in host runtime for IRON that dispatches designs on
the AMD XDNA NPU through **`libhrx`** (the IREE-based runtime with an `amdxdna`
HAL), consuming the same `aiecc` artifacts (`final.xclbin` + `insts.bin`) as the
default [XRT](iron_configuration.md) path. It requires no XRT userspace at
runtime, and — combined with the bundled XRT-free `xclbinutil` — lets the whole
build-and-run flow work on a machine with no XRT install at all.

HRX plugs into the same [`aie.utils.Tensor`](../python/utils/hostruntime/tensor_class.py)
and [`HostRuntime`](../python/utils/hostruntime/hostruntime.py) abstractions the
XRT backend uses, so example and design code is unchanged — the backend is chosen
by an environment/make variable, not by editing sources.

## Enabling HRX

HRX is **strictly opt-in**. It is never auto-selected: an XRT-less host degrades
to CPU-only tensors rather than silently switching to HRX.

A single `NPU_RUNTIME` variable drives both flows:

| Flow | Example |
|------|---------|
| IRON / Python | `NPU_RUNTIME=hrx python my_design.py` |
| C++ examples (`make`) | `NPU_RUNTIME=hrx make run` |

The IRON/Python flow reads `NPU_RUNTIME` at import time (a runtime switch), while
the C++ example `make` flow reads the same variable to build the HRX host stack (a
build-time switch) — one selector, two mechanisms. The full list of `NPU_RUNTIME`
values and every HRX environment variable
(`HRX_DIR`, `LIBHRX_DIR`, `HRX_LIBHRX`, `IRON_HRX_DEVICE`, `HRX_EXE_CACHE_SIZE`,
`IRON_HRX_TIMEOUT`) is documented in
[IRON Configuration](iron_configuration.md#host-runtime-backend-selection-npu_runtime).

## Provisioning `libhrx`

Fetch, checksum-verify, and extract the pinned HRX release (coordinates in
`utils/hrx-release.env`) with the helper script, then source the environment it
prints:

```bash
source "$(utils/fetch-hrx-release.sh)"
# or, in one step:  eval "$(utils/fetch-hrx-release.sh --print-env)"
```

Verify HRX is discoverable (path-only probe — no `dlopen`, no device init):

```bash
python3 -c "import aie.utils as u; print('has_hrx:', u.has_hrx)"   # -> True
```

## Architecture

The Python package
[`aie.utils.hostruntime.hrxruntime`](../python/utils/hostruntime/hrxruntime/README.md)
is split into focused modules:

| Module | Role |
|--------|------|
| `discovery` | Locate `libhrx` on disk (no `dlopen`); backs `aie.utils.has_hrx`. |
| `_bindings` | The C ABI layer: constants, `ctypes` struct mirrors, and bound `hrx_*` entry points. |
| `context` | `HRXContext` — the process-wide device + dispatch-stream singleton (buffers, executables, chained dispatch). |
| `tensor` | `HRXTensor` — a persistent host-mapped, device-visible buffer with explicit flush/invalidate coherence. |
| `hostruntime` | `HRXHostRuntime` (uncached) and `CachedHRXRuntime` (LRU executable cache) — the IRON `HostRuntime` implementations. |

Importing the package is side-effect-free; `libhrx` is bound lazily on first
`HRXContext` creation.

### Multi-dispatch chains (`run_chain`)

`HRXHostRuntime.run_chain([(handle, args), ...])` records several dispatches into
one HRX command buffer — in order, with an execution + memory barrier between
them — and submits the batch with a single `synchronize` (the amdxdna HAL lowers
it into one `ERT_CMD_CHAIN`). Because of the barrier, a later run observes an
earlier run's device writes, so producer→consumer chains work. This is the HRX
analogue of the XRT runlist; there is no equivalent in the XRT *Python* runtime.

## Concurrency and multi-tenancy

`HRXContext` is a per-process singleton (thread-safe lazy creation). Separate
processes — including different users — are fully isolated: each builds its own
context and buffers, and the amdxdna driver isolates each process's hardware
context and device memory. The only shared resource is the finite, system-wide
pool of amdxdna hardware contexts, which can be exhausted under heavy
parallelism (a capacity limit, not a data-safety issue). Within one process the
dispatch stream is not built for concurrent dispatch, so callers must serialize
dispatch on a single context — the same expectation the XRT Python runtime has.

## Further reading

- [HRX runbook](../python/utils/hostruntime/hrxruntime/README.md) — the
  self-contained, step-by-step guide for building and running the IRON (Python)
  and C++ tests on HRX, including the bundled XRT-free `xclbinutil`.
- [IRON Configuration](iron_configuration.md) — all runtime configuration
  variables.
