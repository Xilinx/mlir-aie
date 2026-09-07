<!---//===- hsa_runtime.md ----------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# HSA Runtime (ROCR)

The **HSA** backend is an opt-in host runtime for IRON that dispatches designs on
the AMD XDNA NPU through **ROCR** (`libhsa-runtime64.so`), using ROCR's native
AIE support (`HSA_DEVICE_TYPE_AIE`, `hsa_amd_aie_kernel_dispatch_packet_t`). It
needs no XRT userspace at runtime.

Unlike the [XRT](iron_configuration.md) and [HRX](hrx_runtime.md) backends,
which load `final.xclbin` + `insts.bin`, HSA loads the **PDI** (`main.pdi`) +
`insts.bin` produced by the same `aiecc` build. Designs are unchanged: the
backend plugs into the same [`aie.utils.Tensor`](../python/utils/hostruntime/tensor_class.py)
and [`HostRuntime`](../python/utils/hostruntime/hostruntime.py) abstractions, so
it is chosen by an environment variable, not by editing sources.

The backend is pure Python (ctypes bindings to ROCR); there is no C++ host stack
and no build-system component.

## Requirements

- An AMD XDNA NPU with the `amdxdna` kernel driver loaded.
- A ROCm providing `libhsa-runtime64.so` **with AIE support** (ROCR must expose
  `HSA_DEVICE_TYPE_AIE`). Install it from a wheel — see below.
- The usual IRON toolchain to build designs (`aiecc`), including `xclbinutil` —
  `aiecc` always builds the xclbin edge even though HSA consumes the PDI.

Linux only: ROCR's AIE agent exists on no other platform today, so the backend
reports itself unavailable elsewhere.

## Enabling HSA

HSA is **strictly opt-in**. It is never auto-selected: an XRT-less host degrades
to CPU-only tensors rather than silently switching backends.

This requires a ROCm providing `libhsa-runtime64.so` with AIE support. The
recommended way to get one is the pip wheel from
[TheRock](https://github.com/ROCm/TheRock/blob/main/RELEASES.md), installed into
the environment you run designs from:

```bash
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ rocm
NPU_RUNTIME=hsa python my_design.py
```

The base `rocm` package is enough — it pulls `rocm-sdk-core`, which carries the
HSA runtime; the `[libraries]` and `[device-gfx…]` extras are for GPU workloads.
Discovery picks the wheel up from site-packages with no path to configure.

Failing that, discovery looks for an installation *root*, in order: `ROCM_PATH`,
the pip-installed ROCm above, then `/opt/rocm`. **`ROCM_PATH` overrides the
wheel**, and with neither present an old system `/opt/rocm` is what you get. See
[Configuration options](iron_configuration.md#hsarocr-runtime-configuration) for
the full variable list.

## Running a design

```bash
NPU_RUNTIME=hsa python programming_examples/basic/vector_scalar_add/vector_scalar_add.py
# expect: PASS!
```

`NPU_RUNTIME=hsa` selects `HSATensor` + `CachedHSAHostRuntime`. To confirm what
was selected, and which ROCm won discovery:

```bash
NPU_RUNTIME=hsa python -c "
from aie.utils.tensor_factory import DEFAULT_TENSOR_CLASS
from aie.utils.hostruntime.hsaruntime import discovery
print(DEFAULT_TENSOR_CLASS.__name__)   # -> HSATensor
print(discovery.find_libhsa())"
```

## Architecture

`HSAContext` is a process-wide singleton owning the ROCR objects: the AIE and
CPU agents, two memory pools, one in-order AIE queue, and one completion signal.

| Resource | Allocation | Why |
|----------|-----------|-----|
| PDI, `insts.bin` | Device-heap pool (`hsa_amd_memory_pool_allocate`) | The kernel driver's `aie2_config_cu` accepts only `AMDXDNA_BO_DEV` buffer objects for a CU's PDI; ROCR produces those only from the DEVICE_SVM pool. Anything else fails the submit ioctl with `EIO`. |
| I/O tensors | vmem, mapped for CPU + AIE | Zero-copy: `HSATensor` is a numpy view over the mapped VA, so there is no host staging copy and the sync hooks are no-ops. |
| Kernargs | Fixed slot pool, one slot per queue ring slot | Removes an allocation from every dispatch. Slot *i* belongs to ring slot *i*, so reuse is safe for free — a ring slot is only rewritten once the device has consumed its previous packet, which is exactly when it has finished reading those kernargs. |

A dispatch fills an AIE packet (instruction address/size, PDI address, kernarg
pointer), publishes it at the queue's write index, and rings the doorbell.

> **The ROCR AIE doorbell is synchronous.** Ringing it submits the pending
> packets *and blocks* until they complete (`SubmitPackets` → `SubmitCmdChain`,
> which issues both the submit and the wait). The completion-signal wait that
> follows is therefore normally a no-op.

### Multi-dispatch chains (`run_chain`)

`run_chain` records N dispatches onto the single in-order queue sharing one
completion signal initialized to N, so a single wait covers the batch. Ordering
is guaranteed by the in-order queue plus the system-scope acquire/release fences
already in every packet header — no barrier packets are needed.

Because the doorbell submits everything pending as one hardware command chain,
the doorbell is rung once per batch rather than once per packet; that is what
makes a chain cheaper than N separate dispatches. Chains longer than the queue
wrap around safely, since each ring also drains ring slots.

## Concurrency

The single in-order AIE queue and its doorbell are **not** safe for concurrent
dispatch from multiple threads; callers must serialize dispatches. This is the
same constraint the HRX backend documents.

## Limitations

- **Trace capture is not supported.** A design carrying a `trace_config` is
  rejected before anything is dispatched; use `NPU_RUNTIME=xrt` for trace.

## Troubleshooting

| Symptom | Cause |
|---------|-------|
| `ImportError: NPU_RUNTIME=hsa was requested but libhsa-runtime64.so could not be located` | No ROCm found in any of the three roots. Install the wheel, or set `ROCM_PATH`. |
| `HSAError: No HSA AIE agent found` | The ROCm that was found has no AIE support, or the `amdxdna` driver is not loaded. Check which library was picked (see [Running a design](#running-a-design)); if it is an old `/opt/rocm`, install the wheel. |
| `HSAError: Cannot map HSA AIE agent name ... to a device generation` | An agent name this code does not recognize. Override with `IRON_HSA_DEVICE=npu1\|npu2`. |
| Dispatch hangs, then the kernel log shows `aie2_tdr_work: Device isn't making progress` | The design was compiled for the wrong NPU generation. Check `IRON_HSA_DEVICE` and the `aie.device(...)` in the generated `aie.mlir`. |
| Submit fails with `EIO`; kernel log shows `aie2_config_cu: Invalid BO type` | The PDI was not allocated from the device heap. |

## Package layout

The backend lives in
[`python/utils/hostruntime/hsaruntime/`](../python/utils/hostruntime/hsaruntime).

| File | Responsibility |
|------|----------------|
| `discovery.py` | Locate a ROCm install and its `libhsa-runtime64.so`. No `dlopen`, so it doubles as the cheap `has_hsa` probe. |
| `_bindings.py` | The raw C ABI: constants, ctypes structs, and the lazily-bound `lib` handle. Importing the package performs no `dlopen` and no device init. |
| `context.py` | `HSAContext` — the process-wide singleton owning agents, memory pools, the queue, the kernarg slot pool, and the completion signal; fills and enqueues dispatch packets. |
| `tensor.py` | `HSATensor` — a vmem allocation mapped for CPU + AIE, exposed as a zero-copy numpy view. |
| `hostruntime.py` | `HSAHostRuntime` / `CachedHSAHostRuntime` — the IRON `HostRuntime` interface: load a design's PDI + insts, run it, run chains. |

## Further reading

- [Configuration options](iron_configuration.md#hsarocr-runtime-configuration) —
  every `NPU_RUNTIME=hsa` environment variable.
- [HRX Runtime (amdxdna)](hrx_runtime.md) — the other opt-in backend.
