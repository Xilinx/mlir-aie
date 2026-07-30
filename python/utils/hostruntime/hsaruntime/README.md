<!---//===- README.md ---------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Running IRON on the **HSA/ROCR** runtime

This package dispatches IRON designs on the AMD XDNA NPU through **ROCR**
(`libhsa-runtime64.so`) instead of XRT, using ROCR's native AIE support. It is
pure Python (ctypes); there is no C++ component and nothing to build.

For the architecture overview see
[HSA Runtime (ROCR)](../../../../programming_guide/hsa_runtime.md); for every
environment variable see
[Configuration options](../../../../programming_guide/iron_configuration.md#hsarocr-runtime-configuration).

## 1. Prerequisites

- An AMD XDNA NPU with the `amdxdna` kernel driver loaded.
- A ROCm install providing `libhsa-runtime64.so` **with AIE support** (ROCR must
  expose `HSA_DEVICE_TYPE_AIE`).
- The usual IRON toolchain to build designs (`aiecc`), including `xclbinutil` —
  `aiecc` always builds the xclbin edge even though HSA consumes the PDI.

## 2. Point IRON at the right ROCm

Discovery looks for a ROCm *installation root*, in this order:

1. `ROCM_PATH` — an explicit root you choose.
2. A pip-installed ROCm from
   [TheRock](https://github.com/ROCm/TheRock/blob/main/RELEASES.md), e.g.
   ```bash
   pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
       "rocm[libraries,device-gfx1152]"
   ```
   whose runtime tree lives in site-packages.
3. A system install at `/opt/rocm`.

The first root that actually contains the library wins.

> **Set `ROCM_PATH` if you have more than one ROCm.** A system `/opt/rocm` is
> the *last* root tried, but it still wins whenever `ROCM_PATH` is unset and no
> ROCm wheel is installed — so an older system ROCm shadows a newer local build
> that discovery has no way to find. If the ROCm that wins is too old for AIE,
> the failure surfaces later as an opaque HSA error rather than a version
> complaint.

Both `libhsa-runtime64.so` and the bare SONAME `libhsa-runtime64.so.1` are
accepted. TheRock's runtime wheels contain no symlinks at all, so a
pip-installed ROCm ships only the versioned name.

## 3. Run a design

```bash
NPU_RUNTIME=hsa python programming_examples/basic/vector_scalar_add/vector_scalar_add.py
# expect: PASS!
```

`NPU_RUNTIME=hsa` selects `HSATensor` + `CachedHSAHostRuntime`. The backend is
strictly opt-in — `auto` never selects it, so an XRT-less host degrades to
CPU-only tensors instead.

To confirm what was selected:

```bash
NPU_RUNTIME=hsa python -c "
from aie.utils.tensor_factory import DEFAULT_TENSOR_CLASS
print(DEFAULT_TENSOR_CLASS.__name__)"
# -> HSATensor
```

## 4. Environment (optional)

| Variable | Default | Meaning |
|----------|---------|---------|
| `ROCM_PATH` | unset | Explicit ROCm installation root. |
| `IRON_HSA_DEVICE` | auto-detect | Force `npu1` / `npu2` instead of detecting from the HSA agent name. |
| `HSA_EXE_CACHE_SIZE` | `32` | LRU cap on loaded designs. |
| `IRON_HSA_TIMEOUT` | `0` (disabled) | Seconds bounding the completion wait. |

## 5. Limitations

- **No trace capture.** A design with a `trace_config` is rejected up front; use
  `NPU_RUNTIME=xrt` for trace-enabled designs.
- **Serialize dispatches.** The single in-order AIE queue and doorbell are not
  safe for concurrent dispatch from multiple threads.

## 6. Troubleshooting

| Symptom | Cause |
|---------|-------|
| `ImportError: NPU_RUNTIME=hsa was requested but libhsa-runtime64.so could not be located` | No ROCm found in any of the three roots. Set `ROCM_PATH`. |
| `HSAError: No HSA AIE agent found` | The ROCm that was found has no AIE support, or the `amdxdna` driver is not loaded. Check which library was picked (see below). |
| `HSAError: Cannot map HSA AIE agent name ... to a device generation` | An agent name this code does not recognize. Override with `IRON_HSA_DEVICE=npu1\|npu2`. |
| Dispatch hangs, then the kernel log shows `aie2_tdr_work: Device isn't making progress` | The design was compiled for the wrong NPU generation. Check `IRON_HSA_DEVICE` and the `aie.device(...)` in the generated `aie.mlir`. |
| Submit fails with `EIO`; kernel log shows `aie2_config_cu: Invalid BO type` | The PDI was not allocated from the device heap. |

To see which library was actually loaded:

```bash
NPU_RUNTIME=hsa python -c "
from aie.utils.hostruntime.hsaruntime import discovery
print(discovery.find_libhsa())"
```

## 7. Files in this package

| File | Responsibility |
|------|----------------|
| `discovery.py` | Locate a ROCm install and its `libhsa-runtime64.so`. No `dlopen`, so it doubles as the cheap `has_hsa` probe. |
| `_bindings.py` | The raw C ABI: constants, ctypes structs, and the lazily-bound `lib` handle. Importing the package performs no `dlopen` and no device init. |
| `context.py` | `HSAContext` — the process-wide singleton owning agents, memory pools, the queue, the kernarg slot pool, and the completion signal; fills and enqueues dispatch packets. |
| `tensor.py` | `HSATensor` — a vmem allocation mapped for CPU + AIE, exposed as a zero-copy numpy view. |
| `hostruntime.py` | `HSAHostRuntime` / `CachedHSAHostRuntime` — the IRON `HostRuntime` interface: load a design's PDI + insts, run it, run chains. |
