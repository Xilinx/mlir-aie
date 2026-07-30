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

## Enabling HSA

HSA is **strictly opt-in**. It is never auto-selected: an XRT-less host degrades
to CPU-only tensors rather than silently switching backends.

```bash
NPU_RUNTIME=hsa python my_design.py
```

This requires a ROCm install providing `libhsa-runtime64.so`. Discovery looks
for an installation *root*, in order: `ROCM_PATH`, a pip-installed ROCm from
[TheRock](https://github.com/ROCm/TheRock/blob/main/RELEASES.md), then
`/opt/rocm`. **Set `ROCM_PATH` when the machine has more than one ROCm** — a
system `/opt/rocm` will otherwise shadow a newer local build. See
[Configuration options](iron_configuration.md#hsarocr-runtime-configuration) for
the full variable list.

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

## Further reading

- [HSA runtime README](../python/utils/hostruntime/hsaruntime/README.md) — package
  layout, prerequisites, and troubleshooting.
- [Configuration options](iron_configuration.md#hsarocr-runtime-configuration) —
  every `NPU_RUNTIME=hsa` environment variable.
- [HRX Runtime (amdxdna)](hrx_runtime.md) — the other opt-in backend.
