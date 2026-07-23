<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Host Runtime (`aie.utils.hostruntime`)

Host-side helpers for selecting a device, allocating NPU-accessible tensors,
running compiled designs, and sharing common CLI / argument plumbing across
programming examples.

Import the public surface from `aie.utils.hostruntime` (or via the
`aie.utils` / `iron` re-exports where noted). The symbols below are rendered
from the Python package under `python/utils/hostruntime/` using the same
mkdocstrings path layout as the other Python API pages (`utils.*`, not the
installed `aie.utils.*` import name).

---

## Runtime abstractions

Abstract runtime types that concrete backends (for example XRT) implement.

::: utils.hostruntime.hostruntime
    options:
      show_root_heading: false
      docstring_options:
        warn_missing_types: false

---

## Tensor and device utilities

Device selection, the abstract `Tensor` type, and numerical helpers used when
comparing host results.

::: utils.hostruntime
    options:
      show_root_heading: false
      members:
        - Tensor
        - set_current_device
        - bfloat16_safe_allclose
      docstring_options:
        warn_missing_types: false

::: utils.hostruntime.tensor_class
    options:
      show_root_heading: false
      members:
        - Tensor
      docstring_options:
        warn_missing_types: false

---

## Shared CLI/argument helpers

Reusable `argparse` flag groups and the standard design CLI dispatcher used by
the programming examples.

::: utils.hostruntime.argparse
    options:
      show_root_heading: false
      docstring_options:
        warn_missing_types: false

::: utils.hostruntime.cli
    options:
      show_root_heading: false
      docstring_options:
        warn_missing_types: false

---

## XRT runtime implementation

Concrete XRT-backed runtime, tensor, and scratchpad types. These modules import
`pyxrt`, so they are only importable in environments where XRT / PyXRT is
installed. MkDocs CI does not install XRT, so these symbols are summarized here
(same approach as the non-importable JIT helpers on the IRON page) with links to
source. On a machine with PyXRT available they can also be rendered with
mkdocstrings, for example:

```markdown
::: utils.hostruntime.xrtruntime.hostruntime
    options:
      show_root_heading: false
```

| Symbol | Module | Summary |
|--------|--------|---------|
| `XRTKernelHandle` | [`xrtruntime.hostruntime`](../../python/utils/hostruntime/xrtruntime/hostruntime.py) | Handle for a loaded XRT kernel. |
| `XRTKernelResult` | [`xrtruntime.hostruntime`](../../python/utils/hostruntime/xrtruntime/hostruntime.py) | Result wrapper for a PyXRT kernel run. |
| `XRTHostRuntime` | [`xrtruntime.hostruntime`](../../python/utils/hostruntime/xrtruntime/hostruntime.py) | Singleton manager for AIE XRT resources. |
| `CachedXRTKernelHandle` | [`xrtruntime.hostruntime`](../../python/utils/hostruntime/xrtruntime/hostruntime.py) | Cached handle for a loaded XRT kernel. |
| `CachedXRTRuntime` | [`xrtruntime.hostruntime`](../../python/utils/hostruntime/xrtruntime/hostruntime.py) | Cached `XRTHostRuntime` that reuses contexts for the same xclbin. |
| `XRTTensor` | [`xrtruntime.tensor`](../../python/utils/hostruntime/xrtruntime/tensor.py) | Tensor backed by NPU/CPU-accessible memory managed with PyXRT. |
| `ParameterScratchpad` | [`xrtruntime.parameter_scratchpad`](../../python/utils/hostruntime/xrtruntime/parameter_scratchpad.py) | Write named runtime parameters to the NPU scratchpad buffer. |
