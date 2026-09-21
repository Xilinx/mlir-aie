<!---//===- README.md ---------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# HSA/ROCR host runtime

This package dispatches IRON designs on the AMD XDNA NPU through **ROCR**
(`libhsa-runtime64.so`) instead of XRT, using ROCR's native AIE support. It is
pure Python (ctypes); there is no C++ component and nothing to build.

The documentation lives in the programming guide:

- [HSA Runtime (ROCR)](../../../../programming_guide/hsa_runtime.md) —
  requirements, installing ROCm, running a design, architecture, package
  layout, and troubleshooting.
- [Configuration options](../../../../programming_guide/iron_configuration.md#hsarocr-runtime-configuration)
  — every `NPU_RUNTIME=hsa` environment variable.

