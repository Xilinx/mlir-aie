<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Install via FastFlow SDK

The [AMD FastFlow SDK](https://rocm.github.io/FastFlow-SDK/) is a single `pip` package for Ryzen™ AI NPUs on Windows and Linux. It contains a prebuilt MLIR-AIE under the name *iron-tools* (`aie.iron`, `aiecc`, `aie-opt`, and Peano), the IRON operator library, the [HRX runtime](../programming_guide/hrx_runtime.md), and FastFlowLM (`flm`) for running language models on the NPU. Use it to write, compile, and run IRON designs without checking out this repository or building anything.

To change MLIR-AIE itself, or to use features newer than the revision the SDK pins, follow one of the [host setup guides](../README.md#ryzen-ai-host-setup) instead.

## Requirements

| | |
| --- | --- |
| Operating system | Linux or Windows 11, x86_64. On Windows, use native PowerShell; WSL is not supported. |
| NPU | Ryzen™ AI with an XDNA2 (NPU2) NPU. See [Hardware and drivers](https://rocm.github.io/FastFlow-SDK/reference/hardware/). |
| Driver | The `amdxdna` driver on Linux, the AMD NPU driver on Windows. |
| Python | 3.12, 3.13, or 3.14. |

## Install

Create and activate a virtual environment, then install the SDK from PyPI.

Linux:

```bash
python3.12 -m venv ~/fastflow
source ~/fastflow/bin/activate
python -m pip install fastflow-sdk
fastflow doctor
```

Windows (PowerShell):

```powershell
py -3.12 -m venv $HOME\fastflow
& $HOME\fastflow\Scripts\Activate.ps1
python -m pip install fastflow-sdk
fastflow doctor
```

`fastflow doctor` checks the driver, the NPU, and the install, and exits 0 when everything passes. If a check fails, look it up in [Troubleshooting](https://rocm.github.io/FastFlow-SDK/reference/troubleshooting/).

## Run a design

[Hello NPU](https://rocm.github.io/FastFlow-SDK/hello-npu/) is the smallest complete IRON program: it copies a buffer through a memory tile and back. Download [hello_npu.py](https://rocm.github.io/FastFlow-SDK/examples/hello_npu.py) and run it:

```bash
python hello_npu.py
# PASS: 4096 int32 elements copied through the NPU
```

## Differences from a source build

- The SDK pins one MLIR-AIE revision per release; [Component versions](https://rocm.github.io/FastFlow-SDK/reference/versions/) lists it.
- The [programming examples](../programming_examples/) are written for a source build and follow `main`, not the SDK's pinned revision. Many print MLIR that their Makefile compiles with `aiecc` and runs with a separate host program, so follow each example's README. For a design that compiles and runs from Python alone, start from Hello NPU above.

See the [FastFlow SDK documentation](https://rocm.github.io/FastFlow-SDK/) for the operator library, FastFlowLM, and release notes.
