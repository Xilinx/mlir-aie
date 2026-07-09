<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Native Windows Setup and Build Instructions

This guide covers the **native Windows** setup for **IRON/mlir-aie**, which is the recommended path on Windows 11. These instructions will guide you through installing and configuring everything required to both build and execute programs on the Ryzen™ AI NPU, entirely within Windows and without the need for a POSIX environment. If you prefer, you may instead use the [WSL2 setup](buildHostWin.md) to build and run IRON in a Linux environment on Windows.

Use an **x64 Native Tools Command Prompt for Visual Studio** in `cmd.exe` for the commands in this guide. It initializes MSVC, the linker, and the Windows SDK in one place. Visual Studio and Visual Studio Build Tools install this environment automatically and create a shortcut in the Start menu. PowerShell is also supported; see [Addendum A](#addendum-a-powershell).

> **Python note:** The Windows XRT SDK supplies `pyxrt` bindings for **CPython 3.13**. Use Python 3.13 with the SDK. Do not choose another Python version unless you already have an XRT distribution with matching bindings. Building and packaging XRT from source on Windows is an advanced task outside the scope of this guide.

## Contents

1. [Install the Windows development environment](#1-install-the-windows-development-environment)
2. [Update and verify the NPU driver](#2-update-and-verify-the-npu-driver)
3. [Install the Windows XRT SDK](#3-install-the-windows-xrt-sdk)
4. [Set up IRON](#4-set-up-iron)
5. [Run a complete NPU program](#5-run-a-complete-npu-program)
6. [Addendum A: PowerShell](#addendum-a-powershell)
7. [Addendum B: Build `mlir-aie` wheels locally](#addendum-b-build-mlir-aie-wheels-locally)

## 1. Install the Windows development environment

Together, the items below form the expected native Windows development environment for this repository.

You need:

- A Windows 11 system with a supported Ryzen™ AI / XDNA™ NPU.
- **Visual Studio 2026** (preferred) or **Visual Studio 2022**. The full IDE and the matching Build Tools package are both supported.
- **Python 3.13**. This may be through an ordinary install or a Conda / Miniforge environment. See [Addendum A](#addendum-a-powershell) for Conda / Miniforge usage.
- **CMake**. The CMake supplied by a current Visual Studio installation should be fine; a current Kitware download is also suitable.
- **Git for Windows**. Install it with Visual Studio or as a separate package.
- The latest Ryzen™ AI / XDNA™ NPU driver and the Windows **XRT SDK**.

### 1.1 Visual Studio components

When using the Visual Studio Installer (below), select **Desktop development with C++** and confirm that the following components are installed. Use the search box under the **Individual components** tab when needed:

- MSVC x64/x86 build tools
- Windows SDK
- C++ CMake tools for Windows
- C++ Clang Compiler for Windows
- MSBuild support for LLVM (`clang-cl`) toolset
- Git for Windows, when Git is not installed separately

The Clang and LLVM components support CMake configurations to build native C++ host applications.

### 1.2 Install the tools

Most tools may be installed via the Windows command-line package manager, `winget`:

```bat
REM Choose one: the full IDE or the matching Build Tools package.
winget install -e --id Microsoft.VisualStudio.Community
REM winget install -e --id Microsoft.VisualStudio.BuildTools

REM Python 3.13 (CPython)
winget install -e --id Python.Python.3.13

REM CMake
winget install -e --id Kitware.CMake

REM Git (optional unless not selected in VS installer)
winget install -e --id Git.Git
```

Manual downloads are also available here:

```text
Visual Studio: https://visualstudio.microsoft.com/downloads/
Python:        https://www.python.org/downloads/windows/
CMake:         https://cmake.org/download/
Git:           https://git-scm.com/download/win
```

## 2. Update and verify the NPU driver

Chipset driver updates for Ryzen™ AI / XDNA™ APUs are regularly available through the AMD™ Software / Adrenalin™ application. Ensure you have the latest driver version for your system and verify your NPU is accessible by:

```bat
"C:\Windows\System32\AMD\xrt-smi.exe" examine
```

NPU Driver Version 32.0.20101.3760 (XRT Version 2.21.0) is the minimum supported by this repository on Windows. Older versions may function in some cases, but they are not recommended.

## 3. Install the Windows XRT SDK

The XRT SDK provides the native Windows headers, import libraries, and tools used by C++ host applications. It also supplies the `pyxrt` binding used by Python JIT designs.

Download:

```text
https://github.com/Xilinx/XRT/releases/download/2.21.75/xrt_windows_sdk.zip
```

Extract the SDK such that `xrt_sdk\xrt` becomes:

```text
C:\Xilinx\XRT
```
> This is the canonical location. If you install the SDK elsewhere, pass that location to `iron_setup.py` in the next section. The activation helper it generates will record the selected XRT installation.

## 4. Set up IRON

Clone the `mlir-aie` repository, then create the local IRON environment:

```bat
cd /d C:\dev

git clone --recurse-submodules https://github.com/Xilinx/mlir-aie.git
cd mlir-aie

python utils\iron_setup.py
call .\iron_env.cmd
```

The two commands have different jobs. `iron_setup.py` creates or updates the checkout-local `ironenv`, including all necessary dependencies; `iron_env.cmd` activates it in the current prompt and supplies the IRON and XRT paths. Call `iron_env.cmd` in every new Native Tools prompt. Rerun setup after updating the checkout or changing the XRT SDK location; it refreshes the existing environment and rewrites the helpers.

For normal use, no options are needed. Add `--dev` when preparing a contributor checkout: it installs the pinned development tools and the repository's pre-commit and pre-push hooks. Add `--extras` for the PyTorch-based material or Jupyter notebooks: it installs CPU PyTorch, Notebook, and an `ironenv` Jupyter kernel. These options work alone or in combination.

If you installed the XRT SDK to a different location, supply that location when creating the helper. It is saved in the generated helpers, so later shells still only need `iron_env.cmd`:

```bat
python utils\iron_setup.py --xrt-root D:\tools\XRT
call .\iron_env.cmd
```

## 5. Run a complete NPU program

Many example programs are available in the `programming_examples` and `programming_guide` directories. Try running the SAXPY example to exercise the complete `mlir-aie` toolchain. The design computes `Z = 3X + Y` on one AI Engine tile. Its Python file describes the data movement and runtime sequence; the adjacent C++ file contains the vectorized AI Engine kernel. `@iron.jit` compiles them into an NPU program.

```bat
cd programming_examples\getting_started\01_SAXPY
python saxpy.py
```

The script compiles the design, runs it on the attached NPU, and checks the result against a NumPy reference. If it prints `PASS!`, you have just successfully run your first IRON NPU program on Windows!

Most existing learning material uses direct Python JIT scripts, so continue with the [mini tutorial](../programming_guide/mini_tutorial/) or feel free to try any other Python scripts in `programming_examples` or `programming_guide`.


---

## Addendum A: PowerShell

Visual Studio's Native Tools environment is also available in PowerShell by using the "Developer PowerShell for VS" shortcut in the Start menu. It is equivalent to the Native Tools `cmd.exe` prompt.

Alternatively, since PowerShell uses the same toolchain as `cmd.exe`, you may enter it at any time from the Native Tools `cmd.exe` prompt:

```bat
pwsh
```

The compiler environment will be inherited automatically. To set up from PowerShell, run the same setup command and dot-source the generated helper:

```powershell
python .\utils\iron_setup.py
. .\iron_env.ps1
```

In later PowerShell sessions, dot-source `iron_env.ps1` again. Dot-sourcing keeps the activated environment in the current shell.

However, if you are following this guide using PowerShell, please keep the syntactical differences between the two shells in mind. For instance, PowerShell environment variables use `$env:NAME` while `cmd.exe` uses `%NAME%`. Likewise, PS uses `&` to invoke programs while `cmd.exe` uses `call` for batch files. Etc.

### Conda or Miniforge Python

A dedicated Conda or Miniforge environment works with the SDK when it uses Python 3.13. Activate it before running `iron_setup.py`; the helper creates `ironenv` with that interpreter.

```powershell
conda create -n iron python=3.13
conda activate iron
python .\utils\iron_setup.py
. .\iron_env.ps1
```

`iron_setup.py` uses its running interpreter only when it creates `ironenv`. Remove and recreate `ironenv` before switching the interpreter used by an existing environment.

## Addendum B: Build `mlir-aie` wheels locally

Build local wheels when altering core `mlir-aie` files, testing a local commit, or working on packaging.

### B.1 Install OpenSSL

Local wheel builds compile components that link against OpenSSL. Install the full **Win64 OpenSSL** package from Shining Light Productions; it supplies the headers and libraries required by the build and is much faster and easier than compiling from source. **Do not** use the "Light" package.

```text
https://slproweb.com/products/Win32OpenSSL.html
```

From an x64 Native Tools prompt in a configured checkout, set the OpenSSL location and the CMake arguments for the build:

```bat
set "OPENSSL_ROOT_DIR=C:\Program Files\OpenSSL-Win64"
set "PATH=%OPENSSL_ROOT_DIR%\bin;%PATH%"
set "CMAKE_ARGS=-DOPENSSL_ROOT_DIR=%OPENSSL_ROOT_DIR% -DOPENSSL_USE_STATIC_LIBS=TRUE"
```

### B.2 Build the wheels

Activate the IRON environment, install the local wheel-build tools, and build one Python version. This example uses Python 3.13:

```bat
cd /d C:\dev\mlir-aie

call .\iron_env.cmd
python -m pip install --require-hashes -r python\requirements_dev.lock

python utils\mlir_aie_wheels\scripts\build_local.py --cp313
```

The wheelhouse is:

```text
utils\mlir_aie_wheels\wheelhouse
```

Install the locally built wheels into IRON, then refresh the shell environment:

```bat
python utils\iron_setup.py --wheelhouse utils\mlir_aie_wheels\wheelhouse
call .\iron_env.cmd
```

The local-wheel path force-reinstalls `mlir_aie` from the selected wheelhouse. The rest of setup still reconciles the repository's declared requirements.

Use `--cp312` or `--cp314` only when the selected XRT distribution supplies matching `pyxrt` bindings. The local builder manages the staging directories. Remove `utils\mlir_aie_wheels\wheelhouse` and `C:\tmp` to clean up when you are done.
