<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Native Windows Setup and Build Instructions

This guide covers the native Windows setup for **IRON/mlir-aie** and is the recommended path for Windows 11 users. It supports building and running programs on a Ryzen™ AI NPU entirely within Windows, without requiring a POSIX environment. The [WSL2 setup](buildHostWin.md) is available for users who prefer a POSIX-style development environment.

Use an **x64 Native Tools Command Prompt for Visual Studio** in `cmd.exe` for the commands in this guide. It provides MSVC, the linker, and the Windows SDK in one configured environment. Visual Studio and Visual Studio Build Tools both install that prompt, as well as adding a shortcut to the Start menu.

PowerShell is also supported. While the main instructions use `cmd.exe`, the corresponding PowerShell commands and shell-specific differences are collected in [Addendum A](#addendum-a-powershell).

> **Python note:** The Windows XRT SDK supplies `pyxrt` bindings for **CPython 3.13**. Use Python 3.13 with this SDK. Another Python version requires an XRT distribution with matching bindings. Building and packaging XRT from source on Windows is possible, but it is an advanced task outside the scope of this guide.

## Contents

1. [Install the Windows development environment](#1-install-the-windows-development-environment)
2. [Update and verify the NPU driver](#2-update-and-verify-the-npu-driver)
3. [Install the Windows XRT SDK](#3-install-the-windows-xrt-sdk)
4. [Set up IRON](#4-set-up-iron)
5. [Run a complete NPU program](#5-run-a-complete-npu-program)
6. [Addendum A: PowerShell](#addendum-a-powershell)
7. [Addendum B: Build `mlir-aie` wheels locally](#addendum-b-build-mlir-aie-wheels-locally)

## 1. Install the Windows development environment

A fully usable native Windows checkout requires the following:

- A Windows 11 system with a supported Ryzen™ AI / XDNA™ NPU.
- **Visual Studio 2026** (preferred) or **Visual Studio 2022**. Either the full IDE or the matching **Build Tools** package may be used.
- **Python 3.13**, installed directly or through Conda / Miniforge. Conda and Miniforge users should also read [Addendum A](#addendum-a-powershell).
- **CMake**. The version included with a Visual Studio 2026 installation is normally sufficient; a current Kitware release is also suitable.
- **Git for Windows**, installed with Visual Studio or separately.
- The latest Ryzen™ AI / XDNA™ NPU driver and the Windows **XRT SDK**.

### 1.1 Visual Studio components

In the Visual Studio Installer, select **Desktop development with C++** and confirm that the following components are installed. The search box under **Individual components** is useful when checking an existing installation:

- MSVC x64/x86 build tools
- Windows SDK
- C++ CMake tools for Windows
- C++ Clang Compiler for Windows
- MSBuild support for the LLVM (`clang-cl`) toolset
- Git for Windows, unless Git is installed separately

The Clang and LLVM components are used by CMake configurations that build native C++ host applications.

### 1.2 Install the tools

Most of the required tools are available through the Windows package manager, `winget`:

```bat
REM Choose one: the full IDE or the matching Build Tools package
winget install -e --id Microsoft.VisualStudio.Community
REM winget install -e --id Microsoft.VisualStudio.BuildTools

REM Python 3.13 (CPython)
winget install -e --id Python.Python.3.13

REM CMake
winget install -e --id Kitware.CMake

REM Git (optional unless not selected in the Visual Studio installer)
winget install -e --id Git.Git
```

The same tools may be downloaded directly:

- [Visual Studio](https://visualstudio.microsoft.com/downloads/)
- [Python](https://www.python.org/downloads/windows/)
- [CMake](https://cmake.org/download/)
- [Git](https://git-scm.com/download/win)


## 2. Update and verify the NPU driver

Ryzen™ AI / XDNA™ chipset driver updates are distributed through the AMD™ Software / Adrenalin™ application. Install the latest driver available for the system, then verify that the NPU is accessible:

```bat
"C:\Windows\System32\AMD\xrt-smi.exe" examine
```

NPU Driver Version **32.0.20101.3760** (XRT Version **2.21.0**) is the minimum supported by this repository on Windows. Older versions may work in many cases, but they are not recommended.

## 3. Install the Windows XRT SDK

The Windows XRT SDK provides the headers, import libraries, tools, and `pyxrt` bindings used by native host applications and Python JIT designs.

Download the SDK:

```text
https://github.com/Xilinx/XRT/releases/download/2.21.75/xrt_windows_sdk.zip
```

Extract the archive so that its `xrt_sdk\xrt` directory becomes:

```text
C:\Xilinx\XRT
```

> `C:\Xilinx\XRT` is the canonical location. A different location is also supported; pass it to `iron_setup.py` in the next section. The generated activation helpers retain the selected path, so it does not need to be supplied again in each shell.

## 4. Set up IRON

Clone the `mlir-aie` repository and create the checkout-local IRON environment:

```bat
REM Choose a working directory for the checkout. This example uses C:\dev
mkdir C:\dev
cd C:\dev

git clone --recurse-submodules https://github.com/Xilinx/mlir-aie.git
cd mlir-aie

python utils\iron_setup.py
call .\iron_env.cmd
```

The final two commands are separate because they perform different tasks. `iron_setup.py` creates or updates the checkout-local `ironenv` and installs the required dependencies. `iron_env.cmd` activates that environment in the current prompt and supplies the IRON and XRT paths.

`call` `iron_env.cmd` in each new `Native Tools` prompt. Rerun `iron_setup.py` after updating the checkout or changing the XRT SDK location; it refreshes the existing environment and rewrites the activation helpers.

No options are needed for normal use. Two optional setup modes are available:

- `--dev` installs the pinned development tools and the repository's pre-commit and pre-push hooks. It also installs or upgrades `mlir_aie` to the latest rolling development wheel unless `--wheelhouse` is supplied. Use this option only when actively developing the repository.
- `--extras` installs CPU PyTorch, Notebook, and an `ironenv` Jupyter kernel for the PyTorch-based examples and notebooks.

The options may be used independently or together.

If the XRT SDK is installed somewhere other than `C:\Xilinx\XRT`, provide that location during setup:

```bat
python utils\iron_setup.py --xrt-root D:\tools\XRT
call .\iron_env.cmd
```

The selected path is written into the generated helpers. Later shells still need only `iron_env.cmd`.

## 5. Run a complete NPU program

The repository includes runnable examples under `programming_examples` and `programming_guide`. The SAXPY example is a useful first check because it exercises the complete `mlir-aie` toolchain.

The design computes `Z = 3X + Y` on one AI Engine tile. Its Python file describes the data movement and runtime sequence, while the adjacent C++ file contains the vectorized AI Engine kernel. `@iron.jit` compiles them into an NPU program.

```bat
cd programming_examples\getting_started\01_SAXPY
python saxpy.py
```

The script compiles the design, runs it on the attached NPU, and checks the result against a NumPy reference. A final `PASS!` confirms that the native Windows toolchain, XRT installation, and NPU are working together correctly.

Most of the existing learning material uses direct Python JIT scripts. Continue with the [mini tutorial](../programming_guide/mini_tutorial/) or continue to explore by trying another Python example under `programming_examples` or `programming_guide`.

---

<a id="addendum-a-powershell"></a>
<details>
<summary><strong>Addendum A: PowerShell</strong></summary>

Visual Studio also installs a **Developer PowerShell for VS** shortcut. It provides the same compiler environment as the Native Tools `cmd.exe` prompt.

PowerShell may also be started from an existing Native Tools prompt:

```bat
pwsh
```

The compiler environment is inherited automatically. Run setup and dot-source the generated activation helper:

```powershell
python .\utils\iron_setup.py
. .\iron_env.ps1
```

Dot-source `iron_env.ps1` again in each new PowerShell session. The leading dot is *required*: it keeps the activated environment in the current shell rather than as a child scope.

The toolchain is the same, but the shell syntax differs in the usual PowerShell ways. For example:

- PowerShell environment variables use `$env:NAME`; `cmd.exe` uses `%NAME%`.
- PowerShell uses `&` to invoke a command through an expression; `cmd.exe` uses `call` for batch files.
- Path quoting and command composition are not always interchangeable between the two shells.

### Conda or Miniforge Python

A dedicated Conda or Miniforge environment works with the SDK when it uses Python 3.13. Activate it before running `iron_setup.py`; the helper uses the active interpreter when it creates `ironenv`.

```powershell
conda create -n iron python=3.13
conda activate iron
python .\utils\iron_setup.py
. .\iron_env.ps1
```

`iron_setup.py` will select the named interpreter when `ironenv` is created. Remove and recreate `ironenv` before changing the interpreter used by an existing checkout.

</details>

<a id="addendum-b-build-mlir-aie-wheels-locally"></a>
<details>
<summary><strong>Addendum B: Build <code>mlir-aie</code> wheels locally</strong></summary>

Build local wheels when changing core `mlir-aie` files, testing a local commit, or working on packaging. This is not part of the normal setup path. **Normal users should not need to build local wheels**!

### B.1 Install OpenSSL

Local wheel builds compile components that link against OpenSSL. Install the full **Win64 OpenSSL** package from Shining Light Productions. It provides the required headers and libraries and is considerably simpler than building OpenSSL from source.

**Do not use the "Light" package.** It does not contain the development files required by this build.

```text
https://slproweb.com/products/Win32OpenSSL.html
```

From an x64 Native Tools prompt in a configured checkout, set the OpenSSL location and CMake arguments:

```bat
set "OPENSSL_ROOT_DIR=C:\Program Files\OpenSSL-Win64"
set "PATH=%OPENSSL_ROOT_DIR%\bin;%PATH%"
set "CMAKE_ARGS=-DOPENSSL_ROOT_DIR=%OPENSSL_ROOT_DIR% -DOPENSSL_USE_STATIC_LIBS=TRUE"
```

### B.2 Build the wheels

Activate the IRON environment, install the local wheel-build tools, and build one Python version. This example targets Python 3.13:

```bat
cd /d C:\dev\mlir-aie

call .\iron_env.cmd
python -m pip install --require-hashes -r python\requirements_dev.lock

python utils\mlir_aie_wheels\scripts\build_local.py --cp313
```

The completed wheels are written to:

```text
utils\mlir_aie_wheels\wheelhouse
```

Install the local wheels into IRON, then refresh the shell environment:

```bat
python utils\iron_setup.py --wheelhouse utils\mlir_aie_wheels\wheelhouse
call .\iron_env.cmd
```

The local-wheel path force-reinstalls `mlir_aie` from the selected wheelhouse. The rest of setup continues to reconcile the repository's declared requirements.

Use `--cp312` or `--cp314` only when the selected XRT distribution supplies matching `pyxrt` bindings. The local builder manages its staging directories. Remove the following when the build artifacts are no longer needed:

```text
utils\mlir_aie_wheels\wheelhouse
C:\tmp\aiewhls
```

</details>
