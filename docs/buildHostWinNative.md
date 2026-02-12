# Native Windows Setup and Build Instructions

This guide covers the **native Windows** (not WSL) setup for **IRON/mlir-aie**.

These instructions will guide you through installing and configuring everything required to both build and execute programs on the Ryzen™ AI NPU, entirely within Windows and without the need for a POSIX environment. The instructions were tested on a GMKtec Evo X-2 running Windows 11 Pro and using Visual Studio 2026 with Ryzen™ AI NPU driver version 1.6.1.

During the course of this guide, you will:
1) Prepare the Windows toolchain to collect dependencies and compile C++ projects.
2) Manually compile the Windows Xilinx™ RunTime (XRT) directly from source.
3) Set up your native Windows mlir-aie/IRON environment.
4) Learn how to compile your own mlir-aie wheels.
5) Compile and run an IRON example test project directly on Windows.

> NOTE: IRON on native Windows is still **experimental** and requires significantly more complicated/manual steps than the WSL2 build (`docs/buildHostWin.md`). Additionally, this workflow may be subject to change as it is further developed. Should everything work flawlessly the first time, it will likely take you approximately **two hours** to complete this guide. If you run into problems, this can easily become a weekend project.

## Shell choice (read this first)

Unlike most distributions of Linux, Windows is packaged with *two* shell environments: the legacy DOS `cmd.exe` (Command Prompt), which is barely more than a 'dumb terminal', and PowerShell, which is comparable to Bash on POSIX systems in terms of flexiblity and scripting. While it presents limitations, this guide assumes familiarity with `cmd.exe` and, given that some steps are simpler because of it's terminal-like nature, is written with that in mind. However, PowerShell is strongly recommended for the adventurous as it may make *using* mlir-aie/IRON easier in the long run, especially as all modern version of Windows now use PowerShell as the default prompt.

**Recommended (path of least resistance):**
- Use **"x64 Native Tools Command Prompt for VS"** (**cmd.exe**) for the XRT build and most/all of this guide. This should "just work".

**PowerShell (advanced users):**
- PowerShell is supported for most steps, but the setup is more complex and not completely documented here.
- Because Visual Studio does not currently ship a "Native Tools" prompt version for PowerShell, you must place the build tools on PATH manually. The two easiest ways to do this are to:

  Either open the **Native Tools cmd prompt first**, then run `pwsh` (assumes PowerShell 7, else `powershell` for 5.1) from inside it (PATH and env vars should be inherited)

  **-OR-**

  You can manually set up the env vars by invoking the Visual Studio `vcvarsall.bat` script from within an already-open PowerShell session: 
```powershell
& 'C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvarsall.bat' x64
```
- Further hints for the "PowerShell-first" path are available in the Addendum.

---

## 1) Install base tooling

You will need the following tools installed and accessible on your Windows system:

> NOTE: This guide assumes you are installing everything into default locations, as a single-user (not system-wide/"all users"), and that the option to place binaries on PATH is taken whenever given (notably for Python and CMake). If you install to custom paths, please adjust the commands below accordingly.

- **Visual Studio** (Community or Build Tools)
- **Python >= 3.12** (CPython or Miniforge/Conda: see Addendum)
- **OpenSSL** (used to compile the mlir-aie wheels as well as the Boost dependency for XRT)
- **CMake** (highly recommended, but also installed by VS)
- **Git** (optional unless not selected in VS installer)

Additionally, we will be updating your NPU driver and installing the dependency package manager **vcpkg**.

### 1.1 Visual Studio / build tools components (required)

In the Visual Studio installer, select:

- Workload: **Desktop development with C++**
- Individual components to explicitly check:
  - **C++ x64/x86 Spectre-mitigated libraries** (XRT forces `/Qspectre`)
  - **C++ Clang Compiler for Windows** (Large. Can be skipped if you already have LLVM installed and on PATH)
  - **MSBuild support for LLVM (clang-cl) toolset**
  - **Git for Windows** (Optional, but must be installed separately if not selected here)

### 1.2 Quick install using `winget`

> NOTE: Choose **either** VS Studio Community (recommended) or Build Tools.

```bat
REM Visual Studio Community (recommended) OR Build Tools
winget install -e --id Microsoft.VisualStudio.Community
REM winget install -e --id Microsoft.VisualStudio.BuildTools

REM Python 3.12 (CPython)
winget install -e --id Python.Python.3.12

REM OpenSSL (optional)
REM OpenSSL will be compiled by vcpkg if you don't have it installed.
winget install -e --id OpenSSL.OpenSSL

REM CMake
winget install -e --id Kitware.CMake

REM Git (optional unless not selected in VS installer)
winget install -e --id Git.Git
```

### 1.3 Manual download links

If you don't have/want/like `winget`, download from the following links:

```text
Visual Studio:
  https://visualstudio.microsoft.com/downloads/

Python:
  https://www.python.org/downloads/windows/

OpenSSL:
  https://slproweb.com/products/Win32OpenSSL.html

CMake:
  https://cmake.org/download/

Git:
  https://git-scm.com/download/win
```

### 1.4 Install the Ryzen™ AI / NPU driver

Install **the latest NPU driver** for your machine.

AMD™'s Ryzen™ AI docs provide driver version guidance and installation steps:

```text
Ryzen(TM) AI Software installation instructions:
  https://ryzenai.docs.amd.com/en/1.6/inst.html
```

After install + reboot, verify the driver can talk to the NPU:

```bat
"%WINDIR%/System32/AMD/xrt-smi.exe" examine
```

If `xrt-smi.exe` is missing, or `examine` fails, **stop here** and fix the driver install first.


### 1.5 Clone the repos

Pick a working directory root (these examples use `C:/dev`).

```bat
mkdir C:/dev
cd C:/dev

REM XRT
git clone --recurse-submodules https://github.com/Xilinx/XRT.git

REM mlir-aie
git clone --recurse-submodules https://github.com/Xilinx/mlir-aie.git

REM vcpkg (needed to install XRT dependencies)
git clone https://github.com/microsoft/vcpkg.git
cd C:/dev/vcpkg
bootstrap-vcpkg.bat
```

### 1.6 Configure vcpkg

NOTE: Visual Studio bundles its own minimal vcpkg instance. If you see any errors containing:
```text
Could not locate a manifest (vcpkg.json) ... This vcpkg distribution does not have a classic mode instance.
```
then the above standalone vcpkg clone is not being used. Make sure your `VCPKG_ROOT` env var points to the git clone and that it is on PATH:

```bat
set "VCPKG_ROOT=C:/dev/vcpkg"
set "PATH=%VCPKG_ROOT%;%PATH%"
```

You will need to do this **every time** you open a new Native Tools cmd prompt if you want to use the packages installed via the standalone vcpkg.

---

## 2) Build + install XRT

### 2.1 Install XRT deps via vcpkg and pip

Open:

- **Search bar --> native tools --> x64 Native Tools Command Prompt for VS**

From the Native Tools cmd prompt:

```bat
set "VCPKG_ROOT=C:/dev/vcpkg"
cd "%VCPKG_ROOT%"

REM Force 64-bit libraries (vcpkg default is x86)
REM Omit `openssl` if you installed it from binaries (saves several minutes)
vcpkg.exe install --triplet x64-windows boost opencl openssl protobuf
```
This will take a considerable amount of time (20-30 minutes) as vcpkg has to compile boost, openssl, and protobuf from source.

You must also install pybind11 into your Python environment
```bat
pip install --upgrade pip
pip install pybind11
```

### 2.2 Configure / build / install XRT

Set your paths:

```bat
set "XRT_REPO=C:/dev/XRT"
set "XRT_SRC=%XRT_REPO%/src"
set "XRT_BUILD=%XRT_REPO%/build/WRelease"
set "XRT_ROOT=C:/Xilinx/XRT"
set "VCPKG_ROOT=C:/dev/vcpkg"
```

Configure:

```bat
cmake -S "%XRT_SRC%" -B "%XRT_BUILD%" -A x64 ^
  -DXRT_NPU=1 ^
  -DBOOST_ROOT="%VCPKG_ROOT%/installed/x64-windows" ^
  -DKHRONOS="%VCPKG_ROOT%/installed/x64-windows" ^
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

Build:

```bat
cmake --build "%XRT_BUILD%" --config Release --parallel
```
Building/compiling XRT will also take some time, usually 10-20 minutes depending on the machine.

Install:

```bat
cmake --install "%XRT_BUILD%" --config Release --prefix "%XRT_ROOT%"
```

### 2.3 Patch the XRT install (runtime DLLs + signed `xrt_coreutil`)

#### 2.3.1 Copy required runtime DLLs into `ext/bin`

XRT tools will fail to run if these minimal DLLs aren't available in the `ext/bin` directory.

```bat
set "EXTBIN=%XRT_ROOT%/ext/bin"
if not exist "%EXTBIN%" mkdir "%EXTBIN%"

copy "%VCPKG_ROOT%\installed\x64-windows\bin\boost_filesystem*.dll"      "%EXTBIN%\" >NUL
copy "%VCPKG_ROOT%\installed\x64-windows\bin\boost_program_options*.dll" "%EXTBIN%\" >NUL
copy "%VCPKG_ROOT%\installed\x64-windows\bin\libprotobuf.dll"            "%EXTBIN%\" >NUL
```

#### 2.3.2 Replace `xrt_coreutil.dll` with the driver-signed version

On Windows, the NPU driver installs a **signed** `xrt_coreutil.dll`. In practice you want that one, not the XRT-built version.
Here we copy, with overwrite, from the driver location into your XRT install.

```bat
set "SIGNED_DLL=%WINDIR%\System32\AMD\xrt_coreutil.dll"
if not exist "%SIGNED_DLL%" set "SIGNED_DLL=%WINDIR%\System32\xrt_coreutil.dll"

if not exist "%SIGNED_DLL%" (
  echo ERROR: Signed xrt_coreutil.dll not found in System32. Check your driver install.
  exit /b 1
)

copy "%SIGNED_DLL%" "%XRT_ROOT%\xrt_coreutil.dll" /y
```

#### 2.3.3 Rebuild the import lib for `xrt_coreutil`

We need to be able to link against the signed library. It would be silly at this point to introduce a POSIX environment just to get access to `gendef`, so we will do it manually.

A short PowerShell script, runable from cmd, is provided below to create and place `xrt_coreutil.def` and `xrt_coreutil.lib` for you:

```bat
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$dll=Join-Path $env:XRT_ROOT 'xrt_coreutil.dll';" ^
  "$libDir=Join-Path $env:XRT_ROOT 'lib';" ^
  "Push-Location $libDir;" ^
  "@('LIBRARY xrt_coreutil','EXPORTS')|Set-Content xrt_coreutil.def -Encoding ascii;" ^
  "dumpbin /exports $dll|Where-Object{$_ -match '^\s+(\d+)\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)'}|ForEach-Object{""$($matches[2]) @$($matches[1])""}|Select-Object -Unique|Add-Content xrt_coreutil.def -Encoding ascii;" ^
  "lib /nologo /def:xrt_coreutil.def /machine:x64 /out:xrt_coreutil.lib;" ^
  "Pop-Location"
```

### 2.4 Validate XRT

```bat
REM XRT tool (from your install)
"%XRT_ROOT%/unwrapped/loader.bat" -exec xclbinutil --help
```
If you see the help message, XRT is installed correctly. 

---

## 3) Set up IRON (create the venv + baseline deps)

From the **mlir-aie repo root** (still in the same Native Tools cmd prompt):

```bat
cd C:/dev/mlir-aie

REM Create/refresh the IRON venv and install deps.
python utils/iron_setup.py
```

Activate the venv and apply the IRON toolchain env vars to your current **cmd.exe** session:

```bat
python utils/iron_setup.py env --shell cmd > "%TEMP%\iron_env.bat" && call "%TEMP%\iron_env.bat"
```

> If you prefer PowerShell, use `python utils/iron_setup.py env --shell pwsh | iex`.

---

## 4) Build + install mlir-aie wheels locally

### 4.1 Build the wheels

>NOTE: The build process will create a `/tmp/` directory at the root of the drive containing your mlir-aie repo (e.g. `C:/tmp/`). It can safely be deleted after the build completes, and it **must** be deleted if you want to re-run the build.

Make sure you are in your venv (after `activate.bat`) and have applied the env vars from section 3.1.

Choose what CPython tags to build (example shows CPython 3.12 only):

```bat
set "CIBW_BUILD=cp312-*"
```

And point to your OpenSSL vcpkg install (needed for the wheels):
>NOTE: if you installed pre-built OpenSSL binaries (i.e. not through vcpkg), `OPENSSL_ROOT_DIR` should already be on PATH and this may be skipped. 

```bat
set "OPENSSL_ROOT_DIR=%VCPKG_ROOT%/installed/x64-windows"
```

Build:

```bat
python utils/mlir_aie_wheels/scripts/build_local.py
```
Compliation will take some time (10-15 minutes).

In the output directory:

- `utils/mlir_aie_wheels/wheelhouse/`

You should see **two wheels**:

- `mlir_aie-...whl`
- `mlir_aie_python_bindings-...whl`

### 4.2 Install your local wheels into IRON

Install from the wheelhouse:

```bat
python utils/iron_setup.py install --mlir-aie wheelhouse
```

### 4.3 Re-apply mlir-aie env vars

After installing, re-run `iron_setup.py env` so it exports `MLIR_AIE_INSTALL_DIR` and updates PATH/PYTHONPATH:

```bat
python utils/iron_setup.py env --shell cmd > "%TEMP%\iron_env.bat" && call "%TEMP%\iron_env.bat"
```

> Again, if you prefer PowerShell, use `python utils/iron_setup.py env --shell pwsh | iex`.

---

### 4.4 Validate the mlir-aie install

```bat
python -c "import aie; import importlib.util as u; print('aie.xrt:', bool(u.find_spec('aie.xrt')))"
```

If this prints `aie.xrt: True`, you're good to go.

## 5) Build and run an example

From inside an example directory (recommended):

```bat
cd /d programming_examples/basic/vector_scalar_mul
python ../../../utils/run_example.py build
python ../../../utils/run_example.py run
```

Or from the repo root:

```bat
python utils/run_example.py run --example-dir programming_examples/basic/vector_scalar_mul
```
If this prints `PASS!` at the end, your IRON system is fully configured and functional.

---

## Troubleshooting

### XRT build fails with "could not find any instance of Visual Studio."

With newer versions of VS, this error is usually because older CMake installs occur earlier on PATH. Strawbery Perl, for instance, tends to do this, as can having ad older CMake installed into your Python/Conda environmet.  

Check which CMake is being used:

```bat
where cmake
cmake --version
```
Alter your PATH ordering in Windows "Environment Variables" or uninstall/upgrade conflicting CMake installs. Then delete your XRT `WRelease` build dir and reconfigure.
The standalone CMake installation in this guide is intended to mitigate this problem, but it must be first on PATH to be effective.

### XRT build fails with Spectre-related errors (MSB8042, /Qspectre)

Install **C++ Spectre-mitigated libs** in the Visual Studio installer, then delete your XRT build dir and reconfigure.

### CMake can't find Python (or finds the wrong one)

- Make sure the intended `python.exe` is first on PATH or activate your conda env before configuring XRT.
- Confirm `%PYLIB%` exists (`dir "%PYLIB%"`).
- Wipe the build dir and reconfigure (CMake caches the first Python it sees).

### CMake can't find Boost / Protobuf

- Confirm you have set `%VCPKG_ROOT%` correctly.
- Confirm you installed the packages for the same triplet (`x64-windows`) you're building with.
- If you changed vcpkg settings, wipe the XRT build dir and reconfigure.

### `xrt-smi` missing or fails

- Reinstall/upgrade the NPU driver (run installer as admin; reboot).
- Confirm it landed at `%WINDIR%/System32/AMD/xrt-smi.exe`.

### XRT tools fail to start due to missing DLLs

- You may need the VC++ runtime: install "Visual C++ 2015-2022 Redistributable (x64)".
- Confirm you copied Boost/Protobuf DLLs into `%XRT_ROOT%/ext/bin`.

### You installed XRT somewhere else

Set `XRT_ROOT` accordingly (cmd):

```bat
set "XRT_ROOT=D:/path/to/XRT"
```

and re-run the "cmd-friendly" env snippet in section 3.1.

### Wheels fail to build due to `Ninja` missing: "no such file or directory"

This error message is misleading: it is asctually due to a stale build cache.
Ninja is istalled for you, but you must delete your `/tmp/` directory (e.g. `C:/tmp/`), and the `build` directories under `utils/mlir_aie_wheels/`, **every time** before re-running the build.

---

## Addendum

### Dev Shell for PowerShell 7

If you prefer to use PowerShell, it may be convenient to always launch it with the VS build tools already enabled.
To do so, simply navigate to `C:/Users/%USERNAME%/Documents/PowerShell` and paste the following script in `profile.ps1` (or create it if it does not exist).
Be sure to include the region tags and respect any existing such tags in your profile.

```powershell
#region enter dev tools mode (amd64)
<# VS Developer Shell (amd64) on every pwsh start #>

# Set this in a shell/session to skip DevShell init:
#   $env:DISABLE_VSDEVSHELL = "1"
if (-not $env:DISABLE_VSDEVSHELL) {
    # Avoid re-entering DevShell if we're already in it.
    if (-not $env:VSCMD_VER) {
        $startLocation = Get-Location
        # The install directory should be in the same place, with the same name, for all distributions of VS.
        # Query `vswhere` from that location to find the real/latest VS and launch the DevShell script.
        try {
            $vswhere = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
            if (Test-Path $vswhere) {
                $vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
                $devShell  = Join-Path $vsInstall 'Common7/Tools/Launch-VsDevShell.ps1'
                if (Test-Path $devShell) {
                    & $devShell -Arch amd64 -HostArch amd64
                }
            }
        } catch {
            Write-Verbose "VS Dev Shell init skipped: $($_.Exception.Message)"
        } finally {
            # Keep your original working directory even if DevShell changes it
            Set-Location $startLocation
        }
    }
}

# Prefer standalone/"classic mode" vcpkg.
# Comment out the next line to disable this whole standalone-vcpkg block (env vars + PATH).
$enableStandaloneVcpkg = $true

if ((Get-Variable enableStandaloneVcpkg -ValueOnly -ErrorAction SilentlyContinue)) {
    # This overrides the DevShell env var pointing to the stripped-down vcpkg that ships with VS.
    $preferredVcpkgRoot = "C:/dev/vcpkg"
    if (Test-Path (Join-Path $preferredVcpkgRoot "vcpkg.exe")) {
        $env:VCPKG_ROOT = $preferredVcpkgRoot
        # Match the DevShell architecture (amd64) to prevent implicit x86 builds.
        $env:VCPKG_DEFAULT_TRIPLET = "x64-windows"
        # Add VCPKG_ROOT to PATH (once).
        $vcpkgPath = (Resolve-Path -LiteralPath $env:VCPKG_ROOT).Path
        $pathParts = @($env:Path -split ';' | Where-Object { $_ })
        $vcpkgNorm = $vcpkgPath.TrimEnd('\','/')
        $hasVcpkg  = $pathParts | ForEach-Object { $_.TrimEnd('\','/') } | Where-Object { $_ -ieq $vcpkgNorm } | Select-Object -First 1
        if (-not $hasVcpkg) {
            $env:Path = "$vcpkgPath;$env:Path"
        }
    }
}

# Print output confirming the running version and vcpkg directory.
if ($env:VSCMD_ARG_TGT_ARCH) {
    Write-Host "VS DevShell: Target = $($env:VSCMD_ARG_TGT_ARCH) Host = $($env:VSCMD_ARG_HOST_ARCH) Version = $env:VSCMD_VER"
} else {
    Write-Warning "VS DevShell not active (no VSCMD_* vars found)."
}
Write-Host "VCPKG_ROOT = $env:VCPKG_ROOT"
if ($env:VCPKG_DEFAULT_TRIPLET) { Write-Host "VCPKG_TRIPLET = $env:VCPKG_DEFAULT_TRIPLET" }
#endregion
```

### Notes on PowerShell usage
- Install the latest PowerShell 7+ from either:
winget
```powershell
winget install -e --id Microsoft.Powershell
```
or manually from:
```text
https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell
```
#### Some commands/scripts may need minor syntax changes to run in PowerShell.
- You can run cmd commands from within PowerShell by prefixing them with `cmd /c`, e.g.: 
  ```powershell
  cmd /c "set XRT_ROOT=C:/Xilinx/XRT"
  ``` 
- Alternatively, you can set env vars in PowerShell using `$env:VAR_NAME = "value"`, e.g.:
  ```powershell
  $env:XRT_ROOT = "C:/Xilinx/XRT"
  ```
- Likewise, commands with variables like `%VCPKG_ROOT%` will need to be changed to `$($env:VCPKG_ROOT)` in PowerShell. So: 
  ```powershell
  set "PATH=%VCPKG_ROOT%;%PATH%"
  ```
  becomes:
  ```powershell
  $env:PATH = "$($env:VCPKG_ROOT);$($env:PATH)"
  ```
  But for CMake:
  ```powershell
  cmake -S "$env:XRT_SRC" -B "$env:XRT_BUILD" -A x64 `
    -DXRT_NPU=1 `
    -DBOOST_ROOT="$env:VCPKG_INSTALLED/installed/x64-windows" `
    -DKHRONOS="$env:VCPKG_INSTALLED/installed/x64-windows" `
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  ``` 
- When copying multi-line commands from this doc, ensure that line continuation characters (carrot `^` for cmd, backtick `` ` `` for PowerShell) are adjusted accordingly.
- Some command names slightly differ. For instance, the `cmd.exe` `where` is `Get-Command` (or aliased directly as `where.exe`) in PowerShell, and `copy` is `Copy-Item` (or `cp`), etc.

### Using Conda/Miniforge Python
- If you have Ryzen™ AI Software installed, and you followed the driver install instructions, you very likely have Conda/Miniforge installed as well.
- You can use the Conda Python for building/running XRT and IRON, but it is recommended to create a separate Conda env for IRON development to avoid conflicts with the driver-installed env.
  ```powershell
  conda create -n py312 python=3.12
  ```
- Ensure you activate your conda env every time you enter a new shell before running Python commands:
  ```powershell  
  conda activate py312
  ```
- In this case, you can skip installing Python via `winget` or the official installer.

