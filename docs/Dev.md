<!-- Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Dev

- [Wheels](#wheels)
    * [How to cut a new wheel](#how-to-cut-a-new-wheel)
    * [Developing/extending](#developing-extending)


## Wheels

There are CI/GHA workflows that build

1. a distribution of LLVM+MLIR
   1. [mlirDistro.yml](https://github.com/Xilinx/mlir-aie/blob/main/.github/workflows/mlirDistro.yml)
   2. [Accompanying scripts](https://github.com/Xilinx/mlir-aie/tree/main/utils/mlir_wheels)
2. a distribution of MLIR-AIE
   1. [mlirAIEDistro.yml](https://github.com/Xilinx/mlir-aie/blob/main/.github/workflows/mlirAIEDistro.yml)
   2. [Accompanying scripts](https://github.com/Xilinx/mlir-aie/tree/main/utils/mlir_aie_wheels)

The builds are packaged as [Python wheels](https://packaging.python.org/en/latest/specifications/binary-distribution-format/).
Why package binaries + C++ source as Python wheels? Because doing so enables this:

```shell
$ pip download mlir -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro

Looking in links: https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro
Collecting mlir
  Downloading https://github.com/Xilinx/mlir-aie/releases/download/mlir-distro/mlir-23.0.0.2026071405+46fcb339...
     ╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.7/792.9 MB 14.6 MB/s eta 0:00:54

Saved ./mlir-23.0.0.2026071405+46fcb339...
Successfully downloaded mlir

$ unzip mlir-23.0.0.2026071405+46fcb339...

Archive:  mlir-23.0.0.2026071405+46fcb339...
   creating: mlir/
   creating: mlir.libs/
   creating: mlir/src/
   creating: mlir/share/
   creating: mlir/include/
   creating: mlir/bin/
```

This works for every platform the wheels are built for; there is no need to specify an
architecture or platform, since pip resolves it. A specific version can be requested directly,
e.g. `pip download mlir==23.0.0.2026071405+46fcb339`. The version string follows the scheme
`<llvm_major>.0.0.<datetime>+<llvm_commit_prefix>` defined in
[utils/clone-llvm.sh](https://github.com/Xilinx/mlir-aie/blob/main/utils/clone-llvm.sh); as of
this writing LLVM is sourced from the [ROCm/llvm-project](https://github.com/ROCm/llvm-project)
fork rather than upstream, so the major version tracks that fork.

The wheels are currently built for

* Linux
  * x86_64 ([manylinux_2_27](https://github.com/pypa/manylinux))
  * aarch64
* Windows
  * AMD64

### How to cut a new wheel

1. Go to the [Actions tab](https://github.com/Xilinx/mlir-aie/actions) at github.com/Xilinx/mlir-aie.
2. Select **MLIR Distro** in the left-hand column (under **Actions**).
3. Select **Run workflow** at the far right.
4. The dispatch options all have defaults, so they can be left as-is; hit the green **Run workflow** button.
5. A **MLIR Distro** job will appear under the same Actions tab, where progress can be monitored.

The same procedure applies to the **MLIR AIE Distro** workflow for the MLIR-AIE wheels.

### Developing/extending { #developing-extending }

A brief overview: 

* Everything is meant to flow through [cibuildwheel](https://cibuildwheel.readthedocs.io/en/stable) so start by studying the [pyproject.toml](https://github.com/Xilinx/mlir-aie/tree/main/utils/mlir_wheels%2Fpyproject.toml) files;
* CMake is driven through [setup.py](https://github.com/Xilinx/mlir-aie/tree/main/utils/mlir_aie_wheels%2Fsetup.py)s.
* The GitHub actions:
  * All actions related to the wheels use a "base" [action.yml](https://github.com/Xilinx/mlir-aie/blob/main/.github/actions/setup_base/action.yml) to setup the environment; this base action sets up compilers and docker and deletes unnncessary packages and etc.
  * The build process for each of LLVM+MLIR, MLIR-AIE consists of ~three jobs:
    * Building the base distribution (either LLVM+MLIR or MLIR-AIE)
    * Building the python bindings
    * Upload/release
  * aarch64 is handled specially for both LLVM+MLIR and MLIR-AIE:
    * For LLVM+MLIR the wheel is built directly in the runner environment, because the aarch64 sysroot (headers) needed for cross-compiling is not readily available in the manylinux_x86 containers.
    * For MLIR-AIE, the distro wheel is built the same way. The python bindings (the `aie` package installed via `pip install aie`), however, are built in an aarch64-emulated `manylinux_aarch64` container. Emulation is impractical for the larger LLVM+MLIR build, so it is reserved for the bindings.

#### Tips

The wheel-build system is intricate: it packages C++ sources as Python wheels across multiple
platforms and architectures in an automated, reproducible way, and that requirement drives a
fair amount of incidental complexity. Change it deliberately, and expect slow feedback loops
since most issues only surface in CI.

Useful entry points:

* There are [build_local.sh](https://github.com/Xilinx/mlir-aie/tree/main/utils/mlir_wheels%2Fscripts%2Fbuild_local.sh) scripts for both wheels. They only approximate the GitHub environment, but are useful for flushing out major issues locally.
* Both workflows have a path-scoped `pull_request:` trigger that runs them when their own YAML changes, so edits to the workflows are exercised on the PR that makes them.
  * In this mode, the wheels are deposited under the [dev-wheels release page](https://github.com/Xilinx/mlir-aie/releases/tag/dev-wheels).
* The workflows expose `workflow_dispatch` inputs through the **Run workflow** UI:
  * `LLVM_COMMIT`: the LLVM commit to build (defaults to empty, i.e. the commit pinned in the repo).
  * `APPLY_PATCHES`: whether to apply the vendored source patches (defaults to `true`).
  * `DEBUG_ENABLED`: runs the build with [tmate](https://github.com/marketplace/actions/debugging-with-tmate) debugging enabled, allowing an SSH session into the runner using your GitHub SSH key. When the job reaches its end (on success or failure) but before it exits, the log will print connection details such as
    ```shell
    Waiting for session to end
    Notice: SSH: ssh Jj2rULLuwCJkvgRB9324ZbQD6@nyc1.tmate.io
    Notice: or: ssh -i <path-to-private-SSH-key> Jj2rULLuwCJkvgRB9324ZbQD6@nyc1.tmate.io
    ```
    which can be copied into a terminal to connect. Three related inputs control this session:
    * `DEBUG_OS`: which runner OS to run the tmate action in.
    * `DEBUG_ARCH`: which runner architecture to run the tmate action in.
    * `DEBUG_DETACHED`: whether the SSH tunnel is advertised after all jobs have run (detached mode) or at the beginning (attached mode).

##### Known gotchas and non-obvious behaviors

* In many places `PIP_NO_BUILD_ISOLATION=false` appears. Counterintuitively, this *disables* build isolation (equivalent to passing `--no-build-isolation` to `pip wheel`); see [this pip issue](https://github.com/pypa/pip/issues/5229#issuecomment-387301397) for the rationale.
* CMake versions in the 3.28 range have been observed to segfault during `Detecting CXX compiler ABI info` inside cibuildwheel, so the build pins a known-good version (`cmake==4.3.4`) in the wheel [pyproject.toml](https://github.com/Xilinx/mlir-aie/blob/main/utils/mlir_wheels/pyproject.toml) files.
* `caution filename not matched` during `unzip` is caused by a glob that matches multiple files; escape the glob, e.g. `mlir_aie\*.whl`.
* Files created inside a cibuildwheel container (on Linux) can carry future timestamps, which makes `ninja` loop indefinitely during the `cmake .. -G Ninja ...` configure step. To avoid this, the build resets timestamps to a fixed past value, e.g. `find mlir -exec touch -a -m -t 201108231405.14 {} \;` (the exact value is arbitrary, as long as it is in the past).
* The `setup.py` files accommodate several platform-specific requirements:
    * On Windows, to avoid `LNK1170: line in command file contains 131071 or more characters`, the LLVM+MLIR distro is relocated to a short path (`C:/tmp/aiewhls` by default, configurable via the `AIE_WHEEL_BUILD_ROOT` environment variable).
    * On Windows, the C runtime is statically linked to support multithreading.
    * On Windows, `cl.exe` is used because the alternative configuration path exhausts memory.
    * To support cross-compilation, every LLVM+MLIR build also produces a `mlir_native_tools` wheel containing `mlir-tblgen` and related tools, labeled with the commit and platform/arch. These tools are located and injected into the build environment during cross-compilation. This is used in place of LLVM's built-in cross-compilation flow, which did not work for this setup; revisiting that flow remains a possible future improvement.
* cibuildwheel does not let downstream consumers `pip install` a wheel built for a different platform. For example, cross-compiling MLIR-AIE for aarch64 by `pip install`ing a prebuilt aarch64 LLVM+MLIR wheel fails, because `pip install` resolves only the host (x86) wheel. The cibuildwheel maintainers have marked this [won't fix](https://github.com/pypa/cibuildwheel/issues/1547). The workaround is `pip -q download mlir --platform $PLAT --only-binary=:all:` (see [download_mlir.sh](https://github.com/Xilinx/mlir-aie/tree/main/utils/mlir_aie_wheels%2Fscripts%2Fdownload_mlir.sh)).
