# Dev

## Wheels

There are CI/GHA workflows that build

1. a distribution of LLVM+MLIR
   1. [mlirDistro.yml](..%2F.github%2Fworkflows%2FmlirDistro.yml)
   2. [Accompanying scripts](..%2Futils%2Fmlir_wheels)
2. a distribution of MLIR-AIE
   1. [mlirAIEDistro.yml](..%2F.github%2Fworkflows%2FmlirAIEDistro.yml)
   2. [Accompanying scripts](..%2Futils%2Fmlir_aie_wheels)

The builds are packaged as [Python wheels](https://packaging.python.org/en/latest/specifications/binary-distribution-format/).
Why package binaries + C++ source as Python wheels? Because doing so enables this:

```shell
$ pip download mlir -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro

Looking in links: https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro
Collecting mlir
  Downloading https://github.com/Xilinx/mlir-aie/releases/download/mlir-distro/mlir-18.0.0.2023121201+d36b483...
     ╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.7/792.9 MB 14.6 MB/s eta 0:00:54

Saved ./mlir-18.0.0.2023121201+d36b483...
Successfully downloaded mlir

$ unzip mlir-18.0.0.2023121201+d36b483...

Archive:  mlir-18.0.0.2023121201+d36b483...
   creating: mlir/
   creating: mlir.libs/
   creating: mlir/src/
   creating: mlir/share/
   creating: mlir/include/
   creating: mlir/bin/
```

**and this will work for all platforms that the wheels are being built for**. 
I.e., no need to specify arch or platform or whatever (pip takes care of it).
And also, of course, `pip download mlir==18.0.0.2023121201+d36b483` works (`18.0.0.2023121201+d36b483` is the "version" of wheel).

Currently we are building for

* Linux
  * x86_64 ([manylinux_2_27](https://github.com/pypa/manylinux))
  * aarch64
* Windows
  * AMD64
* MacOS
  * x86_64
  * arm64

Why Mac? Because some people do dev on a Mac.

## How to cut a new wheel

1. Go to the [actions tab](https://github.com/Xilinx/mlir-aie/actions) @ github.com/Xilinx/mlir-aie;
2. Select **MLIR Distro** in the left-most column (under **Actions**)
   <p align="center">
    <img width="300" alt="image" src="https://github.com/Xilinx/mlir-aie/assets/5657668/4a1aa2be-7088-4f43-9bc6-4964c46b03a8">
   </p>
3. Select **Run workflow** at the far right
   <p align="center">
    <img width="300" alt="image" src="https://github.com/Xilinx/mlir-aie/assets/5657668/8dce0e03-1756-4ba2-82c9-2e4d8e019e2f">
   </p>
4. Finally (ignoring all of the options) hit the green `Run workflow`
   <p align="center">
    <img width="300" alt="image" src="https://github.com/Xilinx/mlir-aie/assets/5657668/82454733-1661-4963-8ed9-ceea68ebe947">
   </p>
5. A **MLIR Distro** job will appear under the same actions tab (where you can monitor progress).

## Gotchas

1. In many places you will see `PIP_NO_BUILD_ISOLATION=false` - this means the opposite of what it says i.e., this actually turns off build isolation (i.e., equivalent to passing `--no-build-isolation` to `pip wheel`). [Don't ask me why](https://github.com/pypa/pip/issues/5229#issuecomment-387301397).
2. As of today (12/13/23), CMake will segfault during `Detecting CXX compiler ABI info` on mac for `cmake>3.27.9` inside of cibuildwheel.
3. `caution filename not matched` during `unzip` is due to a glob that matches multiple files; escape the glob like `mlir_aie\*.whl`.
4. Files creating in a cibuildwheel container (i.e., on Linux) have timestamps in the future. This will lead to `ninja` looping forever during a `cmake .. -G Ninja ...` configure step. Hence there's something like `find mlir -exec touch -a -m -t 201108231405.14 {} \;` in various places (where `201108231405.14` is just an arbitrary timestamp in the past).