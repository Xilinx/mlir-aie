# Dev

- [Wheels](#wheels)
    * [How to cut a new wheel](#how-to-cut-a-new-wheel)
    * [Developing/extending](#developing-extending)


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
  Downloading https://github.com/Xilinx/mlir-aie/releases/download/mlir-distro/mlir-19.0.0.2023121201+d36b483...
     ╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.7/792.9 MB 14.6 MB/s eta 0:00:54

Saved ./mlir-19.0.0.2023121201+d36b483...
Successfully downloaded mlir

$ unzip mlir-19.0.0.2023121201+d36b483...

Archive:  mlir-19.0.0.2023121201+d36b483...
   creating: mlir/
   creating: mlir.libs/
   creating: mlir/src/
   creating: mlir/share/
   creating: mlir/include/
   creating: mlir/bin/
```

**and this will work for all platforms that the wheels are being built for**. 
I.e., no need to specify arch or platform or whatever (pip takes care of it).
And also, of course, `pip download mlir==19.0.0.2023121201+d36b483` works (`19.0.0.2023121201+d36b483` is the "version" of wheel).

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

### How to cut a new wheel

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

### Developing/extending

A brief overview: 

* Everything is meant to flow through [cibuildwheel](https://cibuildwheel.readthedocs.io/en/stable) so start by studying the [pyproject.toml](..%2Futils%2Fmlir_wheels%2Fpyproject.toml) files;
* CMake is driven through [setup.py](..%2Futils%2Fmlir_aie_wheels%2Fsetup.py)s.
* The GitHub actions:
  * All actions related to the wheels use a "base" [action.yml](..%2F.github%2Factions%2Fsetup_base%2Faction.yml) to setup the environment; this base action sets up compilers and docker and deletes unnncessary packages and etc.
  * The build process for each of LLVM+MLIR, MLIR-AIE consists of ~three jobs:
    * Building the base distribution (either LLVM+MLIR or MLIR-AIE)
    * Building the python bindings
    * Upload/release
  * aarch64 is handled in a special way for both LLVM+MLIR and MLIR-AIE
    * For LLVM+MLIR the wheel is built in just the runner environment. 
      The reason for this is I can't figure out how to get the aarch64 sysroot (headers), for cross-compiling, in any of the manylinux_x86 containers (AlmaLinux 8, a RHEL derivative does have some kind of header rpms but they're missing something that prevents their use).
    * For MLIR-AIE, for the distro wheel, the above is also true. **But for the python bindings**  (i.e., the thing you might want to `pip install aie`) the wheel is built in an aarch64 emulated environment in a manylinux_aarch64 container.
      Note, you could try to do this for the other wheels but it would definitely take too long for LLVM+MLIR and would probably take tool long for MLIR-AIE.

#### Tips

Probably don't mess with this 🤷. This is the biggest rube-goldberg machine of scripts/code/hacks/tweaks/workarounds I have ever conjured into existence.
If you're reading this at some future date in the hopes of solving a problem you're having, I'm sorry.
I have literally lost whole days of my life to making improvements (days lost to just babysitting GHA to get feedback on a change).
And it likely couldn't be otherwise due to the enormous number of incidental complexities involved (building an unconventional package for C++ sources across multiple platforms _and_ architectures _and_ in an automated/reproducible way).

But if you must:

* There are [build_local.sh](..%2Futils%2Fmlir_wheels%2Fscripts%2Fbuild_local.sh) scripts for both wheels; they are a very poor approximation of the GitHub environment but they're moderately useful for flushing out major issues.
* Uncomment `pull_request:` at the tops of the yaml to get the actions to run on a PR for the actions themselves (there's probably a better to handle this...).
  * In this mode, the wheels will deposited under the [dev-wheels release page](https://github.com/Xilinx/mlir-aie/releases/tag/dev-wheels).
* The actions themselves have knobs that are accessible through the **Run workflow** UI (that's what those fields are in the dropdown):
  * `commit to build`: should be obvious.
  * `Run the build with tmate debugging enabled`: this is your last resort; this enables you to ssh directly into the runner using your GitHub ssh key.
    By default, the way it works is when the job finishes (either crashes or whatever) but before it exits, the log UI will start spamming something like
    ```shell
    Waiting for session to end
    Notice: SSH: ssh Jj2rULLuwCJkvgRB9324ZbQD6@nyc1.tmate.io
    Notice: or: ssh -i <path-to-private-SSH-key> Jj2rULLuwCJkvgRB9324ZbQD6@nyc1.tmate.io
    ```
    which (assuming your ssh credentials work with GitHub) you can just copy+paste into your terminal and hit enter. The three other options in the workflow dispatch UI control aspects of this "experience":
    * `which runner os to run the tmate action in`: obvious
    * `which runner arch to run the tmate action in`: obvious
    * `whether to launch tmate in detached mode`: this controls whether the ssh tunnel is advertised after all the jobs have run (detached mode) or if it is advertised at the beginning (attached mode).

##### A few of the gotchas/surprises that I can remember fighting:

* In many places you will see `PIP_NO_BUILD_ISOLATION=false` - this means the opposite of what it says i.e., this actually turns off build isolation (i.e., equivalent to passing `--no-build-isolation` to `pip wheel`). [Don't ask me why](https://github.com/pypa/pip/issues/5229#issuecomment-387301397).
* As of today (12/13/23), CMake will segfault during `Detecting CXX compiler ABI info` on mac for `cmake>3.27.9` inside of cibuildwheel.
* `caution filename not matched` during `unzip` is due to a glob that matches multiple files; escape the glob like `mlir_aie\*.whl`.
* Files creating in a cibuildwheel container (i.e., on Linux) have timestamps in the future. This will lead to `ninja` looping forever during a `cmake .. -G Ninja ...` configure step. Hence there's something like `find mlir -exec touch -a -m -t 201108231405.14 {} \;` in various places (where `201108231405.14` is just an arbitrary timestamp in the past).
* The `setup.py`s have needed an ungodly amount of fiddling in order to satisfy the pecularities of all platforms/arches;
    * When building MLIR-AIE On Windows, because of `LNK1170: line in command file contains 131071 or more characters`, the LLVM+MLIR distro is moved to `/tmp/m`.
    * On Windows, when the C runtime is statically linked because that's what you need in order to support multithreading (or is it the inverse?).
    * On Windows, `cl.exe` is used because what configuration system you get otherwise would OOM (or something like that).
    * **In order to support cross-compilation**, every build of LLVM+MLIR also a wheel called `mlir_native_tools`, which basically contains `mlir-tblgen` et al. and is labeled with the commit and the platform/arch.
      These tools are then found (by various means) and then injected into the build environment (by various means).
      Why do this instead of using LLVM's native cross-compilation "flow" (which is supposed to build these tools during cross-compile for the host)?
      Because I couldn't get that to work, either because it isn't intended to support what I'm doing or because I was misusing it.
      Either way, this should be revisited.
* On downstream consumers of the LLVM+MLIR (or MLIR-AIE) wheel, cibuildwheel doesn't quite work out because it won't enable you `pip install` wheels from a different platform.
  For example, if you try to cross-compile MLIR-AIE (for aarch64) by `pip install`ing an already built aarch64 LLVM+MLIR wheel you will fail because `pip install` will only grab/find the x86 wheel.
  [The cibuildwheel people have decided this is "won't fix"](https://github.com/pypa/cibuildwheel/issues/1547).
  The workaroud is `pip -q download mlir --platform $PLAT --only-binary=:all:` (as in [download_mlir.sh](..%2Futils%2Fmlir_aie_wheels%2Fscripts%2Fdownload_mlir.sh)).
