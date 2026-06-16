# Building on a non-Ubuntu distro (and/or with the in-tree `amdxdna` driver)

[`buildHostLin.md`](buildHostLin.md) targets **Ubuntu 22.04+** and walks the
*out-of-tree* XDNA™ driver build. This guide covers the two deltas that come up
on other setups:

1. **A distro without `apt`** (Arch, Void, Gentoo, Fedora-from-source, …) — you
   install build dependencies by *function* rather than by Ubuntu package name.
2. **A kernel that already ships the in-tree `amdxdna` driver** (Linux ≥ 6.14 —
   e.g. Ubuntu 25.04, recent Fedora/Arch/Void). You then build **only the XRT
   userspace SHIM**, not a kernel module.

Everything else (the wheels, Peano, running examples) is identical to
`buildHostLin.md`. Worked example below: **Void Linux, kernel 7.0, glibc 2.41**,
on a Ryzen™ AI "Phoenix" (XDNA1) laptop, using **Peano only — no Vitis™**.

> This is a community-contributed guide. Versions/paths are examples; adapt to
> your distro.

## 0. The kernel driver: in-tree is enough

If your kernel already has `amdxdna`:

```bash
sudo dmesg | grep -i amdxdna     # "Load firmware amdnpu/.../npu_*.sbin" + "Initialized amdxdna_accel_driver"
ls /dev/accel/accel0             # the device node exists
```

then you do **not** need the out-of-tree module. Build only the XRT SHIM (next
step, with `-nokmod`) and do **not** install a DKMS package — a newer
out-of-tree module can require newer firmware and break a working in-tree setup
(see [amd/xdna-driver#1074](https://github.com/amd/xdna-driver/issues/1074),
[#1219](https://github.com/amd/xdna-driver/issues/1219)). On a 7.0 kernel the
in-tree driver loads `npu_7.sbin` (firmware protocol 7) cleanly.

## 1. Device permissions

The accel node defaults to `root:root 0600`; XRT needs access and locked memory:

```bash
sudo groupadd -r render 2>/dev/null   # may already exist
sudo usermod -aG render "$USER"       # effective next login; meanwhile use `sg render -c '...'`
echo 'SUBSYSTEM=="accel", KERNEL=="accel*", GROUP="render", MODE="0660"' \
  | sudo tee /etc/udev/rules.d/99-amdxdna.rules
printf '@render - memlock unlimited\n' | sudo tee /etc/security/limits.d/99-amdxdna.conf
sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=accel --action=add
```

## 2. Build XRT (base + `amdxdna` SHIM) from source

Clone with submodules:

```bash
git clone --recursive https://github.com/amd/xdna-driver.git
```

Install the build dependencies — by *function*, since there's no `apt`. On Void
these were: `boost-devel openssl-devel protobuf protobuf-devel rapidjson
libcurl-devel json-c-devel ncurses-devel libuuid-devel libdrm-devel
elfutils-devel OpenCL-Headers OpenCL-CLHPP ocl-icd-devel python3-pybind11
systemtap-devel`. Notes that bite on non-Ubuntu:

- **OpenCL headers** (`CL/cl.h` + `cl2.hpp`) and an ICD loader are required even
  for the `-npu` build.
- **`sys/sdt.h`** (SystemTap USDT header, with the `STAP_PROBE*` macros) is
  needed by XRT's tracing. It ships in your distro's *systemtap* dev package —
  **not** in a DTrace-compat shim.
- **pybind11** ≥ 2.6 with its CMake config (builds `pyxrt`).

Build XRT base, then install:

```bash
cd xdna-driver/xrt/build
./build.sh -npu -opt -disable-werror -j "$(nproc)"   # -disable-werror: GCC >= 14 + -Werror otherwise fails
sudo make -C Release install                          # -> /opt/xilinx/xrt
```

> **Match `pyxrt`'s Python to the one you'll run IRON with.** `pyxrt` is built
> against the Python CMake detects. If that differs from your IRON venv, pass
> `-cmake-flags "-DPython3_EXECUTABLE=<venv-python> -DPython3_ROOT_DIR=<...>"`
> on a **clean** build dir (changing it on a warm cache does not re-detect), or
> simply use the same interpreter for both.

Build and install the SHIM plugin (userspace only):

```bash
source /opt/xilinx/xrt/setup.sh
cd xdna-driver/build
./build.sh -release -nokmod -j "$(nproc)"             # -nokmod: do NOT touch the in-tree kernel module
# install the produced plugin tree into /opt/xilinx/xrt (TGZ on source distros):
sudo tar xzf Release/xrt_plugin.*-amdxdna.tar.gz -C / --strip-components=0
```

> On a distro `xdna-driver` doesn't yet recognize, `build.sh -release` aborts at
> configure with `Unknown Linux package flavor: <id>`. Arch and Void route to a
> TGZ; for others, add your `/etc/os-release` `ID` to the `arch|void` branch in
> `CMake/pkg.cmake` (Void support:
> [amd/xdna-driver#1424](https://github.com/amd/xdna-driver/pull/1424)).

Verify the NPU enumerates:

```bash
xrt-smi examine        # -> Device(s) Present: [....] RyzenAI-npu1  aie2 ...
```

## 3. IRON toolchain (wheels)

Same as `buildHostLin.md`, but pick a Python that matches your `pyxrt` (above).
`uv` makes provisioning one easy:

```bash
uv venv --python 3.12 ironenv && source ironenv/bin/activate
pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4
pip install llvm-aie  -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```

Keep this `mlir-aie` checkout at the commit the wheel was built from (the wheel
version embeds it, e.g. `1.3.3.devN+g<sha>`) so the `programming_examples` match
the installed `aie` package.

> **`llvm-objcopy` gotcha.** IRON renames a symbol in the compiled AIE2 kernel
> object and prefers `llvm-objcopy`, falling back to GNU `objcopy` — which
> **rejects** the AIE2 ELF (`EM_AIE`, machine `0x108`). Peano ships no
> `objcopy`. Install a generic `llvm-objcopy` (any recent LLVM) and put it on
> `PATH`, e.g. on Void `sudo xbps-install -y llvm21` →
> `/usr/lib/llvm/21/bin/llvm-objcopy`.

## 4. Environment

```bash
source ironenv/bin/activate
source /opt/xilinx/xrt/setup.sh                 # puts pyxrt on PYTHONPATH + xrt libs on LD_LIBRARY_PATH
source utils/env_setup.sh "$(...)" "$(...)"      # or set MLIR_AIE_INSTALL_DIR / PEANO_INSTALL_DIR manually
export NPU2=0                                    # Phoenix/Hawk = npu1 (set 1 for Strix/npu2)
```

`utils/env_setup.sh` works once `xrt-smi` is on `PATH` (it queries it to pick
NPU1 vs NPU2). If `xrt-smi` is absent it returns early — set
`MLIR_AIE_INSTALL_DIR`, `PEANO_INSTALL_DIR`, `PATH`, `PYTHONPATH`,
`LD_LIBRARY_PATH` and `NPU2` by hand instead.

## 5. Build and run an example

```bash
cd programming_examples/basic/passthrough_kernel
python3 passthrough_kernel.py -i1s 4096          # JIT-compiles, runs on the NPU
# -> prints a benchmark table then: PASS!
```

Run under the `render` group so the process can open `/dev/accel/accel0`
(`sg render -c '...'`, or just re-login after step 1).
