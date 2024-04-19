# ./lit.cfg.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.

import os
import re
import shutil
import subprocess

import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "AIE_REFERENCE_DESIGNS"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%extraAieCcFlags%", config.extraAieCcFlags))
config.substitutions.append(
    (
        "%host_runtime_lib%",
        os.path.join(config.aie_obj_root, "runtime_lib", config.aieHostTarget),
    )
)
config.substitutions.append(("%aietools", config.vitis_aietools_dir))
# for xchesscc_wrapper
llvm_config.with_environment("AIETOOLS", config.vitis_aietools_dir)

# for python
llvm_config.with_environment("PYTHONPATH", os.path.join(config.aie_obj_root, "python"))

run_on_npu = "echo"
xrt_flags = ""

# Not using run_on_board anymore, need more specific per-platform commands
config.substitutions.append(("%run_on_board", "echo"))

if config.hsa_dir and (not ("NOTFOUND" in config.hsa_dir)):
    if not "hsa" in config.aieHostTarget:
        print(
            "ROCm found, but disabled because host target {}".format(
                config.aieHostTarget
            )
        )
        config.substitutions.append(("%run_on_vck5000", "echo"))
        config.substitutions.append(("%link_against_hsa%", ""))
        config.substitutions.append(("%HSA_DIR%", ""))

    else:
        # Getting the path to the ROCm directory. hsa-runtime64 points to the cmake
        # directory so need to go up three directories
        rocm_root = os.path.join(config.hsa_dir, "..", "..", "..")
        print("Found ROCm:", rocm_root)
        config.available_features.add("hsa")
        config.substitutions.append(("%HSA_DIR%", "{}".format(rocm_root)))
        config.substitutions.append(("%link_against_hsa%", "--link_against_hsa"))
        found_vck5000 = False

        if config.enable_board_tests:
            # If board tests are enabled, make sure there is an AIE ROCm device that we can find
            try:
                # Making sure that we use the experimental ROCm install that can see the AIE device
                my_env = os.environ.copy()
                my_env.update(LD_LIBRARY_PATH="{}/lib/".format(rocm_root))
                result = subprocess.run(
                    ["rocminfo"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=my_env,
                )
                result = result.stdout.decode("utf-8").split("\n")

                # Go through result and look for the VCK5000
                for l in result:
                    if "Versal VCK5000" in l:
                        print("Found VCK500 in rocminfo. Enabling on board tests")
                        found_vck5000 = True
                        config.substitutions.append(
                            ("%run_on_vck5000", "flock /tmp/vck5000.lock")
                        )
                        break

                if not found_vck5000:
                    config.substitutions.append(("%run_on_vck5000", "echo"))
                    print(
                        "Enable board set and HSA found but couldn't find device using rocminfo"
                    )

            except:
                print("Enable board set and HSA found but unable to run rocminfo")
                pass
        else:
            print("Skipping execution of unit tests (ENABLE_BOARD_TESTS=OFF)")
            config.substitutions.append(("%run_on_vck5000", "echo"))
else:
    print("ROCm not found")
    config.substitutions.append(("%run_on_vck5000", "echo"))
    config.substitutions.append(("%link_against_hsa%", ""))
    config.substitutions.append(("%HSA_DIR%", ""))


if config.xrt_lib_dir:
    print("xrt found at", os.path.dirname(config.xrt_lib_dir))
    xrt_flags = "-I{} -L{} -luuid -lxrt_coreutil".format(
        config.xrt_include_dir, config.xrt_lib_dir
    )
    try:
        xbutil = os.path.join(config.xrt_bin_dir, "xbutil")
        result = subprocess.run(
            [xbutil, "examine"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        result = result.stdout.decode("utf-8").split("\n")
        # Starting with Linux 6.8 the format is like "[0000:66:00.1]  :  RyzenAI-npu1"
        p = re.compile("\[.+:.+:.+\].+(Phoenix|RyzenAI-(npu\d))")
        for l in result:
            m = p.match(l)
            if m:
                print("Found Ryzen AI device:", m.group().split()[0])
                if len(m.groups()) == 2:
                    # Prepare the future
                    aie_model = m.group(2)
                    print("\tmodel:", aie_model)
                config.available_features.add("ryzen_ai")
                run_on_npu = (
                    f"flock /tmp/npu.lock {config.aie_src_root}/utils/run_on_npu.sh"
                )
    except:
        print("Failed to run xbutil")
        pass
else:
    print("xrt not found")

config.substitutions.append(("%run_on_npu", run_on_npu))
config.substitutions.append(("%xrt_flags", xrt_flags))
config.substitutions.append(("%XRT_DIR", config.xrt_dir))
config.environment["XRT_HACK_UNSECURE_LOADING_XCLBIN"] = "1"

opencv_flags = ""
if config.opencv_include_dir and config.opencv_libs:
    print("opencv found")
    config.available_features.add("opencv")
    opencv_flags = opencv_flags + " -I" + config.opencv_include_dir
    if config.opencv_lib_dir:
        opencv_flags = opencv_flags + " -L" + config.opencv_lib_dir
    libs = config.opencv_libs.split(";")
    opencv_flags = opencv_flags + " " + " ".join(["-l" + l for l in libs])
else:
    print("opencv not found")
    opencv_flags = ""
config.substitutions.append(("%opencv_flags", opencv_flags))

try:
    import torch

    config.available_features.add("torch")
except ImportError:
    print("torch not found", file=sys.stderr)
    pass

VitisSysrootFlag = ""
if "x86_64" in config.aieHostTarget:
    config.substitutions.append(("%aieHostTargetTriplet%", "x86_64-unknown-linux-gnu"))
elif config.aieHostTarget == "aarch64":
    config.substitutions.append(("%aieHostTargetTriplet%", "aarch64-linux-gnu"))
    VitisSysrootFlag = "--sysroot=" + config.vitis_sysroot

config.substitutions.append(("%VitisSysrootFlag%", VitisSysrootFlag))
config.substitutions.append(("%aieHostTargetArch%", config.aieHostTarget))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "Inputs",
    "Examples",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
    "aie.mlir.prj",
    "lit.cfg.py",
]

config.aie_tools_dir = os.path.join(config.aie_obj_root, "bin")


def prepend_path(path):
    global llvm_config
    paths = [path]

    current_paths = llvm_config.config.environment.get("PATH", None)
    if current_paths:
        paths.extend(current_paths.split(os.path.pathsep))
        paths = [os.path.normcase(os.path.normpath(p)) for p in paths]
    else:
        paths = []

    llvm_config.config.environment["PATH"] = os.pathsep.join(paths)


# Setup the path.
prepend_path(config.llvm_tools_dir)
prepend_path(config.peano_tools_dir)
prepend_path(config.aie_tools_dir)
# llvm_config.with_environment('LM_LICENSE_FILE', os.getenv('LM_LICENSE_FILE'))
# llvm_config.with_environment('XILINXD_LICENSE_FILE', os.getenv('XILINXD_LICENSE_FILE'))
if config.vitis_root:
    config.vitis_aietools_bin = os.path.join(config.vitis_aietools_dir, "bin")
    prepend_path(config.vitis_aietools_bin)
    llvm_config.with_environment("VITIS", config.vitis_root)

# Test to see if we have the peano backend.
try:
    result = subprocess.run(
        [os.path.join(config.peano_tools_dir, "llc"), "-mtriple=aie", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if re.search("Xilinx AI Engine", result.stdout.decode("utf-8")) is not None:
        config.available_features.add("peano")
        print("Peano found: " + shutil.which("llc"))
    else:
        print("Peano not found, but expected at ", config.peano_tools_dir)
except Exception as e:
    print("Peano not found, but expected at ", config.peano_tools_dir)

print("Looking for Chess...")
# test if LM_LICENSE_FILE valid
if config.enable_chess_tests:
    result = None
    if config.vitis_root:
        result = shutil.which("xchesscc")

    if result != None:
        print("Chess found: " + result)
        config.available_features.add("chess")
        config.available_features.add("valid_xchess_license")
        lm_license_file = os.getenv("LM_LICENSE_FILE")
        if lm_license_file != None:
            llvm_config.with_environment("LM_LICENSE_FILE", lm_license_file)
        xilinxd_license_file = os.getenv("XILINXD_LICENSE_FILE")
        if xilinxd_license_file != None:
            llvm_config.with_environment("XILINXD_LICENSE_FILE", xilinxd_license_file)

        validate_chess = False
        if validate_chess:
            import subprocess

            result = subprocess.run(
                ["xchesscc", "+v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            validLMLicense = len(result.stderr.decode("utf-8")) == 0
        else:
            validLMLicense = lm_license_file or xilinxd_license_file

        if not lm_license_file and not xilinxd_license_file:
            print(
                "WARNING: no valid xchess license that is required by some of the lit tests"
            )
    elif os.getenv("XILINXD_LICENSE_FILE") is not None:
        print("Chess license found")
        llvm_config.with_environment(
            "XILINXD_LICENSE_FILE", os.getenv("XILINXD_LICENSE_FILE")
        )
    else:
        print("Chess not found")

tool_dirs = [config.aie_tools_dir, config.peano_tools_dir, config.llvm_tools_dir]
tools = [
    "aie-opt",
    "aie-translate",
    "aiecc.py",
    "ld.lld",
    "llc",
    "llvm-objdump",
    "opt",
    "xchesscc_wrapper",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

if config.enable_board_tests:
    lit_config.parallelism_groups["board"] = 1
    config.parallelism_group = "board"
