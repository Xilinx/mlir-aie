# ./lit.cfg.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.

import os
import sys

# Add shared AIE lit utilities to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import lit.formats
import lit.util

from lit.llvm import llvm_config
from aie_lit_utils import LitConfigHelper

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "AIE_PROGRAMMING_GUIDE"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# Setup standard environment (PYTHONPATH, AIETOOLS, system env, etc.)
LitConfigHelper.setup_standard_environment(
    llvm_config, config, config.aie_obj_root, config.vitis_aietools_dir
)

# Basic substitutions
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

# Not using run_on_board anymore, need more specific per-platform commands
config.substitutions.append(("%run_on_board", "echo"))

# Detect ROCm/HSA and VCK5000
rocm_config = LitConfigHelper.detect_rocm(
    config.hsa_dir, config.aieHostTarget, config.enable_board_tests
)

# Detect XRT and Ryzen AI NPU devices
xrt_config, run_on_npu1, run_on_npu2 = LitConfigHelper.detect_xrt(
    config.xrt_lib_dir,
    config.xrt_include_dir,
    config.xrt_bin_dir,
    config.aie_src_root,
)

config.substitutions.append(("%run_on_npu1%", run_on_npu1))
config.substitutions.append(("%run_on_npu2%", run_on_npu2))
config.substitutions.append(("%xrt_flags", xrt_config.flags))

# OpenCV detection
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

# Setup host target triplet and sysroot
triplet, sysroot_flag = LitConfigHelper.setup_host_target_triplet(
    config.aieHostTarget, config.vitis_sysroot
)
config.substitutions.append(("%aieHostTargetTriplet%", triplet))
config.substitutions.append(("%VitisSysrootFlag%", sysroot_flag))
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

# Setup the PATH with all necessary tool directories
LitConfigHelper.prepend_path(llvm_config, config.aie_tools_dir)
if config.vitis_root:
    config.vitis_aietools_bin = os.path.join(config.vitis_aietools_dir, "bin")
    LitConfigHelper.prepend_path(llvm_config, config.vitis_aietools_bin)
    llvm_config.with_environment("VITIS", config.vitis_root)

peano_tools_dir = os.path.join(config.peano_install_dir, "bin")
LitConfigHelper.prepend_path(llvm_config, config.llvm_tools_dir)
LitConfigHelper.prepend_path(llvm_config, peano_tools_dir)
config.substitutions.append(("%LLVM_TOOLS_DIR", config.llvm_tools_dir))

tool_dirs = [config.aie_tools_dir, config.llvm_tools_dir]

# Detect Peano backend
peano_config = LitConfigHelper.detect_peano(
    peano_tools_dir, config.peano_install_dir, llvm_config
)

# Detect Chess compiler
chess_config = LitConfigHelper.detect_chess(
    config.vitis_root, config.enable_chess_tests, llvm_config
)

# Detect aiesimulator
aiesim_config = LitConfigHelper.detect_aiesimulator()

# Apply all hardware/tool configurations
LitConfigHelper.apply_config_to_lit(config, {
    "rocm": rocm_config,
    "xrt": xrt_config,
    "peano": peano_config,
    "chess": chess_config,
    "aiesim": aiesim_config,
})

# Add Vitis components as features
LitConfigHelper.add_vitis_components_features(config, config.vitis_components)

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
prepend_path(config.aie_tools_dir)
# llvm_config.with_environment('LM_LICENSE_FILE', os.getenv('LM_LICENSE_FILE'))
# llvm_config.with_environment('XILINXD_LICENSE_FILE', os.getenv('XILINXD_LICENSE_FILE'))
if config.vitis_root:
    config.vitis_aietools_bin = os.path.join(config.vitis_aietools_dir, "bin")
    prepend_path(config.vitis_aietools_bin)
    llvm_config.with_environment("VITIS", config.vitis_root)

# Prepend path to XRT installation, which contains a more recent `aiebu-asm` than the Vitis installation.
prepend_path(config.xrt_bin_dir)

peano_tools_dir = os.path.join(config.peano_install_dir, "bin")
prepend_path(config.llvm_tools_dir)
prepend_path(peano_tools_dir)
# Certainly the prepend works but I would rather be explicit
config.substitutions.append(("%LLVM_TOOLS_DIR", config.llvm_tools_dir))

prepend_path(config.aie_tools_dir)

tool_dirs = [config.aie_tools_dir, config.llvm_tools_dir]

# Test to see if we have the peano backend.
try:
    result = subprocess.run(
        [os.path.join(peano_tools_dir, "llc"), "-mtriple=aie", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if re.search("Xilinx AI Engine", result.stdout.decode("utf-8")) is not None:
        config.available_features.add("peano")
        config.substitutions.append(("%PEANO_INSTALL_DIR", config.peano_install_dir))
        config.environment["PEANO_INSTALL_DIR"] = config.peano_install_dir
        print("Peano found: " + os.path.join(peano_tools_dir, "llc"))
        tool_dirs.append(os.path.join(peano_tools_dir, "bin"))
    else:
        print("Peano not found, but expected at ", peano_tools_dir)
except Exception as e:
    print("Peano not found, but expected at ", peano_tools_dir)


if not config.enable_chess_tests:
    print("Chess tests disabled")
else:
    print("Looking for Chess...")
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

        # test if LM_LICENSE_FILE valid
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

# look for aiesimulator
result = shutil.which("aiesimulator")
if result != None:
    print("aiesimulator found: " + result)
    config.available_features.add("aiesimulator")
else:
    print("aiesimulator not found")

# add vitis components as available features
for c in config.vitis_components:
    config.available_features.add(f"aietools_{c.lower()}")

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
