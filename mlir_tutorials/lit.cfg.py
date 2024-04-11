# ./lit.cfg.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.

import os
import platform
import re
import shutil
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "AIE_TUTORIALS"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.environment["PYTHONPATH"] = "{}".format(
    os.path.join(config.aie_obj_root, "python")
)

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

if config.enable_board_tests:
    config.substitutions.append(
        ("%run_on_board", "echo %T >> /home/xilinx/testlog | sync | sudo")
    )
else:
    config.substitutions.append(("%run_on_board", "echo"))

VitisSysrootFlag = ""
if config.aieHostTarget == "x86_64":
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
