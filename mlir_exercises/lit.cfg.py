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
config.name = "AIE_TUTORIALS"

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

if config.enable_board_tests:
    config.substitutions.append(
        ("%run_on_board", "echo %T >> /home/xilinx/testlog | sync | sudo")
    )
else:
    config.substitutions.append(("%run_on_board", "echo"))

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
]

config.aie_tools_dir = os.path.join(config.aie_obj_root, "bin")

# Setup the PATH with all necessary tool directories
LitConfigHelper.prepend_path(llvm_config, config.llvm_tools_dir)
LitConfigHelper.prepend_path(llvm_config, config.peano_tools_dir)
LitConfigHelper.prepend_path(llvm_config, config.aie_tools_dir)
if config.vitis_root:
    config.vitis_aietools_bin = os.path.join(config.vitis_aietools_dir, "bin")
    LitConfigHelper.prepend_path(llvm_config, config.vitis_aietools_bin)
    llvm_config.with_environment("VITIS", config.vitis_root)

# Detect Peano backend
peano_config = LitConfigHelper.detect_peano(
    config.peano_tools_dir, config.peano_install_dir, llvm_config
)

# Detect Chess compiler
chess_config = LitConfigHelper.detect_chess(
    config.vitis_root, config.enable_chess_tests, llvm_config
)

# Apply all tool configurations
LitConfigHelper.apply_config_to_lit(
    config,
    {
        "peano": peano_config,
        "chess": chess_config,
    },
)

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
