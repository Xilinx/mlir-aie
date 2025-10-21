# ./lit.cfg.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

import os
import sys

# Add shared AIE lit utilities to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool
from aie_lit_utils import LitConfigHelper

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "AIE_TEST"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".py", ".test"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# Setup standard environment (PYTHONPATH, AIETOOLS, system env, etc.)
LitConfigHelper.setup_standard_environment(
    llvm_config, config, config.aie_obj_root, config.vitis_aietools_dir
)

# Basic substitutions
config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%AIE_SRC_ROOT", config.aie_src_root))
config.substitutions.append(("%PYTHON", config.python_executable))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%extraAieCcFlags%", config.extraAieCcFlags))
config.substitutions.append(
    ("%aie_runtime_lib%", os.path.join(config.aie_obj_root, "aie_runtime_lib"))
)
config.substitutions.append(("%aietools", config.vitis_aietools_dir))

# Setup test library substitutions
LitConfigHelper.setup_test_lib_substitutions(
    config, config.aie_obj_root, config.aieHostTarget
)

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
if config.vitis_root:
    config.vitis_aietools_bin = os.path.join(config.vitis_aietools_dir, "bin")
    LitConfigHelper.prepend_path(llvm_config, config.vitis_aietools_bin)
    llvm_config.with_environment("VITIS", config.vitis_root)

# Prepend path to XRT installation, which contains a more recent `aiebu-asm` than the Vitis installation.
LitConfigHelper.prepend_path(llvm_config, config.xrt_bin_dir)

peano_tools_dir = os.path.join(config.peano_install_dir, "bin")
LitConfigHelper.prepend_path(llvm_config, config.llvm_tools_dir)
LitConfigHelper.prepend_path(llvm_config, peano_tools_dir)
LitConfigHelper.prepend_path(llvm_config, config.aie_tools_dir)
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
LitConfigHelper.apply_config_to_lit(
    config,
    {
        "rocm": rocm_config,
        "xrt": xrt_config,
        "peano": peano_config,
        "chess": chess_config,
        "aiesim": aiesim_config,
    },
)

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
    "txn2mlir.py",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

if config.enable_board_tests:
    lit_config.parallelism_groups["board"] = 1
    config.parallelism_group = "board"

# Concurrency tests control their own parallelism, so run them serially
lit_config.parallelism_groups["concurrency"] = 1

if "LIT_AVAILABLE_FEATURES" in os.environ:
    for feature in os.environ["LIT_AVAILABLE_FEATURES"].split():
        config.available_features.add(feature)
