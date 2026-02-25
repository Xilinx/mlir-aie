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

from lit.llvm import llvm_config
from aie_lit_utils import LitConfigHelper

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "AIE_PROGRAMMING_EXAMPLES"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".lit"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# Setup standard environment (PYTHONPATH, AIETOOLS, system env, etc.)
LitConfigHelper.setup_standard_environment(
    llvm_config, config, config.aie_obj_root, config.vitis_aietools_dir
)

# Basic substitutions
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

# Detect Peano backend first to get supported architectures
peano_tools_dir = os.path.join(config.peano_install_dir, "bin")
peano_config = LitConfigHelper.detect_peano(
    peano_tools_dir, config.peano_install_dir, llvm_config
)

# Add Peano-detected components to vitis_components if not already present
peano_components_str = peano_config.substitutions.get("%peano_components%", "[]")
try:
    peano_components = eval(peano_components_str)
    for comp in peano_components:
        if comp not in config.vitis_components:
            config.vitis_components.append(comp)
except Exception:
    pass

# Add Vitis components as features
LitConfigHelper.add_vitis_components_features(config, config.vitis_components)

# Add Peano components as features
LitConfigHelper.add_peano_components_features(config, config.peano_components)

# Add peano_only_build feature for tests that should be skipped in Peano-only builds
if config.peano_only_build:
    config.available_features.add("peano_only_build")

# Detect ROCm/HSA and VCK5000
rocm_config = LitConfigHelper.detect_rocm(
    config.hsa_dir, config.aieHostTarget, config.enable_board_tests
)

# Detect XRT and Ryzen AI NPU devices
# Pass both Vitis and Peano components so NPU tests work in Peano-only builds
xrt_config = LitConfigHelper.detect_xrt(
    config.xrt_lib_dir,
    config.xrt_include_dir,
    config.xrt_bin_dir,
    config.aie_src_root,
    config.vitis_components,
    config.peano_components,
)

# Detect OpenCV
opencv_config = LitConfigHelper.detect_opencv(
    config.opencv_include_dir, config.opencv_lib_dir, config.opencv_libs
)

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

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "lit.cfg.py",
]

config.aie_tools_dir = os.path.join(config.aie_obj_root, "bin")

# Setup the PATH with all necessary tool directories
LitConfigHelper.prepend_path(llvm_config, config.aie_tools_dir)
if config.vitis_root:
    config.vitis_aietools_bin = os.path.join(config.vitis_aietools_dir, "bin")
    LitConfigHelper.prepend_path(llvm_config, config.vitis_aietools_bin)
    llvm_config.with_environment("VITIS", config.vitis_root)

# Prepend path to XRT installation, which contains a more recent `aiebu-asm` than the Vitis installation.
LitConfigHelper.prepend_path(llvm_config, config.xrt_bin_dir)

LitConfigHelper.prepend_path(llvm_config, config.llvm_tools_dir)
LitConfigHelper.prepend_path(llvm_config, peano_tools_dir)
config.substitutions.append(("%LLVM_TOOLS_DIR", config.llvm_tools_dir))

tool_dirs = [config.aie_tools_dir, config.llvm_tools_dir]

# Detect Chess compiler
chess_config = LitConfigHelper.detect_chess(
    config.vitis_root, config.enable_chess_tests, llvm_config
)

# Apply all hardware/tool configurations
LitConfigHelper.apply_config_to_lit(
    config,
    {
        "rocm": rocm_config,
        "xrt": xrt_config,
        "peano": peano_config,
        "chess": chess_config,
        "opencv": opencv_config,
    },
)

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
