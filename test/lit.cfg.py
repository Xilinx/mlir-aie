# ./lit.cfg.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021-2026 Xilinx Inc.

import os
import shutil
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

# Forward the install prefix when tests are run against wheel-installed tools.
if "MLIR_AIE_INSTALL_DIR" in os.environ:
    llvm_config.with_system_environment("MLIR_AIE_INSTALL_DIR")

# Basic substitutions
config.substitutions.append(("%PYTHON", config.python_executable))
config.substitutions.append(("%extraAieCcFlags%", config.extraAieCcFlags))
config.substitutions.append(
    ("%aie_runtime_lib%", os.path.join(config.aie_obj_root, "aie_runtime_lib"))
)
config.substitutions.append(("%aietools", config.vitis_aietools_dir))
# Show only failures
config.substitutions.append(("%pytest", "pytest -rA"))

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

# Add Vitis components as features
LitConfigHelper.add_vitis_components_features(config, config.vitis_components)
# Host-side tests should use the host LLVM compiler instead of llvm-aie.
host_clang = os.path.join(config.llvm_tools_dir, f"clang{config.llvm_exe_ext}")
if not os.path.exists(host_clang):
    host_clang = shutil.which("clang") or "clang"
config.substitutions.append(("%host_clang", LitConfigHelper._quote_lit_arg(host_clang)))


# Detect Peano before XRT feature gating for systems without Chess/AIETOOLS.
early_peano_tools_dir = os.path.join(config.peano_install_dir, "bin")
early_peano_config = LitConfigHelper.detect_peano(
    early_peano_tools_dir, config.peano_install_dir, llvm_config
)

# Detect XRT and Ryzen AI NPU devices
xrt_config = LitConfigHelper.detect_xrt(
    config.xrt_lib_dir,
    config.xrt_include_dir,
    config.xrt_bin_dir,
    config.aie_src_root,
    config.vitis_components,
    has_peano_backend=early_peano_config.found,
)

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
    "Inputs",
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
# Keep generic tool substitutions working by making both Peano tools and host
# LLVM tools discoverable. Host-side tests use %host_clang instead of relying on
# PATH order to choose the host compiler.
LitConfigHelper.prepend_path(llvm_config, config.llvm_tools_dir)
LitConfigHelper.prepend_path(llvm_config, peano_tools_dir)
LitConfigHelper.prepend_path(llvm_config, config.aie_tools_dir)
config.substitutions.append(("%LLVM_TOOLS_DIR", config.llvm_tools_dir))

tool_dirs = [config.aie_tools_dir]
if early_peano_config.found:
    tool_dirs.append(peano_tools_dir)
tool_dirs.append(config.llvm_tools_dir)

# Reuse the earlier Peano probe after path setup.
peano_config = early_peano_config

# Detect Chess compiler
chess_config = LitConfigHelper.detect_chess(
    config.vitis_root, config.enable_chess_tests, llvm_config
)

# Detect aiesimulator
aiesim_config = LitConfigHelper.detect_aiesimulator(config.aie_obj_root)

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

# Keep generic npu-xrt tests on the existing Chess path when Chess is available,
# but steer them onto the Peano/lld path when Peano is the only AIE backend.
aiecc_backend_flags = ""
if "peano" in config.available_features and "chess" not in config.available_features:
    aiecc_backend_flags = "--no-xchesscc --no-xbridge"
config.substitutions.append(("%aiecc_backend_flags", aiecc_backend_flags))

# Linux hosted tests need librt/libstdc++. Windows hosted tests link against
# CMake-built dynamic MSVC libraries. Match CMake's default /MD selection.
if os.name == "nt":
    host_link_flags = " ".join(
        [
            "-fms-runtime-lib=dll",
            "-Xlinker",
            "/NODEFAULTLIB:libucrt",
            "-Xlinker",
            "/DEFAULTLIB:ucrt",
        ]
    )
else:
    host_link_flags = "-lrt -lstdc++"
config.substitutions.append(("%host_link_flags", host_link_flags))

tools = [
    "aie-opt",
    "aie-translate",
    "ld.lld",
    "llc",
    "llvm-objdump",
    "opt",
    "xchesscc_wrapper",
]

if os.name != "nt":
    tools.extend(["aiecc.py", "txn2mlir.py"])

llvm_config.add_tool_substitutions(tools, tool_dirs)

if os.name == "nt":
    # Lit on Windows struggles with substituting tools with a .py extension.
    # Add these manually and quote them so paths with spaces survive.
    config.substitutions.append(
        (
            "aiecc.py",
            LitConfigHelper._quote_lit_arg(
                os.path.join(config.aie_tools_dir, "aiecc.py")
            ),
        )
    )
    config.substitutions.append(
        (
            "txn2mlir.py",
            LitConfigHelper._quote_lit_arg(
                os.path.join(config.aie_tools_dir, "txn2mlir.py")
            ),
        )
    )

if config.enable_board_tests:
    lit_config.parallelism_groups["board"] = 1
    config.parallelism_group = "board"

# Concurrency tests control their own parallelism, so run them serially
lit_config.parallelism_groups["concurrency"] = 1

# NPU XRT tests should run serially to avoid resource contention
lit_config.parallelism_groups["npu-xrt"] = 1

if config.python_passes:
    config.available_features.add("python_passes")

if config.xrt_python_bindings and LitConfigHelper.can_import_python_module(
    config, config.python_executable, "pyxrt"
):
    config.available_features.add("xrt_python_bindings")

if config.has_mlir_runtime_libraries:
    config.available_features.add("has_mlir_runtime_libraries")

if config.pytorch:
    config.available_features.add("pytorch")

if "LIT_AVAILABLE_FEATURES" in os.environ:
    for feature in os.environ["LIT_AVAILABLE_FEATURES"].split():
        config.available_features.add(feature)
