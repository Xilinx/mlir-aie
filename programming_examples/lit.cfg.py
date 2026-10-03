# ./lit.cfg.py -*- Python -*-
#
# Copyright (C) 2022 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import hashlib
import os
import platform
import sys
from typing import TYPE_CHECKING, Any

# Add shared AIE lit utilities to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import lit.formats  # pyright: ignore[reportMissingImports]
from aie_lit_utils import LitConfigHelper  # pyright: ignore[reportMissingImports]
from lit.llvm import llvm_config  # pyright: ignore[reportMissingImports]

# ``config`` and ``lit_config`` are injected into this file's namespace by the
# lit runner at execution time; declare them under TYPE_CHECKING only so the
# type checker doesn't flag every reference as undefined.
if TYPE_CHECKING:
    config: Any = None
    lit_config: Any = None

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "AIE_PROGRAMMING_EXAMPLES"

config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".lit"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# Windows MAX_PATH (260) is the binding constraint for the cmake examples, and
# only about a third of the budget is ours. lit mirrors the source tree under
# test_exec_root, and CMake's compiler-ABI try-compile then appends ~92 fixed
# characters:
#
#   CMakeFiles/CMakeScratch/TryCompile-xxxxxx/CMakeFiles/cmTC_xxxxx.dir/
#   testCXXCompiler.cxx.obj
#
# Neither cl.exe nor rc.exe is long-path aware, so a deep-enough example fails
# to configure at all -- "fatal error C1083: Cannot open compiler generated
# file: ''" for CXX, "error RC2136" on manifest.rc for C. The deepest example
# lands at 266 characters under the CI checkout path, and the next-deepest
# clears the limit by two.
#
# The mirrored tree and CMake's tail are both fixed, so the only lever is the
# prefix. Rehome it at a short path keyed by a hash of the real one, which
# keeps concurrent build directories from colliding. That trades 68 characters
# of CI checkout path for 19 and puts every cmake lit ~43 under the limit.
if platform.system() == "Windows":
    _tmp_root = os.environ.get("AIE_LIT_TMP_ROOT")
    if not _tmp_root:
        _digest = hashlib.sha1(config.test_exec_root.encode()).hexdigest()[:8]
        _drive = os.path.splitdrive(config.test_exec_root)[0] or "C:"
        _tmp_root = os.path.join(_drive + os.sep, "aie-lit", _digest)
    config.test_exec_root = _tmp_root

# Setup standard environment (PYTHONPATH, AIETOOLS, system env, etc.)
LitConfigHelper.setup_standard_environment(
    llvm_config, config, config.aie_obj_root, config.vitis_aietools_dir
)

LitConfigHelper.add_makefile_examples_feature(config)
LitConfigHelper.add_cmake_examples_feature(config)

# Basic substitutions
config.substitutions.append(("%extraAieCcFlags%", config.extraAieCcFlags))
config.substitutions.append(
    ("%aie_runtime_lib%", os.path.join(config.aie_obj_root, "aie_runtime_lib"))
)
config.substitutions.append(
    (
        "%host_runtime_lib%",
        os.path.join(config.aie_obj_root, "runtime_lib", config.aieHostTarget),
    )
)
config.substitutions.append(("%aietools", config.vitis_aietools_dir))

# Not using run_on_board anymore, need more specific per-platform commands
config.substitutions.append(("%run_on_board", "echo"))

# VCK5000/HSA support has been removed; these substitutions are permanent
# no-ops kept so existing RUN lines referencing them keep working.
config.substitutions.append(("%run_on_vck5000", "echo"))
config.substitutions.append(("%link_against_hsa%", ""))
config.substitutions.append(("%HSA_DIR%", ""))

# Add Vitis components as features
LitConfigHelper.add_vitis_components_features(config, config.vitis_components)

# Detect Peano before XRT feature gating for systems without Chess/AIETOOLS
early_peano_tools_dir = os.path.join(config.peano_install_dir, "bin")
early_peano_config = LitConfigHelper.detect_peano(
    early_peano_tools_dir, config.peano_install_dir, llvm_config
)

# Detect OpenCV
opencv_config = LitConfigHelper.detect_opencv(
    config.opencv_include_dir, config.opencv_lib_dir, config.opencv_libs
)

if config.pytorch:
    config.available_features.add("torch")
    config.available_features.add("pytorch")

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

peano_tools_dir = os.path.join(config.peano_install_dir, "bin")
LitConfigHelper.prepend_path(llvm_config, config.llvm_tools_dir)
LitConfigHelper.prepend_path(llvm_config, peano_tools_dir)
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

# Peano may gate Ryzen AI features only when it is the active fallback backend.
can_use_peano_feature_gate = early_peano_config.found and not chess_config.found

# Detect XRT and Ryzen AI NPU devices
xrt_config = LitConfigHelper.detect_xrt(
    config.xrt_lib_dir,
    config.xrt_include_dir,
    config.xrt_bin_dir,
    config.aie_src_root,
    llvm_config,
    config.vitis_components,
    can_use_peano_feature_gate=can_use_peano_feature_gate,
)

# Apply all hardware/tool configurations
LitConfigHelper.apply_config_to_lit(
    config,
    {
        "xrt": xrt_config,
        "peano": peano_config,
        "chess": chess_config,
        "opencv": opencv_config,
    },
)

LitConfigHelper.setup_host_compiler_substitutions(config)
LitConfigHelper.setup_aiecc_substitution(config)
LitConfigHelper.setup_host_link_substitution(config)

tools = [
    "aie-opt",
    "aie-translate",
    "aiecc",
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
