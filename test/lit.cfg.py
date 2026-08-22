# ./lit.cfg.py -*- Python -*-
#
# Copyright (C) 2021-2022 Xilinx, Inc.
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

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

# HRX discovery runs during lit configuration, but dispatch runs in test
# subprocesses. Keep both in the same provisioned runtime environment.
llvm_config.with_system_environment(
    [
        "AIE_XCLBINUTIL",
        "CMAKE_PREFIX_PATH",
        "HRX_DIR",
        "HRX_LIBHRX",
        "LIBHRX_DIR",
        "LD_LIBRARY_PATH",
    ]
)

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

# VCK5000/HSA support has been removed; these substitutions are permanent
# no-ops kept so existing RUN lines referencing them keep working.
config.substitutions.append(("%run_on_vck5000", "echo"))
config.substitutions.append(("%link_against_hsa%", ""))
config.substitutions.append(("%HSA_DIR%", ""))

# Add Vitis components as features
LitConfigHelper.add_vitis_components_features(config, config.vitis_components)
LitConfigHelper.setup_host_compiler_substitutions(config)

# Detect Peano before XRT feature gating for systems without Chess/AIETOOLS.
early_peano_tools_dir = os.path.join(config.peano_install_dir, "bin")
early_peano_config = LitConfigHelper.detect_peano(
    early_peano_tools_dir, config.peano_install_dir, llvm_config
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
LitConfigHelper.setup_aiecc_substitution(config)

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

# Detect aiesimulator
aiesim_config = LitConfigHelper.detect_aiesimulator(config.aie_obj_root)

# Apply all hardware/tool configurations
LitConfigHelper.apply_config_to_lit(
    config,
    {
        "xrt": xrt_config,
        "peano": peano_config,
        "chess": chess_config,
        "aiesim": aiesim_config,
    },
)

LitConfigHelper.setup_host_link_substitution(config)

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
    tools.extend(["txn2mlir.py"])

llvm_config.add_tool_substitutions(tools, tool_dirs)

if os.name == "nt":
    LitConfigHelper.add_python_tool_substitutions(config, ["txn2mlir.py"])

if config.enable_board_tests:
    lit_config.parallelism_groups["board"] = 1
    config.parallelism_group = "board"

# Concurrency tests control their own parallelism, so run them serially
lit_config.parallelism_groups["concurrency"] = 1

# NPU XRT tests should run serially to avoid resource contention
lit_config.parallelism_groups["npu-xrt"] = 1

# shutil.which picks up the platform's executable suffix (.exe on Windows
# via PATHEXT) so the feature gate fires correctly on every OS.
if shutil.which("aie-lsp-server", path=config.llvm_tools_dir) is not None:
    config.available_features.add("aie-lsp-server")

# The bundled XRT-free hrx-xclbinutil (-DAIE_BUILD_HRXXCLBINUTIL=ON) installs a
# `xclbinutil` into the AIE tools dir. Gate the packaging section-check test on
# its presence so the test only runs when that tool was actually built (and so
# the bare `xclbinutil` it invokes resolves to the bundled copy).
if shutil.which("xclbinutil", path=config.aie_tools_dir) is not None:
    config.available_features.add("hrxxclbinutil")

# aiebu ELF packager: gate tests that feed a runtime-assembled TXN blob to the
# real `aiebu-asm` (the downstream tool that enforces invariants plain XRT
# dispatch does not, e.g. block-write-covers-patch). It ships with XRT. Prefer
# the configured XRT bin dir (config.xrt_bin_dir, also prepended to PATH above),
# then PATH, then the standard install location so the feature still fires on a
# host that has XRT installed but was not built with it wired into cmake. The
# %aiebu_asm substitution wraps the binary with the LD_LIBRARY_PATH its shared
# libaiebu needs so RUN lines invoke it directly. Tests carry
# `// REQUIRES: aiebu` so they skip (not silently pass) where the tool is
# absent, e.g. on a CI runner without XRT installed.
_aiebu_asm = None
for _cand_dir in [config.xrt_bin_dir, None, "/opt/xilinx/xrt/bin"]:
    if _cand_dir == "":
        continue
    _aiebu_asm = shutil.which("aiebu-asm", path=_cand_dir)
    if _aiebu_asm is not None:
        break
if _aiebu_asm is not None:
    config.available_features.add("aiebu")
    # libaiebu lives next to the bin dir; use the configured XRT lib dir when
    # cmake provided one, else derive it from the located binary.
    _aiebu_libdir = config.xrt_lib_dir or os.path.join(
        os.path.dirname(os.path.dirname(_aiebu_asm)), "lib"
    )
    config.substitutions.append(
        (
            "%aiebu_asm",
            f"env LD_LIBRARY_PATH={_aiebu_libdir}:$LD_LIBRARY_PATH {_aiebu_asm}",
        )
    )

# HRX Python runtime: gate the HRX-only Python tests (test/python/npu-hrx) on
# libhrx being locatable, so they only run where the HRX backend can load. This
# checks a runtime value (aie.utils.has_hrx), not just importability.
if LitConfigHelper.python_expr_is_true(
    config, config.python_executable, "__import__('aie.utils').utils.has_hrx"
):
    config.available_features.add("hrx_python_bindings")

# The pure-HRX hardware job has no XRT probe. Its explicit NPU declaration
# enables only HRX hardware tests; XRT tests remain unsupported.
hrx_npu = os.environ.get("AIE_HRX_NPU")
if hrx_npu:
    if hrx_npu not in {"npu1", "npu2"}:
        lit_config.fatal(f"AIE_HRX_NPU must be 'npu1' or 'npu2', got {hrx_npu!r}")
    config.available_features.add(f"hrx_{hrx_npu}")
    if hrx_npu == "npu2":
        llvm_config.with_environment("NPU2", "1")

if config.xrt_python_bindings and LitConfigHelper.can_import_python_module(
    config, config.python_executable, "pyxrt"
):
    config.available_features.add("xrt_python_bindings")
    if LitConfigHelper.python_module_has_attribute(
        config,
        config.python_executable,
        "pyxrt",
        ("run", "get_ctrl_scratchpad_bo"),
    ):
        config.available_features.add("xrt_python_ctrl_scratchpad_bo")

if config.has_mlir_runtime_libraries:
    config.available_features.add("has_mlir_runtime_libraries")

if config.pytorch:
    config.available_features.add("pytorch")

# Per-backend, per-RUN-line device runners for the runtime-agnostic on-device
# Python tests (test/python/npu). lit's REQUIRES is file-level, but those tests
# carry both XRT and HRX RUN lines, and the two backends run on separate CI
# runners with mutually exclusive Python bindings (the pure-HRX runner has no
# pyxrt). So gate each RUN line via a dedicated substitution instead:
#   * %run_on_npu1_xrt% / %run_on_npu2_xrt% -> the XRT device wrapper only when
#     pyxrt is importable, else a no-op "echo" (so the XRT lines skip cleanly on
#     an XRT-free HRX host rather than dispatching CPU tensors on the NPU);
#   * %run_on_npu2_hrx% -> plain "env NPU_RUNTIME=hrx" when libhrx is locatable,
#     else a no-op "echo".
# The HRX substitution deliberately does NOT reuse the XRT %run_on_npu2% device
# wrapper: that wrapper probes for an *XRT* NPU and collapses to "echo" on the
# XRT-free pure-HRX runner, which would silently turn every HRX RUN line into a
# no-op (the tests would appear to pass without ever dispatching). The HRX
# hardware job instead declares its device explicitly via AIE_HRX_NPU (the
# hrx_npu2 feature above), matching the device-decoupled gating the pure-HRX
# npu-hrx tests use, so this line only needs the libhrx capability check.
_xrt_ok = "xrt_python_bindings" in config.available_features
_hrx_ok = "hrx_python_bindings" in config.available_features
_run_on_npu1 = xrt_config.substitutions.get("%run_on_npu1%", "echo")
_run_on_npu2 = xrt_config.substitutions.get("%run_on_npu2%", "echo")
config.substitutions.append(("%run_on_npu1_xrt%", _run_on_npu1 if _xrt_ok else "echo"))
config.substitutions.append(("%run_on_npu2_xrt%", _run_on_npu2 if _xrt_ok else "echo"))
config.substitutions.append(
    ("%run_on_npu2_hrx%", "env NPU_RUNTIME=hrx" if _hrx_ok else "echo")
)

if "LIT_AVAILABLE_FEATURES" in os.environ:
    for feature in os.environ["LIT_AVAILABLE_FEATURES"].split():
        config.available_features.add(feature)
