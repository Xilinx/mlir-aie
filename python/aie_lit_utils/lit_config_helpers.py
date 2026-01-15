# aie/test/utils/lit_config_helpers.py
# -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

"""
Shared utilities for AIE/AIR lit test configuration.
Consolidates hardware detection, path management, and common substitutions.

This module provides a centralized way to handle:
- Hardware detection (ROCm, XRT, NPU devices)
- Tool detection (Chess, Peano, aiesimulator)
- PATH management
- Common substitutions and features

Usage:
    from lit_config_helpers import LitConfigHelper

    helper = LitConfigHelper()
    rocm_config = helper.detect_rocm(config.hsa_dir, config.enable_board_tests)
    helper.apply_config_to_lit(config, {"rocm": rocm_config})
"""

import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class HardwareConfig:
    """
    Configuration for detected hardware or tools.

    Attributes:
        found: Whether the hardware/tool was detected
        features: List of lit features to add
        substitutions: Dictionary of lit substitutions
        flags: Compiler/linker flags for this hardware
        environment: Dictionary of environment variables to set
    """

    found: bool = False
    features: List[str] = field(default_factory=list)
    substitutions: Dict[str, str] = field(default_factory=dict)
    flags: str = ""
    environment: Dict[str, str] = field(default_factory=dict)


class LitConfigHelper:
    """Helper class for managing lit test configurations."""

    # NPU Model mappings - centralized for easy updates
    # Maps generation name to list of model strings that may appear in xrt-smi
    NPU_MODELS = {
        "npu1": ["npu1", "Phoenix"],
        "npu2": ["npu4", "Strix", "npu5", "Strix Halo", "npu6", "Krackan"],
    }

    @staticmethod
    def prepend_path(llvm_config, path: str) -> None:
        """
        Prepend a directory to the PATH environment variable.

        Args:
            llvm_config: LLVM lit config object
            path: Directory path to prepend
        """
        paths = [path]
        current_paths = llvm_config.config.environment.get("PATH", None)
        if current_paths:
            paths.extend(current_paths.split(os.path.pathsep))
        paths = [os.path.normcase(os.path.normpath(p)) for p in paths]
        llvm_config.config.environment["PATH"] = os.pathsep.join(paths)

    @staticmethod
    def detect_rocm(
        hsa_dir: str, aie_host_target: str, enable_board_tests: bool = False
    ) -> HardwareConfig:
        """
        Detect ROCm/HSA installation and VCK5000 hardware.

        Args:
            hsa_dir: Path to HSA runtime directory
            aie_host_target: Host target architecture (must contain 'hsa' for ROCm)
            enable_board_tests: Whether to enable board testing

        Returns:
            HardwareConfig with ROCm detection results
        """
        config = HardwareConfig()

        if not hsa_dir or "NOTFOUND" in hsa_dir:
            print("ROCm not found")
            config.substitutions = {
                "%run_on_vck5000": "echo",
                "%link_against_hsa%": "",
                "%HSA_DIR%": "",
            }
            return config

        if "hsa" not in aie_host_target:
            print(f"ROCm found, but disabled because host target {aie_host_target}")
            config.substitutions = {
                "%run_on_vck5000": "echo",
                "%link_against_hsa%": "",
                "%HSA_DIR%": "",
            }
            return config

        # Getting the path to the ROCm directory
        # hsa-runtime64 points to cmake dir, go up three directories
        rocm_root = os.path.abspath(os.path.join(hsa_dir, "..", "..", ".."))
        print(f"Found ROCm: {rocm_root}")

        config.found = True
        config.features.append("hsa")
        config.substitutions = {
            "%HSA_DIR%": rocm_root,
            "%link_against_hsa%": "--link_against_hsa",
        }

        # Check for VCK5000 hardware
        found_vck5000 = False
        if enable_board_tests:
            try:
                # Use experimental ROCm install that can see the AIE device
                env = os.environ.copy()
                env["LD_LIBRARY_PATH"] = f"{rocm_root}/lib/"
                result = subprocess.run(
                    ["rocminfo"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    timeout=10,
                )
                output = result.stdout.decode("utf-8", errors="ignore").split("\n")

                for line in output:
                    if "Versal VCK5000" in line:
                        print("Found VCK5000 in rocminfo. Enabling on board tests")
                        found_vck5000 = True
                        config.substitutions["%run_on_vck5000"] = (
                            "flock /tmp/vck5000.lock"
                        )
                        break

                if not found_vck5000:
                    print(
                        "Enable board set and HSA found but couldn't find device using rocminfo"
                    )
                    config.substitutions["%run_on_vck5000"] = "echo"
            except subprocess.TimeoutExpired:
                print("Enable board set and HSA found but rocminfo timed out")
                config.substitutions["%run_on_vck5000"] = "echo"
            except FileNotFoundError:
                print("Enable board set and HSA found but rocminfo not found")
                config.substitutions["%run_on_vck5000"] = "echo"
            except Exception as e:
                print(f"Enable board set and HSA found but unable to run rocminfo: {e}")
                config.substitutions["%run_on_vck5000"] = "echo"
        else:
            print("Skipping execution of unit tests (ENABLE_BOARD_TESTS=OFF)")
            config.substitutions["%run_on_vck5000"] = "echo"

        return config

    @staticmethod
    def detect_xrt(
        xrt_lib_dir: str,
        xrt_include_dir: str,
        xrt_bin_dir: str,
        aie_src_root: str,
        vitis_components: Optional[List[str]] = None,
    ) -> HardwareConfig:
        """
        Detect XRT installation and Ryzen AI NPU hardware.

        Args:
            xrt_lib_dir: Path to XRT library directory
            xrt_include_dir: Path to XRT include directory
            xrt_bin_dir: Path to XRT binary directory
            aie_src_root: Path to AIE source root (for run_on_npu.sh script)
            vitis_components: List of available Vitis components for feature filtering

        Returns:
            HardwareConfig with XRT detection results

            HardwareConfig contains:
                - found: True if XRT is detected and valid
                - flags: Compiler/linker flags for XRT (includes -I, -L, and libraries)
                - substitutions: Dictionary with "%xrt_flags", "%run_on_npu1%", and "%run_on_npu2%" mappings
                - features: List of features including "ryzen_ai", "ryzen_ai_npu1", or "ryzen_ai_npu2"
                           based on detected NPU hardware and available Vitis components
        """
        if vitis_components is None:
            vitis_components = []

        config = HardwareConfig()
        run_on_npu1 = "echo"
        run_on_npu2 = "echo"

        if not xrt_lib_dir:
            print("xrt not found")
            config.flags = ""
            config.substitutions["%xrt_flags"] = ""
            config.substitutions["%run_on_npu1%"] = run_on_npu1
            config.substitutions["%run_on_npu2%"] = run_on_npu2
            return config

        print(f"xrt found at {os.path.dirname(xrt_lib_dir)}")
        config.found = True
        config.flags = f"-I{xrt_include_dir} -L{xrt_lib_dir} -luuid -lxrt_coreutil"
        config.substitutions["%xrt_flags"] = config.flags
        # Add XRT library directory to LD_LIBRARY_PATH for runtime linking,
        # preserving any existing entries from the parent environment.
        existing_ld_library_path = os.environ.get("LD_LIBRARY_PATH")
        if existing_ld_library_path:
            new_ld_library_path = existing_ld_library_path + os.pathsep + xrt_lib_dir
        else:
            new_ld_library_path = xrt_lib_dir
        config.environment["LD_LIBRARY_PATH"] = new_ld_library_path

        # Detect NPU hardware
        try:
            xrtsmi = os.path.join(xrt_bin_dir, "xrt-smi")
            result = subprocess.run(
                [xrtsmi, "examine"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )
            output = result.stdout.decode("utf-8", errors="ignore").split("\n")

            # Pattern matches both old and new xrt-smi output formats:
            # Old: "|[0000:41:00.1]  ||RyzenAI-npu1  |"
            # New: "|[0000:41:00.1]  |NPU Phoenix  |"
            # New with spaces: "|[0000:c6:00.1]  |NPU Strix Halo  |"
            pattern = re.compile(
                r"[\|]?(\[.+:.+:.+\]).+\|(RyzenAI-(npu\d)|NPU ([\w ]+?))\s*\|"
            )

            for line in output:
                match = pattern.match(line)
                if not match:
                    continue

                device_id = match.group(1)
                print(f"Found Ryzen AI device: {device_id}")

                # Extract model name from either group 3 or 4
                model = "unknown"
                if match.group(3):
                    model = str(match.group(3))
                elif match.group(4):
                    model = str(match.group(4))

                print(f"\tmodel: '{model}'")

                run_on_npu = f"{aie_src_root}/utils/run_on_npu.sh"

                # Map model to NPU generation and filter by available components
                # Convert vitis_components to uppercase for case-insensitive comparison
                vitis_components_upper = [c.upper() for c in vitis_components]

                if model in LitConfigHelper.NPU_MODELS["npu1"]:
                    if "AIE2" in vitis_components_upper:
                        run_on_npu1 = run_on_npu
                        config.features.extend(["ryzen_ai", "ryzen_ai_npu1"])
                        config.substitutions["%run_on_npu1%"] = run_on_npu1
                        print(f"Running tests on NPU1 with command line: {run_on_npu1}")
                    else:
                        print("NPU1 detected but aietools for aie2 not available")
                elif model in LitConfigHelper.NPU_MODELS["npu2"]:
                    if "AIE2P" in vitis_components_upper:
                        run_on_npu2 = run_on_npu
                        config.features.extend(["ryzen_ai", "ryzen_ai_npu2"])
                        config.substitutions["%run_on_npu2%"] = run_on_npu2
                        print(f"Running tests on NPU2 with command line: {run_on_npu2}")
                    else:
                        print("NPU2 detected but aietools for aie2p not available")
                else:
                    print(f"WARNING: xrt-smi reported unknown NPU model '{model}'.")
                break

        except subprocess.TimeoutExpired:
            print("Failed to run xrt-smi (timeout)")
        except FileNotFoundError:
            print("Failed to run xrt-smi (not found)")
        except Exception as e:
            print(f"Failed to run xrt-smi: {e}")

        config.substitutions["%run_on_npu1%"] = run_on_npu1
        config.substitutions["%run_on_npu2%"] = run_on_npu2

        return config

    @staticmethod
    def detect_chess(
        vitis_root: str, enable_chess_tests: bool, llvm_config
    ) -> HardwareConfig:
        """
        Detect Chess compiler and validate license.

        Args:
            vitis_root: Path to Vitis installation
            enable_chess_tests: Whether chess tests are enabled in config
            llvm_config: LLVM lit config object for environment setup

        Returns:
            HardwareConfig with Chess detection results
        """
        config = HardwareConfig()

        if not enable_chess_tests:
            print("Chess tests disabled")
            return config

        print("Looking for Chess...")
        xchesscc_path = None
        if vitis_root:
            xchesscc_path = shutil.which("xchesscc")

        if not xchesscc_path:
            # Check if license exists anyway
            xilinxd_license_file = os.getenv("XILINXD_LICENSE_FILE")
            if xilinxd_license_file:
                print("Chess license found")
                llvm_config.with_environment(
                    "XILINXD_LICENSE_FILE", xilinxd_license_file
                )
            else:
                print("Chess not found")
            return config

        print(f"Chess found: {xchesscc_path}")
        config.found = True
        config.features.extend(["chess", "valid_xchess_license"])

        # Handle license files
        lm_license_file = os.getenv("LM_LICENSE_FILE")
        xilinxd_license_file = os.getenv("XILINXD_LICENSE_FILE")

        if lm_license_file:
            llvm_config.with_environment("LM_LICENSE_FILE", lm_license_file)
        if xilinxd_license_file:
            llvm_config.with_environment("XILINXD_LICENSE_FILE", xilinxd_license_file)

        if not (lm_license_file or xilinxd_license_file):
            print(
                "WARNING: no valid xchess license that is required by some of the lit tests"
            )

        return config

    @staticmethod
    def detect_peano(
        peano_tools_dir: str, peano_install_dir: str, llvm_config
    ) -> HardwareConfig:
        """
        Detect Peano backend availability and supported AIE architectures.

        Args:
            peano_tools_dir: Path to Peano tools directory
            peano_install_dir: Path to Peano installation root
            llvm_config: LLVM lit config object for environment setup

        Returns:
            HardwareConfig with Peano detection results including supported AIE architectures.
            The supported architectures are stored in config.substitutions["%peano_components%"]
            as a list that can be added to vitis_components.
        """
        config = HardwareConfig()

        try:
            llc_path = os.path.join(peano_tools_dir, "llc")
            result = subprocess.run(
                [llc_path, "-mtriple=aie", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )

            version_output = result.stdout.decode("utf-8", errors="ignore")
            if re.search("Xilinx AI Engine", version_output):
                config.found = True
                config.features.append("peano")
                config.substitutions["%PEANO_INSTALL_DIR"] = peano_install_dir
                # Also set environment variable for tests that need it
                llvm_config.with_environment("PEANO_INSTALL_DIR", peano_install_dir)
                print(f"Peano found: {llc_path}")

                # Detect supported AIE architectures by checking include directories
                # llvm-aie installed via pip will have include dirs like:
                # - aie2-none-unknown-elf/
                # - aie2p-none-unknown-elf/
                supported_components = []
                peano_include_dir = os.path.join(peano_install_dir, "include")

                if os.path.isdir(peano_include_dir):
                    # Check for AIE2 support
                    aie2_include = os.path.join(
                        peano_include_dir, "aie2-none-unknown-elf"
                    )
                    if os.path.isdir(aie2_include):
                        supported_components.append("AIE2")
                        config.features.append("peano_aie2")
                        print("  Peano supports AIE2")

                    # Check for AIE2P support
                    aie2p_include = os.path.join(
                        peano_include_dir, "aie2p-none-unknown-elf"
                    )
                    if os.path.isdir(aie2p_include):
                        supported_components.append("AIE2P")
                        config.features.append("peano_aie2p")
                        print("  Peano supports AIE2P")

                # Store supported components as a Python list string for lit config
                if supported_components:
                    config.substitutions["%peano_components%"] = str(
                        supported_components
                    )
                else:
                    config.substitutions["%peano_components%"] = "[]"

                return config
        except subprocess.TimeoutExpired:
            print(f"Peano detection timed out at {peano_tools_dir}")
        except FileNotFoundError:
            print(f"Peano not found, but expected at {peano_tools_dir}")
        except Exception as e:
            print(f"Peano detection failed: {e}")

        print(f"Peano not found, but expected at {peano_tools_dir}")
        config.substitutions["%peano_components%"] = "[]"
        return config

    @staticmethod
    def detect_aiesimulator(aie_obj_root) -> HardwareConfig:
        """
        Detect aiesimulator availability.

        Returns:
            HardwareConfig with aiesimulator detection results
        """
        config = HardwareConfig()
        sim_path = shutil.which("aiesimulator")

        if sim_path:
            print(f"aiesimulator found: {sim_path}")
            config.found = True
            config.features.append("aiesimulator")
            config.environment["LD_LIBRARY_PATH"] = "{}".format(
                os.path.join(aie_obj_root, "runtime_lib", "x86_64", "xaiengine", "lib")
            )
        else:
            print("aiesimulator not found")
        return config

    @staticmethod
    def detect_opencv(
        opencv_include_dir: str, opencv_lib_dir: str, opencv_libs: str
    ) -> HardwareConfig:
        """
        Detect OpenCV installation and generate compiler flags.

        Args:
            opencv_include_dir: Path to OpenCV include directory
            opencv_lib_dir: Path to OpenCV library directory (optional)
            opencv_libs: Semicolon-separated list of OpenCV libraries

        Returns:
            HardwareConfig with OpenCV detection results and flags
        """
        config = HardwareConfig()

        if not opencv_include_dir or not opencv_libs:
            print("opencv not found")
            config.substitutions["%opencv_flags"] = ""
            return config

        print("opencv found")
        config.found = True
        config.features.append("opencv")

        # Build compiler flags
        flags = f" -I{opencv_include_dir}"
        if opencv_lib_dir:
            flags += f" -L{opencv_lib_dir}"

        libs = opencv_libs.split(";")
        flags += " " + " ".join([f"-l{lib}" for lib in libs])

        config.substitutions["%opencv_flags"] = flags
        return config

    @staticmethod
    def setup_host_target_triplet(
        aie_host_target: str, vitis_sysroot: str = ""
    ) -> Tuple[str, str]:
        """
        Configure host target triplet and sysroot flags.

        Args:
            aie_host_target: Host target architecture
            vitis_sysroot: Path to Vitis sysroot (for cross-compilation)

        Returns:
            Tuple of (triplet_string, sysroot_flag)
        """
        if "x86_64" in aie_host_target:
            return "x86_64-unknown-linux-gnu", ""
        elif aie_host_target == "aarch64":
            sysroot_flag = f"--sysroot={vitis_sysroot}" if vitis_sysroot else ""
            return "aarch64-linux-gnu", sysroot_flag
        else:
            return aie_host_target, ""

    @staticmethod
    def apply_config_to_lit(config_obj, hardware_configs: Dict[str, HardwareConfig]):
        """
        Apply detected hardware configurations to lit config.

        Args:
            config_obj: Lit config object with available_features and substitutions
            hardware_configs: Dictionary mapping names to HardwareConfig objects
        """
        for name, hw_config in hardware_configs.items():
            # Add features
            for feature in hw_config.features:
                config_obj.available_features.add(feature)

            # Add substitutions
            for key, value in hw_config.substitutions.items():
                config_obj.substitutions.append((key, value))

            # Add environment variables
            for key, value in hw_config.environment.items():
                config_obj.environment[key] = value

    @staticmethod
    def setup_standard_environment(
        llvm_config, config_obj, aie_obj_root: str, vitis_aietools_dir: str
    ):
        """
        Set up standard environment variables and paths.

        Args:
            llvm_config: LLVM lit config
            config_obj: Config object
            aie_obj_root: AIE object root directory
            vitis_aietools_dir: Vitis AIE tools directory
        """
        # Python path for AIE Python bindings
        config_obj.environment["PYTHONPATH"] = os.path.join(aie_obj_root, "python")

        # AIE tools environment
        llvm_config.with_environment("AIETOOLS", vitis_aietools_dir)

        # Peano clang needs this
        llvm_config.with_environment("XILINX_VITIS_AIETOOLS", vitis_aietools_dir)

        # System environment variables
        llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

        # JIT cache for compiled designs
        llvm_config.with_system_environment("IRON_CACHE_HOME")

    @staticmethod
    def setup_test_lib_substitutions(
        config_obj, aie_obj_root: str, aie_host_target: str
    ):
        """
        Set up test library substitutions.

        Args:
            config_obj: Config object with substitutions
            aie_obj_root: AIE object root directory
            aie_host_target: Host target architecture
        """
        test_lib_path = os.path.join(
            aie_obj_root, "runtime_lib", aie_host_target, "test_lib"
        )
        config_obj.substitutions.append(
            (
                "%test_lib_flags",
                f"-I{test_lib_path}/include -L{test_lib_path}/lib -ltest_lib",
            )
        )
        config_obj.substitutions.append(
            (
                "%test_utils_flags",
                f"-I{test_lib_path}/include -L{test_lib_path}/lib -ltest_utils",
            )
        )

    @staticmethod
    def add_vitis_components_features(config_obj, vitis_components: List[str]):
        """
        Add Vitis component features.

        Args:
            config_obj: Config object with available_features
            vitis_components: List of Vitis component names
        """
        for component in vitis_components:
            config_obj.available_features.add(f"aietools_{component.lower()}")
