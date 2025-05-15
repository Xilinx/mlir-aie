# compile.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import os
import subprocess
import sys
import aie.compiler.aiecc.main as aiecc

# find Peano compiler
peano_install_dir = os.getenv("PEANO_INSTALL_DIR")
if not peano_install_dir or not os.path.isdir(peano_install_dir):
    raise RuntimeError("PEANO_INSTALL_DIR is not defined or does not exist")
peano_cxx = os.path.join(peano_install_dir, "bin/clang++")
if not os.path.isfile(peano_cxx):
    raise RuntimeError(f"Peano compiler not found in {peano_install_dir}")

# find MLIR-AIE C++ headers
mlir_aie_install_dir = os.getenv("MLIR_AIE_INSTALL_DIR")
if not mlir_aie_install_dir or not os.path.isdir(mlir_aie_install_dir):
    raise RuntimeError("MLIR_AIE_INSTALL_DIR is not defined or does not exist")
mlir_aie_include_dir = os.path.join(mlir_aie_install_dir, "include")
if not os.path.isdir(mlir_aie_include_dir):
    raise RuntimeError(f"MLIR-AIE headers not found in {mlir_aie_include_dir}")


def compile_cxx_core_function(
    source_path: str,
    target_arch: str,
    output_path: str,
    compile_args=None,
    cwd=None,
):
    """
    Compile a C++ core function.

    This function supports only the Peano compiler.

    Parameters:
        source_path (str): Path to C++ source.
        target_arch (str): Target architecture, e.g., aie2.
        output_path (str): Output object file path.
        compile_args (list[str]): Compile arguments to peano.
        cwd (str): Overrides the current working directory.
    """
    cmd = [
        peano_cxx,
        source_path,
        "-c",
        "-o",
        f"{output_path}",
        f"-I{mlir_aie_include_dir}",
        "-std=c++20",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-O2",
        "-DNDEBUG",
        f"--target={target_arch}-none-unknown-elf",
    ]
    if compile_args:
        cmd = cmd + compile_args
    try:
        subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError("[Peano] compilation failed") from e


def compile_mlir_module_to_pdi(
    mlir_module: str, insts_path: str, pdi_path: str, options=None
):
    """
    Compile an MLIR module to instruction and PDI files using the aiecc module.

    This function supports only the Peano compiler.

    Parameters:
        mlir_module (str): MLIR module to compile.
        insts_path (str): Path to the instruction binary file.
        pdi_path (str): Path to the PDI file.
        options (list[str]): List of additional options.
    """

    args = [
        "--aie-generate-pdi",
        "--aie-generate-npu-insts",
        "--no-compile-host",
        "--no-xchesscc",
        "--no-xbridge",
        f"--peano={peano_install_dir}",
        f"--pdi-name={pdi_path}",
        f"--npu-insts-name={insts_path}",
    ]
    if options:
        args = args + options
    try:
        aiecc.run(mlir_module, args)
    except Exception as e:
        raise RuntimeError("[aiecc] Compilation failed") from e
