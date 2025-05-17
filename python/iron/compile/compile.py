# compile.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import subprocess
import sys
import aie.compiler.aiecc.main as aiecc
import aie.utils.config as config


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
        config.peano_cxx_path(),
        source_path,
        "-c",
        "-o",
        f"{output_path}",
        f"-I{config.cxx_header_path()}",
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
        f"--peano={config.peano_install_dir()}",
        f"--pdi-name={pdi_path}",
        f"--npu-insts-name={insts_path}",
    ]
    if options:
        args = args + options
    try:
        aiecc.run(mlir_module, args)
    except Exception as e:
        raise RuntimeError("[aiecc] Compilation failed") from e
