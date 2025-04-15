# compile.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import subprocess
import os
import sys


def compile_mlir_to_binary(
    mlir_path: str, inst_filename: str, xclbin_filename: str, debug: bool = False
):
    """
    Compile an MLIR file to instruction and xclbin files using aiecc.py.

    Parameters:
        mlir_path (str): Path to the MLIR input file.
        inst_filename (str): Name of the instruction binary file (e.g., 'inst.bin').
        xclbin_filename (str): Name of the xclbin file (e.g., 'final.xclbin').
        debug (bool): If True, print the commands being executed. Default is False.
    """

    mlir_dir = os.path.dirname(os.path.abspath(mlir_path))

    cmd = [
        "aiecc.py",
        "--aie-generate-xclbin",
        "--aie-generate-npu-insts",
        "--no-compile-host",
        "--no-xchesscc",
        "--no-xbridge",
        f"--xclbin-name={xclbin_filename}",
        f"--npu-insts-name={inst_filename}",
        "aie.mlir",
    ]

    try:
        subprocess.run(
            cmd,
            cwd=mlir_dir,
            check=True,
            stdout=sys.stdout if debug else subprocess.DEVNULL,
            stderr=sys.stderr if debug else subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"[aiecc] Compilation failed:\n{e}")
