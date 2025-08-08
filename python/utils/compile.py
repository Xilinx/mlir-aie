# compile.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import os
import aie.compiler.aiecc.main as aiecc


def compile_mlir_module_to_binary(
    mlir_module: str, inst_path: str, xclbin_path: str, work_dir: str = None
):
    """
    Compile an MLIR module to instruction and xclbin files using the aiecc module.

    Parameters:
        mlir_module (str): MLIR module to compile.
        inst_path (str): Path to the instruction binary file.
        xclbin_path (str): Path to the xclbin file.
        work_dir (str, optional): Working directory for compilation. Defaults to None.
    """

    args = [
        "--aie-generate-xclbin",
        "--aie-generate-npu-insts",
        "--no-compile-host",
        "--no-xchesscc",
        "--no-xbridge",
        f"--xclbin-name={xclbin_path}",
        f"--npu-insts-name={inst_path}",
    ]

    # Add working directory if specified
    if work_dir:
        args.append(f"--tmpdir={work_dir}")

    try:
        aiecc.run(mlir_module, args)
    except Exception as e:
        raise RuntimeError(f"[aiecc] Compilation failed:\n{e}")
