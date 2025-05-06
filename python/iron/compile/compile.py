# compile.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import aie.compiler.aiecc.main as aiecc


def compile_mlir_module_to_pdi(
    mlir_module: str, insts_path: str, pdi_path: str, options: list[str]
):
    """
    Compile an MLIR module to instruction and PDI files using the aiecc module.

    Parameters:
        mlir_module (str): MLIR module to compile.
        insts_path (str): Path to the instruction binary file.
        pdi_path (str): Path to the PDI file.
        options (list[str]): List of additional options.
    """

    args = [
        "--aie-generate-pdi",
        "--aie-generate-npu-insts",
        f"--pdi-name={pdi_path}",
        f"--npu-insts-name={insts_path}",
    ] + options
    try:
        aiecc.run(mlir_module, args)
    except Exception as e:
        raise RuntimeError("[aiecc] Compilation failed") from e
